import argparse
import csv
import random
from collections import defaultdict
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torch_geometric.nn import GINEConv, knn_graph
from torch_geometric.data import Data, Batch


# ── constants ─────────────────────────────────────────────────────────────────

AMINO_ACIDS = [
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
]
AA_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}


# ── parsing ───────────────────────────────────────────────────────────────────

def parse_mol2_protein(path):
    coords, features = [], []
    in_atom_block = False
    seen_residues = {}

    with open(path) as f:
        for line in f:
            if line.startswith("@<TRIPOS>ATOM"):
                in_atom_block = True
                continue
            if line.startswith("@<TRIPOS>"):
                in_atom_block = False
                continue
            if not in_atom_block:
                continue

            parts = line.split()
            atom_name = parts[1]
            x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
            res_id = parts[6]
            res_name = parts[7][:3].upper()

            if atom_name == "CA" and res_name in AA_TO_IDX:
                if res_id not in seen_residues:
                    seen_residues[res_id] = True
                    one_hot = np.zeros(20)
                    one_hot[AA_TO_IDX[res_name]] = 1.0
                    coords.append([x, y, z])
                    features.append(one_hot)

    pos = torch.tensor(np.array(coords), dtype=torch.float32)
    h = torch.tensor(np.array(features), dtype=torch.float32)
    return pos, h


def parse_mol2_coords(path):
    coords = []
    in_atom_block = False

    with open(path) as f:
        for line in f:
            if line.startswith("@<TRIPOS>ATOM"):
                in_atom_block = True
                continue
            if line.startswith("@<TRIPOS>"):
                in_atom_block = False
                continue
            if not in_atom_block:
                continue

            parts = line.split()
            x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
            coords.append([x, y, z])

    return torch.tensor(np.array(coords), dtype=torch.float32)


# ── dataset ───────────────────────────────────────────────────────────────────

def _load_entry(args):
    entry, threshold = args
    pos, h = parse_mol2_protein(entry / "protein.mol2")
    site_coords = parse_mol2_coords(entry / "site.mol2")
    dists = torch.cdist(pos, site_coords)
    labels = (dists.min(dim=1).values < threshold).float()
    # return numpy to avoid shared memory file descriptor limits
    return h.numpy(), pos.numpy(), labels.numpy()


class ScPDBDataset(Dataset):
    def __init__(self, root_dir: str, cache_workers: int = 8, threshold: float = 1.0):
        self.entries = [
            d for d in Path(root_dir).iterdir()
            if (d / "protein.mol2").exists() and (d / "site.mol2").exists()
        ]
        print(f"caching {len(self.entries)} entries with {cache_workers} workers...", flush=True)
        args = [(entry, threshold) for entry in self.entries]
        with Pool(cache_workers) as pool:
            self.cache = pool.map(_load_entry, args)
        print("caching done", flush=True)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        h, pos, labels = self.cache[idx]
        return torch.from_numpy(h), torch.from_numpy(pos), torch.from_numpy(labels)


def split_dataset(dataset, val_frac=0.1, test_frac=0.1, seed=42):
    random.seed(seed)

    pdb_groups = defaultdict(list)
    for i, entry in enumerate(dataset.entries):
        pdb_id = entry.name[:4]
        pdb_groups[pdb_id].append(i)

    pdb_ids = list(pdb_groups.keys())
    random.shuffle(pdb_ids)

    n = len(pdb_ids)
    n_test = int(test_frac * n)
    n_val  = int(val_frac * n)

    test_ids  = set(pdb_ids[:n_test])
    val_ids   = set(pdb_ids[n_test:n_test + n_val])
    train_ids = set(pdb_ids[n_test + n_val:])

    train_indices = [i for pdb_id in train_ids for i in pdb_groups[pdb_id]]
    val_indices   = [i for pdb_id in val_ids   for i in pdb_groups[pdb_id]]
    test_indices  = [i for pdb_id in test_ids  for i in pdb_groups[pdb_id]]

    return (
        Subset(dataset, train_indices),
        Subset(dataset, val_indices),
        Subset(dataset, test_indices),
    )


def collate_fn(batch):
    data_list = [Data(x=h, pos=pos, y=labels) for h, pos, labels in batch]
    batched = Batch.from_data_list(data_list)
    return batched.x, batched.pos, batched.y, batched.batch


# ── model ─────────────────────────────────────────────────────────────────────

class PocketFinder(nn.Module):
    def __init__(self, in_dim: int = 20, hidden: int = 128, n_layers: int = 3, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Linear(in_dim, hidden)
        self.convs = nn.ModuleList([
            GINEConv(
                nn=nn.Sequential(
                    nn.Linear(hidden, hidden),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden, hidden),
                ),
                edge_dim=1,
            )
            for _ in range(n_layers)
        ])
        self.head = nn.Sequential(
            nn.Linear(hidden, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )
        # initialise output bias for ~10% positive rate: log(0.1/0.9)
        self.head[-1].bias.data.fill_(-2.2)

    def forward(self, h, pos, batch):
        h = self.dropout(self.embedding(h))

        edge_index = knn_graph(pos, k=10, batch=batch, loop=False)
        row, col = edge_index
        edge_attr = (pos[row] - pos[col]).norm(dim=-1, keepdim=True)

        for conv in self.convs:
            h = h + conv(h, edge_index, edge_attr)

        return self.head(h).squeeze(-1)


# ── evaluation ────────────────────────────────────────────────────────────────

def evaluate(model, subset, device):
    model.eval()
    all_f1, all_loss = [], []

    with torch.no_grad():
        for h, pos, labels in subset:
            h, pos, labels = h.to(device), pos.to(device), labels.to(device)
            batch = torch.zeros(len(pos), dtype=torch.long, device=device)
            logits = model(h, pos, batch)

            loss = F.binary_cross_entropy_with_logits(
                logits, labels, pos_weight=torch.tensor(6.0, device=device)
            )
            all_loss.append(loss.item())

            probs = logits.sigmoid()
            preds = (probs > 0.5).float()
            tp = (preds * labels).sum()
            fp = (preds * (1 - labels)).sum()
            fn = ((1 - preds) * labels).sum()
            precision = tp / (tp + fp + 1e-8)
            recall    = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            all_f1.append(f1.item())

    return sum(all_loss) / len(all_loss), sum(all_f1) / len(all_f1)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",    type=str,   default="./data/scPDB")
    parser.add_argument("--epochs",      type=int,   default=100)
    parser.add_argument("--hidden",      type=int,   default=128)
    parser.add_argument("--n_layers",    type=int,   default=3)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--dropout",     type=float, default=0.0)
    parser.add_argument("--patience",    type=int,   default=20)
    parser.add_argument("--pos_weight",  type=float, default=6.0)
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--num_workers",   type=int,   default=4)
    parser.add_argument("--cache_workers", type=int,   default=8)
    parser.add_argument("--batch_size",    type=int,   default=1)
    parser.add_argument("--out_dir",       type=str,   default="./output")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)

    # data
    dataset = ScPDBDataset(args.data_dir, cache_workers=args.cache_workers)
    train_set, val_set, test_set = split_dataset(
        dataset, val_frac=0.1, test_frac=0.1, seed=args.seed
    )
    print(f"train: {len(train_set)}  val: {len(val_set)}  test: {len(test_set)}", flush=True)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        persistent_workers=True,
        pin_memory=True,
    )

    # model
    model = PocketFinder(
        in_dim=len(AMINO_ACIDS),
        hidden=args.hidden,
        n_layers=args.n_layers,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    pos_weight = torch.tensor(args.pos_weight, device=device)

    # csv log — written after every epoch so it's inspectable while running
    log_path = out_dir / "metrics.csv"
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_f1"])

    best_f1   = 0.0
    best_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0

        for h, pos, labels, batch in train_loader:
            h      = h.to(device)
            pos    = pos.to(device)
            labels = labels.to(device)
            batch  = batch.to(device)

            optimizer.zero_grad()
            logits = model(h, pos, batch)
            loss = F.binary_cross_entropy_with_logits(
                logits, labels, pos_weight=pos_weight
            )
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)
        val_loss, val_f1 = evaluate(model, val_set, device)

        print(
            f"epoch {epoch:3d}  "
            f"train_loss {train_loss:.4f}  "
            f"val_loss {val_loss:.4f}  "
            f"val_f1 {val_f1:.3f}",
            flush=True,
        )

        # append to csv immediately
        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, val_loss, val_f1])

        # save best model by val_f1
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), out_dir / "best_model.pt")
            print(f"  → saved best model (val_f1={val_f1:.3f})", flush=True)

        # early stopping by val_loss
        if val_loss < best_loss:
            best_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            print(f"  no loss improvement ({epochs_without_improvement}/{args.patience})", flush=True)
            if epochs_without_improvement >= args.patience:
                print(f"early stopping at epoch {epoch}", flush=True)
                break

    # final test evaluation with best model
    model.load_state_dict(torch.load(out_dir / "best_model.pt"))
    test_loss, test_f1 = evaluate(model, test_set, device)
    print(f"\ntest_loss {test_loss:.4f}  test_f1 {test_f1:.3f}", flush=True)

    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["test", test_loss, "", test_f1])


if __name__ == "__main__":
    main()

