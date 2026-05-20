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

#                        hydro   charge  polar   arom    mw      flex    hbd     hba     pI
PHYSCHEM = {
    "ALA": np.array([    1.8,    0.0,    0.0,    0.0,    89.0,   0.0,    0.0,    0.0,    6.0 ]),
    "ARG": np.array([   -4.5,    1.0,    1.0,    0.0,   174.0,   1.0,    5.0,    1.0,   10.8 ]),
    "ASN": np.array([   -3.5,    0.0,    1.0,    0.0,   132.0,   0.0,    2.0,    2.0,    5.4 ]),
    "ASP": np.array([   -3.5,   -1.0,    1.0,    0.0,   133.0,   0.0,    1.0,    3.0,    2.8 ]),
    "CYS": np.array([    2.5,    0.0,    0.0,    0.0,   121.0,   0.0,    1.0,    1.0,    5.1 ]),
    "GLN": np.array([   -3.5,    0.0,    1.0,    0.0,   146.0,   1.0,    2.0,    2.0,    5.7 ]),
    "GLU": np.array([   -3.5,   -1.0,    1.0,    0.0,   147.0,   1.0,    1.0,    3.0,    3.2 ]),
    "GLY": np.array([   -0.4,    0.0,    0.0,    0.0,    75.0,   0.0,    0.0,    0.0,    6.0 ]),
    "HIS": np.array([   -3.2,    0.0,    1.0,    1.0,   155.0,   0.0,    2.0,    1.0,    7.6 ]),
    "ILE": np.array([    4.5,    0.0,    0.0,    0.0,   131.0,   0.0,    0.0,    0.0,    6.0 ]),
    "LEU": np.array([    3.8,    0.0,    0.0,    0.0,   131.0,   0.0,    0.0,    0.0,    6.0 ]),
    "LYS": np.array([   -3.9,    1.0,    1.0,    0.0,   146.0,   1.0,    3.0,    1.0,    9.7 ]),
    "MET": np.array([    1.9,    0.0,    0.0,    0.0,   149.0,   1.0,    0.0,    1.0,    5.7 ]),
    "PHE": np.array([    2.8,    0.0,    0.0,    1.0,   165.0,   0.0,    0.0,    0.0,    5.5 ]),
    "PRO": np.array([   -1.6,    0.0,    0.0,    0.0,   115.0,   0.0,    0.0,    0.0,    6.3 ]),
    "SER": np.array([   -0.8,    0.0,    1.0,    0.0,   105.0,   0.0,    1.0,    2.0,    5.7 ]),
    "THR": np.array([   -0.7,    0.0,    1.0,    0.0,   119.0,   0.0,    1.0,    2.0,    5.6 ]),
    "TRP": np.array([   -0.9,    0.0,    1.0,    1.0,   204.0,   0.0,    2.0,    1.0,    5.9 ]),
    "TYR": np.array([   -1.3,    0.0,    1.0,    1.0,   181.0,   0.0,    2.0,    2.0,    5.7 ]),
    "VAL": np.array([    4.2,    0.0,    0.0,    0.0,   117.0,   0.0,    0.0,    0.0,    6.0 ]),
}

# total input feature dimension: 9 physchem only
IN_DIM = 9


# ── parsing ───────────────────────────────────────────────────────────────────

def parse_mol2_protein(path):
    coords, residue_names = [], []
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
                    coords.append([x, y, z])
                    residue_names.append(res_name)

    coords = np.array(coords, dtype=np.float32)

    # physicochemical features only (9)
    physchem = np.array([PHYSCHEM[r] for r in residue_names], dtype=np.float32)

    def norm(x):
        std = x.std(axis=0)
        std[std < 1e-8] = 1.0
        return (x - x.mean(axis=0)) / std

    h = norm(physchem)  # (N, 9)

    pos = torch.tensor(coords, dtype=torch.float32)
    h   = torch.tensor(h,      dtype=torch.float32)
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
    def __init__(self, in_dim: int = IN_DIM, hidden: int = 128, n_layers: int = 3, dropout: float = 0.0):
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
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden) for _ in range(n_layers)
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

        for conv, norm in zip(self.convs, self.norms):
            h = norm(h + conv(h, edge_index, edge_attr))

        return self.head(h).squeeze(-1)


# ── evaluation ────────────────────────────────────────────────────────────────

def evaluate(model, subset, device, beta=2.0):
    model.eval()
    all_loss, all_f1, all_f2, all_p, all_r = [], [], [], [], []

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
            f2 = (1 + beta**2) * precision * recall / (beta**2 * precision + recall + 1e-8)

            all_f1.append(f1.item())
            all_f2.append(f2.item())
            all_p.append(precision.item())
            all_r.append(recall.item())

    return (
        sum(all_loss) / len(all_loss),
        sum(all_f1)   / len(all_f1),
        sum(all_f2)   / len(all_f2),
        sum(all_p)    / len(all_p),
        sum(all_r)    / len(all_r),
    )


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",    type=str,   default="./data/scPDB")
    parser.add_argument("--epochs",      type=int,   default=100)
    parser.add_argument("--hidden",      type=int,   default=128)
    parser.add_argument("--n_layers",    type=int,   default=3)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--dropout",     type=float, default=0.0)
    parser.add_argument("--beta",        type=float, default=2.0,  help="F-beta weight for recall")
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
        in_dim=IN_DIM,
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
        writer.writerow(["epoch", "train_loss", "val_loss", "val_f1", "val_f2", "val_precision", "val_recall"])

    best_f2   = 0.0
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
        val_loss, val_f1, val_f2, val_p, val_r = evaluate(model, val_set, device, beta=args.beta)

        print(
            f"epoch {epoch:3d}  "
            f"train_loss {train_loss:.4f}  "
            f"val_loss {val_loss:.4f}  "
            f"val_f1 {val_f1:.3f}  "
            f"val_f2 {val_f2:.3f}  "
            f"P {val_p:.3f}  R {val_r:.3f}",
            flush=True,
        )

        # append to csv immediately
        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, val_loss, val_f1, val_f2, val_p, val_r])

        # save best model by val_f2 (recall-weighted)
        if val_f2 > best_f2:
            best_f2 = val_f2
            torch.save(model.state_dict(), out_dir / "best_model.pt")
            print(f"  → saved best model (val_f2={val_f2:.3f})", flush=True)

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
    test_loss, test_f1, test_f2, test_p, test_r = evaluate(model, test_set, device, beta=args.beta)
    print(
        f"\ntest_loss {test_loss:.4f}  "
        f"test_f1 {test_f1:.3f}  "
        f"test_f2 {test_f2:.3f}  "
        f"P {test_p:.3f}  R {test_r:.3f}",
        flush=True,
    )

    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["test", test_loss, "", test_f1, test_f2, test_p, test_r])


if __name__ == "__main__":
    main()
