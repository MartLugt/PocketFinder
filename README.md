# PocketFinder
## ML4CHEM

---

## Data

I have not included the training dataset (scPDB) as it is ±30 GB. It can be downloaded from http://bioinfo-pharma.u-strasbg.fr/scPDB/.

The coach420 dataset is included (under `data/p2rank-datasets`), and models can be evaluated using the `eval_coach420.py` notebook.

Model definition and training is in `train.py`, which is a copy of model 15 in `./models` and `./models/submission_script.py`. Weights for this final model are in `./models/submission_weights.pt` and `./models/15_18052026.pt`.

## Files

| File                       | Description                                                                  |
|----------------------------|------------------------------------------------------------------------------|
| `train.py`                 | Main training script                                                         |
| `models/`                  | Model files (`.py` architecture + `.pt` weights + `.csv` training logs)      |
| `eval_scpdb.ipynb`         | Evaluate model on scPDB test split (F1/F2, PR curve, DCA)                    |
| `eval_coach420.ipynb`      | Evaluate model on COACH420 benchmark (F1/F2, DCA)                            |
| `infer_single.ipynb`       | Run inference on a single PDB file                                           |
| `infer_single_single.ipynb` | Run inference on a single PDB file and output ChimeraX string for selection. |
| `simple_figs.ipynb`        | Figures for the report (training curves, ablation, DCA comparison)           |
| `sample.ipynb`             | Scratch / exploratory notebook                                               |
