# PocketFinder
## ML4CHEM

---

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
