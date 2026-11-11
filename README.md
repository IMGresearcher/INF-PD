# INF-PD: Connectome-guided motor phenotype learning

Offical implementation of a clinically informed two-stage motor-aware fusion framework for PD diagnosis using multimodal structural connectomes. The repository contains the five user-provided Python modules.

## Overview

- **Stage I:** graph Transformer using continuous connectome edge weights, motor-aware fusion, and multi-task regression of five motor sub-domain scores.
- **Stage II:** diagnosis classification (Control/Prodromal/PD, including three pairwise comparisons); total MDS-UPDRS Part III score regression; and Hoehn–Yahr pairwise classification (0/1, 1/2, 0/2).
- **CV:** participant-level fold separation (`PATNO`) and out-of-fold (OOF) prediction files. Training of one Stage I fold uses only the other folds' participants.

## Repository layout

```text
INF-PD-GitHub/
├── model_connectome.py          # Stage-I graph neural network
├── train_stage1.py             # Stage-I five-fold training and OOF prediction
├── train_stage2_diagnosis.py   # Stage-II diagnosis and pairwise labels
├── train_stage2_total.py       # Stage-II total score regression
├── train_stage2_hy.py          # Stage-II Hoehn–Yahr pairwise classification
├── tests/                      # Synthetic unit tests (no medical data)
├── requirements.txt
├── requirements-dev.txt
├── pyproject.toml
├── .gitignore
└── LICENSE
```

## Installation

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
# or: pip install -e .
```

Install a PyTorch wheel compatible with your CPU/CUDA environment. For syntax, imports, and synthetic tests:

```bash
pip install -r requirements-dev.txt
python -m pytest -q
python train_stage1.py --help
```

## Required private data

Place **deidentified**, authorized data under `data/` (ignored by Git). The Stage-I script expects the original subject directory layout used by the supplied training script: subject folders containing `DTI_B2000_64dir_PA/<scan>/tract_out*/tracks_500k_ifod2.tck`, with corresponding tractography-derived connectivity matrices in the existing `connectome/<tract_id>` mapping (see `find_connectome_path_for_tck` in `train_stage1.py`). The tractography and parcellation pipeline itself is **not** supplied.

Clinical CSVs must contain subject/visit identifiers and the exact five motor-score columns expected by the original scripts.

## Run Stage I

Run from repository root. Point to your local (unpublished) input files:

```bash
python train_stage1.py \
  --data_base_dir ./data/connectomes \
  --csv_path ./data/UPDRS_all.csv \
  --save_dir ./outputs/stage1 \
  --cv_folds 5 \
  --seed 42 \
  --device cpu
```

The script writes `outputs/stage1/val_predictions_best_models/val_predictions_fold_*.csv`, fold checkpoints, metrics, and optional plots. For training with a GPU use e.g. `--device cuda:0`.

For inference from saved fold checkpoints:

```bash
python train_stage1.py \
  --data_base_dir ./data/connectomes \
  --csv_path ./data/UPDRS_all.csv \
  --save_dir ./outputs/stage1 \
  --checkpoint_dir ./outputs/stage1 \
  --inference_only
```

## Run Stage II

Use only OOF predictions from Stage I for the downstream **held-out** evaluation. All commands run from repository root and expect `val_predictions_fold_*.csv` under the `--pred_dir` path.

```bash
# Diagnosis classification
python train_stage2_diagnosis.py \
  --csv_path ./data/UPDRS_all.csv \
  --pred_dir ./outputs/stage1/val_predictions_best_models \
  --out_dir ./outputs/stage2_diagnosis \
  --enable_filter 0 --enable_prodromal_to_control 0

# Total MDS-UPDRS III regression
python train_stage2_total.py \
  --csv_path ./data/UPDRS_all.csv \
  --pred_dir ./outputs/stage1/val_predictions_best_models \
  --out_dir ./outputs/stage2_total \
  --enable_filter 0

# Hoehn–Yahr 0 vs 1 / 1 vs 2 / 0 vs 2
python train_stage2_hy.py \
  --csv_path ./data/UPDRS_all.csv \
  --pred_dir ./outputs/stage1/val_predictions_best_models \
  --out_dir ./outputs/stage2_hy \
  --enable_filter 0
```
