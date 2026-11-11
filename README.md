# INF-PD: Connectome-guided motor phenotype learning

Research implementation of a **two-stage motor phenotype learning framework** for Parkinsonian assessment using structural connectomes. The repository contains the five user-provided Python modules, cleaned for public release and integrated through a common on-disk OOF prediction interface.

**Important:** This release contains **source code only**. No PPMI or hospital records, tractography matrices, model checkpoints, or precomputed results are included. The authors should review licensing and patient-data governance before GitHub publication. This code is not a medical device.

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

Clinical CSVs must contain subject/visit identifiers and the exact five motor-score columns expected by the original scripts:

- `运动迟缓`, `肌强直`, `静止性震颤`, `姿势性/动作性震颤`, `中轴`
- Diagnosis: `label` (`control`, `prodromal`, `pd`), as applicable
- Total score: `UPDRS总分`, as applicable
- Staging: `HY分级`, as applicable
- Matching keys for Stage II: `PATNO`, `EVENT_ID`, `INFODT`, `EXAMDT`

**Do not upload clinical CSVs or output directories to a public repository.** Scanning/matching assumptions are inherited from the research implementation; adapt the data loader as needed for your institution.

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

### Evaluation and leakage cautions

1. **Removed:** `build_all_data_train_ids_with_fold_val_exclusion` and its companion exclusion printer. The original routine assembled training IDs from **all matched participants**, excluding only a validation-label-dependent subset. This is unsuitable for ordinary held-out CV. The released Stage I strictly uses `fold_info['train_ids']` and checks participant and sample ID disjointness.
2. **Changed defaults:** `--enable_filter 0` in all Stage-II scripts. Filtering samples based on the absolute error between *true and predicted* motor scores uses outcome information and gives optimistically biased results. This legacy option remains explicitly opt-in **for exploratory analysis only**, not unbiased benchmark reporting.
3. **Changed default:** `--enable_prodromal_to_control 0` in diagnosis. Relabeling participants using UPDRS total changes the clinical target and must not be used as a standard diagnosis benchmark.
4. The Stage II scripts take OOF Stage-I predictions. Although the same participant's Stage-I model excluded that participant, a separate Stage-II model is cross-validated over these OOF features. **This is not a fully nested two-stage evaluation** because Stage-I fold models may have learned from individuals assigned to the downstream held-out fold for other OOF rows. Strict end-to-end validation needs an outer loop and an inner cross-fitting loop on outer-training participants. Do not claim the supplied pipeline provides nested or external test performance.
5. No model results are guaranteed to match previously reported tables after correcting fold separation and disabling target-aware filtering. Retrain and recompute all metrics from the actual fold predictions, with per-fold counts and confusion matrices.
6. Validation labels are used by the Stage-I early-stopping/model-selection routine. For an unbiased outer-fold performance estimate, implement an **inner validation split for early stopping**, reserving the outer fold strictly for final evaluation.
7. Do not share training output CSVs or checkpoint files containing participant identifiers without explicit permission.

## Scope and reproducibility

The code reproduces the *supplied implementation*, with targeted leakage prevention and public-release cleanup, not every published experimental comparison. Direct-connectome baselines, an external hospital inference pipeline, and structural connectome generation are **not provided** in these five files. Pretrained weights and input files were not supplied; no real dataset training/inference has been executed here.

For feature-level comparability, document the exact imaging protocol, parcellation/tractography processing, sample inclusion criteria, fold assignments, Python environment, and reported per-fold results when releasing the accompanying paper.

## License

MIT license is included as a proposed permissive open-source license. Confirm authorization and copyright ownership before publishing; change it if your lab/institution requires a different license.
