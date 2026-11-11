# Release changes and audit notes

## Differences from uploaded scripts

- Five source modules were renamed to stable, readable filenames; `model_connectome.py` is kept as the import name used by Stage I.
- Removed `build_all_data_train_ids_with_fold_val_exclusion` and `print_fold_train_exclusion_summary` entirely, and removed associated metrics key `excluded_from_all_data_train_count`.
- Stage I now uses the proper held-out participant split `fold_info["train_ids"]`; validation participants are checked to be absent from training (both `sample_id` and `PATNO`). **This is a scientific correction; results may change.**
- Changed absolute data path defaults to repository-relative paths and stage output directories.
- Stage I no longer writes an uncontrolled `./result.csv`; the result is written inside the selected checkpoint or output directory.
- Downstream error-based data filtering is OFF by default, and diagnosis relabeling is OFF by default. Turning these on produces exploratory results that should not be called unbiased testing.
- Added `.gitignore`, packaging metadata, requirements, license template, documentation and synthetic tests.

## Not changed or not independently verified

- Stage-I model architecture, training losses, data matching and fold assignment algorithms were otherwise preserved.
- Stage-II classifier/regressor implementations, fold-level summary functions and saved report schemas are otherwise preserved.
- Underlying medical preprocessing, fold assignments on real cohorts, private file paths, saved models and performance numbers are not included.
- **Caveat:** Stage-I checkpoint selection currently uses validation performance; end-to-end two-stage nested validation needs a separate research revision.
- **Caveat:** Exact-match keys currently include the five *true* motor score columns in addition to visit fields; this requires ground-truth scores for every downstream sample. For an inference-only application, a patient/visit-only matching procedure must be implemented and validated.

## File mapping
- `connectome_updrs_V1_regression_final(20261111-145240).py` -> `train_stage1.py`
- `mlp_predict_label_filtered_threeclass_pairwise(6).py` -> `train_stage2_diagnosis.py`
- `mlp_predict_updrs_total_from_oof_subscores_filtered(7).py` -> `train_stage2_total.py`
- `mlp_predict_hy_pairwise_remove_HY3(6).py` -> `train_stage2_hy.py`
- `model_connectome(20261111-145239).py` -> `model_connectome.py`
