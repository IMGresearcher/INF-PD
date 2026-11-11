"""Stage II: UPDRS-III total score regression from predicted motor scores.

Research implementation; use deidentified inputs and review evaluation protocol.
"""


import os
import re
import json
import math
import glob
import random
import argparse
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


EPS = 1e-8
SCORE_COLS = ["运动迟缓", "肌强直", "静止性震颤", "姿势性/动作性震颤", "中轴"]
KEY_COLS = ["PATNO", "EVENT_ID", "INFODT", "EXAMDT"]
TARGET_COL = "UPDRS总分"


# -------------------------
# Utilities
# -------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def read_csv_flexible(csv_path: str) -> pd.DataFrame:
    last_err = None
    for enc in ["utf-8-sig", "gbk", "gb18030", "utf-8", "latin1"]:
        try:
            return pd.read_csv(csv_path, encoding=enc)
        except Exception as e:
            last_err = e
    raise last_err


def canonical_str(x: Any) -> str:
    if pd.isna(x):
        return ""
    if isinstance(x, (np.integer, int)):
        return str(int(x))
    if isinstance(x, (np.floating, float)):
        xf = float(x)
        if abs(xf - round(xf)) <= 1e-8:
            return str(int(round(xf)))
        s = f"{xf:.8f}".rstrip("0").rstrip(".")
        return s
    s = str(x).strip()
    try:
        xf = float(s)
        if abs(xf - round(xf)) <= 1e-8:
            return str(int(round(xf)))
        return f"{xf:.8f}".rstrip("0").rstrip(".")
    except Exception:
        return s


def build_precise_match_key(df: pd.DataFrame, score_cols: List[str]) -> pd.Series:
    parts = []
    for c in KEY_COLS + score_cols:
        if c not in df.columns:
            raise ValueError(f"Missing required column for precise matching: {c}")
        parts.append(df[c].map(canonical_str))
    key = parts[0]
    for s in parts[1:]:
        key = key + "||" + s
    return key


def detect_prediction_files(pred_dir: str) -> List[str]:
    patterns = [
        os.path.join(pred_dir, "val_predictions_fold_*.csv"),
        os.path.join(pred_dir, "*.csv"),
    ]
    found: List[str] = []
    for pat in patterns:
        found.extend(glob.glob(pat))
    found = sorted(set(found))
    found = [p for p in found if re.search(r"fold[_-]?\d+", os.path.basename(p), flags=re.IGNORECASE)]
    if len(found) == 0:
        raise FileNotFoundError(f"No fold prediction csv files found in: {pred_dir}")
    return found


def infer_fold_from_filename(path: str) -> int:
    name = os.path.basename(path)
    m = re.search(r"fold[_-]?(\d+)", name, flags=re.IGNORECASE)
    if m is None:
        raise ValueError(f"Cannot infer fold id from filename: {name}")
    return int(m.group(1))


# -------------------------
# Reading and precise matching
# -------------------------
def prepare_target_table(csv_path: str) -> pd.DataFrame:
    df = read_csv_flexible(csv_path).copy()
    required = KEY_COLS + SCORE_COLS + [TARGET_COL]
    missing = [c for c in required if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"CSV is missing required columns: {missing}")

    for c in KEY_COLS:
        df[c] = df[c].map(canonical_str)
    for c in SCORE_COLS + [TARGET_COL]:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    df["precise_match_key"] = build_precise_match_key(df, SCORE_COLS)

    rows = []
    duplicated_total_conflicts = []
    for key, sub in df.groupby("precise_match_key", sort=False):
        uniq_targets = sorted(set([float(x) for x in sub[TARGET_COL].dropna().tolist()]))
        if len(uniq_targets) == 0:
            continue
        if len(uniq_targets) > 1:
            duplicated_total_conflicts.append((key, uniq_targets))
            continue
        row = sub.iloc[0][KEY_COLS + SCORE_COLS].to_dict()
        row["precise_match_key"] = key
        row[TARGET_COL] = float(uniq_targets[0])
        rows.append(row)

    if len(duplicated_total_conflicts) > 0:
        preview = duplicated_total_conflicts[:10]
        raise ValueError(
            "Found duplicated source rows with the same precise key but different UPDRS总分. "
            f"Examples: {preview}"
        )

    out = pd.DataFrame(rows)
    if len(out) == 0:
        raise ValueError("No usable rows remained after building the target table.")
    return out


def read_and_match_prediction_files(pred_dir: str, target_df: pd.DataFrame) -> pd.DataFrame:
    files = detect_prediction_files(pred_dir)
    target_lookup = target_df[["precise_match_key", TARGET_COL]].drop_duplicates().copy()

    all_rows = []
    seen_global_keys = set()
    expected_true_cols = [f"{c}_true" for c in SCORE_COLS]
    expected_pred_cols = [f"{c}_pred" for c in SCORE_COLS]

    for path in files:
        fold = infer_fold_from_filename(path)
        pred_df = read_csv_flexible(path).copy()

        required = KEY_COLS + expected_true_cols + expected_pred_cols
        missing = [c for c in required if c not in pred_df.columns]
        if len(missing) > 0:
            raise ValueError(f"Prediction file {path} missing columns: {missing}")

        for c in KEY_COLS:
            pred_df[c] = pred_df[c].map(canonical_str)
        for c in expected_true_cols + expected_pred_cols:
            pred_df[c] = pd.to_numeric(pred_df[c], errors='coerce')

        tmp = pred_df.copy()
        for src, dst in zip(expected_true_cols, SCORE_COLS):
            tmp[dst] = tmp[src]
        tmp["precise_match_key"] = build_precise_match_key(tmp, SCORE_COLS)
        tmp["fold"] = fold

        dup_counts = tmp["precise_match_key"].value_counts()
        bad_dup = dup_counts[dup_counts > 1]
        if len(bad_dup) > 0:
            raise ValueError(
                f"Prediction file {path} contains duplicated precise keys. "
                f"Examples: {bad_dup.head(10).to_dict()}"
            )

        overlap = set(tmp["precise_match_key"].tolist()) & seen_global_keys
        if len(overlap) > 0:
            raise ValueError(
                f"Precise keys appear in more than one fold prediction file. "
                f"Examples: {list(sorted(overlap))[:10]}"
            )
        seen_global_keys.update(tmp["precise_match_key"].tolist())

        merged = tmp.merge(target_lookup, on="precise_match_key", how="left", validate="one_to_one")
        unmatched = merged[merged[TARGET_COL].isna()].copy()
        if len(unmatched) > 0:
            preview_cols = ["sample_id", "PATNO", "EVENT_ID", "INFODT", "EXAMDT"] if "sample_id" in unmatched.columns else ["PATNO", "EVENT_ID", "INFODT", "EXAMDT"]
            raise ValueError(
                f"Prediction file {path} has {len(unmatched)} rows that cannot be precisely matched to the source csv. "
                f"Examples: {unmatched[preview_cols].head(10).to_dict(orient='records')}"
            )

        keep_cols = []
        for c in ["sample_id", "PATNO", "EVENT_ID", "INFODT", "EXAMDT", "fold", "precise_match_key"]:
            if c in merged.columns:
                keep_cols.append(c)
        keep_cols += expected_true_cols + expected_pred_cols + [TARGET_COL]
        all_rows.append(merged[keep_cols].copy())

    out = pd.concat(all_rows, axis=0, ignore_index=True)
    out = out.sort_values(["fold", "PATNO", "EVENT_ID", "INFODT", "EXAMDT"]).reset_index(drop=True)
    return out


# -------------------------
# Gap filtering after matching
# -------------------------
def add_subscore_gap_columns(df_matched: pd.DataFrame) -> pd.DataFrame:
    df = df_matched.copy()
    err_cols = []
    sq_err_cols = []

    for c in SCORE_COLS:
        true_c = f"{c}_true"
        pred_c = f"{c}_pred"
        if true_c not in df.columns or pred_c not in df.columns:
            raise ValueError(f"Missing columns for error calculation: {true_c}, {pred_c}")

        err_c = f"{c}_abs_error"
        sq_err_c = f"{c}_sq_error"
        diff = pd.to_numeric(df[pred_c], errors="coerce") - pd.to_numeric(df[true_c], errors="coerce")
        df[err_c] = diff.abs()
        df[sq_err_c] = diff.pow(2)
        err_cols.append(err_c)
        sq_err_cols.append(sq_err_c)

    df["subscore_mae"] = df[err_cols].mean(axis=1)
    df["subscore_maxae"] = df[err_cols].max(axis=1)
    df["subscore_rmse"] = np.sqrt(df[sq_err_cols].mean(axis=1))
    return df


def filter_large_gap_samples(
    df_matched: pd.DataFrame,
    out_dir: str,
    enable_filter: bool = True,
    filter_mode: str = "auto",
    mae_quantile: float = 0.90,
    maxae_quantile: float = 0.95,
    mae_threshold: float = -1.0,
    maxae_threshold: float = -1.0,
) -> pd.DataFrame:
    os.makedirs(out_dir, exist_ok=True)
    df = add_subscore_gap_columns(df_matched)

    df.to_csv(
        os.path.join(out_dir, "matched_predictions_with_total_and_gap_metrics.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    if not enable_filter:
        df["keep_after_gap_filter"] = 1
        pd.DataFrame([{
            "enable_filter": 0,
            "filter_mode": str(filter_mode),
            "mae_threshold_used": np.nan,
            "maxae_threshold_used": np.nan,
            "n_before": int(len(df)),
            "n_removed": 0,
            "n_after": int(len(df)),
        }]).to_csv(
            os.path.join(out_dir, "gap_filter_summary.csv"),
            index=False,
            encoding="utf-8-sig",
        )
        return df

    filter_mode = str(filter_mode).strip().lower()
    if filter_mode not in {"auto", "manual"}:
        raise ValueError("filter_mode must be one of {'auto', 'manual'}.")

    if filter_mode == "auto":
        mae_thr = float(df["subscore_mae"].quantile(float(mae_quantile)))
        maxae_thr = float(df["subscore_maxae"].quantile(float(maxae_quantile)))
    else:
        if float(mae_threshold) <= 0 and float(maxae_threshold) <= 0:
            raise ValueError("In manual mode, at least one of mae_threshold or maxae_threshold must be > 0.")
        mae_thr = float(mae_threshold) if float(mae_threshold) > 0 else float("inf")
        maxae_thr = float(maxae_threshold) if float(maxae_threshold) > 0 else float("inf")

    keep_mask = (df["subscore_mae"] <= mae_thr) & (df["subscore_maxae"] <= maxae_thr)
    df["keep_after_gap_filter"] = keep_mask.astype(int)

    kept_df = df.loc[keep_mask].copy()
    removed_df = df.loc[~keep_mask].copy()

    if len(kept_df) == 0:
        raise ValueError("All matched rows were filtered out. Please relax the filtering thresholds.")

    fold_counts = kept_df["fold"].astype(int).value_counts().to_dict()
    missing_folds = [f for f in sorted(df["fold"].astype(int).unique().tolist()) if f not in fold_counts]
    if len(missing_folds) > 0:
        raise ValueError(f"After filtering, these folds became empty: {missing_folds}. Please relax the thresholds.")

    kept_df.to_csv(
        os.path.join(out_dir, "matched_predictions_with_total_filtered.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    removed_df.to_csv(
        os.path.join(out_dir, "removed_large_gap_ids.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    summary_rows = [{
        "enable_filter": 1,
        "filter_mode": filter_mode,
        "mae_quantile": float(mae_quantile),
        "maxae_quantile": float(maxae_quantile),
        "mae_threshold_input": float(mae_threshold),
        "maxae_threshold_input": float(maxae_threshold),
        "mae_threshold_used": float(mae_thr),
        "maxae_threshold_used": float(maxae_thr),
        "n_before": int(len(df)),
        "n_removed": int((~keep_mask).sum()),
        "n_after": int(keep_mask.sum()),
    }]
    for fold in sorted(df["fold"].astype(int).unique().tolist()):
        summary_rows.append({
            "fold": int(fold),
            "n_before": int((df["fold"].astype(int) == int(fold)).sum()),
            "n_removed": int(((df["fold"].astype(int) == int(fold)) & (~keep_mask)).sum()),
            "n_after": int(((df["fold"].astype(int) == int(fold)) & keep_mask).sum()),
            "mae_threshold_used": float(mae_thr),
            "maxae_threshold_used": float(maxae_thr),
        })

    pd.DataFrame(summary_rows).to_csv(
        os.path.join(out_dir, "gap_filter_summary.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    return kept_df.reset_index(drop=True)


# -------------------------
# MLP second-stage regression
# -------------------------
def build_mlp(random_state: int) -> TransformedTargetRegressor:
    base = Pipeline(
        steps=[
            ("x_scaler", StandardScaler()),
            (
                "mlp",
                MLPRegressor(
                    hidden_layer_sizes=(64, 32),
                    activation="relu",
                    solver="adam",
                    alpha=1e-4,
                    batch_size="auto",
                    learning_rate_init=1e-3,
                    max_iter=3000,
                    early_stopping=True,
                    validation_fraction=0.15,
                    n_iter_no_change=80,
                    random_state=random_state,
                ),
            ),
        ]
    )
    reg = TransformedTargetRegressor(
        regressor=base,
        transformer=StandardScaler(),
        check_inverse=False,
    )
    return reg


def fit_oof_mlp(df_matched: pd.DataFrame, out_dir: str, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    os.makedirs(out_dir, exist_ok=True)

    pred_feature_cols = [f"{c}_pred" for c in SCORE_COLS]
    for c in pred_feature_cols + [TARGET_COL, "fold"]:
        if c not in df_matched.columns:
            raise ValueError(f"Missing required column for MLP stage: {c}")

    fold_ids = sorted(df_matched["fold"].astype(int).unique().tolist())
    if fold_ids != [1, 2, 3, 4, 5]:
        print(f"[Warn] Fold ids are {fold_ids}, not exactly [1,2,3,4,5]. The script will still use them as provided.")

    all_pred_rows = []
    metrics_rows = []

    for fold in fold_ids:
        train_df = df_matched[df_matched["fold"].astype(int) != int(fold)].copy().reset_index(drop=True)
        val_df = df_matched[df_matched["fold"].astype(int) == int(fold)].copy().reset_index(drop=True)
        if len(train_df) == 0 or len(val_df) == 0:
            raise ValueError(f"Fold {fold}: empty train or val set after fold split.")

        X_train = train_df[pred_feature_cols].values.astype(np.float32)
        y_train = train_df[TARGET_COL].values.astype(np.float32)
        X_val = val_df[pred_feature_cols].values.astype(np.float32)
        y_val = val_df[TARGET_COL].values.astype(np.float32)

        model = build_mlp(random_state=seed + int(fold))
        model.fit(X_train, y_train)
        val_pred = model.predict(X_val).astype(np.float32)

        fold_mae = float(mean_absolute_error(y_val, val_pred))
        fold_rmse = float(math.sqrt(mean_squared_error(y_val, val_pred)))
        fold_r2 = float(r2_score(y_val, val_pred)) if len(y_val) >= 2 else 0.0

        fold_out = val_df[[c for c in ["sample_id", "PATNO", "EVENT_ID", "INFODT", "EXAMDT", "fold", "precise_match_key"] if c in val_df.columns]].copy()
        for c in SCORE_COLS:
            fold_out[f"{c}_true"] = val_df[f"{c}_true"].values
            fold_out[f"{c}_pred_input"] = val_df[f"{c}_pred"].values
        fold_out["UPDRS总分_true"] = y_val
        fold_out["UPDRS总分_pred"] = val_pred
        fold_out["abs_error"] = np.abs(y_val - val_pred)
        fold_out["sq_error"] = (y_val - val_pred) ** 2
        fold_out.to_csv(os.path.join(out_dir, f"mlp_total_predictions_fold_{fold}.csv"), index=False, encoding="utf-8-sig")

        model_path = os.path.join(out_dir, f"mlp_total_fold_{fold}.joblib")
        try:
            import joblib
            joblib.dump(model, model_path)
        except Exception as e:
            print(f"[Warn] Failed to save joblib model for fold {fold}: {e}")

        metrics_rows.append(
            {
                "fold": int(fold),
                "train_size": int(len(train_df)),
                "val_size": int(len(val_df)),
                "mae": fold_mae,
                "rmse": fold_rmse,
                "r2": fold_r2,
            }
        )
        all_pred_rows.append(fold_out)
        print(f"[Fold {fold}] train={len(train_df)} val={len(val_df)} MAE={fold_mae:.6f} RMSE={fold_rmse:.6f} R2={fold_r2:.6f}")

    pred_all = pd.concat(all_pred_rows, axis=0, ignore_index=True)
    metrics_df = pd.DataFrame(metrics_rows)

    overall_mae = float(mean_absolute_error(pred_all["UPDRS总分_true"].values, pred_all["UPDRS总分_pred"].values))
    overall_rmse = float(math.sqrt(mean_squared_error(pred_all["UPDRS总分_true"].values, pred_all["UPDRS总分_pred"].values)))
    overall_r2 = float(r2_score(pred_all["UPDRS总分_true"].values, pred_all["UPDRS总分_pred"].values)) if len(pred_all) >= 2 else 0.0

    metrics_df.to_csv(os.path.join(out_dir, "mlp_total_5fold_metrics.csv"), index=False, encoding="utf-8-sig")
    pred_all.to_csv(os.path.join(out_dir, "mlp_total_oof_predictions_all_folds.csv"), index=False, encoding="utf-8-sig")

    summary = {
        "overall_mae": overall_mae,
        "overall_rmse": overall_rmse,
        "overall_r2": overall_r2,
        "fold_mae_mean": float(metrics_df["mae"].mean()),
        "fold_mae_std": float(metrics_df["mae"].std(ddof=0)),
        "num_samples": int(len(pred_all)),
        "num_folds": int(len(metrics_df)),
    }
    with open(os.path.join(out_dir, "mlp_total_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n[Overall]")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return pred_all, metrics_df


# -------------------------
# Plotting
# -------------------------
def save_regression_plot(pred_df: pd.DataFrame, out_path: str) -> None:
    y_true = pred_df["UPDRS总分_true"].values.astype(np.float32)
    y_pred = pred_df["UPDRS总分_pred"].values.astype(np.float32)
    folds = pred_df["fold"].astype(int).values if "fold" in pred_df.columns else np.ones(len(pred_df), dtype=int)

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(math.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else 0.0
    corr = float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) >= 2 and np.std(y_true) > 1e-12 and np.std(y_pred) > 1e-12 else 0.0

    fig = plt.figure(figsize=(7.5, 7.0), dpi=180)
    ax = fig.add_subplot(111)

    unique_folds = sorted(set(folds.tolist()))
    for fold in unique_folds:
        mask = folds == fold
        ax.scatter(y_true[mask], y_pred[mask], s=28, alpha=0.75, label=f"Fold {fold}")

    lo = float(min(np.min(y_true), np.min(y_pred)))
    hi = float(max(np.max(y_true), np.max(y_pred)))
    pad = max((hi - lo) * 0.05, 1.0)
    lo -= pad
    hi += pad
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.5, label="y = x")

    if len(y_true) >= 2:
        coef = np.polyfit(y_true, y_pred, deg=1)
        xs = np.linspace(lo, hi, 200)
        ys = coef[0] * xs + coef[1]
        ax.plot(xs, ys, linewidth=1.5, label=f"fit: y={coef[0]:.3f}x+{coef[1]:.3f}")

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("True UPDRS Total")
    ax.set_ylabel("Predicted UPDRS Total")
    ax.set_title("5-Fold OOF MLP Regression for UPDRS Total")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=9)
    ax.text(
        0.03,
        0.97,
        f"MAE = {mae:.4f}\nRMSE = {rmse:.4f}\nR² = {r2:.4f}\nCorr = {corr:.4f}\nN = {len(y_true)}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Use 5 predicted UPDRS sub-scores from saved fold validation predictions to predict UPDRS total with a 5-fold MLP stacking model. Filter large-gap ids after precise matching and before training."
    )
    parser.add_argument("--csv_path", type=str, default="./data/UPDRS_all.csv")
    parser.add_argument(
        "--pred_dir",
        type=str,
        default="./outputs/stage1/val_predictions_best_models",
        help="Directory containing val_predictions_fold_1.csv ... val_predictions_fold_5.csv",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./outputs/stage2_total",
    )
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--enable_filter", type=int, default=0, help="Optional exploratory filtering using true score errors; DISABLE for valid unbiased evaluation (default: 0).")
    parser.add_argument("--filter_mode", type=str, default="auto", choices=["auto", "manual"])
    parser.add_argument("--filter_mae_quantile", type=float, default=0.95, help="Used in auto mode.")
    parser.add_argument("--filter_maxae_quantile", type=float, default=0.95, help="Used in auto mode.")
    parser.add_argument("--filter_mae_threshold", type=float, default=-1.0, help="Used in manual mode.")
    parser.add_argument("--filter_maxae_threshold", type=float, default=-1.0, help="Used in manual mode.")

    args = parser.parse_args()

    set_seed(args.seed)
    if int(args.enable_filter):
        print("[WARNING] Filtering by the error between predicted and TRUE motor scores "
              "uses outcome information; results are NOT suitable for unbiased held-out evaluation.")
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[Info] Reading source csv: {args.csv_path}")
    target_df = prepare_target_table(args.csv_path)
    print(f"[Info] Source target table size after precise de-duplication: {len(target_df)}")

    print(f"[Info] Reading prediction files from: {args.pred_dir}")
    matched_pred_df = read_and_match_prediction_files(args.pred_dir, target_df)
    matched_save_path = os.path.join(args.out_dir, "matched_predictions_with_total.csv")
    matched_pred_df.to_csv(matched_save_path, index=False, encoding="utf-8-sig")
    print(f"[Saved] {matched_save_path}")

    print("[Info] Filtering ids with overly large gaps between predicted and true sub-scores ...")
    filtered_pred_df = filter_large_gap_samples(
        df_matched=matched_pred_df,
        out_dir=args.out_dir,
        enable_filter=bool(int(args.enable_filter)),
        filter_mode=args.filter_mode,
        mae_quantile=float(args.filter_mae_quantile),
        maxae_quantile=float(args.filter_maxae_quantile),
        mae_threshold=float(args.filter_mae_threshold),
        maxae_threshold=float(args.filter_maxae_threshold),
    )
    print(f"[Info] Rows after filtering: {len(filtered_pred_df)}")

    pred_all, metrics_df = fit_oof_mlp(filtered_pred_df, args.out_dir, seed=args.seed)
    plot_path = os.path.join(args.out_dir, "mlp_total_regression_plot.png")
    save_regression_plot(pred_all, plot_path)
    print(f"[Saved] {plot_path}")

    print("\n[Per-fold MAE]")
    print(metrics_df[["fold", "mae"]].to_string(index=False))
    print(f"\n[Overall MAE] {mean_absolute_error(pred_all['UPDRS总分_true'], pred_all['UPDRS总分_pred']):.6f}")


if __name__ == "__main__":
    main()
