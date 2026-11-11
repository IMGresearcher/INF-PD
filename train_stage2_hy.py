"""Stage II: pairwise Hoehn-Yahr classification from predicted motor scores.

Research implementation; use deidentified inputs and review evaluation protocol.
"""


import os
import re
import json
import glob
import random
import argparse
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

try:
    import torch
    import torch.nn as nn
except Exception:
    torch = None
    nn = None


SCORE_COLS = ["运动迟缓", "肌强直", "静止性震颤", "姿势性/动作性震颤", "中轴"]
KEY_COLS = ["PATNO", "EVENT_ID", "INFODT", "EXAMDT"]
TARGET_COL = "HY分级"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))


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
        return f"{xf:.8f}".rstrip("0").rstrip(".")
    s = str(x).strip()
    try:
        xf = float(s)
        if abs(xf - round(xf)) <= 1e-8:
            return str(int(round(xf)))
        return f"{xf:.8f}".rstrip("0").rstrip(".")
    except Exception:
        return s


def canonical_hy_value(x: Any) -> str:
    if pd.isna(x):
        return ""
    try:
        xf = float(x)
        if abs(xf - round(xf)) <= 1e-8:
            return str(int(round(xf)))
        return f"{xf:.8f}".rstrip("0").rstrip(".")
    except Exception:
        return str(x).strip()


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
    files: List[str] = []
    for pat in patterns:
        files.extend(glob.glob(pat))
    files = sorted(set(files))
    files = [p for p in files if re.search(r"fold[_-]?\d+", os.path.basename(p), flags=re.IGNORECASE)]
    if len(files) == 0:
        raise FileNotFoundError(f"No fold prediction csv files found in: {pred_dir}")
    return files


def infer_fold_from_filename(path: str) -> int:
    name = os.path.basename(path)
    m = re.search(r"fold[_-]?(\d+)", name, flags=re.IGNORECASE)
    if m is None:
        raise ValueError(f"Cannot infer fold id from filename: {name}")
    return int(m.group(1))


def prepare_hy_table(csv_path: str) -> Tuple[pd.DataFrame, Dict[str, int], Dict[int, str]]:
    df = read_csv_flexible(csv_path).copy()
    required = KEY_COLS + SCORE_COLS + [TARGET_COL]
    missing = [c for c in required if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"CSV is missing required columns: {missing}")

    for c in KEY_COLS:
        df[c] = df[c].map(canonical_str)
    for c in SCORE_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df[TARGET_COL] = df[TARGET_COL].map(canonical_hy_value)
    df = df[df[TARGET_COL].astype(str) != ""].copy()

    hy_values = sorted(df[TARGET_COL].dropna().astype(str).unique().tolist(), key=lambda v: float(v) if re.match(r"^-?\d+(\.\d+)?$", v) else v)
    if len(hy_values) != 4:
        print(f"[Warn] Detected HY classes are {hy_values}, not exactly 4 classes. The script will still use the detected classes.")

    class_to_idx = {v: i for i, v in enumerate(hy_values)}
    idx_to_class = {i: v for v, i in class_to_idx.items()}

    df["target_class"] = df[TARGET_COL].map(class_to_idx).astype(int)
    df["precise_match_key"] = build_precise_match_key(df, SCORE_COLS)

    rows = []
    conflicts = []
    for key, sub in df.groupby("precise_match_key", sort=False):
        uniq_targets = sorted(set([int(x) for x in sub["target_class"].dropna().tolist()]))
        uniq_hy = sorted(set([str(x) for x in sub[TARGET_COL].dropna().tolist()]))
        if len(uniq_targets) == 0:
            continue
        if len(uniq_targets) > 1:
            conflicts.append((key, uniq_hy))
            continue
        row = sub.iloc[0][KEY_COLS + SCORE_COLS].to_dict()
        row["precise_match_key"] = key
        row[TARGET_COL] = str(sub.iloc[0][TARGET_COL])
        row["target_class"] = int(uniq_targets[0])
        rows.append(row)

    if len(conflicts) > 0:
        raise ValueError(
            "Found duplicated source rows with the same precise key but different HY classes. "
            f"Examples: {conflicts[:10]}"
        )

    out = pd.DataFrame(rows)
    if len(out) == 0:
        raise ValueError("No usable rows remained after building the HY table.")
    return out, class_to_idx, idx_to_class


def read_and_match_prediction_files(pred_dir: str, hy_df: pd.DataFrame) -> pd.DataFrame:
    files = detect_prediction_files(pred_dir)
    hy_lookup = hy_df[["precise_match_key", TARGET_COL, "target_class"]].drop_duplicates().copy()

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
            pred_df[c] = pd.to_numeric(pred_df[c], errors="coerce")

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

        merged = tmp.merge(hy_lookup, on="precise_match_key", how="left", validate="one_to_one")
        unmatched = merged[merged["target_class"].isna()].copy()
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
        keep_cols += expected_true_cols + expected_pred_cols + [TARGET_COL, "target_class"]
        all_rows.append(merged[keep_cols].copy())

    out = pd.concat(all_rows, axis=0, ignore_index=True)
    out["target_class"] = out["target_class"].astype(int)
    out = out.sort_values(["fold", "PATNO", "EVENT_ID", "INFODT", "EXAMDT"]).reset_index(drop=True)
    return out


def add_subscore_gap_columns(df_matched: pd.DataFrame) -> pd.DataFrame:
    df = df_matched.copy()
    err_cols = []
    for c in SCORE_COLS:
        true_c = f"{c}_true"
        pred_c = f"{c}_pred"
        if true_c not in df.columns or pred_c not in df.columns:
            raise ValueError(f"Missing columns for error calculation: {true_c}, {pred_c}")
        err_c = f"{c}_abs_error"
        df[err_c] = (pd.to_numeric(df[pred_c], errors="coerce") - pd.to_numeric(df[true_c], errors="coerce")).abs()
        err_cols.append(err_c)

    df["subscore_mae"] = df[err_cols].mean(axis=1)
    df["subscore_maxae"] = df[err_cols].max(axis=1)
    df["subscore_rmse"] = np.sqrt((df[err_cols] ** 2).mean(axis=1))
    return df


def filter_large_gap_samples(
    df_matched: pd.DataFrame,
    out_dir: str,
    enable_filter: bool = True,
    filter_mode: str = "auto",
    mae_quantile: float = 0.90,
    maxae_quantile: float = 0.90,
    mae_threshold: float = -1.0,
    maxae_threshold: float = -1.0,
) -> pd.DataFrame:
    os.makedirs(out_dir, exist_ok=True)
    df = add_subscore_gap_columns(df_matched)

    if not enable_filter:
        df["keep_after_gap_filter"] = 1
        df.to_csv(os.path.join(out_dir, "matched_predictions_with_hy_and_gap_metrics.csv"), index=False, encoding="utf-8-sig")
        pd.DataFrame([{
            "enable_filter": 0,
            "filter_mode": filter_mode,
            "mae_threshold_used": np.nan,
            "maxae_threshold_used": np.nan,
            "n_before": int(len(df)),
            "n_removed": 0,
            "n_after": int(len(df)),
        }]).to_csv(os.path.join(out_dir, "gap_filter_summary.csv"), index=False, encoding="utf-8-sig")
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

    removed_df = df.loc[~keep_mask].copy()
    kept_df = df.loc[keep_mask].copy()

    if len(kept_df) == 0:
        raise ValueError("All matched rows were filtered out. Please relax the filtering thresholds.")

    fold_counts = kept_df["fold"].value_counts().to_dict()
    missing_folds = [f for f in sorted(df["fold"].unique().tolist()) if f not in fold_counts]
    if len(missing_folds) > 0:
        raise ValueError(f"After filtering, these folds became empty: {missing_folds}. Please relax the thresholds.")

    df.to_csv(os.path.join(out_dir, "matched_predictions_with_hy_and_gap_metrics.csv"), index=False, encoding="utf-8-sig")
    kept_df.to_csv(os.path.join(out_dir, "matched_predictions_with_hy_filtered.csv"), index=False, encoding="utf-8-sig")
    removed_df.to_csv(os.path.join(out_dir, "removed_large_gap_ids.csv"), index=False, encoding="utf-8-sig")

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
            "enable_filter": 1,
            "filter_mode": filter_mode,
            "fold": int(fold),
            "n_before": int((df["fold"].astype(int) == int(fold)).sum()),
            "n_removed": int(((df["fold"].astype(int) == int(fold)) & (~keep_mask)).sum()),
            "n_after": int(((df["fold"].astype(int) == int(fold)) & keep_mask).sum()),
            "mae_threshold_used": float(mae_thr),
            "maxae_threshold_used": float(maxae_thr),
        })

    pd.DataFrame(summary_rows).to_csv(os.path.join(out_dir, "gap_filter_summary.csv"), index=False, encoding="utf-8-sig")
    return kept_df.reset_index(drop=True)


class TorchMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def parse_class_weights(class_weights: str, num_classes: int) -> Optional[np.ndarray]:
    s = str(class_weights).strip()
    if s == "" or s.lower() in {"none", "auto", "-1"}:
        return None
    parts = [p.strip() for p in s.split(",") if p.strip() != ""]
    if len(parts) != int(num_classes):
        raise ValueError(f"--class_weights must contain exactly {num_classes} comma-separated values, got {len(parts)}: {class_weights}")
    arr = np.asarray([float(p) for p in parts], dtype=np.float32)
    if np.any(arr <= 0):
        raise ValueError("--class_weights values must all be > 0 when manually specified.")
    return arr


def resolve_class_weights(
    y_train: np.ndarray,
    num_classes: int,
    class_weights: str = "",
) -> Tuple[np.ndarray, str]:
    manual = parse_class_weights(class_weights, num_classes=num_classes)
    if manual is not None:
        return manual.astype(np.float32), "manual"

    y_train = np.asarray(y_train).astype(int).reshape(-1)
    counts = np.bincount(y_train, minlength=int(num_classes)).astype(np.float32)
    total = float(max(counts.sum(), 1.0))
    weights = total / np.maximum(float(num_classes) * counts, 1.0)
    weights = weights / max(float(weights.mean()), 1e-6)
    return weights.astype(np.float32), "auto"


def _train_torch_mlp_weighted_ce(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    num_classes: int,
    seed: int,
    device: str,
    class_weights: str = "",
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    max_epochs: int = 500,
    patience: int = 40,
    batch_size: int = 64,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    if torch is None or nn is None:
        raise ImportError("Missing PyTorch. Please install torch in your environment to use weighted CE.")

    rng = np.random.RandomState(seed)
    torch.manual_seed(int(seed))
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

    x_scaler = StandardScaler()
    x_train_s = x_scaler.fit_transform(x_train).astype(np.float32)
    x_val_s = x_scaler.transform(x_val).astype(np.float32)

    y_train = np.asarray(y_train).astype(np.int64).reshape(-1)
    y_val = np.asarray(y_val).astype(np.int64).reshape(-1)

    resolved_weights, weight_mode = resolve_class_weights(
        y_train=y_train,
        num_classes=int(num_classes),
        class_weights=class_weights,
    )

    model = TorchMLP(in_dim=int(x_train_s.shape[1]), out_dim=int(num_classes)).to(device)
    loss_fn = nn.CrossEntropyLoss(
        weight=torch.tensor(resolved_weights, dtype=torch.float32, device=device),
        reduction="mean",
    )
    optim = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))

    best_val_loss = float("inf")
    best_state = None
    bad_epochs = 0

    n = int(x_train_s.shape[0])
    idx = np.arange(n)

    for _epoch in range(int(max_epochs)):
        model.train()
        rng.shuffle(idx)
        for start in range(0, n, int(batch_size)):
            sel = idx[start : start + int(batch_size)]
            xb = torch.from_numpy(x_train_s[sel]).to(device)
            yb = torch.from_numpy(y_train[sel]).long().to(device)

            optim.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optim.step()

        model.eval()
        with torch.no_grad():
            xv = torch.from_numpy(x_val_s).to(device)
            yv = torch.from_numpy(y_val).long().to(device)
            val_logits = model(xv)
            val_loss = float(loss_fn(val_logits, yv).item())

        if val_loss + 1e-8 < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= int(patience):
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        xv = torch.from_numpy(x_val_s).to(device)
        logits = model(xv)
        prob = torch.softmax(logits, dim=1).detach().cpu().numpy().astype(np.float32)
        pred_class = prob.argmax(axis=1).astype(int)

    weight_info = {
        "class_weights": resolved_weights.astype(float).tolist(),
        "weight_mode": str(weight_mode),
    }
    return pred_class, prob, weight_info


def compute_multiclass_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> Dict[str, Any]:
    y_true = np.asarray(y_true).astype(int).reshape(-1)
    y_pred = np.asarray(y_pred).astype(int).reshape(-1)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(int(num_classes))))
    acc = float(accuracy_score(y_true, y_pred))

    per_class = {}
    sens_list = []
    spec_list = []
    total = int(cm.sum())
    for k in range(int(num_classes)):
        tp = int(cm[k, k])
        fn = int(cm[k, :].sum() - tp)
        fp = int(cm[:, k].sum() - tp)
        tn = int(total - tp - fn - fp)

        sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        sens_list.append(sensitivity)
        spec_list.append(specificity)
        per_class[k] = {
            "sensitivity": sensitivity,
            "specificity": specificity,
            "tp": tp,
            "fn": fn,
            "fp": fp,
            "tn": tn,
        }

    return {
        "accuracy": acc,
        "macro_sensitivity": float(np.mean(sens_list)) if sens_list else 0.0,
        "macro_specificity": float(np.mean(spec_list)) if spec_list else 0.0,
        "confusion_matrix": cm,
        "per_class": per_class,
    }



def compute_binary_metrics_from_multiclass(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """
    Binary metrics with label 0 as negative class and label 1 as positive class.
    Specificity = TN / (TN + FP)
    Sensitivity = TP / (TP + FN)
    """
    y_true = np.asarray(y_true).astype(int).reshape(-1)
    y_pred = np.asarray(y_pred).astype(int).reshape(-1)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    acc = float(accuracy_score(y_true, y_pred))
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    return {
        "accuracy": acc,
        "specificity": specificity,
        "sensitivity": sensitivity,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "confusion_matrix": cm,
    }


def save_binary_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    neg_hy: str,
    pos_hy: str,
    out_path: str,
) -> pd.DataFrame:
    cm = confusion_matrix(
        np.asarray(y_true).astype(int).reshape(-1),
        np.asarray(y_pred).astype(int).reshape(-1),
        labels=[0, 1],
    )
    cm_df = pd.DataFrame(
        cm,
        index=[f"true_HY_{neg_hy}(0)", f"true_HY_{pos_hy}(1)"],
        columns=[f"pred_HY_{neg_hy}(0)", f"pred_HY_{pos_hy}(1)"],
    )
    cm_df.to_csv(out_path, index=True, encoding="utf-8-sig")
    return cm_df


def remove_hy3_samples(df_matched: pd.DataFrame, out_dir: str) -> pd.DataFrame:
    """
    直接剔除 HY=3 样本，仅保留 HY=0/1/2，用于后续两两分类任务。
    """
    os.makedirs(out_dir, exist_ok=True)
    df = df_matched.copy()
    df[TARGET_COL] = df[TARGET_COL].astype(str).map(canonical_hy_value)

    removed = df[df[TARGET_COL] == "3"].copy()
    kept = df[df[TARGET_COL].isin(["0", "1", "2"])].copy()

    if len(kept) == 0:
        raise ValueError("No samples remain after removing HY=3 and keeping HY=0/1/2.")

    kept.to_csv(os.path.join(out_dir, "matched_predictions_with_hy_filtered_remove_HY3.csv"), index=False, encoding="utf-8-sig")
    removed.to_csv(os.path.join(out_dir, "removed_HY3_samples.csv"), index=False, encoding="utf-8-sig")

    summary = {
        "rule": "remove HY=3 directly and keep only HY=0, HY=1, HY=2",
        "n_before": int(len(df)),
        "n_removed_HY3": int(len(removed)),
        "n_after": int(len(kept)),
        "counts_before": {str(k): int(v) for k, v in df[TARGET_COL].value_counts().sort_index().to_dict().items()},
        "counts_after": {str(k): int(v) for k, v in kept[TARGET_COL].value_counts().sort_index().to_dict().items()},
    }
    with open(os.path.join(out_dir, "remove_HY3_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return kept.reset_index(drop=True)


def fit_oof_pairwise_hy_classifier(
    df_matched: pd.DataFrame,
    out_dir: str,
    task_name: str,
    neg_hy: str,
    pos_hy: str,
    seed: int = 42,
    class_weights: str = "",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Pairwise HY classification:
      neg_hy -> class 0
      pos_hy -> class 1

    The outer fold split is inherited from the original OOF prediction files.
    """
    os.makedirs(out_dir, exist_ok=True)

    pred_feature_cols = [f"{c}_pred" for c in SCORE_COLS]
    for c in pred_feature_cols + [TARGET_COL, "fold"]:
        if c not in df_matched.columns:
            raise ValueError(f"Missing required column for classification stage: {c}")

    neg_hy = str(neg_hy)
    pos_hy = str(pos_hy)

    df_task = df_matched[df_matched[TARGET_COL].astype(str).isin([neg_hy, pos_hy])].copy().reset_index(drop=True)
    if len(df_task) == 0:
        raise ValueError(f"Task {task_name}: no samples found for HY={neg_hy} vs HY={pos_hy}.")

    df_task["target_class"] = (df_task[TARGET_COL].astype(str) == pos_hy).astype(int)
    df_task["pairwise_negative_HY"] = neg_hy
    df_task["pairwise_positive_HY"] = pos_hy

    df_task.to_csv(os.path.join(out_dir, f"{task_name}_matched_data.csv"), index=False, encoding="utf-8-sig")

    fold_ids = sorted(df_task["fold"].astype(int).unique().tolist())
    if len(fold_ids) != 5:
        print(f"[Warn] Task {task_name}: fold ids are {fold_ids}, not exactly 5 folds. The script will still use them as provided.")

    all_pred_rows = []
    metric_rows = []
    device = "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"

    num_classes = 2
    idx_to_class = {0: neg_hy, 1: pos_hy}

    for fold in fold_ids:
        train_df = df_task[df_task["fold"].astype(int) != int(fold)].copy().reset_index(drop=True)
        val_df = df_task[df_task["fold"].astype(int) == int(fold)].copy().reset_index(drop=True)

        if len(val_df) == 0:
            print(f"[Warn] Task {task_name}, fold {fold}: validation set is empty. Skipping this fold.")
            continue

        x_train = train_df[pred_feature_cols].astype(np.float32).values
        y_train = train_df["target_class"].astype(int).values
        x_val = val_df[pred_feature_cols].astype(np.float32).values
        y_val = val_df["target_class"].astype(int).values

        if len(np.unique(y_train)) < 2:
            raise ValueError(f"Task {task_name}, fold {fold}: training data has fewer than 2 classes after filtering.")

        y_pred, prob, weight_info = _train_torch_mlp_weighted_ce(
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            num_classes=num_classes,
            seed=int(seed) + int(fold),
            device=device,
            class_weights=class_weights,
        )

        metrics = compute_binary_metrics_from_multiclass(y_val, y_pred)
        row = {
            "task": task_name,
            "negative_HY_for_specificity": neg_hy,
            "positive_HY_for_sensitivity": pos_hy,
            "fold": int(fold),
            "n_train": int(len(train_df)),
            "n_val": int(len(val_df)),
            "accuracy": float(metrics["accuracy"]),
            "specificity": float(metrics["specificity"]),
            "sensitivity": float(metrics["sensitivity"]),
            "tn": int(metrics["tn"]),
            "fp": int(metrics["fp"]),
            "fn": int(metrics["fn"]),
            "tp": int(metrics["tp"]),
            "weight_mode": str(weight_info["weight_mode"]),
        }
        for k, w in enumerate(weight_info["class_weights"]):
            row[f"class_weight_HY_{idx_to_class[k]}"] = float(w)
        metric_rows.append(row)

        save_binary_confusion_matrix(
            y_true=y_val,
            y_pred=y_pred,
            neg_hy=neg_hy,
            pos_hy=pos_hy,
            out_path=os.path.join(out_dir, f"{task_name}_confusion_matrix_fold_{fold}.csv"),
        )

        pred_out = val_df.copy()
        pred_out["task"] = task_name
        pred_out["target_binary"] = y_val
        pred_out["pred_binary"] = y_pred
        pred_out["true_pairwise_label"] = np.where(pred_out["target_binary"].values == 1, f"HY_{pos_hy}", f"HY_{neg_hy}")
        pred_out["pred_pairwise_label"] = np.where(pred_out["pred_binary"].values == 1, f"HY_{pos_hy}", f"HY_{neg_hy}")
        pred_out[f"prob_HY_{neg_hy}"] = prob[:, 0]
        pred_out[f"prob_HY_{pos_hy}"] = prob[:, 1]
        pred_out.to_csv(os.path.join(out_dir, f"{task_name}_predictions_fold_{fold}.csv"), index=False, encoding="utf-8-sig")
        all_pred_rows.append(pred_out)

    if len(all_pred_rows) == 0:
        raise ValueError(f"Task {task_name}: no fold predictions generated.")

    all_pred = pd.concat(all_pred_rows, axis=0, ignore_index=True)
    all_pred = all_pred.sort_values(["fold", "PATNO", "EVENT_ID", "INFODT", "EXAMDT"]).reset_index(drop=True)

    overall = compute_binary_metrics_from_multiclass(
        all_pred["target_binary"].astype(int).values,
        all_pred["pred_binary"].astype(int).values,
    )

    overall_row = {
        "task": task_name,
        "negative_HY_for_specificity": neg_hy,
        "positive_HY_for_sensitivity": pos_hy,
        "fold": "overall",
        "n_train": np.nan,
        "n_val": int(len(all_pred)),
        "accuracy": float(overall["accuracy"]),
        "specificity": float(overall["specificity"]),
        "sensitivity": float(overall["sensitivity"]),
        "tn": int(overall["tn"]),
        "fp": int(overall["fp"]),
        "fn": int(overall["fn"]),
        "tp": int(overall["tp"]),
    }

    resolved_weights, weight_mode = resolve_class_weights(
        y_train=df_task["target_class"].astype(int).values,
        num_classes=num_classes,
        class_weights=class_weights,
    )
    overall_row["weight_mode"] = str(weight_mode)
    overall_row[f"class_weight_HY_{neg_hy}"] = float(resolved_weights[0])
    overall_row[f"class_weight_HY_{pos_hy}"] = float(resolved_weights[1])
    metric_rows.append(overall_row)

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(os.path.join(out_dir, f"{task_name}_5fold_metrics.csv"), index=False, encoding="utf-8-sig")
    all_pred.to_csv(os.path.join(out_dir, f"{task_name}_oof_predictions_all_folds.csv"), index=False, encoding="utf-8-sig")

    save_binary_confusion_matrix(
        y_true=all_pred["target_binary"].astype(int).values,
        y_pred=all_pred["pred_binary"].astype(int).values,
        neg_hy=neg_hy,
        pos_hy=pos_hy,
        out_path=os.path.join(out_dir, f"{task_name}_confusion_matrix_overall.csv"),
    )

    summary = {
        "task": task_name,
        "negative_HY_for_specificity": neg_hy,
        "positive_HY_for_sensitivity": pos_hy,
        "input_feature_cols": pred_feature_cols,
        "target_col": TARGET_COL,
        "class_mapping": {f"HY_{neg_hy}": 0, f"HY_{pos_hy}": 1},
        "class_weight_setting": {
            "class_weights_input": str(class_weights),
            "class_weights_used_overall": resolved_weights.astype(float).tolist(),
            "weight_mode": str(weight_mode),
        },
        "fold_ids": fold_ids,
        "overall_accuracy": float(overall["accuracy"]),
        "overall_specificity": float(overall["specificity"]),
        "overall_sensitivity": float(overall["sensitivity"]),
        "overall_confusion_matrix": overall["confusion_matrix"].tolist(),
    }
    with open(os.path.join(out_dir, f"{task_name}_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return all_pred, metrics_df



def main():
    ap = argparse.ArgumentParser(
        description=(
            "Use five predicted UPDRS sub-scores from OOF fold files for three pairwise HY classification tasks. "
            "HY=3 samples are removed directly. Tasks: HY=0 vs HY=1, HY=1 vs HY=2, HY=0 vs HY=2."
        )
    )
    ap.add_argument("--csv_path", type=str, default="./data/UPDRS_all.csv")
    ap.add_argument("--pred_dir", type=str, default="./outputs/stage1/val_predictions_best_models")
    ap.add_argument("--out_dir", type=str, default="./outputs/stage2_hy")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--enable_filter", type=int, default=0, help="Optional exploratory filtering using true score errors; DISABLE for valid unbiased evaluation (default: 0).")
    ap.add_argument("--filter_mode", type=str, default="auto", choices=["auto", "manual"])
    ap.add_argument("--filter_mae_quantile", type=float, default=0.85, help="Used in auto mode.")
    ap.add_argument("--filter_maxae_quantile", type=float, default=0.85, help="Used in auto mode.")
    ap.add_argument("--filter_mae_threshold", type=float, default=-1.0, help="Used in manual mode.")
    ap.add_argument("--filter_maxae_threshold", type=float, default=-1.0, help="Used in manual mode.")

    ap.add_argument(
        "--class_weights",
        type=str,
        default="",
        help=(
            "Optional manual class weights for each binary task in order negative,positive. "
            "Example: '1,2'. Empty means auto balanced weights."
        ),
    )

    args = ap.parse_args()

    set_seed(args.seed)
    if int(args.enable_filter):
        print("[WARNING] Filtering by the error between predicted and TRUE motor scores "
              "uses outcome information; results are NOT suitable for unbiased held-out evaluation.")
    os.makedirs(args.out_dir, exist_ok=True)

    print("[1/6] Reading source csv and building precise HY table ...")
    hy_df, class_to_idx, idx_to_class = prepare_hy_table(args.csv_path)
    print(f"[Info] Source HY table rows after dedup: {len(hy_df)}")
    print(f"[Info] HY class mapping from source: {class_to_idx}")

    print("[2/6] Reading fold prediction files and performing precise matching ...")
    matched_df = read_and_match_prediction_files(args.pred_dir, hy_df)
    matched_df.to_csv(os.path.join(args.out_dir, "matched_predictions_with_hy.csv"), index=False, encoding="utf-8-sig")
    print(f"[Info] Precisely matched OOF prediction rows: {len(matched_df)}")
    print(f"[Info] HY counts after matching: {matched_df[TARGET_COL].astype(str).value_counts().sort_index().to_dict()}")

    print("[3/6] Filtering ids with overly large gaps between predicted and true sub-scores ...")
    filtered_df = filter_large_gap_samples(
        df_matched=matched_df,
        out_dir=args.out_dir,
        enable_filter=bool(int(args.enable_filter)),
        filter_mode=args.filter_mode,
        mae_quantile=float(args.filter_mae_quantile),
        maxae_quantile=float(args.filter_maxae_quantile),
        mae_threshold=float(args.filter_mae_threshold),
        maxae_threshold=float(args.filter_maxae_threshold),
    )
    print(f"[Info] Rows after gap filtering: {len(filtered_df)}")
    print(f"[Info] HY counts after gap filtering: {filtered_df[TARGET_COL].astype(str).value_counts().sort_index().to_dict()}")

    print("[4/6] Removing HY=3 samples and keeping HY=0/1/2 ...")
    kept_df = remove_hy3_samples(filtered_df, args.out_dir)
    print(f"[Info] Rows after removing HY=3: {len(kept_df)}")
    print(f"[Info] HY counts after removing HY=3: {kept_df[TARGET_COL].astype(str).value_counts().sort_index().to_dict()}")

    print("[5/6] Training three pairwise HY classifiers ...")
    tasks = [
        ("hy0_vs_hy1", "0", "1"),
        ("hy1_vs_hy2", "1", "2"),
        ("hy0_vs_hy2", "0", "2"),
    ]

    overall_rows = []
    for task_name, neg_hy, pos_hy in tasks:
        task_dir = os.path.join(args.out_dir, task_name)
        _, metrics_df = fit_oof_pairwise_hy_classifier(
            kept_df,
            out_dir=task_dir,
            task_name=task_name,
            neg_hy=neg_hy,
            pos_hy=pos_hy,
            seed=args.seed,
            class_weights=args.class_weights,
        )
        overall = metrics_df[metrics_df["fold"].astype(str) == "overall"].iloc[0].to_dict()
        overall_rows.append(overall)
        print(
            f"[{task_name}] Accuracy={float(overall['accuracy']):.6f}, "
            f"Specificity={float(overall['specificity']):.6f}, "
            f"Sensitivity={float(overall['sensitivity']):.6f}"
        )

    print("[6/6] Saving summary ...")
    overall_df = pd.DataFrame(overall_rows)
    overall_df.to_csv(os.path.join(args.out_dir, "all_pairwise_hy_overall_metrics.csv"), index=False, encoding="utf-8-sig")

    confusion_files = []
    for root, _, files in os.walk(args.out_dir):
        for fn in files:
            if "confusion_matrix" in fn and fn.endswith(".csv"):
                rel = os.path.relpath(os.path.join(root, fn), args.out_dir)
                confusion_files.append(rel)
    confusion_files = sorted(confusion_files)
    pd.DataFrame({"confusion_matrix_file": confusion_files}).to_csv(
        os.path.join(args.out_dir, "all_pairwise_hy_confusion_matrix_files.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    summary = {
        "task": "pairwise HY classification after removing HY=3",
        "removed_class": "HY=3",
        "pairwise_tasks": [
            {"task": "hy0_vs_hy1", "negative_HY_for_specificity": "0", "positive_HY_for_sensitivity": "1"},
            {"task": "hy1_vs_hy2", "negative_HY_for_specificity": "1", "positive_HY_for_sensitivity": "2"},
            {"task": "hy0_vs_hy2", "negative_HY_for_specificity": "0", "positive_HY_for_sensitivity": "2"},
        ],
        "final_HY_counts_after_HY3_removed": {str(k): int(v) for k, v in kept_df[TARGET_COL].astype(str).value_counts().sort_index().to_dict().items()},
        "class_weights_input_for_binary_tasks": str(args.class_weights),
        "output_dir": args.out_dir,
        "confusion_matrix_files": confusion_files,
    }
    with open(os.path.join(args.out_dir, "all_pairwise_hy_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Outputs saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
