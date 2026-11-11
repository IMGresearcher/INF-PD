"""Stage II: diagnosis classification from out-of-fold predicted motor scores.

Research implementation; use deidentified inputs and review evaluation protocol.
"""


import os
import re
import json
import glob
import random
import argparse
from typing import Any, Dict, List, Tuple

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
LABEL_COL = "label"
TOTAL_COL = "UPDRS总分"

# 三分类固定顺序：0=control, 1=prodromal, 2=PD
THREE_CLASS_NAMES = ["control", "prodromal", "pd"]
THREE_CLASS_TO_ID = {name: i for i, name in enumerate(THREE_CLASS_NAMES)}
THREE_ID_TO_CLASS = {i: name for name, i in THREE_CLASS_TO_ID.items()}


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


def normalize_label_3class(x: Any) -> str:
    """PD/control/prodromal 三分类标签标准化。"""
    if pd.isna(x):
        return ""
    s = str(x).strip().lower()
    if s in {"pd", "parkinson", "parkinson's disease", "parkinsons"}:
        return "pd"
    if s in {"control", "healthy", "healthy control", "hc"}:
        return "control"
    if s in {"pandromal", "prodromal", "prod"}:
        return "prodromal"
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


def prepare_label_table(csv_path: str) -> pd.DataFrame:
    df = read_csv_flexible(csv_path).copy()
    required = KEY_COLS + SCORE_COLS + [LABEL_COL, TOTAL_COL]
    missing = [c for c in required if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"CSV is missing required columns: {missing}")

    for c in KEY_COLS:
        df[c] = df[c].map(canonical_str)
    for c in SCORE_COLS + [TOTAL_COL]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["original_label"] = df[LABEL_COL].map(normalize_label_3class)
    bad_labels = sorted(set(df.loc[~df["original_label"].isin(THREE_CLASS_NAMES), "original_label"].astype(str).tolist()))
    if len(bad_labels) > 0:
        raise ValueError(f"Unsupported labels after normalization: {bad_labels}")

    df["precise_match_key"] = build_precise_match_key(df, SCORE_COLS)

    rows = []
    conflicts = []
    for key, sub in df.groupby("precise_match_key", sort=False):
        labels = sorted(set(sub["original_label"].dropna().astype(str).tolist()))
        totals = sorted(set([float(x) for x in sub[TOTAL_COL].dropna().tolist()]))

        if len(labels) == 0:
            continue
        if len(labels) > 1:
            conflicts.append((key, "label_conflict", labels))
            continue
        if len(totals) == 0:
            continue
        if len(totals) > 1:
            conflicts.append((key, "total_conflict", totals))
            continue

        row = sub.iloc[0][KEY_COLS + SCORE_COLS].to_dict()
        row["precise_match_key"] = key
        row["original_label"] = str(labels[0])
        row[TOTAL_COL] = float(totals[0])
        rows.append(row)

    if len(conflicts) > 0:
        raise ValueError(
            "Found duplicated source rows with the same precise key but conflicting label or UPDRS total. "
            f"Examples: {conflicts[:10]}"
        )

    out = pd.DataFrame(rows)
    if len(out) == 0:
        raise ValueError("No usable rows remained after building the label table.")
    return out


def read_and_match_prediction_files(pred_dir: str, label_df: pd.DataFrame) -> pd.DataFrame:
    files = detect_prediction_files(pred_dir)
    label_lookup = label_df[["precise_match_key", "original_label", TOTAL_COL]].drop_duplicates().copy()

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

        merged = tmp.merge(label_lookup, on="precise_match_key", how="left", validate="one_to_one")
        unmatched = merged[merged["original_label"].isna()].copy()
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
        keep_cols += expected_true_cols + expected_pred_cols + ["original_label", TOTAL_COL]
        all_rows.append(merged[keep_cols].copy())

    out = pd.concat(all_rows, axis=0, ignore_index=True)
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
        df.to_csv(os.path.join(out_dir, "matched_predictions_with_gap_metrics.csv"), index=False, encoding="utf-8-sig")
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

    df.to_csv(os.path.join(out_dir, "matched_predictions_with_gap_metrics.csv"), index=False, encoding="utf-8-sig")
    kept_df.to_csv(os.path.join(out_dir, "matched_predictions_filtered_before_relabel.csv"), index=False, encoding="utf-8-sig")
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


def apply_low_total_prodromal_to_control(
    df: pd.DataFrame,
    out_dir: str,
    target_control_count: int = 100,
    enable_relabel: bool = True,
) -> pd.DataFrame:
    os.makedirs(out_dir, exist_ok=True)
    out = df.copy()
    out["final_label"] = out["original_label"].astype(str)
    out["reassigned_prodromal_to_control"] = 0

    before_counts = out["final_label"].value_counts().to_dict()
    current_control = int((out["final_label"] == "control").sum())
    n_need = max(int(target_control_count) - current_control, 0)

    selected_keys: List[str] = []
    if enable_relabel and n_need > 0:
        cand = out[out["final_label"] == "prodromal"].copy()
        cand[TOTAL_COL] = pd.to_numeric(cand[TOTAL_COL], errors="coerce")
        cand = cand.sort_values(
            by=[TOTAL_COL, "PATNO", "EVENT_ID", "INFODT", "EXAMDT"],
            ascending=[True, True, True, True, True],
        )
        selected_idx = cand.head(n_need).index.tolist()
        selected_keys = out.loc[selected_idx, "precise_match_key"].astype(str).tolist()
        out.loc[selected_idx, "final_label"] = "control"
        out.loc[selected_idx, "reassigned_prodromal_to_control"] = 1

    bad_labels = sorted(set(out.loc[~out["final_label"].isin(THREE_CLASS_NAMES), "final_label"].astype(str).tolist()))
    if len(bad_labels) > 0:
        raise ValueError(f"Unexpected final labels: {bad_labels}")

    out["target_class"] = out["final_label"].map(THREE_CLASS_TO_ID).astype(int)
    after_counts = out["final_label"].value_counts().to_dict()

    out.to_csv(os.path.join(out_dir, "matched_predictions_filtered_relabelled.csv"), index=False, encoding="utf-8-sig")
    out[out["reassigned_prodromal_to_control"] == 1].to_csv(
        os.path.join(out_dir, "reassigned_low_total_prodromal_to_control.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    summary = {
        "enable_relabel": int(bool(enable_relabel)),
        "target_control_count": int(target_control_count),
        "control_count_before": int(current_control),
        "needed_from_prodromal": int(n_need),
        "actually_reassigned": int(len(selected_keys)),
        "counts_before": {str(k): int(v) for k, v in before_counts.items()},
        "counts_after": {str(k): int(v) for k, v in after_counts.items()},
        "selection_rule": "lowest UPDRS总分 among original prodromal samples after precise matching and gap filtering",
    }
    with open(os.path.join(out_dir, "prodromal_to_control_relabel_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return out.reset_index(drop=True)


class TorchMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 1):
        super().__init__()
        self.out_dim = int(out_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, int(out_dim)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.net(x)
        if self.out_dim == 1:
            return y.view(-1)
        return y


def _balanced_binary_pos_weight(y_train: np.ndarray) -> float:
    y_train = np.asarray(y_train).astype(int).reshape(-1)
    n_pos = float((y_train == 1).sum())
    n_neg = float((y_train == 0).sum())
    return (n_neg / max(n_pos, 1.0)) if (n_pos + n_neg) > 0 else 1.0


def _balanced_multiclass_weights(y_train: np.ndarray, num_classes: int) -> np.ndarray:
    y_train = np.asarray(y_train).astype(int).reshape(-1)
    counts = np.bincount(y_train, minlength=int(num_classes)).astype(np.float32)
    total = float(counts.sum())
    weights = total / np.maximum(float(num_classes) * counts, 1.0)
    weights = weights / max(float(weights.mean()), 1e-8)
    return weights.astype(np.float32)


def _train_torch_mlp_binary(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    seed: int,
    device: str,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    max_epochs: int = 500,
    patience: int = 40,
    batch_size: int = 64,
) -> Tuple[np.ndarray, np.ndarray]:
    if torch is None or nn is None:
        raise ImportError("Missing PyTorch. Please install torch in your environment.")

    rng = np.random.RandomState(seed)
    torch.manual_seed(int(seed))
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

    x_scaler = StandardScaler()
    x_train_s = x_scaler.fit_transform(x_train).astype(np.float32)
    x_val_s = x_scaler.transform(x_val).astype(np.float32)

    y_train = np.asarray(y_train).astype(np.float32).reshape(-1)
    y_val = np.asarray(y_val).astype(np.float32).reshape(-1)

    pos_weight = _balanced_binary_pos_weight(y_train)

    model = TorchMLP(in_dim=int(x_train_s.shape[1]), out_dim=1).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float32, device=device))
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
            yb = torch.from_numpy(y_train[sel]).to(device)
            optim.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optim.step()

        model.eval()
        with torch.no_grad():
            xv = torch.from_numpy(x_val_s).to(device)
            yv = torch.from_numpy(y_val).to(device)
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
        logits = model(xv).detach().cpu().numpy().reshape(-1)
        prob_pos = (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)
        pred_binary = (prob_pos >= 0.5).astype(int)

    return pred_binary, prob_pos


def _train_torch_mlp_multiclass(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    num_classes: int,
    seed: int,
    device: str,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    max_epochs: int = 500,
    patience: int = 40,
    batch_size: int = 64,
) -> Tuple[np.ndarray, np.ndarray]:
    if torch is None or nn is None:
        raise ImportError("Missing PyTorch. Please install torch in your environment.")

    rng = np.random.RandomState(seed)
    torch.manual_seed(int(seed))
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

    x_scaler = StandardScaler()
    x_train_s = x_scaler.fit_transform(x_train).astype(np.float32)
    x_val_s = x_scaler.transform(x_val).astype(np.float32)

    y_train = np.asarray(y_train).astype(np.int64).reshape(-1)
    y_val = np.asarray(y_val).astype(np.int64).reshape(-1)

    class_weights = _balanced_multiclass_weights(y_train, num_classes=num_classes)

    model = TorchMLP(in_dim=int(x_train_s.shape[1]), out_dim=int(num_classes)).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=device))
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

    return pred_class, prob


def compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int).reshape(-1)
    y_pred = np.asarray(y_pred).astype(int).reshape(-1)
    acc = float(accuracy_score(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
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
    }


def compute_multiclass_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> Dict[str, Any]:
    y_true = np.asarray(y_true).astype(int).reshape(-1)
    y_pred = np.asarray(y_pred).astype(int).reshape(-1)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    acc = float(accuracy_score(y_true, y_pred))

    total = int(cm.sum())
    per_class = {}
    sens = []
    spec = []
    for cls in range(num_classes):
        tp = int(cm[cls, cls])
        fn = int(cm[cls, :].sum() - tp)
        fp = int(cm[:, cls].sum() - tp)
        tn = int(total - tp - fn - fp)
        sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        sens.append(sensitivity)
        spec.append(specificity)
        per_class[cls] = {
            "sensitivity": sensitivity,
            "specificity": specificity,
            "tp": tp,
            "fn": fn,
            "fp": fp,
            "tn": tn,
        }
    return {
        "accuracy": acc,
        "macro_sensitivity": float(np.mean(sens)) if len(sens) > 0 else 0.0,
        "macro_specificity": float(np.mean(spec)) if len(spec) > 0 else 0.0,
        "confusion_matrix": cm,
        "per_class": per_class,
    }


def fit_oof_three_class(df_matched: pd.DataFrame, out_dir: str, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    os.makedirs(out_dir, exist_ok=True)

    pred_feature_cols = [f"{c}_pred" for c in SCORE_COLS]
    required = pred_feature_cols + ["target_class", "fold", "final_label"]
    missing = [c for c in required if c not in df_matched.columns]
    if len(missing) > 0:
        raise ValueError(f"Missing required columns for three-class classification: {missing}")

    fold_ids = sorted(df_matched["fold"].astype(int).unique().tolist())
    all_pred_rows = []
    metric_rows = []
    device = "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"
    num_classes = len(THREE_CLASS_NAMES)

    for fold in fold_ids:
        train_df = df_matched[df_matched["fold"].astype(int) != int(fold)].copy().reset_index(drop=True)
        val_df = df_matched[df_matched["fold"].astype(int) == int(fold)].copy().reset_index(drop=True)

        x_train = train_df[pred_feature_cols].astype(np.float32).values
        y_train = train_df["target_class"].astype(int).values
        x_val = val_df[pred_feature_cols].astype(np.float32).values
        y_val = val_df["target_class"].astype(int).values

        if len(val_df) == 0:
            raise ValueError(f"Fold {fold}: validation data is empty.")
        if len(np.unique(y_train)) < 2:
            raise ValueError(f"Fold {fold}: training data has fewer than 2 classes.")

        y_pred, prob = _train_torch_mlp_multiclass(
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            num_classes=num_classes,
            seed=int(seed) + int(fold),
            device=device,
        )

        metrics = compute_multiclass_metrics(y_val, y_pred, num_classes=num_classes)
        row = {
            "task": "three_class",
            "fold": int(fold),
            "n_val": int(len(val_df)),
            "accuracy": float(metrics["accuracy"]),
            "macro_specificity": float(metrics["macro_specificity"]),
            "macro_sensitivity": float(metrics["macro_sensitivity"]),
        }
        for cls_id, cls_name in THREE_ID_TO_CLASS.items():
            row[f"sensitivity_{cls_name}"] = float(metrics["per_class"][cls_id]["sensitivity"])
            row[f"specificity_{cls_name}"] = float(metrics["per_class"][cls_id]["specificity"])
        metric_rows.append(row)

        pred_out = val_df.copy()
        pred_out["pred_class"] = y_pred
        pred_out["pred_label"] = [THREE_ID_TO_CLASS[int(v)] for v in y_pred]
        for cls_id, cls_name in THREE_ID_TO_CLASS.items():
            pred_out[f"prob_{cls_name}"] = prob[:, cls_id]
        pred_out.to_csv(os.path.join(out_dir, f"three_class_predictions_fold_{fold}.csv"), index=False, encoding="utf-8-sig")
        all_pred_rows.append(pred_out)

    all_pred = pd.concat(all_pred_rows, axis=0, ignore_index=True)
    all_pred = all_pred.sort_values(["fold", "PATNO", "EVENT_ID", "INFODT", "EXAMDT"]).reset_index(drop=True)

    overall = compute_multiclass_metrics(
        all_pred["target_class"].astype(int).values,
        all_pred["pred_class"].astype(int).values,
        num_classes=num_classes,
    )
    overall_row = {
        "task": "three_class",
        "fold": "overall",
        "n_val": int(len(all_pred)),
        "accuracy": float(overall["accuracy"]),
        "macro_specificity": float(overall["macro_specificity"]),
        "macro_sensitivity": float(overall["macro_sensitivity"]),
    }
    for cls_id, cls_name in THREE_ID_TO_CLASS.items():
        overall_row[f"sensitivity_{cls_name}"] = float(overall["per_class"][cls_id]["sensitivity"])
        overall_row[f"specificity_{cls_name}"] = float(overall["per_class"][cls_id]["specificity"])
    metric_rows.append(overall_row)

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(os.path.join(out_dir, "three_class_metrics.csv"), index=False, encoding="utf-8-sig")
    all_pred.to_csv(os.path.join(out_dir, "three_class_oof_predictions_all_folds.csv"), index=False, encoding="utf-8-sig")

    cm_df = pd.DataFrame(
        overall["confusion_matrix"],
        index=[f"true_{THREE_ID_TO_CLASS[i]}" for i in range(num_classes)],
        columns=[f"pred_{THREE_ID_TO_CLASS[i]}" for i in range(num_classes)],
    )
    cm_df.to_csv(os.path.join(out_dir, "three_class_confusion_matrix_overall.csv"), index=True, encoding="utf-8-sig")

    return all_pred, metrics_df


def fit_oof_binary_task(
    df_all: pd.DataFrame,
    out_dir: str,
    task_name: str,
    neg_label: str,
    pos_label: str,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    os.makedirs(out_dir, exist_ok=True)

    pred_feature_cols = [f"{c}_pred" for c in SCORE_COLS]
    df_task = df_all[df_all["final_label"].isin([neg_label, pos_label])].copy().reset_index(drop=True)
    if len(df_task) == 0:
        raise ValueError(f"Task {task_name}: no samples after selecting labels {neg_label}, {pos_label}.")

    df_task["target_binary"] = (df_task["final_label"].astype(str) == str(pos_label)).astype(int)

    fold_ids = sorted(df_task["fold"].astype(int).unique().tolist())
    all_pred_rows = []
    metric_rows = []
    device = "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"

    df_task.to_csv(os.path.join(out_dir, f"{task_name}_matched_filtered_data.csv"), index=False, encoding="utf-8-sig")

    for fold in fold_ids:
        train_df = df_task[df_task["fold"].astype(int) != int(fold)].copy().reset_index(drop=True)
        val_df = df_task[df_task["fold"].astype(int) == int(fold)].copy().reset_index(drop=True)

        if len(val_df) == 0:
            print(f"[Warn] Task {task_name}, fold {fold}: empty validation subset, skipping this fold.")
            continue

        x_train = train_df[pred_feature_cols].astype(np.float32).values
        y_train = train_df["target_binary"].astype(int).values
        x_val = val_df[pred_feature_cols].astype(np.float32).values
        y_val = val_df["target_binary"].astype(int).values

        if len(np.unique(y_train)) < 2:
            raise ValueError(f"Task {task_name}, fold {fold}: training data has fewer than 2 classes.")

        y_pred, prob_pos = _train_torch_mlp_binary(
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            seed=int(seed) + int(fold),
            device=device,
        )

        metrics = compute_binary_metrics(y_val, y_pred)
        metrics["task"] = task_name
        metrics["fold"] = int(fold)
        metrics["n_val"] = int(len(val_df))
        metrics["negative_label_for_specificity"] = neg_label
        metrics["positive_label_for_sensitivity"] = pos_label
        metric_rows.append(metrics)

        pred_out = val_df.copy()
        pred_out["task"] = task_name
        pred_out["target_binary"] = y_val
        pred_out["pred_binary"] = y_pred
        pred_out["prob_negative"] = (1.0 - prob_pos).astype(np.float32)
        pred_out["prob_positive"] = prob_pos.astype(np.float32)
        pred_out["pred_label"] = np.where(pred_out["pred_binary"].values == 1, pos_label, neg_label)
        pred_out.to_csv(os.path.join(out_dir, f"{task_name}_predictions_fold_{fold}.csv"), index=False, encoding="utf-8-sig")
        all_pred_rows.append(pred_out)

    if len(all_pred_rows) == 0:
        raise ValueError(f"Task {task_name}: no fold predictions generated.")

    all_pred = pd.concat(all_pred_rows, axis=0, ignore_index=True)
    all_pred = all_pred.sort_values(["fold", "PATNO", "EVENT_ID", "INFODT", "EXAMDT"]).reset_index(drop=True)

    overall = compute_binary_metrics(
        all_pred["target_binary"].astype(int).values,
        all_pred["pred_binary"].astype(int).values,
    )
    overall["task"] = task_name
    overall["fold"] = "overall"
    overall["n_val"] = int(len(all_pred))
    overall["negative_label_for_specificity"] = neg_label
    overall["positive_label_for_sensitivity"] = pos_label
    metric_rows.append(overall)

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(os.path.join(out_dir, f"{task_name}_metrics.csv"), index=False, encoding="utf-8-sig")
    all_pred.to_csv(os.path.join(out_dir, f"{task_name}_oof_predictions_all_folds.csv"), index=False, encoding="utf-8-sig")

    cm = confusion_matrix(
        all_pred["target_binary"].astype(int).values,
        all_pred["pred_binary"].astype(int).values,
        labels=[0, 1],
    )
    cm_df = pd.DataFrame(
        cm,
        index=[f"true_{neg_label}(0)", f"true_{pos_label}(1)"],
        columns=[f"pred_{neg_label}(0)", f"pred_{pos_label}(1)"],
    )
    cm_df.to_csv(os.path.join(out_dir, f"{task_name}_confusion_matrix_overall.csv"), index=True, encoding="utf-8-sig")

    return all_pred, metrics_df


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Use five predicted UPDRS sub-scores from OOF fold files for 3-class classification "
            "and pairwise classification. Low-UPDRS-total prodromal samples are reassigned to control "
            "to increase control count to around the target number."
        )
    )
    ap.add_argument("--csv_path", type=str, default="./data/UPDRS_all.csv")
    ap.add_argument("--pred_dir", type=str, default="./outputs/stage1/val_predictions_best_models")
    ap.add_argument("--out_dir", type=str, default="./outputs/stage2_diagnosis")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--enable_filter", type=int, default=0, help="Optional exploratory filtering using true score errors; DISABLE for valid unbiased evaluation (default: 0).")
    ap.add_argument("--filter_mode", type=str, default="auto", choices=["auto", "manual"])
    ap.add_argument("--filter_mae_quantile", type=float, default=0.85, help="Used in auto mode.")
    ap.add_argument("--filter_maxae_quantile", type=float, default=0.85, help="Used in auto mode.")
    ap.add_argument("--filter_mae_threshold", type=float, default=-1.0, help="Used in manual mode.")
    ap.add_argument("--filter_maxae_threshold", type=float, default=-1.0, help="Used in manual mode.")

    ap.add_argument("--target_control_count", type=int, default=100, help="Target final control count after moving low-UPDRS-total prodromal samples.")
    ap.add_argument("--enable_prodromal_to_control", type=int, default=0, help="Exploratory relabeling based on UPDRS total; not recommended for diagnosis evaluation (default: 0).")

    args = ap.parse_args()

    set_seed(args.seed)
    if int(args.enable_filter):
        print("[WARNING] Filtering by the error between predicted and TRUE motor scores "
              "uses outcome information; results are NOT suitable for unbiased held-out evaluation.")
    if int(args.enable_prodromal_to_control):
        print("[WARNING] Relabeling prodromal participants using UPDRS total changes the "
              "clinical task definition; report this as an exploratory experiment only.")
    os.makedirs(args.out_dir, exist_ok=True)

    print("[1/6] Reading source csv and building original 3-class label table ...")
    label_df = prepare_label_table(args.csv_path)
    print(f"[Info] Source label table rows after dedup: {len(label_df)}")
    print(f"[Info] Original label counts in source table: {label_df['original_label'].value_counts().to_dict()}")

    print("[2/6] Reading fold prediction files and performing precise matching ...")
    matched_df = read_and_match_prediction_files(args.pred_dir, label_df)
    matched_df.to_csv(os.path.join(args.out_dir, "matched_predictions_with_original_labels.csv"), index=False, encoding="utf-8-sig")
    print(f"[Info] Precisely matched OOF prediction rows: {len(matched_df)}")
    print(f"[Info] Original label counts after matching: {matched_df['original_label'].value_counts().to_dict()}")

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
    print(f"[Info] Rows after filtering: {len(filtered_df)}")
    print(f"[Info] Original label counts after filtering: {filtered_df['original_label'].value_counts().to_dict()}")

    print("[4/6] Reassigning low-UPDRS-total prodromal samples to control ...")
    final_df = apply_low_total_prodromal_to_control(
        filtered_df,
        out_dir=args.out_dir,
        target_control_count=int(args.target_control_count),
        enable_relabel=bool(int(args.enable_prodromal_to_control)),
    )
    print(f"[Info] Final label counts for classification: {final_df['final_label'].value_counts().to_dict()}")

    task_summaries = []

    print("[5/6] Training 3-class MLP classifier: control / prodromal / PD ...")
    three_dir = os.path.join(args.out_dir, "three_class")
    _, three_metrics = fit_oof_three_class(final_df, three_dir, seed=args.seed)
    three_overall = three_metrics[three_metrics["fold"].astype(str) == "overall"].iloc[0].to_dict()
    task_summaries.append(three_overall)
    print(
        f"[Three-class] Accuracy={float(three_overall['accuracy']):.6f}, "
        f"Macro-Specificity={float(three_overall['macro_specificity']):.6f}, "
        f"Macro-Sensitivity={float(three_overall['macro_sensitivity']):.6f}"
    )

    print("[6/6] Training pairwise MLP classifiers ...")
    pair_tasks = [
        ("pd_vs_control", "control", "pd"),
        ("control_vs_prodromal", "control", "prodromal"),
        ("pd_vs_prodromal", "prodromal", "pd"),
    ]

    for task_name, neg_label, pos_label in pair_tasks:
        task_dir = os.path.join(args.out_dir, task_name)
        _, metric_df = fit_oof_binary_task(
            final_df,
            out_dir=task_dir,
            task_name=task_name,
            neg_label=neg_label,
            pos_label=pos_label,
            seed=args.seed,
        )
        overall = metric_df[metric_df["fold"].astype(str) == "overall"].iloc[0].to_dict()
        task_summaries.append(overall)
        print(
            f"[{task_name}] Accuracy={float(overall['accuracy']):.6f}, "
            f"Specificity={float(overall['specificity']):.6f}, "
            f"Sensitivity={float(overall['sensitivity']):.6f}"
        )

    summary_df = pd.DataFrame(task_summaries)
    summary_df.to_csv(os.path.join(args.out_dir, "all_task_overall_metrics.csv"), index=False, encoding="utf-8-sig")

    summary = {
        "task_design": {
            "three_class": THREE_CLASS_NAMES,
            "pairwise_tasks": [
                {"task": "pd_vs_control", "negative_label_for_specificity": "control", "positive_label_for_sensitivity": "pd"},
                {"task": "control_vs_prodromal", "negative_label_for_specificity": "control", "positive_label_for_sensitivity": "prodromal"},
                {"task": "pd_vs_prodromal", "negative_label_for_specificity": "prodromal", "positive_label_for_sensitivity": "pd"},
            ],
        },
        "target_control_count": int(args.target_control_count),
        "enable_prodromal_to_control": int(args.enable_prodromal_to_control),
        "final_label_counts": {str(k): int(v) for k, v in final_df["final_label"].value_counts().to_dict().items()},
        "output_dir": args.out_dir,
    }
    with open(os.path.join(args.out_dir, "all_task_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"[Done] Outputs saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
