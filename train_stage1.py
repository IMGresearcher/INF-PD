"""Stage I: participant-level cross-validated motor sub-domain regression.

Research implementation; use deidentified inputs and review evaluation protocol.
"""

import os
import re
import glob
import json
import math
import shutil
import random
import argparse
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold

from model_connectome import ConnectomeSubdomainNet


# =========================
# Utilitiesc
# =========================
EPS = 1e-8


# Common Chinese font candidates across Windows / macOS / Linux
CHINESE_FONT_CANDIDATES = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Noto Sans SC",
    "Source Han Sans SC",
    "WenQuanYi Zen Hei",
    "PingFang SC",
    "Heiti SC",
    "Arial Unicode MS",
]


def configure_matplotlib_for_chinese() -> Optional[str]:
    """
    Try to find an installed Chinese-capable font and configure matplotlib to use it.
    Returns the chosen font name, or None if no candidate is found.
    """
    # Matplotlib may start with a stale / incomplete font cache on some machines.
    # We try to refresh the font manager once so Chinese fonts become discoverable.
    try:
        for fp in font_manager.findSystemFonts(fontext="ttf"):
            try:
                font_manager.fontManager.addfont(fp)
            except Exception:
                pass
    except Exception:
        pass

    try:
        available = {f.name for f in font_manager.fontManager.ttflist}
    except Exception:
        available = set()

    chosen = None
    for name in CHINESE_FONT_CANDIDATES:
        if name in available:
            chosen = name
            break

    if chosen is not None:
        matplotlib.rcParams["font.family"] = "sans-serif"
        matplotlib.rcParams["font.sans-serif"] = [chosen] + [
            f for f in matplotlib.rcParams.get("font.sans-serif", []) if f != chosen
        ]
    matplotlib.rcParams["axes.unicode_minus"] = False
    return chosen


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def read_csv_flexible(csv_path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(csv_path, encoding="utf-8-sig")
    except Exception:
        try:
            return pd.read_csv(csv_path, encoding="gbk")
        except Exception:
            return pd.read_csv(csv_path)


def detect_id_col(df: pd.DataFrame) -> str:
    if "PATNO" in df.columns:
        return "PATNO"
    if "id" in df.columns:
        return "id"
    return df.columns[0]


def detect_score_cols(df: pd.DataFrame, id_col: str) -> List[str]:
    score_cols = ["运动迟缓", "肌强直", "静止性震颤", "姿势性/动作性震颤", "中轴"]
    if all(c in df.columns for c in score_cols):
        return score_cols

    print("[Warn] Chinese score columns not found. Falling back to first 5 numeric columns.")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != id_col]
    if len(numeric_cols) < 5:
        raise ValueError("Cannot detect 5 score columns from CSV.")
    score_cols = numeric_cols[:5]
    print(f"[Info] Using score columns: {score_cols}")
    return score_cols


DEFAULT_SCORE_UPPER_BOUNDS: Dict[str, float] = {
    "运动迟缓": 32.0,
    "肌强直": 12.0,
    "静止性震颤": 12.0,
    "姿势性/动作性震颤": 9.0,
    "中轴": 8.0,
}


def resolve_score_bounds(score_cols: List[str], reference_df: Optional[pd.DataFrame] = None) -> Dict[str, List[float]]:
    lower_bounds: List[float] = []
    upper_bounds: List[float] = []

    for c in score_cols:
        lower = 0.0
        if c in DEFAULT_SCORE_UPPER_BOUNDS:
            upper = float(DEFAULT_SCORE_UPPER_BOUNDS[c])
        else:
            upper = None
            if reference_df is not None and c in reference_df.columns:
                vals = pd.to_numeric(reference_df[c], errors="coerce").values.astype(np.float32)
                vals = vals[np.isfinite(vals)]
                if vals.size > 0:
                    upper = float(np.ceil(np.max(vals)))
            if upper is None or upper <= lower:
                upper = 32.0

        lower_bounds.append(lower)
        upper_bounds.append(float(max(upper, lower + 1.0)))

    return {
        "lower": lower_bounds,
        "upper": upper_bounds,
    }


def mode_or_first(series: pd.Series):
    s = series.dropna()
    if len(s) == 0:
        return None
    m = s.mode()
    if len(m) > 0:
        return m.iloc[0]
    return s.iloc[0]


def normalize_diag_label(x: Any) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    s = str(x).strip().lower()
    if s in {"pd", "parkinson", "parkinson's disease", "parkinsons"}:
        return "pd"
    if s in {"control", "hc", "healthy", "healthy control"}:
        return "control"
    if s in {"pandromal", "prodromal", "prod"}:
        return "prodromal"
    return s


def cohort_to_diag_label(cohort: str) -> str:
    c = str(cohort).strip().lower()
    if "control" in c:
        return "control"
    if "prodromal" in c:
        return "prodromal"
    if c.endswith("-pd") or c == "pd" or "ppmi-pd" in c:
        return "pd"
    return c


def parse_partial_date(x: Any) -> Optional[pd.Timestamp]:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "nat", "none"}:
        return None

    m = re.match(r"^(\d{1,2})/(\d{4})$", s)
    if m:
        mm, yyyy = int(m.group(1)), int(m.group(2))
        if 1 <= mm <= 12:
            return pd.Timestamp(year=yyyy, month=mm, day=1)

    m = re.match(r"^(\d{4})-(\d{2})$", s)
    if m:
        yyyy, mm = int(m.group(1)), int(m.group(2))
        if 1 <= mm <= 12:
            return pd.Timestamp(year=yyyy, month=mm, day=1)

    dt = pd.to_datetime(s, errors="coerce")
    if pd.isna(dt):
        return None
    return pd.Timestamp(dt.year, dt.month, dt.day)


def month_distance(a: Optional[pd.Timestamp], b: Optional[pd.Timestamp]) -> int:
    if a is None or b is None or pd.isna(a) or pd.isna(b):
        return 9999
    return abs((a.year - b.year) * 12 + (a.month - b.month))


# =========================
# Date / path parsing
# =========================
def extract_patno_from_subject_id(subject_id: str) -> str:
    m = re.search(r"_(\d+)$", subject_id)
    if m is None:
        raise ValueError(f"Cannot parse PATNO from subject_id={subject_id}")
    return m.group(1)


def extract_cohort_from_subject_id(subject_id: str) -> str:
    patno = extract_patno_from_subject_id(subject_id)
    suffix = f"_{patno}"
    if not subject_id.endswith(suffix):
        return ""
    return subject_id[:-len(suffix)]


def extract_scan_folder_name_from_tck_path(tck_path: Optional[str]) -> Optional[str]:
    if tck_path is None:
        return None
    p = os.path.normpath(tck_path)
    return os.path.basename(os.path.dirname(os.path.dirname(p)))


def extract_scan_date_from_tck_path(tck_path: Optional[str]) -> Optional[pd.Timestamp]:
    folder = extract_scan_folder_name_from_tck_path(tck_path)
    if folder is None:
        return None

    m = re.match(r"^(\d{4}-\d{2}-\d{2})", folder)
    if m:
        dt = pd.to_datetime(m.group(1), errors="coerce")
        if not pd.isna(dt):
            return pd.Timestamp(dt.year, dt.month, dt.day)

    m = re.search(r"(\d{4}-\d{2}-\d{2})", str(tck_path))
    if m:
        dt = pd.to_datetime(m.group(1), errors="coerce")
        if not pd.isna(dt):
            return pd.Timestamp(dt.year, dt.month, dt.day)
    return None


# =========================
# Connectome path discovery
# =========================
def find_latest_tract_file(subject_dir: str) -> Optional[str]:
    dti_base = os.path.join(subject_dir, "DTI_B2000_64dir_PA")
    if not os.path.exists(dti_base):
        return None

    all_items = [os.path.join(dti_base, item) for item in os.listdir(dti_base)]
    subdirs = [item for item in all_items if os.path.isdir(item)]
    if len(subdirs) == 0:
        return None

    subdirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    latest_subdir = subdirs[0]

    tract_dirs = glob.glob(os.path.join(latest_subdir, "tract_out*"))
    tract_dirs = [d for d in tract_dirs if os.path.isdir(d)]
    if len(tract_dirs) == 0:
        return None

    for tract_dir in tract_dirs:
        tck_file = os.path.join(tract_dir, "tracks_500k_ifod2.tck")
        if os.path.exists(tck_file):
            return tck_file
    return None


def extract_tract_instance_id_from_tck(tck_path: str) -> Optional[str]:
    tract_dir = os.path.basename(os.path.dirname(tck_path))
    m = re.match(r"^tract_out_([A-Za-z]\d+)$", tract_dir, flags=re.IGNORECASE)
    if m is None:
        return None
    return m.group(1)


def find_connectome_path_for_tck(base_dir: str, tck_path: str) -> Tuple[Optional[str], Optional[str]]:
    tract_id = extract_tract_instance_id_from_tck(tck_path)
    if tract_id is None:
        return None, None

    connectome_root = os.path.join(base_dir, "connectome")
    if not os.path.exists(connectome_root):
        return tract_id, None

    candidate_dir = os.path.join(connectome_root, tract_id)
    if not os.path.isdir(candidate_dir):
        found = None
        for d in glob.glob(os.path.join(connectome_root, "*")):
            if os.path.isdir(d) and os.path.basename(d).lower() == tract_id.lower():
                found = d
                break
        candidate_dir = found

    if candidate_dir is None or not os.path.isdir(candidate_dir):
        return tract_id, None

    connectome_path = os.path.join(candidate_dir, "connectome.csv")
    if not os.path.exists(connectome_path):
        connectome_path = None
    return tract_id, connectome_path


def load_connectome_matrix(path: str) -> np.ndarray:
    try:
        mat = np.loadtxt(path, delimiter=",", dtype=np.float32)
    except Exception:
        df = pd.read_csv(path, header=None)
        mat = df.values.astype(np.float32)

    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError(f"Invalid connectome shape: {mat.shape}, expected square matrix, file={path}")

    mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)
    return mat.astype(np.float32)


def compute_laplacian_pe(A: np.ndarray, k: int) -> np.ndarray:
    A = np.nan_to_num(A.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    A = np.maximum(A, 0.0)
    A = np.log1p(A)
    A = 0.5 * (A + A.T)

    R = A.shape[0]
    A = A + np.eye(R, dtype=np.float64)
    deg = np.sum(A, axis=1)
    deg = np.clip(deg, 1e-6, None)
    d_inv_sqrt = 1.0 / np.sqrt(deg)
    A_norm = d_inv_sqrt[:, None] * A * d_inv_sqrt[None, :]
    L = np.eye(R, dtype=np.float64) - A_norm

    evals, evecs = np.linalg.eigh(L)
    idx = np.argsort(evals)
    evecs = evecs[:, idx]

    if R > 1:
        evecs = evecs[:, 1:1 + k]
    else:
        evecs = evecs[:, :k]

    if evecs.shape[1] < k:
        pad = np.zeros((R, k - evecs.shape[1]), dtype=np.float64)
        evecs = np.concatenate([evecs, pad], axis=1)

    return evecs.astype(np.float32)


# =========================
# Build connectome sample table
# =========================
def build_connectome_table(
    data_base_dir: str,
    max_subjects_per_dir: Optional[int] = None,
    seed: int = 42,
) -> pd.DataFrame:
    base_dirs = [
        os.path.join(data_base_dir, "PPMI-Control"),
        os.path.join(data_base_dir, "PPMI-PD"),
        os.path.join(data_base_dir, "PPMI-Prodromal"),
    ]

    rows = []
    for dir_idx, base_dir in enumerate(base_dirs):
        if not os.path.exists(base_dir):
            print(f"[Warn] Base directory not found: {base_dir}, skipping.")
            continue

        ppmi_dir = os.path.join(base_dir, "PPMI")
        if not os.path.exists(ppmi_dir):
            print(f"[Warn] {ppmi_dir} not found, trying {base_dir} directly.")
            ppmi_dir = base_dir

        if not os.path.exists(ppmi_dir):
            print(f"[Warn] Missing PPMI dir: {ppmi_dir}, skipping.")
            continue

        cohort_rows = []
        for item in os.listdir(ppmi_dir):
            subject_path = os.path.join(ppmi_dir, item)
            if not os.path.isdir(subject_path):
                continue

            tck_path = find_latest_tract_file(subject_path)
            if tck_path is None:
                continue

            tract_id, connectome_path = find_connectome_path_for_tck(base_dir, tck_path)
            if connectome_path is None:
                continue

            subject_id = f"{os.path.basename(base_dir)}_{item}"
            scan_date = extract_scan_date_from_tck_path(tck_path)
            scan_folder = extract_scan_folder_name_from_tck_path(tck_path)
            patno = extract_patno_from_subject_id(subject_id)
            cohort = extract_cohort_from_subject_id(subject_id)

            cohort_rows.append(
                {
                    "sample_id": subject_id,
                    "subject_id": subject_id,
                    "PATNO": patno,
                    "cohort": cohort,
                    "diag_from_cohort": cohort_to_diag_label(cohort),
                    "connectome_path": connectome_path,
                    "tck_path": tck_path,
                    "tract_id": tract_id,
                    "scan_folder": scan_folder,
                    "scan_date": scan_date,
                    "scan_date_str": "" if scan_date is None else scan_date.strftime("%Y-%m-%d"),
                }
            )

        if max_subjects_per_dir is not None and len(cohort_rows) > max_subjects_per_dir:
            rng = np.random.RandomState(seed + dir_idx)
            idx = rng.choice(len(cohort_rows), size=max_subjects_per_dir, replace=False)
            cohort_rows = [cohort_rows[i] for i in idx]

        print(f"[Info] {os.path.basename(base_dir)} selected subjects: {len(cohort_rows)}")
        rows.extend(cohort_rows)

    df = pd.DataFrame(rows)
    if len(df) == 0:
        raise ValueError("No valid connectome samples found.")
    return df


# =========================
# Matching logic
# =========================
def build_visit_tables(csv_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, pd.DataFrame], str, List[str]]:
    df = read_csv_flexible(csv_path)
    id_col = detect_id_col(df)
    df[id_col] = df[id_col].astype(str)
    if id_col != "PATNO":
        df = df.rename(columns={id_col: "PATNO"})
        id_col = "PATNO"

    for c in ["EVENT_ID", "INFODT", "EXAMDT"]:
        if c not in df.columns:
            raise ValueError(f"CSV must contain column: {c}")

    score_cols = detect_score_cols(df, id_col)

    df["EVENT_ID"] = df["EVENT_ID"].astype(str)
    df["INFODT"] = df["INFODT"].astype(str)
    df["EXAMDT"] = df["EXAMDT"].astype(str)
    df["INFODT_DT"] = df["INFODT"].apply(parse_partial_date)
    df["EXAMDT_DT"] = df["EXAMDT"].apply(parse_partial_date)

    if "label" in df.columns:
        df["diag_label"] = df["label"].apply(normalize_diag_label)
    else:
        df["diag_label"] = ""

    df["visit_key"] = (
        df["PATNO"].astype(str)
        + "||"
        + df["EVENT_ID"].astype(str)
        + "||"
        + df["INFODT"].astype(str)
        + "||"
        + df["EXAMDT"].astype(str)
    )

    visit_groups: Dict[str, pd.DataFrame] = {}
    visit_rows: List[Dict[str, Any]] = []

    for visit_key, sub in df.groupby("visit_key", sort=False):
        sub = sub.reset_index(drop=True).copy()
        visit_groups[visit_key] = sub
        uniq_scores = sub[score_cols].drop_duplicates().reset_index(drop=True)

        visit_rows.append(
            {
                "visit_key": visit_key,
                "PATNO": str(sub.loc[0, "PATNO"]),
                "EVENT_ID": str(sub.loc[0, "EVENT_ID"]),
                "INFODT": str(sub.loc[0, "INFODT"]),
                "EXAMDT": str(sub.loc[0, "EXAMDT"]),
                "INFODT_DT": sub.loc[0, "INFODT_DT"],
                "EXAMDT_DT": sub.loc[0, "EXAMDT_DT"],
                "diag_label": mode_or_first(sub["diag_label"]) if "diag_label" in sub.columns else "",
                "row_count": int(len(sub)),
                "unique_score_count": int(len(uniq_scores)),
            }
        )

    visit_df = pd.DataFrame(visit_rows)
    return df, visit_df, visit_groups, id_col, score_cols


def resolve_duplicate_rows_within_visit(
    sub: pd.DataFrame,
    score_cols: List[str],
    duplicate_policy: str,
) -> Tuple[Optional[np.ndarray], str]:
    uniq_scores = sub[score_cols].drop_duplicates().reset_index(drop=True)
    if len(sub) == 1:
        return sub.iloc[0][score_cols].astype(np.float32).values, "single_row"
    if len(uniq_scores) == 1:
        return uniq_scores.iloc[0].astype(np.float32).values, "duplicate_same_scores"

    if duplicate_policy == "drop":
        return None, "ambiguous_duplicate_drop"
    if duplicate_policy == "mean":
        return sub[score_cols].mean(axis=0).astype(np.float32).values, "ambiguous_duplicate_mean"
    if duplicate_policy == "first":
        return sub.iloc[0][score_cols].astype(np.float32).values, "ambiguous_duplicate_first"
    if duplicate_policy == "last":
        return sub.iloc[-1][score_cols].astype(np.float32).values, "ambiguous_duplicate_last"
    if duplicate_policy == "max_total":
        total = sub[score_cols].sum(axis=1).astype(float)
        idx = int(total.values.argmax())
        return sub.iloc[idx][score_cols].astype(np.float32).values, "ambiguous_duplicate_max_total"

    raise ValueError(f"Unknown duplicate_policy={duplicate_policy}")


def choose_best_visit_for_sample(
    sample_row: pd.Series,
    visit_df: pd.DataFrame,
    visit_groups: Dict[str, pd.DataFrame],
    score_cols: List[str],
    max_month_gap: int,
    duplicate_policy: str,
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    patno = str(sample_row["PATNO"])
    scan_date = sample_row["scan_date"]

    debug = {
        "sample_id": sample_row["sample_id"],
        "PATNO": patno,
        "scan_date": sample_row["scan_date_str"],
        "status": "unmatched",
        "reason": "",
        "match_note": "",
        "matched_visit_key": "",
        "matched_EVENT_ID": "",
        "matched_INFODT": "",
        "matched_EXAMDT": "",
        "month_gap": None,
    }

    if scan_date is None or pd.isna(scan_date):
        debug["reason"] = "missing_scan_date"
        return None, debug

    cand = visit_df[visit_df["PATNO"].astype(str) == patno].copy()
    if len(cand) == 0:
        debug["reason"] = "no_label_rows_for_patno"
        return None, debug

    cand["exam_gap"] = cand["EXAMDT_DT"].apply(lambda x: month_distance(scan_date, x))
    cand["info_gap"] = cand["INFODT_DT"].apply(lambda x: month_distance(scan_date, x))
    cand["best_gap"] = cand[["exam_gap", "info_gap"]].min(axis=1)
    cand["exact_exam"] = (cand["exam_gap"] == 0).astype(int)
    cand["exact_info"] = (cand["info_gap"] == 0).astype(int)
    cand["exact_both"] = ((cand["exam_gap"] == 0) & (cand["info_gap"] == 0)).astype(int)

    cand = cand[cand["best_gap"] <= int(max_month_gap)].copy()
    if len(cand) == 0:
        debug["reason"] = f"no_visit_within_{max_month_gap}_month"
        return None, debug

    cand = cand.sort_values(
        by=["exact_both", "exact_exam", "exact_info", "best_gap"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)

    best = cand.iloc[0]
    best_rank = cand.iloc[0][["exact_both", "exact_exam", "exact_info", "best_gap"]].tolist()

    ties = cand[
        (cand["exact_both"] == best_rank[0])
        & (cand["exact_exam"] == best_rank[1])
        & (cand["exact_info"] == best_rank[2])
        & (cand["best_gap"] == best_rank[3])
    ].copy()

    if ties["visit_key"].nunique() > 1:
        debug["reason"] = "ambiguous_multiple_visit_keys"
        debug["month_gap"] = int(best["best_gap"])
        return None, debug

    visit_key = str(best["visit_key"])
    sub = visit_groups[visit_key]
    score_vec, match_note = resolve_duplicate_rows_within_visit(sub, score_cols, duplicate_policy)
    if score_vec is None:
        debug["reason"] = match_note
        debug["month_gap"] = int(best["best_gap"])
        return None, debug

    diag_label = str(best["diag_label"]) if "diag_label" in best else ""
    matched = {
        "sample_id": sample_row["sample_id"],
        "subject_id": sample_row["subject_id"],
        "PATNO": patno,
        "cohort": sample_row["cohort"],
        "label": diag_label,
        "connectome_path": sample_row["connectome_path"],
        "tck_path": sample_row["tck_path"],
        "tract_id": sample_row["tract_id"],
        "scan_folder": sample_row["scan_folder"],
        "scan_date": sample_row["scan_date_str"],
        "EVENT_ID": str(best["EVENT_ID"]),
        "INFODT": str(best["INFODT"]),
        "EXAMDT": str(best["EXAMDT"]),
        "visit_key": visit_key,
        "row_count_for_visit": int(best["row_count"]),
        "unique_score_count_for_visit": int(best["unique_score_count"]),
        "month_gap": int(best["best_gap"]),
        "match_note": match_note,
    }
    for i, c in enumerate(score_cols):
        matched[c] = float(score_vec[i])

    debug.update(
        {
            "status": "matched",
            "reason": "ok",
            "match_note": match_note,
            "matched_visit_key": visit_key,
            "matched_EVENT_ID": str(best["EVENT_ID"]),
            "matched_INFODT": str(best["INFODT"]),
            "matched_EXAMDT": str(best["EXAMDT"]),
            "month_gap": int(best["best_gap"]),
        }
    )
    return matched, debug


def build_matched_sample_table(
    data_base_dir: str,
    csv_path: str,
    max_month_gap: int,
    duplicate_policy: str,
    max_subjects_per_dir: Optional[int],
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    sample_df = build_connectome_table(
        data_base_dir=data_base_dir,
        max_subjects_per_dir=max_subjects_per_dir,
        seed=seed,
    )
    _, visit_df, visit_groups, _, score_cols = build_visit_tables(csv_path)

    matched_rows: List[Dict[str, Any]] = []
    debug_rows: List[Dict[str, Any]] = []

    for _, sample_row in sample_df.iterrows():
        matched, debug = choose_best_visit_for_sample(
            sample_row=sample_row,
            visit_df=visit_df,
            visit_groups=visit_groups,
            score_cols=score_cols,
            max_month_gap=max_month_gap,
            duplicate_policy=duplicate_policy,
        )
        debug_rows.append(debug)
        if matched is not None:
            matched_rows.append(matched)

    matched_df = pd.DataFrame(matched_rows)
    debug_df = pd.DataFrame(debug_rows)

    if len(matched_df) == 0:
        raise ValueError("No matched samples after connectome-label matching.")

    matched_df["label"] = matched_df["label"].apply(normalize_diag_label)
    if (matched_df["label"] == "").any():
        idx = matched_df["label"] == ""
        matched_df.loc[idx, "label"] = matched_df.loc[idx, "cohort"].apply(cohort_to_diag_label)

    return matched_df, debug_df, score_cols



# =========================
# Anti-collapse target stats / weighting
# =========================
def get_lds_kernel_window(kernel: str = "gaussian", ks: int = 5, sigma: float = 2.0) -> np.ndarray:
    ks = int(max(1, ks))
    if ks % 2 == 0:
        ks += 1
    sigma = float(max(sigma, 1e-6))
    half = ks // 2
    x = np.arange(-half, half + 1, dtype=np.float32)
    kernel = str(kernel).strip().lower()

    if kernel == "gaussian":
        w = np.exp(-0.5 * (x / sigma) ** 2)
    elif kernel in {"triang", "triangle", "triangular"}:
        w = (half + 1 - np.abs(x)).clip(min=0)
    elif kernel in {"laplace", "laplacian"}:
        w = np.exp(-np.abs(x) / sigma)
    else:
        raise ValueError(f"Unsupported LDS kernel: {kernel}")

    w = w.astype(np.float32)
    w_sum = float(w.sum())
    if w_sum <= 0:
        raise ValueError("Invalid LDS kernel window with non-positive sum.")
    return w / w_sum


def convolve1d_same_constant(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    v = np.asarray(values, dtype=np.float32).reshape(-1)
    w = np.asarray(weights, dtype=np.float32).reshape(-1)
    if len(w) == 0:
        return v.copy()
    if len(w) % 2 == 0:
        raise ValueError("LDS kernel size must be odd.")
    pad = len(w) // 2
    vp = np.pad(v, (pad, pad), mode="constant", constant_values=0.0)
    out = np.convolve(vp, w[::-1], mode="valid")
    return out.astype(np.float32)


def get_label_bin_idx(value: float, bin_min: float, bin_width: float, num_bins: int) -> int:
    if num_bins <= 1:
        return 0
    idx = int(np.floor((float(value) - float(bin_min)) / float(bin_width) + 1e-12))
    idx = max(0, min(int(num_bins) - 1, idx))
    return idx


def build_lds_task_stats(
    labels: np.ndarray,
    bin_width: float = 1.0,
    kernel: str = "gaussian",
    ks: int = 5,
    sigma: float = 2.0,
    reweight: str = "inverse",
    weight_cap: float = 6.0,
) -> Dict[str, Any]:
    labels = np.asarray(labels, dtype=np.float32).reshape(-1)
    if labels.size == 0:
        raise ValueError("Cannot build LDS stats from empty labels.")

    reweight = str(reweight).strip().lower()
    if reweight not in {"none", "inverse", "sqrt_inv"}:
        raise ValueError(f"Unsupported LDS reweight mode: {reweight}")

    bin_width = float(max(bin_width, 1e-6))
    label_min = float(np.floor(labels.min() / bin_width) * bin_width)
    label_max = float(np.ceil(labels.max() / bin_width) * bin_width)
    num_bins = int(round((label_max - label_min) / bin_width)) + 1
    num_bins = max(num_bins, 1)

    bin_indices = np.asarray(
        [get_label_bin_idx(v, label_min, bin_width, num_bins) for v in labels],
        dtype=np.int64,
    )
    emp_label_dist = np.bincount(bin_indices, minlength=num_bins).astype(np.float32)

    base_dist = emp_label_dist.copy()
    if reweight == "sqrt_inv":
        base_dist = np.sqrt(base_dist)

    kernel_window = get_lds_kernel_window(kernel=kernel, ks=ks, sigma=sigma)
    eff_label_dist = convolve1d_same_constant(base_dist, kernel_window)
    eff_label_dist = np.maximum(eff_label_dist, 1e-6).astype(np.float32)

    sample_eff_num = eff_label_dist[bin_indices]
    sample_weights = (1.0 / sample_eff_num).astype(np.float32)
    scaling = float(len(sample_weights) / max(sample_weights.sum(), 1e-6))
    sample_weights = sample_weights * scaling
    sample_weights = np.clip(sample_weights, 1e-6, float(weight_cap)).astype(np.float32)

    weight_lookup = np.zeros(num_bins, dtype=np.float32)
    for b in range(num_bins):
        mask = bin_indices == b
        if np.any(mask):
            weight_lookup[b] = float(sample_weights[mask].mean())
        else:
            weight_lookup[b] = float(np.clip((1.0 / eff_label_dist[b]) * scaling, 1e-6, weight_cap))

    return {
        "bin_min": label_min,
        "bin_width": bin_width,
        "num_bins": num_bins,
        "emp_label_dist": emp_label_dist.tolist(),
        "eff_label_dist": eff_label_dist.tolist(),
        "kernel_window": kernel_window.tolist(),
        "weight_lookup": weight_lookup.astype(np.float32).tolist(),
        "sample_weights": sample_weights.astype(np.float32).tolist(),
    }


def build_train_target_stats(
    matched_df: pd.DataFrame,
    train_ids: List[str],
    score_cols: List[str],
    high_quantile: float = 0.75,
    min_std: float = 1e-3,
    sample_weight_alpha: float = 0.35,
    sample_weight_pos_alpha: float = 0.60,
    sample_weight_high_alpha: float = 0.90,
    sample_weight_cap: float = 5.0,
    enable_lds: bool = True,
    lds_kernel: str = "gaussian",
    lds_ks: int = 5,
    lds_sigma: float = 2.0,
    lds_bin_width: float = 1.0,
    lds_reweight: str = "inverse",
    lds_task_weight_cap: float = 6.0,
    lds_use_scalar_sample_weights: bool = False,
    lds_agg: str = "mean",
) -> Dict[str, Any]:
    train_df = matched_df[matched_df["sample_id"].isin(train_ids)].copy()
    if len(train_df) == 0:
        raise ValueError("Empty train_df when building target stats.")

    order_map = {sid: i for i, sid in enumerate(train_ids)}
    train_df["_order"] = train_df["sample_id"].map(order_map)
    train_df = train_df.sort_values("_order").reset_index(drop=True)

    y = train_df[score_cols].astype(np.float32).values
    label_mean = y.mean(axis=0).astype(np.float32)
    label_std = y.std(axis=0).astype(np.float32)
    label_std = np.maximum(label_std, min_std).astype(np.float32)

    pos_rate = []
    high_thr = []
    high_rate = []
    for t in range(y.shape[1]):
        vals = y[:, t]
        pos_mask = vals > 0
        pr = float(pos_mask.mean())
        pos_rate.append(pr)

        pos_vals = vals[pos_mask]
        if len(pos_vals) == 0:
            thr = float("inf")
            hr = 0.0
        else:
            thr = float(np.quantile(pos_vals, high_quantile))
            thr = max(thr, float(np.min(pos_vals)))
            high_mask = vals >= thr
            hr = float(high_mask.mean())
        high_thr.append(thr)
        high_rate.append(hr)

    pos_rate = np.asarray(pos_rate, dtype=np.float32)
    high_thr = np.asarray(high_thr, dtype=np.float32)
    high_rate = np.asarray(high_rate, dtype=np.float32)

    legacy_sample_weights = []
    for i in range(y.shape[0]):
        per_task = np.ones(y.shape[1], dtype=np.float32)
        for t in range(y.shape[1]):
            pr = max(float(pos_rate[t]), 1e-4)
            hr = max(float(high_rate[t]), 1e-4)
            if y[i, t] > 0:
                per_task[t] += sample_weight_pos_alpha * (1.0 - pr) / pr
            if np.isfinite(high_thr[t]) and y[i, t] >= high_thr[t]:
                per_task[t] += sample_weight_high_alpha * (1.0 - hr) / hr
        sw = 1.0 + sample_weight_alpha * float(per_task.mean() - 1.0)
        sw = min(sw, sample_weight_cap)
        legacy_sample_weights.append(sw)

    legacy_sample_weights = np.asarray(legacy_sample_weights, dtype=np.float32)
    legacy_sample_weights = legacy_sample_weights / max(float(legacy_sample_weights.mean()), 1e-6)
    legacy_sample_weight_map = {
        sid: float(w) for sid, w in zip(train_df["sample_id"].tolist(), legacy_sample_weights.tolist())
    }

    output_bounds = resolve_score_bounds(score_cols=score_cols, reference_df=matched_df)

    train_target_stats: Dict[str, Any] = {
        "label_mean": label_mean.tolist(),
        "label_std": label_std.tolist(),
        "pos_rate": pos_rate.tolist(),
        "high_thr": high_thr.tolist(),
        "high_rate": high_rate.tolist(),
        "output_lower_bounds": output_bounds["lower"],
        "output_upper_bounds": output_bounds["upper"],
        "sample_weight_map": legacy_sample_weight_map,
        "legacy_sample_weight_map": legacy_sample_weight_map,
        "sampler_weight_map": legacy_sample_weight_map,
        "enable_lds": bool(enable_lds),
        "lds_task_stats": [],
        "lds_sample_weight_map": {sid: 1.0 for sid in train_df["sample_id"].tolist()},
        "lds_sampler_weight_map": {sid: 1.0 for sid in train_df["sample_id"].tolist()},
        "lds_config": {
            "kernel": str(lds_kernel),
            "ks": int(lds_ks),
            "sigma": float(lds_sigma),
            "bin_width": float(lds_bin_width),
            "reweight": str(lds_reweight),
            "agg": str(lds_agg),
        },
    }

    if not enable_lds:
        return train_target_stats

    lds_agg = str(lds_agg).strip().lower()
    if lds_agg not in {"mean", "gmean", "max"}:
        raise ValueError(f"Unsupported LDS aggregation mode: {lds_agg}")

    task_stats = []
    task_sample_weights = []
    for t in range(y.shape[1]):
        stats_t = build_lds_task_stats(
            labels=y[:, t],
            bin_width=lds_bin_width,
            kernel=lds_kernel,
            ks=lds_ks,
            sigma=lds_sigma,
            reweight=lds_reweight,
            weight_cap=lds_task_weight_cap,
        )
        task_stats.append(stats_t)
        task_sample_weights.append(np.asarray(stats_t["sample_weights"], dtype=np.float32))

    task_sample_weights = np.stack(task_sample_weights, axis=1).astype(np.float32)  # (N, T)

    if lds_agg == "mean":
        agg_weights = task_sample_weights.mean(axis=1)
    elif lds_agg == "max":
        agg_weights = task_sample_weights.max(axis=1)
    else:
        agg_weights = np.exp(np.log(np.clip(task_sample_weights, 1e-6, None)).mean(axis=1))

    agg_weights = agg_weights.astype(np.float32)
    agg_weights = agg_weights / max(float(agg_weights.mean()), 1e-6)
    agg_weights = np.clip(agg_weights, 1e-6, float(sample_weight_cap)).astype(np.float32)

    lds_sampler_weight_map = {
        sid: float(w) for sid, w in zip(train_df["sample_id"].tolist(), agg_weights.tolist())
    }
    if lds_use_scalar_sample_weights:
        lds_sample_weight_map = lds_sampler_weight_map
    else:
        lds_sample_weight_map = {sid: 1.0 for sid in train_df["sample_id"].tolist()}

    train_target_stats["sample_weight_map"] = lds_sample_weight_map
    train_target_stats["sampler_weight_map"] = lds_sampler_weight_map
    train_target_stats["lds_sample_weight_map"] = lds_sample_weight_map
    train_target_stats["lds_sampler_weight_map"] = lds_sampler_weight_map
    train_target_stats["lds_task_stats"] = task_stats
    return train_target_stats


class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 0.0, warmup_epochs: int = 0):
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.warmup_epochs = int(warmup_epochs)
        self.best = float("inf")
        self.num_bad = 0

    def step(self, metric: float, epoch_idx: int) -> bool:
        if epoch_idx < self.warmup_epochs:
            if metric < self.best:
                self.best = metric
                self.num_bad = 0
            return False

        improved = metric < (self.best - self.min_delta)
        if improved:
            self.best = metric
            self.num_bad = 0
        else:
            self.num_bad += 1
        return self.num_bad >= self.patience


# =========================
# Dataset
# =========================
class ConnectomeDataset(Dataset):
    def __init__(
        self,
        matched_df: pd.DataFrame,
        sample_ids: List[str],
        score_cols: List[str],
        sample_weight_map: Optional[Dict[str, float]] = None,
        lap_pe_dim: int = 16,
        preload: bool = True,
    ):
        self.score_cols = list(score_cols)
        self.sample_weight_map = sample_weight_map or {}
        self.sample_ids = [str(x) for x in sample_ids]
        self.lap_pe_dim = int(lap_pe_dim)
        self.preload = bool(preload)

        df = matched_df.copy()
        df["sample_id"] = df["sample_id"].astype(str)
        df = df[df["sample_id"].isin(self.sample_ids)].copy().reset_index(drop=True)
        if len(df) == 0:
            raise ValueError("ConnectomeDataset received zero matched samples.")

        missing = [sid for sid in self.sample_ids if sid not in set(df["sample_id"].tolist())]
        if missing:
            raise ValueError(f"Missing matched rows for sample_ids: {missing[:10]}{'...' if len(missing) > 10 else ''}")

        order_map = {sid: i for i, sid in enumerate(self.sample_ids)}
        df["_order"] = df["sample_id"].map(order_map)
        df = df.sort_values("_order").drop(columns=["_order"]).reset_index(drop=True)
        self.records = df

        self.connectome_cache: Dict[str, torch.Tensor] = {}
        self.lap_pe_cache: Dict[str, torch.Tensor] = {}
        self.num_rois: Optional[int] = None

        if self.preload:
            for _, row in tqdm(self.records.iterrows(), total=len(self.records), desc="Preloading connectomes", leave=False):
                sid = str(row["sample_id"])
                conn_path = row["connectome_path"]
                A = load_connectome_matrix(conn_path)
                lap_pe = compute_laplacian_pe(A, self.lap_pe_dim)

                if self.num_rois is None:
                    self.num_rois = A.shape[0]
                else:
                    if A.shape[0] != self.num_rois:
                        raise ValueError(f"Inconsistent ROI size: got {A.shape[0]}, expected {self.num_rois}")

                self.connectome_cache[sid] = torch.from_numpy(A).float()
                self.lap_pe_cache[sid] = torch.from_numpy(lap_pe).float()

        if self.num_rois is None and len(self.records) > 0:
            A = load_connectome_matrix(self.records.iloc[0]["connectome_path"])
            self.num_rois = A.shape[0]

    def __len__(self):
        return len(self.records)

    def _load_one(self, sample_id: str, conn_path: str):
        if sample_id in self.connectome_cache:
            return self.connectome_cache[sample_id], self.lap_pe_cache[sample_id]

        A = load_connectome_matrix(conn_path)
        lap_pe = compute_laplacian_pe(A, self.lap_pe_dim)
        self.connectome_cache[sample_id] = torch.from_numpy(A).float()
        self.lap_pe_cache[sample_id] = torch.from_numpy(lap_pe).float()
        return self.connectome_cache[sample_id], self.lap_pe_cache[sample_id]

    def __getitem__(self, idx: int):
        row = self.records.iloc[idx]
        sample_id = str(row["sample_id"])
        conn_path = row["connectome_path"]

        A, lap_pe = self._load_one(sample_id, conn_path)
        label = torch.tensor(row[self.score_cols].astype(np.float32).values, dtype=torch.float32)
        sw = float(self.sample_weight_map.get(sample_id, 1.0))
        sample_weight = torch.tensor(sw, dtype=torch.float32)

        meta = {
            "PATNO": str(row["PATNO"]),
            "EVENT_ID": str(row["EVENT_ID"]),
            "INFODT": str(row["INFODT"]),
            "EXAMDT": str(row["EXAMDT"]),
            "scan_date": str(row["scan_date"]),
            "match_note": str(row["match_note"]),
            "month_gap": int(row["month_gap"]),
            "tck_path": str(row["tck_path"]),
            "tract_id": str(row["tract_id"]),
            "connectome_path": str(conn_path),
        }

        return {
            "connectome": A,
            "lap_pe": lap_pe,
            "label": label,
            "sample_id": sample_id,
            "meta": meta,
            "sample_weight": sample_weight,
        }


def collate_fn(batch):
    connectomes = torch.stack([b["connectome"] for b in batch], dim=0)
    lap_pes = torch.stack([b["lap_pe"] for b in batch], dim=0)
    labels = torch.stack([b["label"] for b in batch], dim=0)
    sample_ids = [b["sample_id"] for b in batch]
    metas = [b["meta"] for b in batch]
    sample_weights = torch.stack([b["sample_weight"] for b in batch], dim=0)

    return {
        "connectome": connectomes,
        "lap_pe": lap_pes,
        "label": labels,
        "sample_id": sample_ids,
        "meta": metas,
        "sample_weight": sample_weights,
    }


# =========================
# Loss / metrics / evaluation
# =========================

def get_stat_tensors(train_stats: Dict[str, Any], device: torch.device) -> Dict[str, torch.Tensor]:
    label_mean = train_stats["label_mean"]
    default_lower = [0.0 for _ in label_mean]
    default_upper = [float(max(1.0, m + 3.0 * s)) for m, s in zip(train_stats["label_mean"], train_stats["label_std"])]

    stat_tensors = {
        "mean": torch.tensor(train_stats["label_mean"], dtype=torch.float32, device=device),
        "std": torch.tensor(train_stats["label_std"], dtype=torch.float32, device=device),
        "lower": torch.tensor(train_stats.get("output_lower_bounds", default_lower), dtype=torch.float32, device=device),
        "upper": torch.tensor(train_stats.get("output_upper_bounds", default_upper), dtype=torch.float32, device=device),
        "pos_rate": torch.tensor(train_stats.get("pos_rate", []), dtype=torch.float32, device=device),
        "high_thr": torch.tensor(train_stats.get("high_thr", []), dtype=torch.float32, device=device),
        "high_rate": torch.tensor(train_stats.get("high_rate", []), dtype=torch.float32, device=device),
        "enable_lds": bool(train_stats.get("enable_lds", False) and len(train_stats.get("lds_task_stats", [])) > 0),
    }

    if stat_tensors["enable_lds"]:
        task_stats = train_stats["lds_task_stats"]
        max_bins = max(int(ts["num_bins"]) for ts in task_stats)
        weight_lookup = np.ones((len(task_stats), max_bins), dtype=np.float32)
        bin_min = np.zeros(len(task_stats), dtype=np.float32)
        bin_width = np.ones(len(task_stats), dtype=np.float32)
        num_bins = np.ones(len(task_stats), dtype=np.int64)

        for t, ts in enumerate(task_stats):
            nb = int(ts["num_bins"])
            weight_lookup[t, :nb] = np.asarray(ts["weight_lookup"], dtype=np.float32)
            bin_min[t] = float(ts["bin_min"])
            bin_width[t] = float(ts["bin_width"])
            num_bins[t] = nb

        stat_tensors["lds_weight_lookup"] = torch.tensor(weight_lookup, dtype=torch.float32, device=device)
        stat_tensors["lds_bin_min"] = torch.tensor(bin_min, dtype=torch.float32, device=device)
        stat_tensors["lds_bin_width"] = torch.tensor(bin_width, dtype=torch.float32, device=device)
        stat_tensors["lds_num_bins"] = torch.tensor(num_bins, dtype=torch.long, device=device)

    return stat_tensors


def standardize_labels(labels: torch.Tensor, stat_tensors: Dict[str, torch.Tensor]) -> torch.Tensor:
    return (labels - stat_tensors["mean"].unsqueeze(0)) / (stat_tensors["std"].unsqueeze(0) + EPS)


def destandardize_outputs(outputs_z: torch.Tensor, stat_tensors: Dict[str, torch.Tensor]) -> torch.Tensor:
    return outputs_z * stat_tensors["std"].unsqueeze(0) + stat_tensors["mean"].unsqueeze(0)



def clamp_raw_predictions(preds_raw: torch.Tensor, stat_tensors: Dict[str, torch.Tensor]) -> torch.Tensor:
    lower = stat_tensors["lower"].unsqueeze(0)
    upper = stat_tensors["upper"].unsqueeze(0)
    return torch.minimum(torch.maximum(preds_raw, lower), upper)


def compute_output_range_penalty(
    preds_raw: torch.Tensor,
    stat_tensors: Dict[str, torch.Tensor],
    power: float = 2.0,
    margin: float = 0.0,
) -> torch.Tensor:
    lower = stat_tensors["lower"].unsqueeze(0) - float(margin)
    upper = stat_tensors["upper"].unsqueeze(0) + float(margin)
    width = torch.clamp(upper - lower, min=1.0)

    lower_violation = torch.relu(lower - preds_raw) / width
    upper_violation = torch.relu(preds_raw - upper) / width

    if abs(float(power) - 1.0) < 1e-8:
        penalty = lower_violation + upper_violation
    else:
        penalty = lower_violation.pow(float(power)) + upper_violation.pow(float(power))
    return penalty.mean()


def build_task_rarity_weight(
    labels_raw: torch.Tensor,
    stat_tensors: Dict[str, torch.Tensor],
    pos_alpha: float,
    high_alpha: float,
    max_task_weight: float,
) -> torch.Tensor:
    pos_rate = torch.clamp(stat_tensors["pos_rate"], min=1e-4)
    high_rate = torch.clamp(stat_tensors["high_rate"], min=1e-4)
    high_thr = stat_tensors["high_thr"]

    pos_term = 1.0 + pos_alpha * ((1.0 - pos_rate) / pos_rate).unsqueeze(0) * (labels_raw > 0).float()
    high_mask = torch.isfinite(high_thr).unsqueeze(0) & (labels_raw >= high_thr.unsqueeze(0))
    high_term = 1.0 + high_alpha * ((1.0 - high_rate) / high_rate).unsqueeze(0) * high_mask.float()
    rarity = pos_term * high_term
    return torch.clamp(rarity, max=max_task_weight)


def build_lds_task_weight(
    labels_raw: torch.Tensor,
    stat_tensors: Dict[str, torch.Tensor],
    max_task_weight: float,
) -> torch.Tensor:
    if not bool(stat_tensors.get("enable_lds", False)):
        return torch.ones_like(labels_raw)

    # labels_raw: (B, T)
    # lds_weight_lookup: (T, max_bins)
    # We need a per-task lookup for each sample, so use task-wise advanced
    # indexing instead of torch.gather over dim=1.
    bin_min = stat_tensors["lds_bin_min"].unsqueeze(0)
    bin_width = torch.clamp(stat_tensors["lds_bin_width"].unsqueeze(0), min=1e-6)
    num_bins = stat_tensors["lds_num_bins"].unsqueeze(0)

    idx = torch.floor((labels_raw - bin_min) / bin_width + 1e-12).long()
    idx = torch.clamp(idx, min=0)
    idx = torch.minimum(idx, num_bins - 1)

    lookup = stat_tensors["lds_weight_lookup"]  # (T, max_bins)
    num_tasks = lookup.shape[0]
    task_ids = torch.arange(num_tasks, device=labels_raw.device).unsqueeze(1)  # (T, 1)

    # idx.T: (T, B) -> output (T, B) -> transpose back to (B, T)
    lds_w = lookup[task_ids, idx.transpose(0, 1)].transpose(0, 1)
    return torch.clamp(lds_w, min=1e-6, max=max_task_weight)


def pearson_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if pred.shape[0] < 2:
        return pred.new_tensor(0.0)

    pred_centered = pred - pred.mean(dim=0, keepdim=True)
    target_centered = target - target.mean(dim=0, keepdim=True)

    pred_std = pred_centered.pow(2).mean(dim=0).sqrt()
    target_std = target_centered.pow(2).mean(dim=0).sqrt()
    valid = (pred_std > 1e-8) & (target_std > 1e-8)
    if valid.sum() == 0:
        return pred.new_tensor(0.0)

    corr = (pred_centered * target_centered).mean(dim=0) / (pred_std * target_std + EPS)
    corr = corr[valid].clamp(-1.0, 1.0)
    return 1.0 - corr.mean()


def ccc_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if pred.shape[0] < 2:
        return pred.new_tensor(0.0)

    mu_x = pred.mean(dim=0)
    mu_y = target.mean(dim=0)
    vx = pred.var(dim=0, unbiased=False)
    vy = target.var(dim=0, unbiased=False)
    cov = ((pred - mu_x.unsqueeze(0)) * (target - mu_y.unsqueeze(0))).mean(dim=0)
    denom = vx + vy + (mu_x - mu_y).pow(2) + EPS
    ccc = 2.0 * cov / denom

    valid = (vy > 1e-10)
    if valid.sum() == 0:
        return pred.new_tensor(0.0)

    ccc = ccc[valid].clamp(-1.0, 1.0)
    return 1.0 - ccc.mean()



class AntiCollapseLoss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def forward(
        self,
        outputs_z: torch.Tensor,
        labels_raw: torch.Tensor,
        sample_weights: torch.Tensor,
        train_stats: Dict[str, Any],
    ) -> Dict[str, torch.Tensor]:
        stat_tensors = get_stat_tensors(train_stats, outputs_z.device)
        labels_z = standardize_labels(labels_raw, stat_tensors)
        preds_raw_unbounded = destandardize_outputs(outputs_z, stat_tensors)
        preds_raw_clipped = clamp_raw_predictions(preds_raw_unbounded, stat_tensors)

        huber_elem = nn.functional.smooth_l1_loss(
            outputs_z,
            labels_z,
            reduction="none",
            beta=self.args.huber_beta,
        )
        raw_mae_elem = torch.abs(preds_raw_unbounded - labels_raw) / (stat_tensors["std"].unsqueeze(0) + EPS)

        legacy_rarity_w = build_task_rarity_weight(
            labels_raw=labels_raw,
            stat_tensors=stat_tensors,
            pos_alpha=self.args.loss_pos_alpha,
            high_alpha=self.args.loss_high_alpha,
            max_task_weight=self.args.max_task_weight,
        )
        lds_rarity_w = build_lds_task_weight(
            labels_raw=labels_raw,
            stat_tensors=stat_tensors,
            max_task_weight=self.args.max_task_weight,
        )

        if bool(getattr(self.args, "enable_lds", False)) and bool(train_stats.get("enable_lds", False)):
            if bool(getattr(self.args, "combine_lds_with_legacy_rarity", False)):
                rarity_w = lds_rarity_w * legacy_rarity_w
            else:
                rarity_w = lds_rarity_w
        else:
            rarity_w = legacy_rarity_w

        rarity_w = torch.clamp(rarity_w, min=1e-6, max=self.args.max_task_weight)

        elem = self.args.loss_huber_weight * huber_elem + self.args.loss_mae_weight * raw_mae_elem
        elem = elem * rarity_w
        per_sample = elem.mean(dim=1)
        point_loss = (per_sample * sample_weights).sum() / (sample_weights.sum() + EPS)

        corr_loss = ccc_loss(preds_raw_unbounded, labels_raw)
        pr_loss = pearson_loss(preds_raw_unbounded, labels_raw)
        range_loss = compute_output_range_penalty(
            preds_raw=preds_raw_unbounded,
            stat_tensors=stat_tensors,
            power=self.args.range_penalty_power,
            margin=self.args.range_margin,
        )
        var_loss = variance_matching_loss(preds_raw_clipped, labels_raw)
        rank_loss = pairwise_ranking_loss(
            preds_raw_clipped,
            labels_raw,
            max_pairs=self.args.rank_pairs_per_task,
            pred_margin=self.args.rank_pred_margin,
            target_margin=self.args.rank_target_margin,
        )
        tail_loss = tail_asymmetric_loss(
            preds_raw=preds_raw_clipped,
            labels_raw=labels_raw,
            stat_tensors=stat_tensors,
            tail_under_weight=self.args.tail_under_weight,
            tail_over_weight=self.args.tail_over_weight,
            zero_over_weight=self.args.zero_over_weight,
        )

        total = (
            point_loss
            + self.args.loss_ccc_weight * corr_loss
            + self.args.loss_pearson_weight * pr_loss
            + self.args.range_penalty_weight * range_loss
            + self.args.loss_variance_weight * var_loss
            + self.args.loss_ranking_weight * rank_loss
            + self.args.loss_tail_weight * tail_loss
        )

        return {
            "loss": total,
            "point_loss": point_loss.detach(),
            "ccc_loss": corr_loss.detach(),
            "pearson_loss": pr_loss.detach(),
            "range_loss": range_loss.detach(),
            "variance_loss": var_loss.detach(),
            "ranking_loss": rank_loss.detach(),
            "tail_loss": tail_loss.detach(),
            "pred_raw": preds_raw_clipped.detach(),
            "pred_raw_unbounded": preds_raw_unbounded.detach(),
            "labels_z": labels_z.detach(),
            "rarity_mean": rarity_w.mean().detach(),
            "legacy_rarity_mean": legacy_rarity_w.mean().detach(),
            "lds_rarity_mean": lds_rarity_w.mean().detach(),
        }


def safe_corrcoef(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2:
        return 0.0
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def variance_matching_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if pred.shape[0] < 2:
        return pred.new_tensor(0.0)
    pred_std = pred.std(dim=0, unbiased=False)
    true_std = target.std(dim=0, unbiased=False)
    valid = true_std > 1e-8
    if valid.sum() == 0:
        return pred.new_tensor(0.0)
    ratio = (pred_std[valid] + EPS) / (true_std[valid] + EPS)
    return torch.abs(torch.log(ratio)).mean()


def pairwise_ranking_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    max_pairs: int = 64,
    pred_margin: float = 0.25,
    target_margin: float = 0.5,
) -> torch.Tensor:
    bsz, num_tasks = pred.shape
    if bsz < 2 or max_pairs <= 0:
        return pred.new_tensor(0.0)
    losses = []
    for t in range(num_tasks):
        y = target[:, t]
        p = pred[:, t]
        i = torch.randint(0, bsz, (max_pairs,), device=pred.device)
        j = torch.randint(0, bsz, (max_pairs,), device=pred.device)
        valid = (i != j) & (torch.abs(y[i] - y[j]) >= float(target_margin))
        if valid.any():
            i = i[valid]
            j = j[valid]
            sign = torch.sign(y[i] - y[j])
            margin_term = (p[i] - p[j]) * sign
            losses.append(torch.relu(float(pred_margin) - margin_term).mean())
    if len(losses) == 0:
        return pred.new_tensor(0.0)
    return torch.stack(losses).mean()


def tail_asymmetric_loss(
    preds_raw: torch.Tensor,
    labels_raw: torch.Tensor,
    stat_tensors: Dict[str, torch.Tensor],
    tail_under_weight: float = 2.0,
    tail_over_weight: float = 1.0,
    zero_over_weight: float = 0.75,
) -> torch.Tensor:
    losses = []
    std = stat_tensors["std"]
    high_thr = stat_tensors["high_thr"]
    num_tasks = labels_raw.shape[1]
    for t in range(num_tasks):
        denom = std[t] + EPS
        yt = labels_raw[:, t]
        yp = preds_raw[:, t]
        tail_mask = torch.isfinite(high_thr[t]) & (yt >= high_thr[t])
        if tail_mask.any():
            err = yp[tail_mask] - yt[tail_mask]
            under = torch.relu(-err) / denom
            over = torch.relu(err) / denom
            losses.append((float(tail_under_weight) * under + float(tail_over_weight) * over).mean())
        zero_mask = yt <= 1e-8
        if zero_mask.any():
            zero_over = torch.relu(yp[zero_mask]) / denom
            losses.append(float(zero_over_weight) * zero_over.mean())
    if len(losses) == 0:
        return preds_raw.new_tensor(0.0)
    return torch.stack(losses).mean()


def compute_tail_distribution_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    score_cols: List[str],
    train_stats: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if train_stats is None:
        high_thr = np.array([np.inf] * len(score_cols), dtype=np.float32)
        label_std = np.array([1.0] * len(score_cols), dtype=np.float32)
    else:
        high_thr = np.asarray(train_stats.get("high_thr", [np.inf] * len(score_cols)), dtype=np.float32)
        label_std = np.asarray(train_stats.get("label_std", [1.0] * len(score_cols)), dtype=np.float32)

    per_task = {}
    tail_mae_norms = []
    tail_under_ratios = []
    for j, c in enumerate(score_cols):
        thr = float(high_thr[j]) if j < len(high_thr) else float("inf")
        denom = float(label_std[j]) if j < len(label_std) else 1.0
        denom = max(denom, 1e-6)
        if np.isfinite(thr):
            mask = y_true[:, j] >= thr - 1e-8
        else:
            mask = np.zeros(len(y_true), dtype=bool)
        if mask.sum() > 0:
            tail_true = y_true[mask, j]
            tail_pred = y_pred[mask, j]
            tail_mae = float(np.mean(np.abs(tail_true - tail_pred)))
            tail_mae_norm = float(tail_mae / denom)
            true_mean = float(np.mean(tail_true))
            pred_mean = float(np.mean(tail_pred))
            tail_under_ratio = float(pred_mean / max(true_mean, 1e-6)) if true_mean > 1e-6 else 1.0
        else:
            tail_mae = 0.0
            tail_mae_norm = 0.0
            true_mean = 0.0
            pred_mean = 0.0
            tail_under_ratio = 1.0
        per_task[c] = {
            "tail_threshold": thr,
            "tail_count": int(mask.sum()),
            "tail_mae": tail_mae,
            "tail_mae_norm": tail_mae_norm,
            "tail_true_mean": true_mean,
            "tail_pred_mean": pred_mean,
            "tail_under_ratio": tail_under_ratio,
        }
        tail_mae_norms.append(tail_mae_norm)
        tail_under_ratios.append(tail_under_ratio)

    return {
        "tail_metrics": per_task,
        "tail_mae_norm": float(np.mean(tail_mae_norms)) if tail_mae_norms else 0.0,
        "tail_under_ratio": float(np.mean(tail_under_ratios)) if tail_under_ratios else 1.0,
    }


def compute_selection_metric(
    metrics: Dict[str, Any],
    args,
    epoch_idx: Optional[int] = None,
) -> Dict[str, Any]:
    warmup_epochs = int(max(0, getattr(args, "selection_warmup_epochs", getattr(args, "early_warmup", 10))))
    base_selection = float(metrics["mae"] + args.selection_corr_weight * (1.0 - metrics["macro_corr"]))
    metrics["selection_metric_base"] = base_selection

    if epoch_idx is not None and int(epoch_idx) < warmup_epochs:
        metrics["selection_eligible"] = True
        metrics["selection_ineligible_reason"] = "warmup"
        metrics["selection_metric"] = base_selection
        metrics["distribution_penalty"] = 0.0
        return metrics

    min_task_std_ratio = float(metrics.get("min_task_std_ratio", metrics.get("macro_std_ratio", 0.0)))
    macro_std_ratio = float(metrics.get("macro_std_ratio", 0.0))
    macro_corr = float(metrics.get("macro_corr", 0.0))
    tail_under_ratio = float(metrics.get("tail_under_ratio", 1.0))
    tail_mae_norm = float(metrics.get("tail_mae_norm", 0.0))

    reasons = []
    if macro_std_ratio < float(args.selection_min_macro_std_ratio):
        reasons.append("macro_std")
    if min_task_std_ratio < float(args.selection_min_task_std_ratio):
        reasons.append("task_std")
    if macro_corr < float(args.selection_min_macro_corr):
        reasons.append("macro_corr")
    if tail_under_ratio < float(args.selection_min_tail_under_ratio):
        reasons.append("tail_under")

    distribution_penalty = (
        float(args.selection_std_weight) * abs(math.log(max(macro_std_ratio, 1e-6)))
        + float(args.selection_tail_mae_weight) * tail_mae_norm
        + float(args.selection_tail_under_weight) * max(0.0, 1.0 - min(tail_under_ratio, 1.0))
    )
    selection = base_selection + distribution_penalty
    eligible = len(reasons) == 0
    if not eligible:
        selection += float(args.selection_ineligible_penalty)

    metrics["selection_metric"] = float(selection)
    metrics["selection_eligible"] = bool(eligible)
    metrics["selection_ineligible_reason"] = "|".join(reasons) if reasons else ""
    metrics["distribution_penalty"] = float(distribution_penalty)
    return metrics


def summarize_predictions(y_true: np.ndarray, y_pred: np.ndarray, score_cols: List[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    out["mae"] = float(np.mean(np.abs(y_true - y_pred)))
    out["mse"] = float(np.mean((y_true - y_pred) ** 2))
    out["rmse"] = float(np.sqrt(out["mse"]))
    out["pred_mean"] = float(np.mean(y_pred))
    out["pred_std"] = float(np.std(y_pred))
    out["true_mean"] = float(np.mean(y_true))
    out["true_std"] = float(np.std(y_true))

    task_metrics = {}
    corr_list = []
    std_ratio_list = []
    for j, c in enumerate(score_cols):
        yt = y_true[:, j]
        yp = y_pred[:, j]
        corr = safe_corrcoef(yt, yp)
        pred_std = float(np.std(yp))
        true_std = float(np.std(yt))
        std_ratio = float(pred_std / true_std) if true_std > 1e-12 else 0.0
        corr_list.append(corr)
        std_ratio_list.append(std_ratio)
        task_metrics[c] = {
            "mae": float(np.mean(np.abs(yt - yp))),
            "rmse": float(np.sqrt(np.mean((yt - yp) ** 2))),
            "corr": corr,
            "pred_mean": float(np.mean(yp)),
            "pred_std": pred_std,
            "true_mean": float(np.mean(yt)),
            "true_std": true_std,
            "std_ratio": std_ratio,
        }

    out["macro_corr"] = float(np.mean(corr_list)) if corr_list else 0.0
    out["macro_std_ratio"] = float(np.mean(std_ratio_list)) if std_ratio_list else 0.0
    out["min_task_std_ratio"] = float(np.min(std_ratio_list)) if std_ratio_list else 0.0
    out["task_metrics"] = task_metrics
    return out



def safe_filename(name: str) -> str:
    s = str(name)
    s = re.sub(r'[\/:*?"<>|]+', '_', s)
    s = re.sub(r"\s+", "_", s.strip())
    return s or "unnamed"



def _build_discrete_count_table(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    vals = pd.to_numeric(df[score_col], errors="coerce").dropna().astype(float)
    if len(vals) == 0:
        return pd.DataFrame(columns=["label_value", "count", "proportion"])

    vals = np.rint(vals.values).astype(int)
    min_v = int(np.min(vals))
    max_v = int(np.max(vals))
    full_index = np.arange(min_v, max_v + 1, dtype=int)
    counts = pd.Series(vals).value_counts().sort_index().reindex(full_index, fill_value=0)
    total = float(max(counts.sum(), 1))
    out = pd.DataFrame({
        "label_value": counts.index.astype(int),
        "count": counts.values.astype(int),
    })
    out["proportion"] = out["count"].astype(float) / total
    return out



def save_overall_distribution_artifacts(
    matched_df: pd.DataFrame,
    score_cols: List[str],
    out_dir: str,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    summary_rows: List[Dict[str, Any]] = []

    for score_col in score_cols:
        cnt_df = _build_discrete_count_table(matched_df, score_col)
        for _, row in cnt_df.iterrows():
            summary_rows.append({
                "split": "overall",
                "subdomain": score_col,
                "label_value": int(row["label_value"]),
                "count": int(row["count"]),
                "proportion": float(row["proportion"]),
            })

        fig = plt.figure(figsize=(9, 5))
        ax = fig.add_subplot(111)
        ax.bar(cnt_df["label_value"].astype(str).tolist(), cnt_df["count"].astype(float).tolist())
        ax.set_title(f"整体数据分布 - {score_col}")
        ax.set_xlabel("标签值")
        ax.set_ylabel("样本数")
        ax.grid(axis="y", alpha=0.25)
        ax.text(
            0.98,
            0.95,
            f"N = {len(matched_df)}\nUnique PATNO = {matched_df['PATNO'].astype(str).nunique()}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"overall_{safe_filename(score_col)}_distribution.png"), dpi=200, bbox_inches="tight")
        plt.close(fig)

    pd.DataFrame(summary_rows).to_csv(
        os.path.join(out_dir, "overall_distribution_summary.csv"),
        index=False,
        encoding="utf-8-sig",
    )



def save_fold_distribution_artifacts(
    matched_df: pd.DataFrame,
    score_cols: List[str],
    train_ids: List[str],
    val_ids: List[str],
    fold: int,
    out_dir: str,
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    train_df = matched_df[matched_df["sample_id"].astype(str).isin(set(map(str, train_ids)))].copy()
    val_df = matched_df[matched_df["sample_id"].astype(str).isin(set(map(str, val_ids)))].copy()

    summary_rows: List[Dict[str, Any]] = []
    for score_col in score_cols:
        train_cnt = _build_discrete_count_table(train_df, score_col)
        val_cnt = _build_discrete_count_table(val_df, score_col)
        label_values = sorted(set(train_cnt["label_value"].tolist()) | set(val_cnt["label_value"].tolist()))
        cmp_df = pd.DataFrame({"label_value": label_values})
        cmp_df = cmp_df.merge(train_cnt.rename(columns={"count": "train_count", "proportion": "train_proportion"}), on="label_value", how="left")
        cmp_df = cmp_df.merge(val_cnt.rename(columns={"count": "val_count", "proportion": "val_proportion"}), on="label_value", how="left")
        cmp_df = cmp_df.fillna(0.0)

        for _, row in cmp_df.iterrows():
            summary_rows.append({
                "fold": int(fold),
                "subdomain": score_col,
                "label_value": int(row["label_value"]),
                "train_count": int(round(float(row["train_count"]))),
                "train_proportion": float(row["train_proportion"]),
                "val_count": int(round(float(row["val_count"]))),
                "val_proportion": float(row["val_proportion"]),
            })

        x = np.arange(len(cmp_df), dtype=float)
        width = 0.38
        fig = plt.figure(figsize=(10, 5))
        ax1 = fig.add_subplot(121)
        ax1.bar(x - width / 2, cmp_df["train_count"].astype(float).values, width=width, label="Train")
        ax1.bar(x + width / 2, cmp_df["val_count"].astype(float).values, width=width, label="Val")
        ax1.set_xticks(x)
        ax1.set_xticklabels(cmp_df["label_value"].astype(int).astype(str).tolist(), rotation=0)
        ax1.set_title(f"Fold {fold} 样本数分布 - {score_col}")
        ax1.set_xlabel("标签值")
        ax1.set_ylabel("样本数")
        ax1.grid(axis="y", alpha=0.25)
        ax1.legend()

        ax2 = fig.add_subplot(122)
        ax2.bar(x - width / 2, cmp_df["train_proportion"].astype(float).values, width=width, label="Train")
        ax2.bar(x + width / 2, cmp_df["val_proportion"].astype(float).values, width=width, label="Val")
        ax2.set_xticks(x)
        ax2.set_xticklabels(cmp_df["label_value"].astype(int).astype(str).tolist(), rotation=0)
        ax2.set_title(f"Fold {fold} 占比分布 - {score_col}")
        ax2.set_xlabel("标签值")
        ax2.set_ylabel("占比")
        ax2.grid(axis="y", alpha=0.25)
        ax2.legend()

        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"fold_{fold}_{safe_filename(score_col)}_train_val_distribution.png"), dpi=200, bbox_inches="tight")
        plt.close(fig)

    pd.DataFrame(summary_rows).to_csv(
        os.path.join(out_dir, f"fold_{fold}_train_val_distribution_summary.csv"),
        index=False,
        encoding="utf-8-sig",
    )


def build_patno_joint_strat_table(
    matched_df: pd.DataFrame,
    score_cols: List[str],
    n_total_bins: int = 5,
    n_splits: int = 5,
) -> pd.DataFrame:
    """
    Build a PATNO-level table for multi-task balanced fold assignment.

    Each PATNO is represented by:
      - diagnosis label
      - sample_count
      - Total mean / total_bin
      - per-task mean score and ordinal bin
      - per-task zero-pattern bin and high-tail pattern bin

    The returned table is later used by a greedy fold assignment routine that
    balances the marginal distributions needed by the downstream regression task.
    """
    del n_splits  # kept for interface compatibility

    df = matched_df.copy()
    df["PATNO"] = df["PATNO"].astype(str)
    df["label"] = df["label"].astype(str).fillna("").str.strip().str.lower()
    df["Total"] = df[score_cols].sum(axis=1).astype(float)

    score_bounds = resolve_score_bounds(score_cols=score_cols, reference_df=matched_df)
    global_high_thr: Dict[str, float] = {}
    for c in score_cols:
        vals = pd.to_numeric(df[c], errors="coerce").astype(float)
        pos_vals = vals[vals > 0]
        if len(pos_vals) > 0:
            thr = float(np.quantile(pos_vals, 0.75))
            thr = max(thr, float(np.min(pos_vals)))
        else:
            thr = float("inf")
        global_high_thr[c] = thr

    patno_rows = []
    for patno, sub in df.groupby("PATNO", sort=False):
        row = {
            "PATNO": str(patno),
            "label": str(mode_or_first(sub["label"])),
            "sample_count": int(len(sub)),
            "Total_for_bin": float(sub["Total"].mean()),
            "Total_min": float(sub["Total"].min()),
            "Total_max": float(sub["Total"].max()),
        }
        for j, c in enumerate(score_cols):
            vals = sub[c].astype(float).values
            mean_score = float(np.mean(vals))
            upper = float(score_bounds["upper"][j])
            score_bin = int(np.clip(np.rint(mean_score), 0, int(round(upper))))

            zero_prop = float(np.mean(vals <= 1e-8))
            if zero_prop >= 0.999:
                zero_bin = 2
            elif zero_prop <= 1e-8:
                zero_bin = 0
            else:
                zero_bin = 1

            thr = global_high_thr[c]
            high_prop = float(np.mean(vals >= thr)) if np.isfinite(thr) else 0.0
            if high_prop <= 1e-8:
                high_bin = 0
            elif high_prop < 0.50:
                high_bin = 1
            else:
                high_bin = 2

            row[f"{c}_mean"] = mean_score
            row[f"{c}_bin"] = score_bin
            row[f"{c}_zero_bin"] = int(zero_bin)
            row[f"{c}_high_bin"] = int(high_bin)
        patno_rows.append(row)

    patno_df = pd.DataFrame(patno_rows)
    if len(patno_df) == 0:
        raise ValueError("Empty PATNO table for stratification.")

    n_total_bins = int(max(1, n_total_bins))
    n_total_bins_eff = min(n_total_bins, len(patno_df))
    if n_total_bins_eff <= 1:
        patno_df["total_bin"] = 0
    else:
        ranked = patno_df["Total_for_bin"].rank(method="first")
        patno_df["total_bin"] = pd.qcut(
            ranked,
            q=n_total_bins_eff,
            labels=False,
            duplicates="drop",
        ).astype(int)

    weight_maps: Dict[str, Dict[str, float]] = {}
    for key in ["label", "total_bin"] + [f"{c}_bin" for c in score_cols] + [f"{c}_zero_bin" for c in score_cols] + [f"{c}_high_bin" for c in score_cols]:
        counts = patno_df.groupby(key)["sample_count"].sum().to_dict()
        weight_maps[key] = {str(k): float(1.0 / max(v, 1.0)) for k, v in counts.items()}

    rarity_scores = []
    for _, row in patno_df.iterrows():
        rarity = float(row["sample_count"]) * (
            weight_maps["label"].get(str(row["label"]), 0.0)
            + 0.35 * weight_maps["total_bin"].get(str(row["total_bin"]), 0.0)
            + sum(weight_maps[f"{c}_bin"].get(str(row[f"{c}_bin"]), 0.0) for c in score_cols)
            + 0.40 * sum(weight_maps[f"{c}_zero_bin"].get(str(row[f"{c}_zero_bin"]), 0.0) for c in score_cols)
            + 0.35 * sum(weight_maps[f"{c}_high_bin"].get(str(row[f"{c}_high_bin"]), 0.0) for c in score_cols)
        )
        rarity_scores.append(rarity)

    patno_df["balance_priority"] = rarity_scores
    patno_df["joint_key_raw"] = patno_df["label"].astype(str) + "__TOT" + patno_df["total_bin"].astype(str)
    patno_df["joint_key_final"] = "multitask_balanced"
    patno_df["is_stratifiable"] = 1
    return patno_df


def _build_balance_attribute_specs(
    patno_df: pd.DataFrame,
    score_cols: List[str],
) -> List[Dict[str, Any]]:
    del patno_df
    attr_specs: List[Dict[str, Any]] = [
        {"name": "label", "weight": 1.25},
        {"name": "total_bin", "weight": 0.40},
    ]
    for c in score_cols:
        attr_specs.append({"name": f"{c}_bin", "weight": 1.00})
        attr_specs.append({"name": f"{c}_zero_bin", "weight": 0.65})
        attr_specs.append({"name": f"{c}_high_bin", "weight": 0.55})
    return attr_specs


def build_joint_stratified_kfold_splits(
    matched_df: pd.DataFrame,
    score_cols: List[str],
    n_splits: int = 5,
    n_total_bins: int = 5,
    seed: int = 42,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """
    PATNO-level multi-task balanced CV split.

    Instead of only stratifying by `label + Total_bin`, this routine greedily
    assigns PATNOs to folds so that each validation fold has a similar sample-level
    marginal distribution over:
      - diagnosis label
      - Total severity bin
      - each of the five subdomain ordinal bins

    This is more aligned with the downstream objective of 5-task regression.
    """
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")

    patno_df = build_patno_joint_strat_table(
        matched_df=matched_df,
        score_cols=score_cols,
        n_total_bins=n_total_bins,
        n_splits=n_splits,
    )
    if len(patno_df) < n_splits:
        raise ValueError(
            f"Number of unique PATNO ({len(patno_df)}) is smaller than n_splits ({n_splits})."
        )

    rng = np.random.RandomState(seed)
    attr_specs = _build_balance_attribute_specs(patno_df=patno_df, score_cols=score_cols)

    target_sample_count = float(patno_df["sample_count"].sum()) / float(n_splits)
    target_patno_count = float(len(patno_df)) / float(n_splits)

    target_attr_counts: Dict[str, Dict[Any, float]] = {}
    for spec in attr_specs:
        name = spec["name"]
        target_attr_counts[name] = (
            patno_df.groupby(name)["sample_count"].sum().astype(float).div(float(n_splits)).to_dict()
        )

    fold_attr_counts: List[Dict[str, Dict[Any, float]]] = []
    fold_sample_counts = [0.0 for _ in range(n_splits)]
    fold_patno_counts = [0 for _ in range(n_splits)]
    fold_val_patnos: List[List[str]] = [[] for _ in range(n_splits)]

    for _ in range(n_splits):
        fold_attr_counts.append({spec["name"]: defaultdict(float) for spec in attr_specs})

    order_df = patno_df.sort_values(
        by=["balance_priority", "sample_count", "Total_for_bin", "PATNO"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)

    size_weight = 0.85
    patno_size_weight = 0.30
    overshoot_weight = 1.50

    for _, row in order_df.iterrows():
        row_sample_count = float(row["sample_count"])
        candidate_costs = []

        for fold_idx in range(n_splits):
            sample_after = fold_sample_counts[fold_idx] + row_sample_count
            patno_after = fold_patno_counts[fold_idx] + 1

            cost = 0.0
            cost += size_weight * ((sample_after - target_sample_count) / max(target_sample_count, 1.0)) ** 2
            cost += patno_size_weight * ((patno_after - target_patno_count) / max(target_patno_count, 1.0)) ** 2

            if sample_after > target_sample_count:
                cost += overshoot_weight * ((sample_after - target_sample_count) / max(target_sample_count, 1.0)) ** 2

            for spec in attr_specs:
                name = spec["name"]
                attr_weight = float(spec["weight"])
                cat = row[name]
                target = float(target_attr_counts[name].get(cat, 0.0))
                current = float(fold_attr_counts[fold_idx][name].get(cat, 0.0))
                after = current + row_sample_count

                denom = max(target, 1.0)
                before_err = ((current - target) ** 2) / denom
                after_err = ((after - target) ** 2) / denom
                cost += attr_weight * (after_err - before_err)

            candidate_costs.append(cost)

        min_cost = float(min(candidate_costs))
        best_folds = [i for i, c in enumerate(candidate_costs) if abs(float(c) - min_cost) <= 1e-12]
        if len(best_folds) > 1:
            best_folds = sorted(
                best_folds,
                key=lambda i: (
                    fold_sample_counts[i],
                    fold_patno_counts[i],
                    rng.rand(),
                ),
            )
        chosen_fold = int(best_folds[0])

        fold_val_patnos[chosen_fold].append(str(row["PATNO"]))
        fold_sample_counts[chosen_fold] += row_sample_count
        fold_patno_counts[chosen_fold] += 1
        for spec in attr_specs:
            name = spec["name"]
            cat = row[name]
            fold_attr_counts[chosen_fold][name][cat] += row_sample_count

    fold_infos: List[Dict[str, Any]] = []
    patno_to_fold = {}

    for fold_idx in range(1, n_splits + 1):
        val_patnos = sorted(set(map(str, fold_val_patnos[fold_idx - 1])))
        for p in val_patnos:
            patno_to_fold[p] = fold_idx
        val_patno_set = set(val_patnos)
        train_patnos = sorted(set(patno_df["PATNO"].astype(str).tolist()) - val_patno_set)
        train_patno_set = set(train_patnos)

        train_ids = matched_df[
            matched_df["PATNO"].astype(str).isin(train_patno_set)
        ]["sample_id"].astype(str).tolist()

        val_ids = matched_df[
            matched_df["PATNO"].astype(str).isin(val_patno_set)
        ]["sample_id"].astype(str).tolist()

        overlap_patno = train_patno_set & val_patno_set
        overlap_ids = set(train_ids) & set(val_ids)
        if len(overlap_patno) > 0:
            raise ValueError(f"Fold {fold_idx}: PATNO leakage detected: {list(overlap_patno)[:10]}")
        if len(overlap_ids) > 0:
            raise ValueError(f"Fold {fold_idx}: sample_id leakage detected: {list(overlap_ids)[:10]}")

        fold_infos.append(
            {
                "fold": fold_idx,
                "train_patnos": train_patnos,
                "val_patnos": val_patnos,
                "train_ids": train_ids,
                "val_ids": val_ids,
            }
        )

    patno_df["assigned_fold"] = patno_df["PATNO"].astype(str).map(patno_to_fold).fillna(-1).astype(int)
    return patno_df, fold_infos


def validate_fold_separation(
    matched_df: pd.DataFrame,
    train_ids: List[str],
    val_ids: List[str],
    fold: int,
) -> None:
    """Fail on subject/sample leakage or missing IDs in a held-out CV fold."""
    train_set = set(map(str, train_ids))
    val_set = set(map(str, val_ids))
    overlap = train_set & val_set
    if overlap:
        raise ValueError(f"Fold {fold}: overlapping train/validation sample IDs: {sorted(overlap)[:5]}")
    indexed = matched_df.copy()
    indexed["sample_id"] = indexed["sample_id"].astype(str)
    all_ids = set(indexed["sample_id"])
    if train_set - all_ids or val_set - all_ids:
        raise ValueError(f"Fold {fold}: split contains IDs absent from matched samples")
    train_patnos = set(indexed.loc[indexed["sample_id"].isin(train_set), "PATNO"].astype(str))
    val_patnos = set(indexed.loc[indexed["sample_id"].isin(val_set), "PATNO"].astype(str))
    overlap_patnos = train_patnos & val_patnos
    if overlap_patnos:
        raise ValueError(f"Fold {fold}: subject leakage: {sorted(overlap_patnos)[:5]}")
    if not train_set or not val_set:
        raise ValueError(f"Fold {fold}: empty training or validation set")


def print_one_fold_summary(
    fold: int,
    matched_df: pd.DataFrame,
    score_cols: List[str],
    train_ids: List[str],
    val_ids: List[str],
) -> None:
    train_df = matched_df[matched_df["sample_id"].astype(str).isin(set(map(str, train_ids)))].copy()
    val_df = matched_df[matched_df["sample_id"].astype(str).isin(set(map(str, val_ids)))].copy()

    train_df["Total"] = train_df[score_cols].sum(axis=1)
    val_df["Total"] = val_df[score_cols].sum(axis=1)

    print("\n" + "-" * 70)
    print(f"Fold {fold} summary")
    print("-" * 70)
    print(
        f"Train PATNO: {train_df['PATNO'].astype(str).nunique()} | "
        f"Val PATNO: {val_df['PATNO'].astype(str).nunique()} | "
        f"Train samples: {len(train_df)} | Val samples: {len(val_df)}"
    )

    print("\nPATNO-level label distribution:")
    print("Train:")
    print(train_df.groupby("label")["PATNO"].nunique())
    print("Val:")
    print(val_df.groupby("label")["PATNO"].nunique())

    print("\nSample-level label distribution:")
    print("Train:")
    print(train_df["label"].value_counts())
    print("Val:")
    print(val_df["label"].value_counts())

    print("\nTotal score distribution:")
    print(
        f"Train Total mean={train_df['Total'].mean():.4f}, std={train_df['Total'].std():.4f}, "
        f"min={train_df['Total'].min():.4f}, max={train_df['Total'].max():.4f}"
    )
    print(
        f"Val   Total mean={val_df['Total'].mean():.4f}, std={val_df['Total'].std():.4f}, "
        f"min={val_df['Total'].min():.4f}, max={val_df['Total'].max():.4f}"
    )


def summarize_cv_predictions(pred_df: pd.DataFrame, score_cols: List[str]) -> Dict[str, Any]:
    y_true = pred_df[[f"{c}_true" for c in score_cols]].values.astype(np.float32)
    y_pred = pred_df[[f"{c}_pred" for c in score_cols]].values.astype(np.float32)
    metrics = summarize_predictions(y_true, y_pred, score_cols)
    return metrics


def save_fold_loss_plot(history_df: pd.DataFrame, fold: int, save_dir: str) -> Optional[str]:
    """
    Save per-fold train/val loss curve.
    - Train curve uses `train_loss`.
    - Val curve uses `val_selection_metric` as validation-side loss proxy.
    """
    if history_df is None or len(history_df) == 0:
        return None
    required_cols = {"epoch", "train_loss", "val_selection_metric"}
    if not required_cols.issubset(set(history_df.columns)):
        return None

    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f"loss_curve_fold_{fold}.png")

    fig = plt.figure(figsize=(8, 5), dpi=150)
    x = history_df["epoch"].values
    plt.plot(x, history_df["train_loss"].values, label="Train Loss", linewidth=1.8)
    plt.plot(x, history_df["val_selection_metric"].values, label="Val Loss", linewidth=1.8)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Fold {fold} Train/Val Loss")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


@torch.no_grad()
def evaluate_loader_exact(
    model: nn.Module,
    loader: DataLoader,
    score_cols: List[str],
    train_stats: Dict[str, Any],
    device: torch.device,
    selection_corr_weight: float,
    include_rows: bool = False,
    fold: Optional[int] = None,
    clip_to_valid_range: bool = True,
    args=None,
    epoch_idx: Optional[int] = None,
) -> Dict[str, Any]:
    model.eval()
    stat_tensors = get_stat_tensors(train_stats, device)

    y_true_list = []
    y_pred_list = []
    rows = []

    for batch in loader:
        A = batch["connectome"].to(device)
        lap_pe = batch["lap_pe"].to(device)
        labels = batch["label"].to(device)
        sample_ids = batch["sample_id"]
        metas = batch["meta"]

        outputs_z = model(A, lap_pe)
        preds = destandardize_outputs(outputs_z, stat_tensors)
        if clip_to_valid_range:
            preds = clamp_raw_predictions(preds, stat_tensors)

        y_true = labels.cpu().numpy()
        y_pred = preds.cpu().numpy()
        y_true_list.append(y_true)
        y_pred_list.append(y_pred)

        if include_rows:
            for i, sid in enumerate(sample_ids):
                meta = metas[i]
                row = {
                    "sample_id": sid,
                    "PATNO": meta["PATNO"],
                    "EVENT_ID": meta["EVENT_ID"],
                    "INFODT": meta["INFODT"],
                    "EXAMDT": meta["EXAMDT"],
                    "scan_date": meta["scan_date"],
                    "match_note": meta["match_note"],
                    "month_gap": meta["month_gap"],
                    "fold": fold if fold is not None else 1,
                }
                for j, c in enumerate(score_cols):
                    row[f"{c}_true"] = float(y_true[i, j])
                for j, c in enumerate(score_cols):
                    row[f"{c}_pred"] = float(y_pred[i, j])
                row["MAE"] = float(np.mean(np.abs(y_true[i] - y_pred[i])))
                row["MSE"] = float(np.mean((y_true[i] - y_pred[i]) ** 2))
                row["PredTotal"] = float(np.sum(y_pred[i]))
                row["TrueTotal"] = float(np.sum(y_true[i]))
                rows.append(row)

    y_true = np.concatenate(y_true_list, axis=0)
    y_pred = np.concatenate(y_pred_list, axis=0)
    metrics = summarize_predictions(y_true, y_pred, score_cols)
    metrics.update(compute_tail_distribution_metrics(y_true, y_pred, score_cols, train_stats=train_stats))
    if args is not None:
        metrics = compute_selection_metric(metrics, args=args, epoch_idx=epoch_idx)
    else:
        metrics["selection_metric"] = metrics["mae"] + selection_corr_weight * (1.0 - metrics["macro_corr"])
        metrics["selection_metric_base"] = metrics["selection_metric"]
        metrics["selection_eligible"] = True
        metrics["selection_ineligible_reason"] = ""
        metrics["distribution_penalty"] = 0.0
    if include_rows:
        metrics["rows_df"] = pd.DataFrame(rows)
    return metrics


# =========================
# Inference
# =========================
@torch.no_grad()
def inference_on_validation(
    fold: int,
    val_ids: List[str],
    matched_df: pd.DataFrame,
    score_cols: List[str],
    model_path: str,
    model: nn.Module,
    device: torch.device,
    args,
) -> pd.DataFrame:
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    train_stats = checkpoint["train_target_stats"]

    val_dataset = ConnectomeDataset(
        matched_df=matched_df,
        sample_ids=val_ids,
        score_cols=score_cols,
        sample_weight_map=None,
        lap_pe_dim=args.lap_pe_dim,
        preload=args.preload_connectomes,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    metrics = evaluate_loader_exact(
        model=model,
        loader=val_loader,
        score_cols=score_cols,
        train_stats=train_stats,
        device=device,
        selection_corr_weight=args.selection_corr_weight,
        include_rows=True,
        fold=fold,
        clip_to_valid_range=args.range_clip_eval,
        args=args,
        epoch_idx=None,
    )
    df_out = metrics["rows_df"]

    col_order = [
        "sample_id", "PATNO", "EVENT_ID", "INFODT", "EXAMDT", "scan_date", "match_note", "month_gap", "fold"
    ]
    for c in score_cols:
        col_order.append(f"{c}_true")
    for c in score_cols:
        col_order.append(f"{c}_pred")
    col_order += ["TrueTotal", "PredTotal", "MAE", "MSE"]
    return df_out[col_order]




def _flatten_feature_tensor(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if x is None:
        return None
    if not torch.is_tensor(x):
        return None
    x = x.detach()
    if x.ndim == 0:
        x = x.view(1, 1)
    elif x.ndim == 1:
        x = x.unsqueeze(1)
    elif x.ndim > 2:
        x = x.reshape(x.shape[0], -1)
    return x.float().cpu()


def _select_first_tensor(obj: Any) -> Optional[torch.Tensor]:
    if torch.is_tensor(obj):
        return obj
    if isinstance(obj, (list, tuple)):
        for item in obj:
            t = _select_first_tensor(item)
            if t is not None:
                return t
    if isinstance(obj, dict):
        for key in ["x", "feat", "feature", "hidden", "representation", "emb", "embedding", "token", "tokens"]:
            if key in obj:
                t = _select_first_tensor(obj[key])
                if t is not None:
                    return t
        for _, value in obj.items():
            t = _select_first_tensor(value)
            if t is not None:
                return t
    return None


def _find_task_head_modules(model: nn.Module, num_tasks: int) -> List[Tuple[str, nn.Module]]:
    attr_candidates = [
        "task_heads", "heads", "regression_heads", "regressors", "subdomain_heads",
        "output_heads", "pred_heads", "task_regressors",
    ]
    for attr in attr_candidates:
        mod = getattr(model, attr, None)
        if isinstance(mod, nn.ModuleList) and len(mod) == num_tasks:
            return [(f"{attr}.{i}", mod[i]) for i in range(num_tasks)]
        if isinstance(mod, (list, tuple)) and len(mod) == num_tasks and all(isinstance(m, nn.Module) for m in mod):
            return [(f"{attr}.{i}", mod[i]) for i in range(num_tasks)]

    for name, mod in model.named_modules():
        if isinstance(mod, nn.ModuleList) and len(mod) == num_tasks:
            return [(f"{name}.{i}", mod[i]) for i in range(num_tasks)]
    return []


def _find_global_head_module(model: nn.Module) -> Optional[Tuple[str, nn.Module]]:
    attr_candidates = ["global_head", "shared_head", "head", "regressor", "output_head", "pred_head"]
    for attr in attr_candidates:
        mod = getattr(model, attr, None)
        if isinstance(mod, nn.Module):
            return attr, mod
    return None


def _find_final_vector_head_module(model: nn.Module, num_tasks: int) -> Optional[Tuple[str, nn.Module]]:
    global_head = _find_global_head_module(model)
    if global_head is not None:
        return global_head

    linear_candidates: List[Tuple[str, nn.Module]] = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and int(getattr(mod, "out_features", -1)) == int(num_tasks):
            linear_candidates.append((name, mod))
    if len(linear_candidates) > 0:
        return linear_candidates[-1]
    return None


class _PreHeadHookCollector:
    def __init__(self, model: nn.Module, num_tasks: int):
        self.model = model
        self.num_tasks = int(num_tasks)
        self.handles: List[Any] = []
        self.task_hidden: Dict[int, torch.Tensor] = {}
        self.global_hidden: Optional[torch.Tensor] = None
        self.pre_head_hidden: Optional[torch.Tensor] = None
        self.extraction_mode = "hooks"
        self._register()

    def _register(self) -> None:
        task_modules = _find_task_head_modules(self.model, self.num_tasks)
        if len(task_modules) > 0:
            for task_idx, (_, module) in enumerate(task_modules):
                def _make_hook(idx: int):
                    def _hook(mod, inputs):
                        x = _select_first_tensor(inputs)
                        if x is not None:
                            self.task_hidden[idx] = x.detach()
                    return _hook
                self.handles.append(module.register_forward_pre_hook(_make_hook(task_idx)))

        global_module = _find_global_head_module(self.model)
        if global_module is not None:
            _, mod = global_module
            def _global_hook(mod, inputs):
                x = _select_first_tensor(inputs)
                if x is not None:
                    self.global_hidden = x.detach()
                    self.pre_head_hidden = x.detach()
            self.handles.append(mod.register_forward_pre_hook(_global_hook))

        if self.pre_head_hidden is None:
            vector_head = _find_final_vector_head_module(self.model, self.num_tasks)
            if vector_head is not None:
                _, mod = vector_head
                def _vector_hook(mod, inputs):
                    x = _select_first_tensor(inputs)
                    if x is not None:
                        self.pre_head_hidden = x.detach()
                self.handles.append(mod.register_forward_pre_hook(_vector_hook))

    def remove(self) -> None:
        for h in self.handles:
            try:
                h.remove()
            except Exception:
                pass
        self.handles = []


def _stack_task_hidden(task_hidden_dict: Dict[int, torch.Tensor], num_tasks: int) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    if len(task_hidden_dict) == 0:
        return None, None
    task_list: List[Optional[torch.Tensor]] = []
    flat_list: List[torch.Tensor] = []
    for i in range(num_tasks):
        x = task_hidden_dict.get(i)
        x = _flatten_feature_tensor(x)
        task_list.append(x)
        if x is not None:
            flat_list.append(x)
    valid = [x for x in task_list if x is not None]
    if len(valid) == 0:
        return None, None
    same_dim = all(x.shape == valid[0].shape for x in valid)
    task_hidden = None
    if same_dim and len(valid) == num_tasks:
        task_hidden = torch.stack(task_list, dim=1)
    task_hidden_flat = torch.cat(flat_list, dim=1) if len(flat_list) > 0 else None
    return task_hidden, task_hidden_flat


def _forward_model_with_hidden(
    model: nn.Module,
    A: torch.Tensor,
    lap_pe: torch.Tensor,
    num_tasks: int,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    features: Dict[str, Any] = {
        "global_hidden": None,
        "task_hidden": None,
        "task_hidden_flat": None,
        "pre_head_hidden": None,
        "extraction_mode": "unknown",
    }

    try:
        out = model(A, lap_pe, return_features=True)
        if isinstance(out, dict):
            outputs_z = None
            for key in ["outputs_z", "pred_z", "logits", "output", "pred"]:
                if key in out and torch.is_tensor(out[key]):
                    outputs_z = out[key]
                    break
            if outputs_z is None:
                outputs_z = _select_first_tensor(out)
            if outputs_z is None:
                raise ValueError("Cannot parse outputs from model return_features dict.")
            gh = out.get("global_hidden", out.get("global_feature", out.get("shared_hidden")))
            th = out.get("task_hidden", out.get("task_features"))
            ph = out.get("pre_head_hidden", out.get("hidden"))
            features["global_hidden"] = _flatten_feature_tensor(_select_first_tensor(gh))
            if isinstance(th, dict):
                task_hidden, task_hidden_flat = _stack_task_hidden({int(k): v for k, v in th.items()}, num_tasks)
                features["task_hidden"] = task_hidden
                features["task_hidden_flat"] = task_hidden_flat
            elif isinstance(th, (list, tuple)):
                task_dict = {i: t for i, t in enumerate(th) if torch.is_tensor(t)}
                task_hidden, task_hidden_flat = _stack_task_hidden(task_dict, num_tasks)
                features["task_hidden"] = task_hidden
                features["task_hidden_flat"] = task_hidden_flat
            elif torch.is_tensor(th):
                thf = _flatten_feature_tensor(th)
                features["task_hidden"] = thf.unsqueeze(1) if thf is not None and thf.ndim == 2 else None
                features["task_hidden_flat"] = thf
            features["pre_head_hidden"] = _flatten_feature_tensor(_select_first_tensor(ph))
            features["extraction_mode"] = "return_features"
            return outputs_z, features
        if isinstance(out, (list, tuple)) and len(out) >= 1 and torch.is_tensor(out[0]):
            outputs_z = out[0]
            if len(out) >= 2:
                features["pre_head_hidden"] = _flatten_feature_tensor(_select_first_tensor(out[1]))
            features["extraction_mode"] = "return_features_tuple"
            return outputs_z, features
    except TypeError:
        pass
    except Exception:
        pass

    collector = _PreHeadHookCollector(model, num_tasks=num_tasks)
    try:
        outputs_z = model(A, lap_pe)
    finally:
        collector.remove()

    features["global_hidden"] = _flatten_feature_tensor(collector.global_hidden)
    task_hidden, task_hidden_flat = _stack_task_hidden(collector.task_hidden, num_tasks)
    features["task_hidden"] = task_hidden
    features["task_hidden_flat"] = task_hidden_flat
    features["pre_head_hidden"] = _flatten_feature_tensor(collector.pre_head_hidden)
    features["extraction_mode"] = collector.extraction_mode

    if features["pre_head_hidden"] is None and features["task_hidden_flat"] is None and features["global_hidden"] is None:
        features["pre_head_hidden"] = _flatten_feature_tensor(outputs_z)
        features["extraction_mode"] = "fallback_outputs_z"
    return outputs_z, features


@torch.no_grad()
def inference_on_validation_with_hidden(
    fold: int,
    val_ids: List[str],
    matched_df: pd.DataFrame,
    score_cols: List[str],
    model_path: str,
    model: nn.Module,
    device: torch.device,
    args,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    train_stats = checkpoint["train_target_stats"]
    stat_tensors = get_stat_tensors(train_stats, device)

    val_dataset = ConnectomeDataset(
        matched_df=matched_df,
        sample_ids=val_ids,
        score_cols=score_cols,
        sample_weight_map=None,
        lap_pe_dim=args.lap_pe_dim,
        preload=args.preload_connectomes,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    rows: List[Dict[str, Any]] = []
    sample_ids_all: List[str] = []
    metas_all: List[Dict[str, Any]] = []
    y_true_all: List[torch.Tensor] = []
    y_pred_all: List[torch.Tensor] = []
    global_hidden_all: List[torch.Tensor] = []
    task_hidden_all: List[torch.Tensor] = []
    task_hidden_flat_all: List[torch.Tensor] = []
    pre_head_hidden_all: List[torch.Tensor] = []
    extraction_modes: List[str] = []

    for batch in val_loader:
        A = batch["connectome"].to(device)
        lap_pe = batch["lap_pe"].to(device)
        labels = batch["label"].to(device)
        batch_sample_ids = batch["sample_id"]
        metas = batch["meta"]

        outputs_z, feat_dict = _forward_model_with_hidden(model, A, lap_pe, num_tasks=len(score_cols))
        preds = destandardize_outputs(outputs_z, stat_tensors)
        if args.range_clip_eval:
            preds = clamp_raw_predictions(preds, stat_tensors)

        y_true = labels.detach().cpu()
        y_pred = preds.detach().cpu()
        y_true_all.append(y_true)
        y_pred_all.append(y_pred)
        sample_ids_all.extend(batch_sample_ids)
        metas_all.extend(metas)
        extraction_modes.extend([str(feat_dict.get("extraction_mode", "unknown"))] * len(batch_sample_ids))

        gh = feat_dict.get("global_hidden")
        th = feat_dict.get("task_hidden")
        thf = feat_dict.get("task_hidden_flat")
        ph = feat_dict.get("pre_head_hidden")
        if gh is not None:
            global_hidden_all.append(gh)
        if th is not None:
            task_hidden_all.append(th)
        if thf is not None:
            task_hidden_flat_all.append(thf)
        if ph is not None:
            pre_head_hidden_all.append(ph)

        y_true_np = y_true.numpy()
        y_pred_np = y_pred.numpy()
        for i, sid in enumerate(batch_sample_ids):
            meta = metas[i]
            row = {
                "sample_id": sid,
                "PATNO": meta["PATNO"],
                "EVENT_ID": meta["EVENT_ID"],
                "INFODT": meta["INFODT"],
                "EXAMDT": meta["EXAMDT"],
                "scan_date": meta["scan_date"],
                "match_note": meta["match_note"],
                "month_gap": meta["month_gap"],
                "fold": fold,
            }
            for j, c in enumerate(score_cols):
                row[f"{c}_true"] = float(y_true_np[i, j])
            for j, c in enumerate(score_cols):
                row[f"{c}_pred"] = float(y_pred_np[i, j])
            row["MAE"] = float(np.mean(np.abs(y_true_np[i] - y_pred_np[i])))
            row["MSE"] = float(np.mean((y_true_np[i] - y_pred_np[i]) ** 2))
            row["PredTotal"] = float(np.sum(y_pred_np[i]))
            row["TrueTotal"] = float(np.sum(y_true_np[i]))
            rows.append(row)

    df_out = pd.DataFrame(rows)
    col_order = [
        "sample_id", "PATNO", "EVENT_ID", "INFODT", "EXAMDT", "scan_date", "match_note", "month_gap", "fold"
    ]
    for c in score_cols:
        col_order.append(f"{c}_true")
    for c in score_cols:
        col_order.append(f"{c}_pred")
    col_order += ["TrueTotal", "PredTotal", "MAE", "MSE"]
    df_out = df_out[col_order]

    payload: Dict[str, Any] = {
        "fold": int(fold),
        "model_path": str(model_path),
        "sample_ids": sample_ids_all,
        "meta": metas_all,
        "score_cols": list(score_cols),
        "y_true": torch.cat(y_true_all, dim=0) if len(y_true_all) > 0 else torch.empty(0, len(score_cols)),
        "y_pred": torch.cat(y_pred_all, dim=0) if len(y_pred_all) > 0 else torch.empty(0, len(score_cols)),
        "prediction_table": df_out.to_dict(orient="records"),
        "feature_extraction_mode": extraction_modes,
    }
    if len(global_hidden_all) > 0:
        payload["global_hidden"] = torch.cat(global_hidden_all, dim=0)
    if len(task_hidden_all) > 0:
        payload["task_hidden"] = torch.cat(task_hidden_all, dim=0)
    if len(task_hidden_flat_all) > 0:
        payload["task_hidden_flat"] = torch.cat(task_hidden_flat_all, dim=0)
    if len(pre_head_hidden_all) > 0:
        payload["pre_head_hidden"] = torch.cat(pre_head_hidden_all, dim=0)

    return df_out, payload

def run_inference_mode(matched_df: pd.DataFrame, score_cols: List[str], checkpoint_dir: str, device: torch.device, args):
    print("\n" + "=" * 60)
    print("Inference Mode")
    print("=" * 60)

    if not os.path.exists(checkpoint_dir):
        raise FileNotFoundError(f"Checkpoint dir not found: {checkpoint_dir}")

    fold_dirs = []
    for name in os.listdir(checkpoint_dir):
        p = os.path.join(checkpoint_dir, name)
        if os.path.isdir(p) and re.fullmatch(r"fold_\d+", name):
            fold_dirs.append((int(name.split("_")[-1]), p))
    fold_dirs = sorted(fold_dirs, key=lambda x: x[0])

    if len(fold_dirs) >= 2:
        print(f"Detected {len(fold_dirs)} fold directories under {checkpoint_dir}. Running CV inference.")
        _, fold_infos = build_joint_stratified_kfold_splits(
            matched_df=matched_df,
            score_cols=score_cols,
            n_splits=args.cv_folds,
            n_total_bins=args.n_total_bins,
            seed=args.seed,
        )

        first_conn = load_connectome_matrix(matched_df.iloc[0]["connectome_path"])
        num_rois_default = first_conn.shape[0]

        all_pred_dfs = []
        for fold, fold_dir in fold_dirs:
            if fold > len(fold_infos):
                print(f"[Warn] Skip fold_{fold}: fold index exceeds reconstructed split count.")
                continue

            mp = os.path.join(fold_dir, "best_model.pt")
            if not os.path.exists(mp):
                mp = os.path.join(fold_dir, f"best_model_fold_{fold}.pt")
            if not os.path.exists(mp):
                print(f"[Warn] Skip fold_{fold}: no best model found.")
                continue

            ckpt = torch.load(mp, map_location=device)
            cfg = ckpt.get("config", {})
            num_rois = cfg.get("num_rois", num_rois_default)

            model = ConnectomeSubdomainNet(
                num_rois=num_rois,
                num_tasks=cfg.get("num_tasks", args.num_tasks),
                embed_dim=cfg.get("embed_dim", args.embed_dim),
                depth=cfg.get("depth", args.depth),
                num_heads=cfg.get("num_heads", args.num_heads),
                ff_mult=cfg.get("ff_mult", args.ff_mult),
                dropout=cfg.get("dropout", args.dropout),
                lap_pe_dim=cfg.get("lap_pe_dim", args.lap_pe_dim),
            ).to(device)

            fold_info = fold_infos[fold - 1]
            val_ids = ckpt.get("val_ids", fold_info["val_ids"])

            df_pred = inference_on_validation(
                fold=fold,
                val_ids=val_ids,
                matched_df=matched_df,
                score_cols=score_cols,
                model_path=mp,
                model=model,
                device=device,
                args=args,
            )

            fold_pred_path = os.path.join(fold_dir, f"val_predictions_fold_{fold}_inference.csv")
            df_pred.to_csv(fold_pred_path, index=False, encoding="utf-8-sig")
            print(f"[Saved] {fold_pred_path}")
            all_pred_dfs.append(df_pred)

        if len(all_pred_dfs) == 0:
            raise FileNotFoundError(f"No fold predictions were generated from {checkpoint_dir}")

        all_pred_df = pd.concat(all_pred_dfs, axis=0, ignore_index=True)
        all_pred_path = os.path.join(checkpoint_dir, "all_predictions_5fold_inference.csv")
        all_pred_df.to_csv(all_pred_path, index=False, encoding="utf-8-sig")

        out_path = os.path.join(checkpoint_dir, "result.csv")
        all_pred_df.to_csv(out_path, index=False, encoding="utf-8-sig")

        metrics = summarize_cv_predictions(all_pred_df, score_cols)
        print(f"\nSaved: {out_path}")
        print(f"Saved: {all_pred_path}")
        print(
            f"Total: {len(all_pred_df)} | MAE: {metrics['mae']:.4f} | "
            f"RMSE: {metrics['rmse']:.4f} | MacroCorr: {metrics['macro_corr']:.4f} | "
            f"MacroStdRatio: {metrics['macro_std_ratio']:.4f}"
        )
        return

    mp = os.path.join(checkpoint_dir, "best_model.pt")
    if not os.path.exists(mp):
        mp = os.path.join(checkpoint_dir, "best_model_fold_1.pt")
    if not os.path.exists(mp):
        raise FileNotFoundError(
            f"No model file found in {checkpoint_dir}. Expected best_model.pt or best_model_fold_1.pt."
        )

    ckpt = torch.load(mp, map_location=device)
    cfg = ckpt.get("config", {})
    num_rois = cfg.get("num_rois", None)
    if num_rois is None:
        first_conn = load_connectome_matrix(matched_df.iloc[0]["connectome_path"])
        num_rois = first_conn.shape[0]

    model = ConnectomeSubdomainNet(
        num_rois=num_rois,
        num_tasks=cfg.get("num_tasks", args.num_tasks),
        embed_dim=cfg.get("embed_dim", args.embed_dim),
        depth=cfg.get("depth", args.depth),
        num_heads=cfg.get("num_heads", args.num_heads),
        ff_mult=cfg.get("ff_mult", args.ff_mult),
        dropout=cfg.get("dropout", args.dropout),
        lap_pe_dim=cfg.get("lap_pe_dim", args.lap_pe_dim),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    val_ids = ckpt.get("val_ids", None)
    if val_ids is None:
        val_ids = matched_df["sample_id"].tolist()

    print(f"\nSingle-model inference on {len(val_ids)} samples")
    df_pred = inference_on_validation(
        fold=1,
        val_ids=val_ids,
        matched_df=matched_df,
        score_cols=score_cols,
        model_path=mp,
        model=model,
        device=device,
        args=args,
    )

    out_path = os.path.join(checkpoint_dir, "result.csv")
    df_pred.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\nSaved: {out_path}")
    print(f"Total: {len(df_pred)}; Avg MAE: {df_pred['MAE'].mean():.4f}; Avg MSE: {df_pred['MSE'].mean():.4f}")


# =========================
# Training
# =========================
def train_one_fold(
    fold: int,
    train_ids: List[str],
    val_ids: List[str],
    matched_df: pd.DataFrame,
    score_cols: List[str],
    model: nn.Module,
    device: torch.device,
    args,
) -> Dict[str, Any]:
    train_target_stats = build_train_target_stats(
        matched_df=matched_df,
        train_ids=train_ids,
        score_cols=score_cols,
        high_quantile=args.high_quantile,
        min_std=args.label_min_std,
        sample_weight_alpha=args.sample_weight_alpha,
        sample_weight_pos_alpha=args.sample_weight_pos_alpha,
        sample_weight_high_alpha=args.sample_weight_high_alpha,
        sample_weight_cap=args.sample_weight_cap,
        enable_lds=args.enable_lds,
        lds_kernel=args.lds_kernel,
        lds_ks=args.lds_ks,
        lds_sigma=args.lds_sigma,
        lds_bin_width=args.lds_bin_width,
        lds_reweight=args.lds_reweight,
        lds_task_weight_cap=args.max_task_weight,
        lds_use_scalar_sample_weights=args.lds_use_scalar_sample_weights,
        lds_agg=args.lds_agg,
    )
    sample_weight_map = train_target_stats["sample_weight_map"]

    train_dataset = ConnectomeDataset(
        matched_df=matched_df,
        sample_ids=train_ids,
        score_cols=score_cols,
        sample_weight_map=sample_weight_map,
        lap_pe_dim=args.lap_pe_dim,
        preload=args.preload_connectomes,
    )
    train_eval_dataset = ConnectomeDataset(
        matched_df=matched_df,
        sample_ids=train_ids,
        score_cols=score_cols,
        sample_weight_map=None,
        lap_pe_dim=args.lap_pe_dim,
        preload=args.preload_connectomes,
    )
    val_dataset = ConnectomeDataset(
        matched_df=matched_df,
        sample_ids=val_ids,
        score_cols=score_cols,
        sample_weight_map=None,
        lap_pe_dim=args.lap_pe_dim,
        preload=args.preload_connectomes,
    )

    sampler = None
    if args.enable_task_sampler:
        sampler_weight_map = train_target_stats.get("sampler_weight_map", sample_weight_map)
        sampler_w = torch.tensor([sampler_weight_map[sid] for sid in train_dataset.sample_ids], dtype=torch.double)
        sampler = WeightedRandomSampler(weights=sampler_w, num_samples=len(train_dataset), replacement=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False if sampler is not None else True,
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    train_eval_loader = DataLoader(
        train_eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=max(args.lr * 0.05, 1e-6))
    loss_fn = AntiCollapseLoss(args)

    early = EarlyStopping(
        patience=args.early_patience,
        min_delta=args.early_min_delta,
        warmup_epochs=args.early_warmup,
    )

    best_selection = float("inf")
    best_val_mae = float("inf")
    best_val_loss_proxy = float("inf")
    best_epoch = 0
    best_model_path = None
    history_rows: List[Dict[str, Any]] = []

    for epoch in range(args.epochs):
        model.train()
        train_loss_meter = 0.0
        point_loss_meter = 0.0
        ccc_loss_meter = 0.0
        pearson_loss_meter = 0.0
        range_loss_meter = 0.0
        variance_loss_meter = 0.0
        ranking_loss_meter = 0.0
        tail_loss_meter = 0.0
        rarity_meter = 0.0
        legacy_rarity_meter = 0.0
        lds_rarity_meter = 0.0

        pbar = tqdm(train_loader, desc=f"Fold {fold} Epoch {epoch + 1}/{args.epochs} [Train]")
        for batch in pbar:
            A = batch["connectome"].to(device)
            lap_pe = batch["lap_pe"].to(device)
            labels = batch["label"].to(device)
            sw = torch.clamp(batch["sample_weight"].to(device), min=1e-6)

            optimizer.zero_grad(set_to_none=True)
            outputs_z = model(A, lap_pe)
            loss_out = loss_fn(outputs_z, labels, sw, train_target_stats)
            loss = loss_out["loss"]
            loss.backward()

            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            train_loss_meter += float(loss.item())
            point_loss_meter += float(loss_out["point_loss"].item())
            ccc_loss_meter += float(loss_out["ccc_loss"].item())
            pearson_loss_meter += float(loss_out["pearson_loss"].item())
            range_loss_meter += float(loss_out.get("range_loss", torch.tensor(0.0)).item())
            variance_loss_meter += float(loss_out.get("variance_loss", torch.tensor(0.0)).item())
            ranking_loss_meter += float(loss_out.get("ranking_loss", torch.tensor(0.0)).item())
            tail_loss_meter += float(loss_out.get("tail_loss", torch.tensor(0.0)).item())
            rarity_meter += float(loss_out["rarity_mean"].item())
            legacy_rarity_meter += float(loss_out.get("legacy_rarity_mean", loss_out["rarity_mean"]).item())
            lds_rarity_meter += float(loss_out.get("lds_rarity_mean", loss_out["rarity_mean"]).item())

            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "point": f"{loss_out['point_loss'].item():.4f}",
                    "var": f"{loss_out.get('variance_loss', torch.tensor(0.0)).item():.4f}",
                    "tail": f"{loss_out.get('tail_loss', torch.tensor(0.0)).item():.4f}",
                }
            )

        scheduler.step()

        num_batches = max(1, len(train_loader))
        epoch_train_loss = train_loss_meter / num_batches
        epoch_point_loss = point_loss_meter / num_batches
        epoch_ccc_loss = ccc_loss_meter / num_batches
        epoch_pearson_loss = pearson_loss_meter / num_batches
        epoch_range_loss = range_loss_meter / num_batches
        epoch_variance_loss = variance_loss_meter / num_batches
        epoch_ranking_loss = ranking_loss_meter / num_batches
        epoch_tail_loss = tail_loss_meter / num_batches
        epoch_rarity = rarity_meter / num_batches
        epoch_legacy_rarity = legacy_rarity_meter / num_batches
        epoch_lds_rarity = lds_rarity_meter / num_batches

        train_metrics = evaluate_loader_exact(
            model=model,
            loader=train_eval_loader,
            score_cols=score_cols,
            train_stats=train_target_stats,
            device=device,
            selection_corr_weight=args.selection_corr_weight,
            include_rows=False,
            clip_to_valid_range=args.range_clip_eval,
            args=args,
            epoch_idx=epoch,
        )
        val_metrics = evaluate_loader_exact(
            model=model,
            loader=val_loader,
            score_cols=score_cols,
            train_stats=train_target_stats,
            device=device,
            selection_corr_weight=args.selection_corr_weight,
            include_rows=False,
            clip_to_valid_range=args.range_clip_eval,
            args=args,
            epoch_idx=epoch,
        )

        train_selection = train_metrics["selection_metric"]
        val_selection = val_metrics["selection_metric"]

        history_rows.append(
            {
                "epoch": epoch + 1,
                "train_loss": epoch_train_loss,
                "train_point_loss": epoch_point_loss,
                "train_ccc_loss": epoch_ccc_loss,
                "train_pearson_loss": epoch_pearson_loss,
                "train_range_loss": epoch_range_loss,
                "train_variance_loss": epoch_variance_loss,
                "train_ranking_loss": epoch_ranking_loss,
                "train_tail_loss": epoch_tail_loss,
                "rarity_mean": epoch_rarity,
                "legacy_rarity_mean": epoch_legacy_rarity,
                "lds_rarity_mean": epoch_lds_rarity,
                "train_eval_mae": train_metrics["mae"],
                "train_eval_rmse": train_metrics["rmse"],
                "train_eval_macro_corr": train_metrics["macro_corr"],
                "train_eval_macro_std_ratio": train_metrics["macro_std_ratio"],
                "train_eval_tail_mae_norm": train_metrics.get("tail_mae_norm", 0.0),
                "train_eval_tail_under_ratio": train_metrics.get("tail_under_ratio", 1.0),
                "train_eval_min_task_std_ratio": train_metrics.get("min_task_std_ratio", 0.0),
                "train_selection_metric": train_selection,
                "train_selection_eligible": int(bool(train_metrics.get("selection_eligible", True))),
                "val_eval_mae": val_metrics["mae"],
                "val_eval_rmse": val_metrics["rmse"],
                "val_eval_macro_corr": val_metrics["macro_corr"],
                "val_eval_macro_std_ratio": val_metrics["macro_std_ratio"],
                "val_eval_tail_mae_norm": val_metrics.get("tail_mae_norm", 0.0),
                "val_eval_tail_under_ratio": val_metrics.get("tail_under_ratio", 1.0),
                "val_eval_min_task_std_ratio": val_metrics.get("min_task_std_ratio", 0.0),
                "val_selection_metric": val_selection,
                "val_selection_eligible": int(bool(val_metrics.get("selection_eligible", True))),
                "val_selection_distribution_penalty": val_metrics.get("distribution_penalty", 0.0),
                "lr": optimizer.param_groups[0]["lr"],
            }
        )

        print(
            f"Fold {fold} Epoch {epoch + 1}/{args.epochs} - "
            f"TrainLoss: {epoch_train_loss:.4f} (Point {epoch_point_loss:.4f}, CCC {epoch_ccc_loss:.4f}, P {epoch_pearson_loss:.4f}, Range {epoch_range_loss:.4f}, Var {epoch_variance_loss:.4f}, Rank {epoch_ranking_loss:.4f}, Tail {epoch_tail_loss:.4f}) | "
            f"TrainEval MAE: {train_metrics['mae']:.4f}, Corr: {train_metrics['macro_corr']:.4f}, Pred/True STD: {train_metrics['macro_std_ratio']:.4f}, TailUnder: {train_metrics.get('tail_under_ratio', 1.0):.4f} | "
            f"ValEval MAE: {val_metrics['mae']:.4f}, Corr: {val_metrics['macro_corr']:.4f}, Pred/True STD: {val_metrics['macro_std_ratio']:.4f}, TailUnder: {val_metrics.get('tail_under_ratio', 1.0):.4f}, "
            f"Eligible: {int(bool(val_metrics.get('selection_eligible', True)))}, Select: {val_selection:.4f}"
        )

        if val_selection < best_selection:
            best_selection = val_selection
            best_val_mae = val_metrics["mae"]
            best_val_loss_proxy = epoch_train_loss
            best_epoch = epoch + 1

            if args.save_dir:
                os.makedirs(args.save_dir, exist_ok=True)
                best_model_path = os.path.join(args.save_dir, f"best_model_fold_{fold}.pt")
                torch.save(
                    {
                        "fold": fold,
                        "epoch": best_epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_selection_metric": best_selection,
                        "val_mae": best_val_mae,
                        "train_loss_proxy": best_val_loss_proxy,
                        "train_ids": train_ids,
                        "val_ids": val_ids,
                        "train_target_stats": train_target_stats,
                        "score_cols": score_cols,
                        "config": {
                            "num_rois": train_dataset.num_rois,
                            "embed_dim": args.embed_dim,
                            "num_tasks": args.num_tasks,
                            "depth": args.depth,
                            "num_heads": args.num_heads,
                            "ff_mult": args.ff_mult,
                            "dropout": args.dropout,
                            "lap_pe_dim": args.lap_pe_dim,
                            "loss_huber_weight": args.loss_huber_weight,
                            "loss_mae_weight": args.loss_mae_weight,
                            "loss_ccc_weight": args.loss_ccc_weight,
                            "loss_pearson_weight": args.loss_pearson_weight,
                            "loss_variance_weight": args.loss_variance_weight,
                            "loss_ranking_weight": args.loss_ranking_weight,
                            "loss_tail_weight": args.loss_tail_weight,
                            "rank_pairs_per_task": args.rank_pairs_per_task,
                            "rank_pred_margin": args.rank_pred_margin,
                            "rank_target_margin": args.rank_target_margin,
                            "tail_under_weight": args.tail_under_weight,
                            "tail_over_weight": args.tail_over_weight,
                            "zero_over_weight": args.zero_over_weight,
                            "range_penalty_weight": args.range_penalty_weight,
                            "range_penalty_power": args.range_penalty_power,
                            "range_margin": args.range_margin,
                            "range_clip_eval": args.range_clip_eval,
                            "loss_pos_alpha": args.loss_pos_alpha,
                            "loss_high_alpha": args.loss_high_alpha,
                            "max_task_weight": args.max_task_weight,
                            "high_quantile": args.high_quantile,
                            "enable_task_sampler": args.enable_task_sampler,
                            "enable_lds": args.enable_lds,
                            "lds_kernel": args.lds_kernel,
                            "lds_ks": args.lds_ks,
                            "lds_sigma": args.lds_sigma,
                            "lds_bin_width": args.lds_bin_width,
                            "lds_reweight": args.lds_reweight,
                            "lds_agg": args.lds_agg,
                            "lds_use_scalar_sample_weights": args.lds_use_scalar_sample_weights,
                            "combine_lds_with_legacy_rarity": args.combine_lds_with_legacy_rarity,
                            "selection_corr_weight": args.selection_corr_weight,
                            "selection_std_weight": args.selection_std_weight,
                            "selection_tail_mae_weight": args.selection_tail_mae_weight,
                            "selection_tail_under_weight": args.selection_tail_under_weight,
                            "selection_min_macro_std_ratio": args.selection_min_macro_std_ratio,
                            "selection_min_task_std_ratio": args.selection_min_task_std_ratio,
                            "selection_min_macro_corr": args.selection_min_macro_corr,
                            "selection_min_tail_under_ratio": args.selection_min_tail_under_ratio,
                            "selection_warmup_epochs": args.selection_warmup_epochs,
                            "selection_ineligible_penalty": args.selection_ineligible_penalty,
                            "input": "subject_connectome",
                            "label_transform": "zscore_by_train_set",
                            "loss": "smoothl1_z + raw_mae/std + ccc + pearson + variance + ranking + tail_asym",
                            "selection": "mae+corr+std_ratio+tail_mae+tail_under with warmup and hard gates",
                            "max_month_gap": args.max_month_gap,
                            "duplicate_policy": args.duplicate_policy,
                        },
                    },
                    best_model_path,
                )

        if early.step(val_selection, epoch_idx=epoch):
            print(f"[EarlyStopping] Stop at epoch {epoch + 1}. Best selection={early.best:.4f}")
            break

    history_df = pd.DataFrame(history_rows)
    if args.save_dir and len(history_df) > 0:
        history_df.to_csv(os.path.join(args.save_dir, f"training_history_fold_{fold}.csv"), index=False, encoding="utf-8-sig")
        loss_plot_path = save_fold_loss_plot(history_df=history_df, fold=fold, save_dir=args.save_dir)
        if loss_plot_path is not None:
            print(f"[Saved] {loss_plot_path}")

    return {
        "best_selection_metric": best_selection,
        "best_val_mae": best_val_mae,
        "best_epoch": best_epoch,
        "final_train_loss": epoch_train_loss,
        "best_model_path": best_model_path,
        "train_target_stats": train_target_stats,
    }


# =========================
# Main
# =========================

def main():
    chosen_font = configure_matplotlib_for_chinese()
    if chosen_font is not None:
        print(f"[Info] Matplotlib Chinese font: {chosen_font}")
    else:
        print("[Warn] No Chinese-capable font found. Chinese glyphs may be missing in plots.")

    parser = argparse.ArgumentParser(
        description="Anti-mean-collapse training for subject-level connectome -> 5 UPDRS subdomains (multitask-balanced CV + range-aware outputs)"
    )

    parser.add_argument("--data_base_dir", type=str, default="./data/connectomes")
    parser.add_argument("--csv_path", type=str, default="./data/UPDRS_all.csv")
    parser.add_argument("--save_dir", type=str, default="./outputs/stage1")
    parser.add_argument("--device", type=str, default="cuda:0")

    # Matching
    parser.add_argument("--max_month_gap", type=int, default=0)
    parser.add_argument(
        "--duplicate_policy",
        type=str,
        default="drop",
        choices=["drop", "mean", "first", "last", "max_total"],
    )
    parser.add_argument("--max_subjects_per_dir", type=int, default=None)

    # Model
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--num_tasks", type=int, default=5)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--ff_mult", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lap_pe_dim", type=int, default=16)

    # Train
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--preload_connectomes", action="store_true")

    # CV split
    parser.add_argument("--cv_folds", type=int, default=5)
    parser.add_argument("--n_total_bins", type=int, default=5)
    parser.set_defaults(save_distribution_plots=True)
    parser.add_argument("--save_distribution_plots", dest="save_distribution_plots", action="store_true")
    parser.add_argument("--disable_distribution_plots", dest="save_distribution_plots", action="store_false")


    # Anti-collapse label stats / weighting
    parser.add_argument("--label_min_std", type=float, default=1e-3)
    parser.add_argument("--high_quantile", type=float, default=0.75)
    parser.add_argument("--sample_weight_alpha", type=float, default=0.35)
    parser.add_argument("--sample_weight_pos_alpha", type=float, default=0.60)
    parser.add_argument("--sample_weight_high_alpha", type=float, default=0.90)
    parser.add_argument("--sample_weight_cap", type=float, default=5.0)
    parser.add_argument("--enable_task_sampler", action="store_true")

    # LDS (official DIR-style label distribution smoothing)
    parser.set_defaults(enable_lds=True)
    parser.add_argument("--enable_lds", dest="enable_lds", action="store_true")
    parser.add_argument("--disable_lds", dest="enable_lds", action="store_false")
    parser.add_argument("--lds_kernel", type=str, default="gaussian", choices=["gaussian", "triang", "triangle", "triangular", "laplace", "laplacian"])
    parser.add_argument("--lds_ks", type=int, default=5)
    parser.add_argument("--lds_sigma", type=float, default=2.0)
    parser.add_argument("--lds_bin_width", type=float, default=1.0)
    parser.add_argument("--lds_reweight", type=str, default="inverse", choices=["none", "inverse", "sqrt_inv"])
    parser.add_argument("--lds_agg", type=str, default="mean", choices=["mean", "gmean", "max"])
    parser.add_argument("--lds_use_scalar_sample_weights", action="store_true")
    parser.add_argument("--combine_lds_with_legacy_rarity", action="store_true")

    # Loss
    parser.add_argument("--huber_beta", type=float, default=0.5)
    parser.add_argument("--range_penalty_weight", type=float, default=0.35)
    parser.add_argument("--range_penalty_power", type=float, default=2.0)
    parser.add_argument("--range_margin", type=float, default=0.0)
    parser.set_defaults(range_clip_eval=True)
    parser.add_argument("--range_clip_eval", dest="range_clip_eval", action="store_true")
    parser.add_argument("--disable_range_clip_eval", dest="range_clip_eval", action="store_false")
    parser.add_argument("--loss_huber_weight", type=float, default=0.55)
    parser.add_argument("--loss_mae_weight", type=float, default=0.45)
    parser.add_argument("--loss_ccc_weight", type=float, default=0.25)
    parser.add_argument("--loss_pearson_weight", type=float, default=0.10)
    parser.add_argument("--loss_variance_weight", type=float, default=0.45)
    parser.add_argument("--loss_ranking_weight", type=float, default=0.20)
    parser.add_argument("--loss_tail_weight", type=float, default=0.35)
    parser.add_argument("--rank_pairs_per_task", type=int, default=64)
    parser.add_argument("--rank_pred_margin", type=float, default=0.25)
    parser.add_argument("--rank_target_margin", type=float, default=0.50)
    parser.add_argument("--tail_under_weight", type=float, default=2.25)
    parser.add_argument("--tail_over_weight", type=float, default=1.00)
    parser.add_argument("--zero_over_weight", type=float, default=0.75)
    parser.add_argument("--loss_pos_alpha", type=float, default=0.60)
    parser.add_argument("--loss_high_alpha", type=float, default=1.20)
    parser.add_argument("--max_task_weight", type=float, default=6.0)

    # Model selection / early stopping
    parser.add_argument("--selection_corr_weight", type=float, default=0.30)
    parser.add_argument("--selection_std_weight", type=float, default=0.55)
    parser.add_argument("--selection_tail_mae_weight", type=float, default=0.35)
    parser.add_argument("--selection_tail_under_weight", type=float, default=0.40)
    parser.add_argument("--selection_min_macro_std_ratio", type=float, default=0.78)
    parser.add_argument("--selection_min_task_std_ratio", type=float, default=0.55)
    parser.add_argument("--selection_min_macro_corr", type=float, default=0.10)
    parser.add_argument("--selection_min_tail_under_ratio", type=float, default=0.65)
    parser.add_argument("--selection_ineligible_penalty", type=float, default=1e6)
    parser.add_argument("--selection_warmup_epochs", type=int, default=12)
    parser.add_argument("--early_patience", type=int, default=20)
    parser.add_argument("--early_min_delta", type=float, default=1e-4)
    parser.add_argument("--early_warmup", type=int, default=10)

    # Inference
    parser.add_argument("--inference_only", action="store_true")
    parser.add_argument("--checkpoint_dir", type=str, default="./outputs/stage1")

    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    matched_df, debug_df, score_cols = build_matched_sample_table(
        data_base_dir=args.data_base_dir,
        csv_path=args.csv_path,
        max_month_gap=args.max_month_gap,
        duplicate_policy=args.duplicate_policy,
        max_subjects_per_dir=args.max_subjects_per_dir,
        seed=args.seed,
    )

    os.makedirs(args.save_dir, exist_ok=True)
    matched_df.to_csv(os.path.join(args.save_dir, "matched_samples.csv"), index=False, encoding="utf-8-sig")
    debug_df.to_csv(os.path.join(args.save_dir, "matching_debug.csv"), index=False, encoding="utf-8-sig")

    if args.save_distribution_plots:
        overall_plot_dir = os.path.join(args.save_dir, "distribution_plots", "overall")
        save_overall_distribution_artifacts(
            matched_df=matched_df,
            score_cols=score_cols,
            out_dir=overall_plot_dir,
        )
        print(f"[Saved] overall distribution plots -> {overall_plot_dir}")

    if args.inference_only:
        run_inference_mode(
            matched_df=matched_df,
            score_cols=score_cols,
            checkpoint_dir=args.checkpoint_dir,
            device=device,
            args=args,
        )
        return

    print("\n" + "=" * 60)
    print("5-Fold CV split (PATNO-level multitask balanced assignment)")
    print("=" * 60)
    print(f"Matched samples: {len(matched_df)}")
    print(f"Unique PATNO: {matched_df['PATNO'].astype(str).nunique()}")
    print("Label distribution (unique PATNO):")
    print(matched_df.groupby("label")["PATNO"].nunique())

    first_conn = load_connectome_matrix(matched_df.iloc[0]["connectome_path"])
    num_rois = first_conn.shape[0]

    patno_strat_df, fold_infos = build_joint_stratified_kfold_splits(
        matched_df=matched_df,
        score_cols=score_cols,
        n_splits=args.cv_folds,
        n_total_bins=args.n_total_bins,
        seed=args.seed,
    )

    patno_strat_path = os.path.join(args.save_dir, "patno_multitask_balance_table.csv")
    patno_strat_df.to_csv(patno_strat_path, index=False, encoding="utf-8-sig")
    print(f"[Saved] {patno_strat_path}")

    fold_assign_rows = []
    for fold_info in fold_infos:
        for patno in fold_info["val_patnos"]:
            fold_assign_rows.append({"PATNO": str(patno), "fold": int(fold_info["fold"])})
    fold_assign_df = pd.DataFrame(fold_assign_rows)
    fold_assign_path = os.path.join(args.save_dir, "patno_fold_assignments.csv")
    fold_assign_df.to_csv(fold_assign_path, index=False, encoding="utf-8-sig")
    print(f"[Saved] {fold_assign_path}")

    print("\nAssigned fold counts (PATNO-level):")
    print(patno_strat_df["assigned_fold"].value_counts().sort_index())

    base_save_dir = args.save_dir
    cv_rows = []
    all_pred_dfs = []

    for fold_info in fold_infos:
        fold = fold_info["fold"]
        val_ids = fold_info["val_ids"]
        # Use only the participant-disjoint training set from the outer CV split.
        # The original all-data/validation-label exclusion routine was removed:
        # it could include held-out participants in the training set.
        train_ids = fold_info["train_ids"]
        validate_fold_separation(
            matched_df=matched_df,
            train_ids=train_ids,
            val_ids=val_ids,
            fold=int(fold),
        )
        print_one_fold_summary(
            fold=fold,
            matched_df=matched_df,
            score_cols=score_cols,
            train_ids=train_ids,
            val_ids=val_ids,
        )

        fold_save_dir = os.path.join(base_save_dir, f"fold_{fold}")
        if args.save_distribution_plots:
            fold_plot_dir = os.path.join(fold_save_dir, "distribution_plots")
            save_fold_distribution_artifacts(
                matched_df=matched_df,
                score_cols=score_cols,
                train_ids=train_ids,
                val_ids=val_ids,
                fold=fold,
                out_dir=fold_plot_dir,
            )
            print(f"[Saved] fold {fold} train/val distribution plots -> {fold_plot_dir}")
        os.makedirs(fold_save_dir, exist_ok=True)
        args.save_dir = fold_save_dir

        model = ConnectomeSubdomainNet(
            num_rois=num_rois,
            num_tasks=args.num_tasks,
            embed_dim=args.embed_dim,
            depth=args.depth,
            num_heads=args.num_heads,
            ff_mult=args.ff_mult,
            dropout=args.dropout,
            lap_pe_dim=args.lap_pe_dim,
        ).to(device)

        res = train_one_fold(
            fold=fold,
            train_ids=train_ids,
            val_ids=val_ids,
            matched_df=matched_df,
            score_cols=score_cols,
            model=model,
            device=device,
            args=args,
        )

        actual_train_patno_count = matched_df[
            matched_df["sample_id"].astype(str).isin(set(map(str, train_ids)))
        ]["PATNO"].astype(str).nunique()

        cv_rows.append(
            {
                "fold": fold,
                "train_patno_count": int(actual_train_patno_count),
                "val_patno_count": len(set(map(str, fold_info["val_patnos"]))),
                "train_sample_count": len(train_ids),
                "val_sample_count": len(val_ids),
                "best_epoch": res["best_epoch"],
                "best_val_mae": res["best_val_mae"],
                "best_selection_metric": res["best_selection_metric"],
                "final_train_loss": res["final_train_loss"],
            }
        )

        print(
            f"[Fold {fold}] Best Val MAE: {res['best_val_mae']:.4f} "
            f"(Epoch {res['best_epoch']})"
        )

        if res["best_model_path"] and os.path.exists(res["best_model_path"]):
            ckpt_fold_path = os.path.join(fold_save_dir, "best_model.pt")
            try:
                if os.path.abspath(res["best_model_path"]) != os.path.abspath(ckpt_fold_path):
                    shutil.copyfile(res["best_model_path"], ckpt_fold_path)
            except Exception as e:
                print(f"[Warn] Failed to save {ckpt_fold_path}: {e}")

            root_best_model_dir = os.path.join(base_save_dir, "best_models_5fold")
            root_pred_dir = os.path.join(base_save_dir, "val_predictions_best_models")
            root_hidden_dir = os.path.join(base_save_dir, "val_hidden_vectors_best_models")
            os.makedirs(root_best_model_dir, exist_ok=True)
            os.makedirs(root_pred_dir, exist_ok=True)
            os.makedirs(root_hidden_dir, exist_ok=True)

            root_ckpt_path = os.path.join(root_best_model_dir, f"best_model_fold_{fold}.pt")
            try:
                if os.path.abspath(res["best_model_path"]) != os.path.abspath(root_ckpt_path):
                    shutil.copyfile(res["best_model_path"], root_ckpt_path)
            except Exception as e:
                print(f"[Warn] Failed to save {root_ckpt_path}: {e}")

            df_pred, hidden_payload = inference_on_validation_with_hidden(
                fold=fold,
                val_ids=val_ids,
                matched_df=matched_df,
                score_cols=score_cols,
                model_path=res["best_model_path"],
                model=model,
                device=device,
                args=args,
            )

            pred_path = os.path.join(fold_save_dir, f"val_predictions_fold_{fold}.csv")
            df_pred.to_csv(pred_path, index=False, encoding="utf-8-sig")
            print(f"[Saved] {pred_path}")

            root_pred_path = os.path.join(root_pred_dir, f"val_predictions_fold_{fold}.csv")
            df_pred.to_csv(root_pred_path, index=False, encoding="utf-8-sig")
            print(f"[Saved] {root_pred_path}")

            hidden_path = os.path.join(fold_save_dir, f"val_hidden_vectors_fold_{fold}.pt")
            torch.save(hidden_payload, hidden_path)
            print(f"[Saved] {hidden_path}")

            root_hidden_path = os.path.join(root_hidden_dir, f"val_hidden_vectors_fold_{fold}.pt")
            torch.save(hidden_payload, root_hidden_path)
            print(f"[Saved] {root_hidden_path}")

            all_pred_dfs.append(df_pred)

    args.save_dir = base_save_dir

    cv_df = pd.DataFrame(cv_rows)
    cv_path = os.path.join(base_save_dir, "cv_metrics_5fold.csv")
    cv_df.to_csv(cv_path, index=False, encoding="utf-8-sig")
    print(f"\n[Saved] {cv_path}")
    print("\n5-Fold summary:")
    print(cv_df)

    if len(all_pred_dfs) > 0:
        all_pred_df = pd.concat(all_pred_dfs, axis=0, ignore_index=True)

        dup_count = all_pred_df["sample_id"].astype(str).duplicated().sum()
        if dup_count > 0:
            print(f"[Warn] Duplicated sample_id detected in merged CV predictions: {dup_count}")

        all_pred_path = os.path.join(base_save_dir, "all_predictions_5fold.csv")
        all_pred_df.to_csv(all_pred_path, index=False, encoding="utf-8-sig")
        print(f"[Saved] {all_pred_path}")

        out_path = os.path.join(base_save_dir, "result.csv")
        all_pred_df.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"[Saved] {out_path}")

        overall_metrics = summarize_cv_predictions(all_pred_df, score_cols)
        overall_rows = {
            "mae": overall_metrics["mae"],
            "rmse": overall_metrics["rmse"],
            "macro_corr": overall_metrics["macro_corr"],
            "macro_std_ratio": overall_metrics["macro_std_ratio"],
        }
        overall_path = os.path.join(base_save_dir, "overall_metrics_5fold.json")
        with open(overall_path, "w", encoding="utf-8") as f:
            json.dump(overall_rows, f, indent=2, ensure_ascii=False)
        print(f"[Saved] {overall_path}")

        print("\nOverall 5-Fold CV metrics:")
        print(
            f"MAE={overall_metrics['mae']:.4f}, "
            f"RMSE={overall_metrics['rmse']:.4f}, "
            f"MacroCorr={overall_metrics['macro_corr']:.4f}, "
            f"MacroStdRatio={overall_metrics['macro_std_ratio']:.4f}"
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
