"""Small tests that require no private patient data."""
import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from model_connectome import ConnectomeSubdomainNet, preprocess_connectome
from train_stage1 import validate_fold_separation

ROOT = Path(__file__).resolve().parents[1]


def test_removed_function_not_present():
    for path in ROOT.glob("*.py"):
        names = {n.name for n in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
                 if isinstance(n, ast.FunctionDef)}
        assert "build_all_data_train_ids_with_fold_val_exclusion" not in names


def test_fold_separation_accepts_disjoint_subjects():
    df = pd.DataFrame({"sample_id": ["a1", "a2", "b1", "c1"],
                       "PATNO": ["a", "a", "b", "c"]})
    validate_fold_separation(df, ["a1", "a2", "b1"], ["c1"], 1)


def test_fold_separation_rejects_shared_subject():
    df = pd.DataFrame({"sample_id": ["a1", "a2", "b1"],
                       "PATNO": ["a", "a", "b"]})
    with pytest.raises(ValueError, match="subject leakage"):
        validate_fold_separation(df, ["a1", "b1"], ["a2"], 1)


def test_fold_separation_rejects_shared_id():
    df = pd.DataFrame({"sample_id": ["a1", "b1"], "PATNO": ["a", "b"]})
    with pytest.raises(ValueError, match="overlapping"):
        validate_fold_separation(df, ["a1", "b1"], ["b1"], 1)


def test_weighted_preprocessing_is_symmetric():
    x = torch.rand(2, 8, 8)
    a = preprocess_connectome(x)
    assert a.shape == (2, 8, 8)
    assert torch.allclose(a, a.transpose(-1,-2), atol=1e-6)
    assert torch.all(a >= 0)


def test_model_forward_backward_shapes():
    model = ConnectomeSubdomainNet(num_rois=8, num_tasks=5, embed_dim=16,
                                   depth=1, num_heads=4, lap_pe_dim=4, dropout=0.0)
    a = torch.rand(2, 8, 8)
    pe = torch.zeros(2, 8, 4)
    output = model(a, pe)
    assert output.shape == (2,5)
    output.mean().backward()
    assert model.task_queries.grad is not None
