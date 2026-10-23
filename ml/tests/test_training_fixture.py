"""End-to-end fixture test: split, train, export ONNX + schema + manifest.

Runs the real `train_full` pipeline on a tiny synthetic dataset so the whole
path (chronological split -> fit -> evaluate -> export) is exercised without
network access or the full processed parquet.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from training import train_full

pytest.importorskip("xgboost")
pytest.importorskip("onnxmltools")
pytest.importorskip("onnxruntime")


def _write_fixture(path) -> None:
    n = 120
    frame = pd.DataFrame(
        {
            "match_id": list(range(1000, 1000 + n)),
            "start_time": [1_700_000_000 + i * 60 for i in range(n)],
            "radiant_win": [(i * 7) % 2 for i in range(n)],
            "hero_a": [float(i % 5) for i in range(n)],
            "hero_b": [float((i * 3) % 7) for i in range(n)],
            "patch": pd.Categorical(["7.40" if i % 2 else "7.41" for i in range(n)]),
        }
    )
    frame.to_parquet(path, index=False)


def _run(tmp_path, extra_args=()):
    data = tmp_path / "features.parquet"
    _write_fixture(data)
    args = train_full.build_parser().parse_args(
        [
            "--data",
            str(data),
            "--model-dir",
            str(tmp_path / "models"),
            "--model-stem",
            "fixture_model",
            "--train-frac",
            "0.6",
            "--val-frac",
            "0.2",
            "--test-frac",
            "0.2",
            "--n-estimators",
            "8",
            "--seed",
            "1",
            *extra_args,
        ]
    )
    exit_code = train_full.run(args)
    assert exit_code == 0
    return tmp_path / "models"


def test_training_pipeline_exports_onnx_schema_and_manifest(tmp_path):
    model_dir = _run(tmp_path)

    manifest = json.loads((model_dir / "fixture_model_manifest.json").read_text())
    assert manifest["metrics"]["test"]["rows"] == 24
    assert manifest["metrics"]["validation"]["rows"] == 24
    assert manifest["split"]["used_start_time"] is True
    assert manifest["split"]["sizes"] == {"train": 72, "validation": 24, "test": 24}
    assert manifest["dataset"]["sha256"]
    assert manifest["feature_schema"]["feature_count"] == 3
    assert (model_dir / "fixture_model_manifest.json").exists()

    schema = json.loads((model_dir / "fixture_model_schema.json").read_text())
    assert schema["feature_count"] == manifest["feature_schema"]["feature_count"]
    assert "patch" in schema["categoricals"]

    import onnxruntime as ort

    session = ort.InferenceSession(
        str(model_dir / "fixture_model.onnx"), providers=["CPUExecutionProvider"]
    )
    shape = session.get_inputs()[0].shape
    assert shape[1] == manifest["feature_schema"]["feature_count"]
    outputs = session.run(
        None,
        {session.get_inputs()[0].name: np.zeros((2, shape[1]), dtype=np.float32)},
    )
    assert len(outputs) == 2


def test_skip_train_reuses_model_and_writes_manifest(tmp_path):
    model_dir = _run(tmp_path)
    before = json.loads((model_dir / "fixture_model_manifest.json").read_text())

    args = train_full.build_parser().parse_args(
        [
            "--data",
            str(tmp_path / "features.parquet"),
            "--model-dir",
            str(model_dir),
            "--model-stem",
            "fixture_model",
            "--skip-train",
        ]
    )
    assert train_full.run(args) == 0

    after = json.loads((model_dir / "fixture_model_manifest.json").read_text())
    assert after["best_iteration"] is None
    assert after["metrics"]["test"]["roc_auc"] == pytest.approx(
        before["metrics"]["test"]["roc_auc"]
    )
