"""Fast tests for the model-family comparison harness.

Guarantees pinned here:

* preprocessing (standardisation) is fitted on training rows only, so validation
  statistics cannot influence the linear model;
* inference is deterministic and keyed by feature *name*, not column position;
* probability output is validated rather than trusted;
* the paired bootstrap reports a real interval for the difference and stays
  silent (zero-width, no improvement) when the two prediction vectors are equal;
* the decision rule refuses to call a within-noise difference an improvement;
* the quality policy's numeric floor is reported separately from "best family",
  and a family can pass selection while failing the policy;
* a full run reports validation metrics only, never touches the test partition,
  and exports a gate-shaped ONNX graph whose predictions match the source model.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ML_ROOT = Path(__file__).resolve().parents[1]
if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))

pytest.importorskip("xgboost")

from sklearn.linear_model import LogisticRegression  # noqa: E402

from training import model_families  # noqa: E402
from training.quality_gate import QualityPolicy  # noqa: E402
from training.splits import file_sha256  # noqa: E402

MINI_FEATURES = [
    "duration",
    "avg_mmr",
    "game_mode",
    "lobby_type",
    "patch",
    "patch_version",
    "rad_pick_0",
    "rad_pick_1",
    "rad_pick_2",
    "rad_pick_3",
    "rad_pick_4",
    "dire_pick_0",
    "dire_pick_1",
    "dire_pick_2",
    "dire_pick_3",
    "dire_pick_4",
    "rad_has_anti_mage",
    "rad_has_axe",
    "dire_has_axe",
    "rad_has_shadow_fiend",
    "rad_has_phantom_lancer",
    "dire_has_puck",
    "dire_has_pudge",
]

HERO_IDS = (1, 2, 11, 12, 13, 14)


def mini_schema() -> dict:
    names = list(MINI_FEATURES)
    return {
        "schema_version": 1,
        "model": "mini",
        "feature_count": len(names),
        "feature_names": names,
        "categoricals": {"patch": ["7.40", "7.41"]},
        "defaults": {name: 0.0 for name in names} | {"patch": 1.0, "duration": 2163.0},
        "positive_class_index": 1,
        "input_name": "float_input",
        "label_output": "label",
        "probability_output": "probabilities",
        "source": {},
    }


def mini_frame(rows: int = 400, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rad = rng.choice(HERO_IDS[:3], size=(rows, 5))
    dire = rng.choice(HERO_IDS[3:], size=(rows, 5))
    frame = pd.DataFrame(
        {
            "match_id": np.arange(1, rows + 1),
            "start_time": 1_600_000_000 + np.arange(rows) * 60,
            "radiant_win": rng.integers(0, 2, size=rows),
            "duration": rng.integers(1200, 3400, size=rows),
            "avg_mmr": np.zeros(rows),
            "game_mode": np.full(rows, 22),
            "lobby_type": np.zeros(rows),
            "patch": np.where(np.arange(rows) < rows // 2, 0.0, 1.0),
            "patch_version": np.ones(rows),
        }
    )
    for index in range(5):
        frame[f"rad_pick_{index}"] = rad[:, index]
        frame[f"dire_pick_{index}"] = dire[:, index]
    for name in (
        "rad_has_anti_mage",
        "rad_has_axe",
        "dire_has_axe",
        "rad_has_shadow_fiend",
        "rad_has_phantom_lancer",
        "dire_has_puck",
        "dire_has_pudge",
    ):
        frame[name] = rng.integers(0, 2, size=rows).astype(float)
    return frame


def _write_split_config(path: Path, frame: pd.DataFrame, data_path: Path) -> None:
    rows = len(frame)
    train_max = float(frame["start_time"].iloc[int(rows * 0.5) - 1])
    val_max = float(frame["start_time"].iloc[int(rows * 0.75) - 1])
    train_mask = frame["start_time"] <= train_max
    val_mask = (frame["start_time"] > train_max) & (frame["start_time"] <= val_max)
    config = {
        "schema_version": 1,
        "dataset": {"path": str(data_path), "sha256": file_sha256(data_path)},
        "rule": {"fractions": {"train": 0.5, "validation": 0.25, "test": 0.25}},
        "partitions": {
            "train": {"start_time_max": train_max, "rows": int(train_mask.sum())},
            "validation": {"start_time_max": val_max, "rows": int(val_mask.sum())},
            "test": {"rows": int((~train_mask & ~val_mask).sum())},
        },
        "contamination": {"untouched": False, "reason": "test fixture"},
    }
    path.write_text(json.dumps(config), encoding="utf-8")


def test_standardisation_uses_training_rows_only():
    rng = np.random.default_rng(0)
    X_train = rng.normal(10.0, 3.0, size=(200, 4)).astype(np.float32)
    X_val = rng.normal(1000.0, 50.0, size=(50, 4)).astype(np.float32)
    y_train = (rng.random(200) > 0.5).astype(int)

    pipeline = model_families.fit_logistic(X_train, y_train, C=0.1, seed=42)
    scaler = pipeline.named_steps["scale"]

    np.testing.assert_allclose(scaler.mean_, X_train.mean(axis=0), rtol=1e-5)
    # The validation shift must not leak into the fitted statistics.
    assert not np.allclose(scaler.mean_, X_val.mean(axis=0), rtol=1e-3)
    assert isinstance(pipeline.named_steps["clf"], LogisticRegression)
    assert pipeline.named_steps["clf"].C == 0.1


def test_predictions_are_deterministic_and_valid():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(300, 6)).astype(np.float32)
    y = rng.integers(0, 2, size=300)
    model = model_families.fit_logistic(X, y, C=0.1, seed=7)

    first = model_families._predict(model, X)
    second = model_families._predict(model, X)

    np.testing.assert_array_equal(first, second)
    assert np.isfinite(first).all()
    assert first.min() >= 0.0 and first.max() <= 1.0
    assert first.shape == (300,)


def test_predict_rejects_invalid_probabilities():
    class OutOfRange:
        def predict_proba(self, matrix):  # noqa: ARG002
            return np.array([[0.2, 1.4], [0.5, 0.5]])

    class NotFinite:
        def predict_proba(self, matrix):  # noqa: ARG002
            return np.array([[0.5, np.nan], [0.5, 0.5]])

    class SingleColumn:
        def predict_proba(self, matrix):  # noqa: ARG002
            return np.array([[0.5], [0.5]])

    for model, message in (
        (OutOfRange(), "outside"),
        (NotFinite(), "non-finite"),
        (SingleColumn(), "two-class"),
    ):
        with pytest.raises(SystemExit, match=message):
            model_families._predict(model, np.zeros((2, 3), dtype=np.float32))


def test_paired_bootstrap_interval_and_null_case():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, size=3000)
    reference = np.full(3000, 0.5)
    candidate = np.clip(0.5 + 0.3 * (y - 0.5) + rng.normal(0, 0.05, 3000), 0, 1)

    better = model_families.paired_bootstrap(y, reference, candidate, replicates=400)
    assert better["auc_diff"]["ci95"][0] > 0
    assert better["brier_diff"]["ci95"][1] < 0
    assert better["auc_diff"]["fraction_positive"] == 1.0

    same = model_families.paired_bootstrap(y, reference, reference.copy(), replicates=200)
    assert same["auc_diff"]["ci95"] == [0.0, 0.0]
    assert same["auc_diff"]["fraction_positive"] == 0.0


def _family(name: str, auc: float, brier: float) -> dict:
    return {
        "name": name,
        "aggregate": {
            "roc_auc_mean": auc,
            "brier_score_mean": brier,
            "roc_auc_std": 0.0,
            "brier_score_std": 0.0,
        },
    }


def test_decision_requires_both_auc_interval_and_brier_margin():
    families = [
        _family("xgboost_combined", 0.5481, 0.2473),
        _family("logistic_l2", 0.5479, 0.2472),
    ]
    overlap = [
        {
            "candidate": "logistic_l2",
            "auc_diff": {"mean": 0.0002, "ci95": [-0.004, 0.004]},
            "brier_diff": {"mean": -0.0001, "ci95": [-0.001, 0.001]},
        }
    ]
    decision = model_families.decide(families, overlap)
    assert decision["material_improvement_over_reference"] is False
    # Within the AUC tolerance the lower Brier still names the leader, but the
    # report must not call that a material gain.
    assert decision["selected"] == "logistic_l2"

    separated = [
        {
            "candidate": "logistic_l2",
            "auc_diff": {"mean": 0.02, "ci95": [0.015, 0.025]},
            "brier_diff": {"mean": -0.01, "ci95": [-0.012, -0.008]},
        }
    ]
    better_families = [
        _family("xgboost_combined", 0.5481, 0.2473),
        _family("logistic_l2", 0.5681, 0.2410),
    ]
    assert (
        model_families.decide(better_families, separated)[
            "material_improvement_over_reference"
        ]
        is True
    )
    # A real AUC gain with a sub-margin Brier gain is still not material: both
    # conditions must hold, so a marginal calibration win cannot carry the vote.
    assert (
        model_families.decide(families, separated)["material_improvement_over_reference"]
        is False
    )


def test_decision_prefers_reference_when_it_leads_on_auc():
    families = [
        _family("xgboost_combined", 0.5600, 0.2460),
        _family("logistic_l2", 0.5400, 0.2480),
    ]
    comparisons = [
        {
            "candidate": "logistic_l2",
            "auc_diff": {"mean": -0.02, "ci95": [-0.03, -0.01]},
            "brier_diff": {"mean": 0.002, "ci95": [0.001, 0.003]},
        }
    ]
    decision = model_families.decide(families, comparisons)
    assert decision["selected"] == "xgboost_combined"
    assert decision["material_improvement_over_reference"] is False


def test_policy_floor_is_reported_separately_from_selection():
    policy = QualityPolicy()
    failing = {
        "rows": 10000,
        "roc_auc": 0.55,
        "brier_score": 0.2473,
        "calibration": {"ece": 0.02},
    }
    status = model_families.policy_status(failing, policy)
    assert status["passes_numeric_floor"] is False
    assert status["checks"]["roc_auc"]["pass"] is False
    assert status["checks"]["brier_score"]["pass"] is False
    assert status["checks"]["rows"]["pass"] is True
    assert status["evaluated_on"] == "validation"

    passing = dict(failing, roc_auc=0.62, brier_score=0.23)
    assert model_families.policy_status(passing, policy)["passes_numeric_floor"] is True


def test_run_scores_validation_only_and_exports_parity(tmp_path):
    frame = mini_frame()
    data_path = tmp_path / "mini.parquet"
    schema_path = tmp_path / "mini_schema.json"
    split_path = tmp_path / "split.json"
    report_path = tmp_path / "report.json"
    frame.to_parquet(data_path, index=False)
    schema_path.write_text(json.dumps(mini_schema()), encoding="utf-8")
    _write_split_config(split_path, frame, data_path)

    code = model_families.main(
        [
            "--data",
            str(data_path),
            "--schema",
            str(schema_path),
            "--split-config",
            str(split_path),
            "--report",
            str(report_path),
            "--model-dir",
            str(tmp_path / "families"),
            "--seeds",
            "42",
            "--bootstrap",
            "50",
            "--max-estimators",
            "10",
            "--hist-max-iter",
            "12",
            "--threads",
            "2",
            "--calibration-bins",
            "5",
        ]
    )

    assert code == 0
    report = json.loads(report_path.read_text(encoding="utf-8"))

    # The frozen-split contract and the test partition are explicit.
    assert report["split"]["test_scored_here"] is False
    assert report["protocol"]["selection_partition"] == "validation"
    assert [family["name"] for family in report["families"]] == list(
        model_families.FAMILY_NAMES
    )
    assert all("test" not in family for family in report["families"])
    for family in report["families"]:
        assert all("test" not in fit for fit in family["fits"])
    assert report["policy"]["promotion"] is False

    # Feature set is the combined ablation: match context and pick slots dropped.
    assert report["feature_set"]["feature_count"] == 7
    assert "duration" in report["feature_set"]["dropped_features"]
    assert "rad_pick_0" in report["feature_set"]["dropped_features"]

    # Every family is scored on the same rows with the same baselines.
    rows = {family["fits"][0]["validation"]["inference"]["metrics"]["rows"] for family in report["families"]}
    assert len(rows) == 1
    for family in report["families"]:
        baselines = family["fits"][0]["validation"]["inference"]["baselines"]
        assert set(baselines) == {"constant_train_prior", "constant_half"}
        assert set(family["fits"][0]["partial_draft"]) == {
            str(level) for level in model_families.PROGRESS_LEVELS
        }

    # Partial-draft stages shrink monotonically in information, not in rows.
    for family in report["families"]:
        fit = family["fits"][0]
        assert (
            fit["partial_draft"]["0"]["metrics"]["rows"]
            == fit["validation"]["inference"]["metrics"]["rows"]
        )

    # ONNX export must reproduce the source model, and stay off the bundled path.
    exported = {record["family"]: record for record in report["onnx_export"]}
    assert exported["logistic_l2"]["exported"] is True
    assert exported["logistic_l2"]["parity_ok"] is True
    assert exported["logistic_l2"]["input_type"] == "tensor(float)"
    assert "probabilities" in exported["logistic_l2"]["output_names"]
    assert Path(exported["logistic_l2"]["onnx"]).is_file()
    assert "ml/models/draft_model_xgboost_full" not in exported["logistic_l2"]["onnx"]
    for record in report["onnx_export"]:
        if record.get("exported"):
            assert record["parity_ok"] is True
            assert record["max_abs_diff"] < 1e-4
        else:
            assert record.get("error")


def test_frozen_split_rejects_a_foreign_dataset(tmp_path):
    frame = mini_frame(rows=60)
    data_path = tmp_path / "mini.parquet"
    split_path = tmp_path / "split.json"
    frame.to_parquet(data_path, index=False)
    _write_split_config(split_path, frame, data_path)

    # Rewriting the data invalidates the recorded checksum: the split must
    # refuse to apply rather than silently re-cut.
    frame.assign(radiant_win=1 - frame["radiant_win"]).to_parquet(data_path, index=False)
    schema_path = tmp_path / "mini_schema.json"
    schema_path.write_text(json.dumps(mini_schema()), encoding="utf-8")
    with pytest.raises(SystemExit, match="does not match the frozen split"):
        model_families.main(
            [
                "--data",
                str(data_path),
                "--schema",
                str(schema_path),
                "--split-config",
                str(split_path),
                "--report",
                str(tmp_path / "report.json"),
                "--seeds",
                "42",
                "--bootstrap",
                "10",
                "--no-export",
            ]
        )
