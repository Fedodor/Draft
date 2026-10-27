"""Quality-gate behaviour: thresholds, failures, and promotion blocking.

The integration cases run the real training/export path on a tiny synthetic
parquet, then point the gate at the exported bundle. No full dataset and no
network access are involved.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from training import quality_gate as gate
from training import train_full

pytest.importorskip("xgboost")
pytest.importorskip("onnxmltools")
pytest.importorskip("onnxruntime")


def _write_fixture(path: Path, rows: int = 160) -> None:
    frame = pd.DataFrame(
        {
            "match_id": list(range(1_000, 1_000 + rows)),
            "start_time": [1_700_000_000 + index * 60 for index in range(rows)],
            "radiant_win": [index % 2 for index in range(rows)],
            "hero_a": [float(index % 2) + 0.001 * (index % 7) for index in range(rows)],
            "hero_b": [float((index * 3) % 5) for index in range(rows)],
            "patch": pd.Categorical(["7.40" if index % 2 else "7.41" for index in range(rows)]),
        }
    )
    frame.to_parquet(path, index=False)


def _train_fixture(tmp_path: Path) -> tuple[Path, Path]:
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
        ]
    )
    assert train_full.run(args) == 0
    return tmp_path / "models", data


def _lenient_policy() -> gate.QualityPolicy:
    return gate.QualityPolicy(
        min_roc_auc=0.0,
        max_brier=1.0,
        max_ece=1.0,
        min_test_rows=1,
        evaluation_mode="inference",
    )


def _run(models: Path, data: Path, **kwargs) -> gate.GateReport:
    return gate.run_gate(
        model_dir=models,
        model_stem="fixture_model",
        data_path=data,
        report_path=models / "fixture_model_quality_report.json",
        **kwargs,
    )


def test_gate_passes_a_valid_bundle_and_writes_a_report(tmp_path):
    models, data = _train_fixture(tmp_path)
    result = _run(models, data, policy=_lenient_policy())

    assert result.exit_code == 0
    assert result.report["evaluation"]["status"] == "completed"
    assert result.report["decision"]["quality_status"] == "passed"
    assert result.report["decision"]["promotion_eligible"] is True
    assert result.report["evaluation"]["modes"]["inference"]["metrics"]["rows"] == 32
    assert result.report["evaluation"]["split"]["match_id_overlaps"] == {
        "train_validation": 0,
        "train_test": 0,
        "validation_test": 0,
    }
    assert result.report["evaluation"]["dataset"]["fingerprint_match"] is True
    assert (
        result.report["checks"]
        and all("identifier" in check and "status" in check for check in result.report["checks"])
    )

    written = json.loads((models / "fixture_model_quality_report.json").read_text())
    assert written["decision"]["promotion_eligible"] is True
    assert written["policy"]["evaluation_mode"] == "inference"


def test_gate_fails_thresholds_without_an_evaluation_error(tmp_path):
    models, data = _train_fixture(tmp_path)
    strict = gate.QualityPolicy(
        min_roc_auc=1.01,
        max_brier=0.0,
        max_ece=0.0,
        min_test_rows=1,
        evaluation_mode="inference",
    )
    result = _run(models, data, policy=strict)

    assert result.exit_code == 1
    assert result.report["evaluation"]["status"] == "completed"
    assert result.report["decision"]["quality_status"] == "failed"
    assert result.report["decision"]["promotion_eligible"] is False
    failed = result.report["decision"]["failed_checks"]
    assert "policy.roc_auc" in failed
    assert "policy.brier_score" in failed
    assert "policy.calibration_ece" in failed


def test_dataset_fingerprint_mismatch_is_not_promotion_eligible(tmp_path):
    models, data = _train_fixture(tmp_path)
    tampered = tmp_path / "tampered.parquet"
    frame = pd.read_parquet(data)
    frame["hero_a"] = frame["hero_a"] + 0.01
    frame.to_parquet(tampered, index=False)

    result = _run(models, data, policy=_lenient_policy())
    assert result.exit_code == 0

    mismatch = gate.run_gate(
        model_dir=models,
        model_stem="fixture_model",
        data_path=tampered,
        policy=_lenient_policy(),
        allow_dataset_mismatch=True,
        report_path=models / "mismatch_report.json",
    )
    assert mismatch.exit_code == 1
    assert mismatch.report["evaluation"]["status"] == "completed"
    assert mismatch.report["evaluation"]["dataset"]["fingerprint_match"] is False
    assert mismatch.report["decision"]["promotion_eligible"] is False
    assert "dataset.fingerprint" in mismatch.report["decision"]["failed_checks"]


def test_missing_or_malformed_artifacts_are_evaluation_errors(tmp_path):
    models, data = _train_fixture(tmp_path)

    manifest = models / "fixture_model_manifest.json"
    original = manifest.read_text()
    manifest.write_text("{ not json")
    malformed = _run(models, data, policy=_lenient_policy())
    assert malformed.exit_code == 2
    assert malformed.report["evaluation"]["status"] == "error"
    assert malformed.report["decision"]["promotion_eligible"] is False
    assert malformed.report["evaluation"]["error"]["kind"] == "evaluation_error"
    manifest.write_text(original)

    onnx = models / "fixture_model.onnx"
    onnx.rename(models / "fixture_model.onnx.bak")
    missing = _run(models, data, policy=_lenient_policy())
    assert missing.exit_code == 2
    assert "ONNX model not found" in missing.report["evaluation"]["error"]["message"]


def test_schema_name_or_order_mismatch_is_rejected(tmp_path):
    models, data = _train_fixture(tmp_path)
    schema_path = models / "fixture_model_schema.json"
    schema = json.loads(schema_path.read_text())
    schema["feature_names"][0], schema["feature_names"][1] = (
        schema["feature_names"][1],
        schema["feature_names"][0],
    )
    schema_path.write_text(json.dumps(schema))

    result = _run(models, data, policy=_lenient_policy())

    assert result.exit_code == 2
    failed = {
        check["identifier"]
        for check in result.report["checks"]
        if check["status"] == "fail" and check["mandatory"]
    }
    assert "contract.features_match_schema" in failed
    assert "contract.manifest_match_schema" in failed
    assert "contract.feature_fingerprint" in failed


def test_missing_labels_single_class_and_bad_predictions_are_rejected():
    labels = np.array([0, 1, 1, 0])
    good = np.array([0.1, 0.8, 0.7, 0.2])
    metrics = gate.compute_metrics(labels, good, calibration_bins=4)
    assert metrics["rows"] == 4
    assert metrics["positives"] == 2
    assert metrics["calibration"]["ece"] == pytest.approx(
        sum(bin_["rows"] * bin_["absolute_gap"] for bin_ in metrics["calibration"]["bins"]) / 4
    )

    with pytest.raises(gate.EvaluationError, match="only 0/1"):
        gate.validate_labels([0, 1, 2])
    with pytest.raises(gate.EvaluationError, match="missing labels"):
        gate.validate_labels([0, float("nan")])
    with pytest.raises(gate.EvaluationError, match="only one target class"):
        gate.validate_labels([1, 1, 1])
    with pytest.raises(gate.EvaluationError, match="empty"):
        gate.validate_labels([])

    with pytest.raises(gate.EvaluationError, match="NaN or infinity"):
        gate.compute_metrics(labels, np.array([0.1, np.nan, 0.7, 0.2]))
    with pytest.raises(gate.EvaluationError, match="\\[0,1\\]"):
        gate.compute_metrics(labels, np.array([0.1, 1.2, 0.7, 0.2]))
    with pytest.raises(gate.EvaluationError, match="shape"):
        gate.compute_metrics(labels, np.array([0.1, 0.2]))


def test_baselines_use_the_training_prior_and_constant_half():
    labels = np.array([0, 0, 1, 1])
    baselines = gate.baseline_metrics(labels, train_positive_rate=0.75)

    prior = baselines["constant_train_prior"]
    assert prior["brier_score"] == pytest.approx(0.3125)
    assert prior["accuracy_at_0_5"] == pytest.approx(0.5)
    half = baselines["constant_half"]
    assert half["brier_score"] == pytest.approx(0.25)


def test_promotion_is_rejected_after_a_failed_gate(tmp_path):
    models, data = _train_fixture(tmp_path)
    strict = gate.QualityPolicy(
        min_roc_auc=1.01, max_brier=0.0, max_ece=0.0, min_test_rows=1
    )
    failed = _run(models, data, policy=strict)
    destination = tmp_path / "promoted"

    with pytest.raises(gate.PromotionRejected, match="did not approve"):
        gate.promote_artifacts(failed.report, models, destination, "fixture_model")
    assert not destination.exists()

    passed = _run(models, data, policy=_lenient_policy())
    copied = gate.promote_artifacts(passed.report, models, destination, "fixture_model")
    assert copied
    assert (destination / "fixture_model.onnx").is_file()
    assert (destination / "fixture_model_schema.json").is_file()
    assert (destination / "fixture_model_quality_report.json").is_file()


def test_leakage_audit_marks_post_draft_features_and_keeps_gain_rank():
    schema = {
        "feature_names": ["duration", "avg_mmr", "rad_pick_0", "hero_is_present"],
        "defaults": {"duration": 2163.0},
        "categoricals": {},
    }
    frame = pd.DataFrame(
        {
            "duration": [1800.0, 2200.0, 2600.0, 3000.0],
            "avg_mmr": [0.0, 0.0, 0.0, 0.0],
            "rad_pick_0": [1.0, 2.0, 3.0, 4.0],
        }
    )
    audit = gate.build_leakage_audit(
        frame,
        np.array([0, 0, 1, 1]),
        schema,
        {"f0": 10.0, "f3": 5.0},
    )
    rows = {row["feature"]: row for row in audit["features"]}

    assert rows["duration"]["available_at_recommendation"] == "post_draft"
    assert rows["duration"]["inference_value"] == "schema_default"
    assert rows["duration"]["gain_rank"] == 1
    assert rows["duration"]["univariate_roc_auc"] == pytest.approx(1.0)
    assert rows["avg_mmr"]["unique_values"] == 1
    assert rows["avg_mmr"]["univariate_roc_auc"] is None
    assert rows["rad_pick_0"]["available_at_recommendation"] == "partial_draft"
    assert rows["rad_pick_0"]["gain_rank"] is None
    assert "hero_is_present" not in rows
