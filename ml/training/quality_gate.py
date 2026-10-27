#!/usr/bin/env python3
"""Automated quality gate for draft models.

The gate answers one question: *may this artifact be promoted to the bundled
production model?* It does not train anything and never rewrites the model or
its schema. It validates the artifact bundle, re-evaluates the model on the
chronological held-out partition declared by the manifest, writes a
machine-readable report, and refuses promotion when a mandatory check fails.

Two failure kinds are kept separate:

* **evaluation error** (exit 2) - the evaluation could not be trusted or run:
  missing/malformed artifacts, schema drift, dataset fingerprint mismatch,
  invalid labels, malformed predictions, empty/single-class data.
* **quality failure** (exit 1) - the evaluation ran, but the model does not
  meet the policy (ROC-AUC / Brier / calibration / sample size).

A quality failure blocks promotion. It does *not* disable manual runtime
opt-in: ML is already disabled by default in the app, and the existing
architecture allows a user to experiment with a local model. That boundary is
deliberate and documented in `docs/ML_TRAINING.md`.

Usage:
    python ml/training/quality_gate.py
    python ml/training/quality_gate.py --report ml/models/local_gate.json
    python ml/training/quality_gate.py --promote-to ml/models/promoted
    python ml/training/quality_gate.py --json

Exit codes: 0 = passed / promotion eligible, 1 = quality failure, 2 = evaluation
error. A failed gate never copies artifacts.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

ML_ROOT = Path(__file__).resolve().parent.parent
if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from training.inference_contract import CONTEXT_FEATURES, encode_matrix
from training.manifest import manifest_paths
from training.splits import file_sha256, schema_fingerprint, time_split

REPORT_VERSION = 1
DEFAULT_MODEL_DIR = ML_ROOT / "models"
DEFAULT_MODEL_STEM = "draft_model_xgboost_full"
DEFAULT_DATA = ML_ROOT / "datasets" / "processed" / "draft_features.parquet"

IDENTITY_COLUMNS = ("match_id", "radiant_win", "start_time", "start_time_utc")
PROMOTED_ARTIFACTS = ("onnx", "features", "schema", "config", "manifest")


class EvaluationError(RuntimeError):
    """The evaluation could not be executed or cannot be trusted."""


class PromotionRejected(RuntimeError):
    """Artifacts were not promoted because the gate did not pass."""


@dataclass(frozen=True)
class QualityPolicy:
    """Initial promotion policy. Thresholds are a floor, not a quality claim."""

    min_roc_auc: float = 0.60
    max_brier: float = 0.235
    max_ece: float = 0.10
    min_test_rows: int = 500
    calibration_bins: int = 10
    evaluation_mode: str = "inference"  # "inference" or "as_trained"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Check:
    identifier: str
    status: str  # pass | fail | skip
    detail: str
    mandatory: bool = True

    @property
    def failed(self) -> bool:
        return self.status == "fail" and self.mandatory


@dataclass
class GateReport:
    report: Dict[str, Any] = field(default_factory=dict)
    exit_code: int = 2

    def as_dict(self) -> Dict[str, Any]:
        return self.report


def _check(checks: List[Check], identifier: str, ok: bool, detail: str, mandatory: bool = True) -> None:
    checks.append(Check(identifier, "pass" if ok else "fail", detail, mandatory))


def _skip(checks: List[Check], identifier: str, detail: str) -> None:
    checks.append(Check(identifier, "skip", detail, mandatory=False))


def _mandatory_failures(checks: Sequence[Check]) -> List[Check]:
    return [check for check in checks if check.failed]


def _read_json(path: Path, label: str) -> Any:
    if not path.is_file():
        raise EvaluationError(f"{label} not found: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        raise EvaluationError(f"{label} is not valid JSON ({path}): {error}") from error


def _require_list_of_strings(value: Any, label: str) -> List[str]:
    if not isinstance(value, list) or not value or not all(isinstance(item, str) for item in value):
        raise EvaluationError(f"{label} must be a non-empty JSON array of strings")
    return value


def validate_artifact_bundle(
    paths: Mapping[str, Path],
    checks: List[Check],
) -> Tuple[Dict[str, Any], Dict[str, Any], List[str]]:
    """Validate manifest, schema, and ordered feature list against each other."""
    manifest = _read_json(paths["manifest"], "manifest")
    if not isinstance(manifest, dict):
        raise EvaluationError("manifest must be a JSON object")
    _check(checks, "manifest.present", True, str(paths["manifest"]))

    schema = _read_json(paths["schema"], "schema")
    if not isinstance(schema, dict):
        raise EvaluationError("schema must be a JSON object")
    _check(checks, "schema.present", True, str(paths["schema"]))

    features = _read_json(paths["features"], "features")
    feature_names = _require_list_of_strings(features, "features file")
    schema_names = _require_list_of_strings(schema.get("feature_names"), "schema.feature_names")

    _check(
        checks,
        "contract.feature_count",
        int(schema.get("feature_count", -1)) == len(schema_names),
        f"schema declares {schema.get('feature_count')}, lists {len(schema_names)}",
    )
    _check(
        checks,
        "contract.features_match_schema",
        feature_names == schema_names,
        "features.json order and names match schema.feature_names",
    )
    manifest_schema = manifest.get("feature_schema", {})
    manifest_names = _require_list_of_strings(
        manifest_schema.get("feature_names"), "manifest.feature_schema.feature_names"
    )
    _check(
        checks,
        "contract.manifest_match_schema",
        manifest_names == schema_names,
        "manifest feature order matches schema feature order",
    )
    _check(
        checks,
        "contract.feature_fingerprint",
        manifest_schema.get("fingerprint") == schema_fingerprint(schema_names),
        "manifest schema fingerprint matches the ordered schema names",
    )
    _check(
        checks,
        "contract.feature_count_manifest",
        int(manifest_schema.get("feature_count", -1)) == len(schema_names),
        f"manifest declares {manifest_schema.get('feature_count')} features",
    )
    _check(
        checks,
        "contract.positive_class",
        int(schema.get("positive_class_index", -1)) >= 0,
        f"positive_class_index={schema.get('positive_class_index')}",
    )
    outputs = (
        schema.get("input_name"),
        schema.get("label_output"),
        schema.get("probability_output"),
    )
    _check(
        checks,
        "contract.output_names",
        all(isinstance(value, str) and value for value in outputs),
        f"input={outputs[0]}, label={outputs[1]}, probability={outputs[2]}",
    )
    return manifest, schema, schema_names


class OnnxPredictor:
    """Thin runtime-contract wrapper around the exported ONNX model."""

    def __init__(self, path: Path, schema: Mapping[str, Any]) -> None:
        if not path.is_file():
            raise EvaluationError(f"ONNX model not found: {path}")
        try:
            import onnxruntime as ort
        except ImportError as error:  # pragma: no cover - dependency is installed in the venv
            raise EvaluationError(f"onnxruntime is not installed: {error}") from error

        self.path = path
        self.session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        inputs = self.session.get_inputs()
        if len(inputs) != 1:
            raise EvaluationError(f"expected exactly one ONNX input, found {len(inputs)}")
        self.input = inputs[0]
        self.feature_count = len(schema["feature_names"])
        feature_count = self.feature_count
        shape = list(self.input.shape)
        if len(shape) != 2:
            raise EvaluationError(f"ONNX input must be 2-D, found shape {shape}")
        if shape[1] not in (None, feature_count):
            raise EvaluationError(
                f"ONNX input expects {shape[1]} features but schema declares {feature_count}"
            )
        if self.input.type != "tensor(float)":
            raise EvaluationError(f"ONNX input type must be tensor(float), found {self.input.type}")
        self.positive_class_index = int(schema.get("positive_class_index", 1))
        self.probability_output = str(schema.get("probability_output", "probabilities"))
        self.output_names = [output.name for output in self.session.get_outputs()]
        if self.positive_class_index < 0:
            raise EvaluationError("positive_class_index must be non-negative")

    def predict(self, matrix: np.ndarray, batch_size: int = 8192) -> np.ndarray:
        if matrix.ndim != 2 or matrix.shape[1] != self.feature_count:
            raise EvaluationError(
                f"feature matrix shape {matrix.shape} does not match "
                f"{self.feature_count} schema features"
            )
        if matrix.size == 0:
            raise EvaluationError("no rows to predict")
        probabilities: List[np.ndarray] = []
        for start in range(0, len(matrix), batch_size):
            batch = np.ascontiguousarray(matrix[start : start + batch_size], dtype=np.float32)
            if not np.isfinite(batch).all():
                raise EvaluationError("feature matrix contains NaN or infinity")
            outputs = self.session.run(None, {self.input.name: batch})
            probabilities.append(self._select_probabilities(outputs, len(batch)))
        stacked = np.concatenate(probabilities)
        if not np.isfinite(stacked).all():
            raise EvaluationError("model returned NaN or infinite probabilities")
        tolerance = 1e-5
        if (stacked < -tolerance).any() or (stacked > 1.0 + tolerance).any():
            raise EvaluationError(
                "model returned probabilities outside [0,1]; check positive_class_index/output"
            )
        return np.clip(stacked, 0.0, 1.0)

    def _select_probabilities(self, outputs: Sequence[np.ndarray], rows: int) -> np.ndarray:
        # Prefer the named probability output; otherwise mirror the runtime's
        # fallback of using the first tensor with a positive-class column.
        named = dict(zip(self.output_names, outputs))
        candidates = []
        if self.probability_output in named:
            candidates.append(named[self.probability_output])
        candidates.extend(outputs)
        for output in candidates:
            array = np.asarray(output)
            if array.ndim != 2 or array.shape[0] != rows:
                continue
            if array.shape[1] <= self.positive_class_index:
                continue
            return array[:, self.positive_class_index].astype(np.float64)
        raise EvaluationError(
            f"no output exposes class {self.positive_class_index} for {rows} rows; "
            f"outputs={self.output_names}"
        )


def validate_labels(labels: Sequence[Any], name: str = "radiant_win") -> np.ndarray:
    """Return labels as ints {0,1} or raise a clear evaluation error."""
    series = pd.Series(labels, name=name)
    if series.empty:
        raise EvaluationError("evaluation set is empty")
    if series.isna().any():
        raise EvaluationError(f"{name} contains {int(series.isna().sum())} missing labels")
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.isna().any():
        raise EvaluationError(f"{name} contains non-numeric labels")
    unique = set(numeric.unique().tolist())
    if not unique.issubset({0.0, 1.0}):
        raise EvaluationError(f"{name} must contain only 0/1, found {sorted(unique)[:10]}")
    result = numeric.to_numpy(dtype=np.int64)
    if len(np.unique(result)) < 2:
        raise EvaluationError("evaluation set contains only one target class")
    return result


def _auc_ci(roc_auc: float, positives: int, negatives: int) -> Dict[str, float]:
    """Hanley-McNeil 95% confidence interval (deterministic, no resampling)."""
    if positives <= 0 or negatives <= 0:
        return {"lower": float("nan"), "upper": float("nan")}
    q1 = roc_auc / (2.0 - roc_auc)
    q2 = 2.0 * roc_auc * roc_auc / (1.0 + roc_auc)
    variance = (
        roc_auc * (1.0 - roc_auc)
        + (positives - 1) * (q1 - roc_auc * roc_auc)
        + (negatives - 1) * (q2 - roc_auc * roc_auc)
    ) / (positives * negatives)
    standard_error = math.sqrt(max(variance, 0.0))
    return {
        "lower": float(max(0.0, roc_auc - 1.96 * standard_error)),
        "upper": float(min(1.0, roc_auc + 1.96 * standard_error)),
    }


def calibration_table(
    labels: np.ndarray,
    probabilities: np.ndarray,
    bins: int = 10,
) -> Dict[str, Any]:
    """Equal-frequency reliability table plus ECE/MCE."""
    order = np.argsort(probabilities, kind="mergesort")
    rows: List[Dict[str, Any]] = []
    total = len(labels)
    weighted_gap = 0.0
    max_gap = 0.0
    for indices in np.array_split(order, bins):
        if len(indices) == 0:
            continue
        bin_labels = labels[indices]
        bin_probabilities = probabilities[indices]
        observed = float(bin_labels.mean())
        predicted = float(bin_probabilities.mean())
        gap = abs(observed - predicted)
        weighted_gap += (len(indices) / total) * gap
        max_gap = max(max_gap, gap)
        rows.append(
            {
                "rows": int(len(indices)),
                "mean_predicted": predicted,
                "observed_rate": observed,
                "absolute_gap": gap,
            }
        )
    return {"bins": rows, "ece": float(weighted_gap), "mce": float(max_gap)}


def compute_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    calibration_bins: int = 10,
) -> Dict[str, Any]:
    """Compute the policy metrics and calibration table for one prediction set."""
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if labels.shape != probabilities.shape:
        raise EvaluationError(
            f"labels shape {labels.shape} does not match predictions shape {probabilities.shape}"
        )
    if len(labels) == 0:
        raise EvaluationError("evaluation set is empty")
    if not np.isfinite(probabilities).all():
        raise EvaluationError("predictions contain NaN or infinity")
    if (probabilities < 0.0).any() or (probabilities > 1.0).any():
        raise EvaluationError("predictions must be probabilities in [0,1]")

    positives = int((labels == 1).sum())
    negatives = int((labels == 0).sum())
    if positives == 0 or negatives == 0:
        raise EvaluationError("ROC-AUC requires both target classes")

    clipped = np.clip(probabilities, 1e-7, 1.0 - 1e-7)
    calibration = calibration_table(labels, probabilities, bins=calibration_bins)
    return {
        "rows": int(len(labels)),
        "positives": positives,
        "negatives": negatives,
        "positive_rate": float(positives / len(labels)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "roc_auc_ci95": _auc_ci(float(roc_auc_score(labels, probabilities)), positives, negatives),
        "brier_score": float(brier_score_loss(labels, probabilities)),
        "log_loss": float(log_loss(labels, clipped)),
        "accuracy_at_0_5": float(((probabilities > 0.5).astype(int) == labels).mean()),
        "mean_prediction": float(probabilities.mean()),
        "calibration": calibration,
    }


def baseline_metrics(labels: np.ndarray, train_positive_rate: float) -> Dict[str, Any]:
    """Constant-probability baselines on the same evaluation rows."""
    baselines: Dict[str, Any] = {}
    for name, probability in (
        ("constant_train_prior", float(train_positive_rate)),
        ("constant_half", 0.5),
    ):
        predictions = np.full(len(labels), probability, dtype=np.float64)
        baselines[name] = {
            "probability": probability,
            "brier_score": float(brier_score_loss(labels, predictions)),
            "log_loss": float(log_loss(labels, np.clip(predictions, 1e-7, 1.0 - 1e-7))),
            "accuracy_at_0_5": float(((predictions > 0.5).astype(int) == labels).mean()),
        }
    return baselines


def policy_checks(
    metrics: Mapping[str, Any],
    policy: QualityPolicy,
    *,
    mode: str,
) -> List[Check]:
    """Turn the measured metrics into mandatory policy checks."""
    checks: List[Check] = []
    roc_auc = float(metrics["roc_auc"])
    brier = float(metrics["brier_score"])
    ece = float(metrics["calibration"]["ece"])
    rows = int(metrics["rows"])
    _check(
        checks,
        "policy.roc_auc",
        math.isfinite(roc_auc) and roc_auc >= policy.min_roc_auc,
        f"{mode} ROC-AUC {roc_auc:.4f} >= {policy.min_roc_auc:.4f}",
    )
    _check(
        checks,
        "policy.brier_score",
        math.isfinite(brier) and brier <= policy.max_brier,
        f"{mode} Brier {brier:.4f} <= {policy.max_brier:.4f}",
    )
    _check(
        checks,
        "policy.calibration_ece",
        math.isfinite(ece) and ece <= policy.max_ece,
        f"{mode} ECE {ece:.4f} <= {policy.max_ece:.4f}",
    )
    _check(
        checks,
        "policy.sample_size",
        rows >= policy.min_test_rows,
        f"{mode} rows {rows} >= {policy.min_test_rows}",
    )
    return checks


def _split_boundaries(frame: pd.DataFrame, column: str) -> Dict[str, float]:
    return {"min": float(frame[column].min()), "max": float(frame[column].max())}


def validate_split(
    split: Any,
    manifest: Mapping[str, Any],
    dataset: pd.DataFrame,
    checks: List[Check],
) -> Dict[str, Any]:
    """Verify the reconstructed split is the one the manifest describes."""
    declared = manifest.get("split", {})
    if not isinstance(declared, dict) or "sizes" not in declared:
        raise EvaluationError("manifest has no split metadata; cannot verify independence")

    sizes = split.sizes()
    _check(
        checks,
        "split.sizes",
        sizes == declared.get("sizes"),
        f"reconstructed {sizes} vs manifest {declared.get('sizes')}",
    )
    if declared.get("fractions") and split.fractions != declared["fractions"]:
        raise EvaluationError(
            f"split fractions differ: {split.fractions} vs manifest {declared['fractions']}"
        )
    _check(
        checks,
        "split.time_column",
        split.time_column == declared.get("time_column"),
        f"time column {split.time_column!r}",
    )

    if "match_id" not in dataset.columns:
        raise EvaluationError("dataset has no match_id; match-level independence is unverifiable")
    duplicate_matches = int(dataset["match_id"].duplicated().sum())
    _check(
        checks,
        "split.match_id_unique",
        duplicate_matches == 0,
        f"{duplicate_matches} duplicated match_id rows",
    )

    partitions = {"train": split.train, "validation": split.val, "test": split.test}
    ids = {name: set(frame["match_id"].tolist()) for name, frame in partitions.items()}
    overlaps = {
        "train_validation": len(ids["train"] & ids["validation"]),
        "train_test": len(ids["train"] & ids["test"]),
        "validation_test": len(ids["validation"] & ids["test"]),
    }
    _check(
        checks,
        "split.match_level_disjoint",
        all(count == 0 for count in overlaps.values()),
        f"match_id overlaps: {overlaps}",
    )

    ordered = (
        split.boundaries["train"]["max"] <= split.boundaries["validation"]["min"]
        and split.boundaries["validation"]["max"] <= split.boundaries["test"]["min"]
    )
    _check(
        checks,
        "split.chronological",
        ordered,
        f"boundaries: {split.boundaries}",
    )
    _check(
        checks,
        "split.used_start_time",
        bool(split.used_start_time),
        f"used_start_time={split.used_start_time} (fallback={split.time_column})",
        mandatory=False,
    )
    return {
        "time_column": split.time_column,
        "used_start_time": bool(split.used_start_time),
        "fractions": dict(split.fractions),
        "sizes": sizes,
        "boundaries": split.boundaries,
        "match_id_overlaps": overlaps,
        "duplicate_match_id_rows": duplicate_matches,
    }


def _feature_values(dataset: pd.DataFrame, feature: str) -> Optional[np.ndarray]:
    if feature not in dataset.columns:
        return None
    if isinstance(dataset[feature].dtype, pd.CategoricalDtype):
        return None
    values = pd.to_numeric(dataset[feature], errors="coerce").to_numpy(dtype=np.float64)
    if np.isnan(values).all():
        return None
    return values


def build_leakage_audit(
    test_frame: pd.DataFrame,
    labels: np.ndarray,
    schema: Mapping[str, Any],
    gain_scores: Optional[Mapping[str, float]] = None,
) -> Dict[str, Any]:
    """Classify the audited feature families with measured evidence.

    The gate does not remove features: changing the schema would invalidate the
    bundled model. This section exists so a promotion review can see which
    inputs are post-draft or only partially available, and how strongly the
    model relies on them.
    """
    audited = {
        "duration": ("post_draft", "Known only after the game; replaced by the schema median at inference."),
        "avg_mmr": ("match_context", "Constant 0 in the current dataset; no live value."),
        "game_mode": ("match_context", "Match setting; replaced by the schema median at inference."),
        "lobby_type": ("match_context", "Match setting; replaced by the schema median at inference."),
        "patch": ("live_context", "Live patch is known, but evaluated rows use historical patch codes."),
        "patch_version": ("live_context", "Live patch is known, but evaluated rows use historical patch codes."),
    }
    for prefix in ("rad_pick_", "dire_pick_"):
        for index in range(5):
            audited[f"{prefix}{index}"] = (
                "partial_draft",
                "Final hero id in slot i; unknown slots fall back to the schema median.",
            )

    names = list(schema["feature_names"])
    gain_rank: Dict[str, int] = {}
    if gain_scores:
        ranked = sorted(gain_scores.items(), key=lambda item: item[1], reverse=True)
        for rank, (key, value) in enumerate(ranked, start=1):
            if key.startswith("f") and key[1:].isdigit():
                index = int(key[1:])
                if index < len(names):
                    gain_rank[names[index]] = rank

    rows: List[Dict[str, Any]] = []
    for feature, (availability, note) in audited.items():
        if feature not in names:
            continue
        values = _feature_values(test_frame, feature)
        univariate_auc: Optional[float] = None
        unique_values: Optional[int] = None
        if values is not None:
            unique_values = int(len(np.unique(values)))
            if unique_values > 1:
                try:
                    univariate_auc = float(roc_auc_score(labels, values))
                except ValueError:
                    univariate_auc = None
        rows.append(
            {
                "feature": feature,
                "available_at_recommendation": availability,
                "inference_value": (
                    "schema_default" if feature in CONTEXT_FEATURES else "live_or_schema_default"
                ),
                "univariate_roc_auc": univariate_auc,
                "unique_values": unique_values,
                "gain_rank": gain_rank.get(feature),
                "note": note,
            }
        )
    return {"features": rows}


def _load_gain_scores(joblib_path: Path, feature_names: Sequence[str]) -> Dict[str, float]:
    """Best-effort XGBoost gain map keyed by `f<index>` (empty when unavailable)."""
    if not joblib_path.is_file():
        return {}
    try:
        import xgboost as xgb

        model = xgb.XGBClassifier()
        model.load_model(str(joblib_path))
        if getattr(model, "n_features_in_", None) not in (None, len(feature_names)):
            return {}
        return model.get_booster().get_score(importance_type="gain")
    except Exception:
        return {}


def run_gate(
    model_dir: Path | str = DEFAULT_MODEL_DIR,
    model_stem: str = DEFAULT_MODEL_STEM,
    data_path: Path | str = DEFAULT_DATA,
    *,
    policy: Optional[QualityPolicy] = None,
    report_path: Optional[Path | str] = None,
    allow_dataset_mismatch: bool = False,
) -> GateReport:
    """Run the gate and return the report (also written to `report_path`)."""
    policy = policy or QualityPolicy()
    model_dir = Path(model_dir)
    data_path = Path(data_path)
    paths = manifest_paths(model_dir, model_stem)
    paths["schema"] = model_dir / f"{model_stem}_schema.json"
    report_path = Path(report_path) if report_path else model_dir / f"{model_stem}_quality_report.json"

    report: Dict[str, Any] = {
        "report_version": REPORT_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": {
            "stem": model_stem,
            "directory": str(model_dir),
            "artifacts": {name: str(path) for name, path in paths.items()},
        },
        "policy": policy.to_dict(),
        "checks": [],
        "evaluation": {
            "status": "error",
            "error": None,
            "dataset": {},
            "split": {},
            "modes": {},
            "leakage": {},
        },
        "decision": {
            "quality_status": "not_evaluated",
            "promotion_eligible": False,
            "failed_checks": [],
            "runtime_opt_in": "allowed",
        },
        "exit_code": 2,
    }
    checks: List[Check] = []

    def finish() -> GateReport:
        report["checks"] = [asdict(check) for check in checks]
        failures = _mandatory_failures(checks)
        report["decision"]["failed_checks"] = [check.identifier for check in failures]
        if report["evaluation"]["status"] == "completed":
            passed = not failures
            report["decision"]["quality_status"] = "passed" if passed else "failed"
            report["decision"]["promotion_eligible"] = bool(passed)
            report["exit_code"] = 0 if passed else 1
        else:
            report["decision"]["quality_status"] = "not_evaluated"
            report["decision"]["promotion_eligible"] = False
            report["exit_code"] = 2
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        return GateReport(report=report, exit_code=int(report["exit_code"]))

    try:
        manifest, schema, feature_names = validate_artifact_bundle(paths, checks)
        report["model"]["manifest_created_at"] = manifest.get("created_at")
        report["model"]["model_type"] = manifest.get("model_type")
        report["model"]["patch_version"] = manifest.get("patch_version")
        _check(
            checks,
            "manifest.model_stem",
            manifest.get("model_stem") in (None, model_stem),
            f"manifest model_stem={manifest.get('model_stem')!r}",
        )
        if _mandatory_failures(checks):
            raise EvaluationError(
                "artifact/contract validation failed: "
                + ", ".join(check.identifier for check in _mandatory_failures(checks))
            )

        predictor = OnnxPredictor(paths["onnx"], schema)
        _check(
            checks,
            "artifact.onnx_input",
            list(predictor.input.shape)[1] in (None, len(feature_names)),
            f"input {predictor.input.name!r} shape={predictor.input.shape} type={predictor.input.type}",
        )
        _check(
            checks,
            "artifact.onnx_output",
            predictor.probability_output in predictor.output_names,
            f"probability output {predictor.probability_output!r} in {predictor.output_names}",
        )
        expected_input_name = schema.get("input_name")
        _check(
            checks,
            "artifact.onnx_input_name",
            predictor.input.name == expected_input_name,
            f"onnx input {predictor.input.name!r} vs schema {expected_input_name!r}",
            mandatory=False,
        )

        joblib_path = paths["joblib"]
        if joblib_path.is_file():
            try:
                import xgboost as xgb

                model = xgb.XGBClassifier()
                model.load_model(str(joblib_path))
                _check(
                    checks,
                    "artifact.joblib_feature_count",
                    int(getattr(model, "n_features_in_", len(feature_names))) == len(feature_names),
                    f"joblib expects {getattr(model, 'n_features_in_', '?')} features",
                )
            except Exception as error:
                _skip(checks, "artifact.joblib_load", f"joblib unreadable: {error}")
        else:
            _skip(checks, "artifact.joblib_load", "no joblib artifact (runtime uses ONNX only)")

        if not data_path.is_file():
            raise EvaluationError(f"dataset not found: {data_path}")
        dataset = pd.read_parquet(data_path)
        dataset_sha = file_sha256(data_path)
        declared_sha = manifest.get("dataset", {}).get("sha256")
        sha_matches = dataset_sha == declared_sha
        _check(
            checks,
            "dataset.fingerprint",
            sha_matches,
            f"dataset sha256 {dataset_sha} vs manifest {declared_sha}",
        )
        if not sha_matches and not allow_dataset_mismatch:
            raise EvaluationError(
                "dataset fingerprint does not match the manifest; evaluation would not be "
                "the model's declared holdout (pass --allow-dataset-mismatch to inspect only)"
            )
        declared_rows = manifest.get("dataset", {}).get("rows")
        _check(
            checks,
            "dataset.rows",
            declared_rows is None or int(declared_rows) == len(dataset),
            f"dataset rows {len(dataset)} vs manifest {declared_rows}",
        )

        declared_split = manifest.get("split", {})
        fractions = declared_split.get("fractions") or {
            "train": 0.8,
            "validation": 0.1,
            "test": 0.1,
        }
        split = time_split(
            dataset,
            train_frac=float(fractions["train"]),
            val_frac=float(fractions["validation"]),
            test_frac=float(fractions["test"]),
            time_col=declared_split.get("time_column"),
        )
        split_info = validate_split(split, manifest, dataset, checks)

        labels = validate_labels(split.test["radiant_win"].to_numpy())
        train_labels = validate_labels(split.train["radiant_win"].to_numpy())
        train_prior = float(train_labels.mean())

        as_trained = encode_matrix(split.test, schema, inference_defaults=False)
        inference = encode_matrix(split.test, schema, inference_defaults=True)
        for name, encoded in (("as_trained", as_trained), ("inference", inference)):
            if encoded.nonfinite_features:
                raise EvaluationError(
                    f"{name} feature matrix contains non-finite values in "
                    f"{list(encoded.nonfinite_features)[:10]}"
                )
            if encoded.missing_columns:
                # Missing columns are filled from schema defaults, but a large
                # gap means the dataset and schema came from different exports.
                _check(
                    checks,
                    f"dataset.columns.{name}",
                    len(encoded.missing_columns) == 0,
                    f"{len(encoded.missing_columns)} schema features absent from dataset",
                    mandatory=False,
                )

        modes: Dict[str, Any] = {}
        for name, encoded in (("as_trained", as_trained), ("inference", inference)):
            probabilities = predictor.predict(encoded.matrix)
            metrics = compute_metrics(
                labels,
                probabilities,
                calibration_bins=policy.calibration_bins,
            )
            modes[name] = {
                "metrics": metrics,
                "baselines": baseline_metrics(labels, train_prior),
                "encoding": {
                    "nan_values_filled": encoded.nan_values,
                    "unknown_categories": encoded.unknown_categories,
                    "missing_columns": list(encoded.missing_columns),
                },
            }

        if policy.evaluation_mode not in modes:
            raise EvaluationError(
                f"unknown evaluation_mode {policy.evaluation_mode!r}; "
                f"choose from {sorted(modes)}"
            )
        selected = modes[policy.evaluation_mode]
        _check(checks, "evaluation.mode", True, policy.evaluation_mode)
        manifest_test_auc = manifest.get("metrics", {}).get("test", {}).get("roc_auc")
        parity = True
        parity_detail = "manifest has no as-trained test ROC-AUC to compare"
        if manifest_test_auc is not None:
            parity = (
                abs(
                    float(modes["as_trained"]["metrics"]["roc_auc"])
                    - float(manifest_test_auc)
                )
                <= 1e-4
            )
            parity_detail = (
                "recomputed as-trained ROC-AUC "
                f"{modes['as_trained']['metrics']['roc_auc']:.6f} vs manifest "
                f"{float(manifest_test_auc):.6f} (tolerance 1e-4)"
            )
        _check(
            checks,
            "evaluation.as_trained_metric_parity",
            parity,
            parity_detail,
            mandatory=False,
        )

        checks.extend(policy_checks(selected["metrics"], policy, mode=policy.evaluation_mode))

        gain_scores = _load_gain_scores(joblib_path, feature_names)
        leakage = build_leakage_audit(split.test, labels, schema, gain_scores)

        report["evaluation"].update(
            {
                "status": "completed",
                "error": None,
                "dataset": {
                    "path": str(data_path),
                    "sha256": dataset_sha,
                    "manifest_sha256": declared_sha,
                    "fingerprint_match": sha_matches,
                    "rows": int(len(dataset)),
                    "label_counts": {
                        "radiant_win_1": int((dataset["radiant_win"] == 1).sum()),
                        "radiant_win_0": int((dataset["radiant_win"] == 0).sum()),
                    },
                },
                "split": {
                    **split_info,
                    "strategy": "chronological contiguous slices",
                    "labels": {
                        "positive_class": "radiant_win=1 (Radiant wins)",
                        "positive_class_index": int(schema.get("positive_class_index", 1)),
                        "probability_output": schema.get("probability_output"),
                        "train_positive_rate": train_prior,
                    },
                },
                "modes": modes,
                "selected_mode": policy.evaluation_mode,
                "leakage": leakage,
            }
        )
        _check(
            checks,
            "evaluation.rows",
            int(selected["metrics"]["rows"]) > 0,
            f"{selected['metrics']['rows']} evaluation rows",
        )
        return finish()
    except EvaluationError as error:
        report["evaluation"]["status"] = "error"
        report["evaluation"]["error"] = {"kind": "evaluation_error", "message": str(error)}
        return finish()
    except Exception as error:  # pragma: no cover - defensive: always write a report
        report["evaluation"]["status"] = "error"
        report["evaluation"]["error"] = {
            "kind": "unexpected_error",
            "message": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc().splitlines()[-8:],
        }
        return finish()


def promote_artifacts(
    report: Mapping[str, Any],
    model_dir: Path | str,
    destination: Path | str,
    model_stem: str,
) -> List[Path]:
    """Copy a gate-approved bundle; refuse anything the gate did not pass."""
    if not report.get("decision", {}).get("promotion_eligible"):
        failed = report.get("decision", {}).get("failed_checks", [])
        raise PromotionRejected(
            "quality gate did not approve promotion"
            + (f" (failed: {', '.join(failed)}" if failed else "")
        )
    model_dir = Path(model_dir)
    destination = Path(destination)
    paths = manifest_paths(model_dir, model_stem)
    paths["schema"] = model_dir / f"{model_stem}_schema.json"
    destination.mkdir(parents=True, exist_ok=True)
    copied: List[Path] = []
    for name in PROMOTED_ARTIFACTS:
        source = paths[name]
        if not source.is_file():
            raise PromotionRejected(f"approved bundle is missing {name}: {source}")
        target = destination / source.name
        shutil.copy2(source, target)
        copied.append(target)
    report_target = destination / f"{model_stem}_quality_report.json"
    report_target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    copied.append(report_target)
    return copied


def _print_summary(report: Mapping[str, Any]) -> None:
    policy = report["policy"]
    decision = report["decision"]
    evaluation = report["evaluation"]
    print("ML model quality gate")
    print(f"  model:   {report['model']['stem']}")
    print(
        "  policy:  ROC-AUC >= {min_roc_auc:.3f}, Brier <= {max_brier:.3f}, "
        "ECE <= {max_ece:.3f}, rows >= {min_test_rows} ({evaluation_mode})".format(**policy)
    )
    if evaluation["status"] == "completed":
        mode = evaluation["selected_mode"]
        metrics = evaluation["modes"][mode]["metrics"]
        baseline = evaluation["modes"][mode]["baselines"]["constant_train_prior"]
        print(
            f"  result:  ROC-AUC={metrics['roc_auc']:.4f} "
            f"(95% CI {metrics['roc_auc_ci95']['lower']:.3f}-{metrics['roc_auc_ci95']['upper']:.3f}), "
            f"Brier={metrics['brier_score']:.4f}, ECE={metrics['calibration']['ece']:.4f}, "
            f"rows={metrics['rows']}"
        )
        print(
            f"  baseline: constant train prior {baseline['probability']:.4f} -> "
            f"Brier={baseline['brier_score']:.4f}"
        )
    else:
        print(f"  evaluation error: {evaluation['error']['message']}")
    print(
        f"  decision: {decision['quality_status']}"
        f" (promotion_eligible={decision['promotion_eligible']})"
    )
    failed = decision.get("failed_checks") or []
    if failed:
        print(f"  failed checks: {', '.join(failed)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate and re-evaluate a draft model before promotion",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--model-stem", default=DEFAULT_MODEL_STEM)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--min-roc-auc", type=float, default=QualityPolicy().min_roc_auc)
    parser.add_argument("--max-brier", type=float, default=QualityPolicy().max_brier)
    parser.add_argument("--max-ece", type=float, default=QualityPolicy().max_ece)
    parser.add_argument("--min-test-rows", type=int, default=QualityPolicy().min_test_rows)
    parser.add_argument(
        "--evaluation-mode",
        choices=("inference", "as_trained"),
        default=QualityPolicy().evaluation_mode,
        help="inference = runtime defaults for match-only fields; as_trained = logged values",
    )
    parser.add_argument(
        "--allow-dataset-mismatch",
        action="store_true",
        help="evaluate even if the dataset fingerprint differs (never promotion-eligible)",
    )
    parser.add_argument(
        "--promote-to",
        type=Path,
        default=None,
        help="copy the bundle here only when the gate passes",
    )
    parser.add_argument("--json", action="store_true", help="print the full report as JSON")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    policy = QualityPolicy(
        min_roc_auc=args.min_roc_auc,
        max_brier=args.max_brier,
        max_ece=args.max_ece,
        min_test_rows=args.min_test_rows,
        evaluation_mode=args.evaluation_mode,
    )
    result = run_gate(
        model_dir=args.model_dir,
        model_stem=args.model_stem,
        data_path=args.data,
        policy=policy,
        report_path=args.report,
        allow_dataset_mismatch=args.allow_dataset_mismatch,
    )
    if args.json:
        print(json.dumps(result.report, indent=2))
    else:
        _print_summary(result.report)
    if args.promote_to:
        if result.exit_code != 0:
            print("Promotion refused: the quality gate did not pass.", file=sys.stderr)
            return result.exit_code
        try:
            copied = promote_artifacts(
                result.report, args.model_dir, args.promote_to, args.model_stem
            )
        except PromotionRejected as error:
            print(f"Promotion refused: {error}", file=sys.stderr)
            return 1
        print(f"Promoted {len(copied)} artifacts to {args.promote_to}")
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
