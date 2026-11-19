#!/usr/bin/env python3
"""Compare model families on the corrected dataset (validation only).

The ablation experiment fixed the model family and varied features. This script
does the opposite: it fixes the feature set and varies the family, to answer
"is a boosted tree ensemble the right tool for draft-time prediction?" before
any more feature work is attempted.

Protocol (frozen *before* any fit, see `DECISION_RULE`):

* dataset: `ml/datasets/processed_v2/draft_features.parquet` (pick/ban fixed);
* split: the boundary timestamps in `ml/configs/eval_split_v2.json`, verified
  against the dataset SHA-256 and the recorded partition sizes;
* features: the winning combined ablation of the 2026-10-28 experiment
  (871 features: no post-match/match-metadata fields, no raw pick slots), so
  every family sees exactly the same information that a live draft provides;
* selection: validation only. The test partition is never encoded, scored, or
  counted - `quality_gate.py` remains the single place a test number is
  produced, and only for a candidate that has already been selected.

Families:

* `xgboost_combined` - reference: the shipped `WIDE_CONFIG` ensemble;
* `logistic_l2`      - L2 logistic regression on train-fitted standardisation;
* `hist_gbdt`        - sklearn histogram gradient boosting (different binning,
  leaf growth, and regularisation from XGBoost; no new dependency).

Every family is fitted on the same rows with the same labels for each seed, and
evaluated twice: `as_trained` (logged values) and `inference` (the values the
Rust runtime would send). Partial-draft stages use the runtime-faithful
reconstruction from `training.ablation`, so win rates, aggregates, semantics
and synergy only ever contain picks the engine already knows.

Usage:
    python ml/training/model_families.py
    python ml/training/model_families.py --population no_duplicates
    python ml/training/model_families.py --families logistic_l2 --seeds 42
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

ML_ROOT = Path(__file__).resolve().parent.parent
if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from training import ablation
from training.inference_contract import encode_matrix
from training.quality_gate import QualityPolicy, baseline_metrics, compute_metrics
from training.splits import file_sha256, schema_fingerprint
from training.train_full import WIDE_CONFIG, _export_onnx

DEFAULT_DATA = ML_ROOT / "datasets" / "processed_v2" / "draft_features.parquet"
DEFAULT_SPLIT_CONFIG = ML_ROOT / "configs" / "eval_split_v2.json"
DEFAULT_SCHEMA = ML_ROOT / "models" / "draft_model_xgboost_full_schema.json"
DEFAULT_MODEL_DIR = ML_ROOT / "models" / "families"

# The combined ablation from the 2026-10-28 experiment: the feature set that won
# on validation, and the only one whose inputs a live draft can supply.
FEATURE_VARIANT = "no_match_context_no_pick_order"

FAMILY_NAMES = ("xgboost_combined", "logistic_l2", "hist_gbdt")
COMPLEXITY = {"logistic_l2": 0, "hist_gbdt": 1, "xgboost_combined": 2}
REFERENCE = "xgboost_combined"

SEEDS = (42, 43, 44)
# Documented, deliberately small: three decades of L2 strength. Selected on
# validation AUC only; the grid is reported whether or not it changes the choice.
LOGISTIC_C_GRID = (0.01, 0.1, 1.0)
LOGISTIC_MAX_ITER = 3000

HISTGB_CONFIG: Dict[str, Any] = {
    "learning_rate": 0.05,
    "max_iter": 600,
    "max_leaf_nodes": 15,
    "min_samples_leaf": 60,
    "l2_regularization": 1.0,
    "early_stopping": True,
    "validation_fraction": 0.1,
    "n_iter_no_change": 30,
}

PROGRESS_LEVELS = (0, 1, 2, 3, 5)

MATERIAL_AUC_MARGIN = 0.002
MATERIAL_BRIER_MARGIN = 0.002
DEFAULT_BOOTSTRAP = 2000
DEFAULT_THREADS = 8

DECISION_RULE = (
    "Rank families by mean validation inference ROC-AUC across seeds. Families "
    "within 0.002 AUC of the leader are compared on mean validation Brier (lower "
    "wins), then on model complexity (logistic < histogram GBDT < XGBoost). A "
    "family counts as a material improvement over the XGBoost reference only if "
    "its paired-bootstrap 95% CI for the AUC difference excludes 0 *and* its mean "
    "Brier improves by more than 0.002. Anything else is reported as "
    "statistically indistinguishable, not as a winner."
)

LIMITATIONS = (
    "Validation only: the frozen test partition is never encoded or scored here, "
    "and it was already inspected in the 2026-10-28 session, so no fresh final "
    "number exists for any family.",
    "One corpus (matches up to 2026-08-28). Seed spread measures optimisation "
    "noise within this corpus, not sampling uncertainty of a future one.",
    "The primary population keeps the 3,870 duplicate-hero rows; the filtered run "
    "is a sensitivity analysis, not a corrected headline.",
    "Partial-draft stages hold bans at their full-draft values, because ban "
    "timing is not modelled; per-hero win rates, aggregates, semantics, synergy "
    "and damage diversity are recomputed from known picks only.",
    "Histogram GBDT stops on an internal 10% split of the training rows (no "
    "validation leakage); XGBoost stops on validation and is refit at the "
    "selected round count. The stopping budgets therefore differ by design.",
    "Paired-bootstrap intervals describe the difference on these validation "
    "rows; they do not include retraining variability, which the per-seed "
    "results show separately.",
)


def _predict(model: Any, matrix: np.ndarray) -> np.ndarray:
    """Predict positive-class probabilities and reject invalid output."""
    probabilities = np.asarray(model.predict_proba(matrix), dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[1] < 2:
        raise SystemExit("model does not expose two-class probabilities")
    positive = probabilities[:, 1]
    if not np.isfinite(positive).all():
        raise SystemExit("model produced non-finite probabilities")
    if ((positive < 0.0) | (positive > 1.0)).any():
        raise SystemExit("model produced probabilities outside [0, 1]")
    return positive


def _metrics(
    predict_fn,
    matrix: np.ndarray,
    labels: np.ndarray,
    *,
    train_prior: float,
    calibration_bins: int,
) -> Dict[str, Any]:
    return {
        "metrics": compute_metrics(
            labels, predict_fn(matrix), calibration_bins=calibration_bins
        ),
        "baselines": baseline_metrics(labels, train_prior),
    }


def fit_logistic(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    C: float,
    seed: int,
) -> Pipeline:
    """Standardise on train rows only, then fit L2 logistic regression."""
    pipeline = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=float(C),
                    solver="lbfgs",
                    max_iter=LOGISTIC_MAX_ITER,
                    random_state=int(seed),
                ),
            ),
        ]
    )
    pipeline.fit(X_train, y_train)
    return pipeline


def fit_hist_gbdt(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    seed: int,
    max_iter: Optional[int] = None,
) -> HistGradientBoostingClassifier:
    config = dict(HISTGB_CONFIG)
    if max_iter is not None:
        config["max_iter"] = int(max_iter)
    model = HistGradientBoostingClassifier(**config, random_state=int(seed))
    model.fit(X_train, y_train)
    return model


def select_logistic_c(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    seed: int,
    calibration_bins: int,
) -> Tuple[float, List[Dict[str, Any]]]:
    """Pick C on validation AUC only, with Brier as the tie-break."""
    grid: List[Dict[str, Any]] = []
    for C in LOGISTIC_C_GRID:
        start = time.time()
        model = fit_logistic(X_train, y_train, C=C, seed=seed)
        probabilities = _predict(model, X_val)
        grid.append(
            {
                "C": float(C),
                "fit_seconds": time.time() - start,
                "validation_inference_roc_auc": float(roc_auc_score(y_val, probabilities)),
                "validation_inference_brier": float(brier_score_loss(y_val, probabilities)),
            }
        )
    best = sorted(
        grid,
        key=lambda row: (-row["validation_inference_roc_auc"], row["validation_inference_brier"]),
    )[0]
    return float(best["C"]), grid


def paired_bootstrap(
    labels: np.ndarray,
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    replicates: int,
    seed: int = 0,
) -> Dict[str, Any]:
    """Paired bootstrap over the same validation rows.

    Returns the mean and 95% percentile interval of `candidate - reference` for
    ROC-AUC and Brier. Negative Brier differences mean the candidate is better.
    Replicates in which a resample loses a class are skipped and counted.
    """
    labels = np.asarray(labels)
    reference = np.asarray(reference)
    candidate = np.asarray(candidate)
    if not (labels.shape == reference.shape == candidate.shape):
        raise ValueError("labels and predictions must share one shape")
    rng = np.random.default_rng(seed)
    n = len(labels)
    auc_diffs: List[float] = []
    brier_diffs: List[float] = []
    skipped = 0
    for _ in range(int(replicates)):
        idx = rng.integers(0, n, size=n)
        y = labels[idx]
        if y.min() == y.max():
            skipped += 1
            continue
        auc_diffs.append(
            float(roc_auc_score(y, candidate[idx]) - roc_auc_score(y, reference[idx]))
        )
        brier_diffs.append(
            float(
                brier_score_loss(y, candidate[idx]) - brier_score_loss(y, reference[idx])
            )
        )
    if not auc_diffs:
        raise ValueError("every bootstrap resample lost a class; cannot compare")
    auc = np.asarray(auc_diffs)
    brier = np.asarray(brier_diffs)
    return {
        "replicates": int(replicates),
        "replicates_used": int(len(auc)),
        "replicates_skipped_single_class": int(skipped),
        "auc_diff": {
            "mean": float(auc.mean()),
            "ci95": [float(np.percentile(auc, 2.5)), float(np.percentile(auc, 97.5))],
            "fraction_positive": float((auc > 0).mean()),
        },
        "brier_diff": {
            "mean": float(brier.mean()),
            "ci95": [float(np.percentile(brier, 2.5)), float(np.percentile(brier, 97.5))],
            "fraction_negative": float((brier < 0).mean()),
        },
    }


def _aggregate(fits: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Mean/std of the headline validation metrics across seeds."""
    keys = ("roc_auc", "brier_score", "log_loss", "accuracy_at_0_5")
    summary: Dict[str, Any] = {"seeds": [fit["seed"] for fit in fits]}
    for key in keys:
        values = np.array(
            [fit["validation"]["inference"]["metrics"][key] for fit in fits], dtype=np.float64
        )
        summary[f"{key}_mean"] = float(values.mean())
        summary[f"{key}_std"] = float(values.std())
    eces = np.array(
        [fit["validation"]["inference"]["metrics"]["calibration"]["ece"] for fit in fits],
        dtype=np.float64,
    )
    summary["ece_mean"] = float(eces.mean())
    summary["ece_std"] = float(eces.std())
    summary["fit_seconds_mean"] = float(np.mean([fit["fit_seconds"] for fit in fits]))
    summary["predict_ms_per_1k_mean"] = float(
        np.mean([fit["predict_ms_per_1k"] for fit in fits])
    )
    summary["partial_draft_auc_mean"] = {
        level: float(
            np.mean([fit["partial_draft"][level]["metrics"]["roc_auc"] for fit in fits])
        )
        for level in fits[0]["partial_draft"]
    }
    return summary


def policy_status(metrics: Mapping[str, Any], policy: QualityPolicy) -> Dict[str, Any]:
    """Apply the promotion policy's numeric floor to one metric block.

    This is *not* a promotion decision: the gate applies the same policy to the
    frozen test partition of an already-selected candidate. A family can be the
    best candidate here and still fail these thresholds, and the report says so
    in both places.
    """
    rows = int(metrics["rows"])
    checks = {
        "roc_auc": {
            "value": float(metrics["roc_auc"]),
            "threshold": policy.min_roc_auc,
            "comparison": ">=",
            "pass": bool(metrics["roc_auc"] >= policy.min_roc_auc),
        },
        "brier_score": {
            "value": float(metrics["brier_score"]),
            "threshold": policy.max_brier,
            "comparison": "<=",
            "pass": bool(metrics["brier_score"] <= policy.max_brier),
        },
        "ece": {
            "value": float(metrics["calibration"]["ece"]),
            "threshold": policy.max_ece,
            "comparison": "<=",
            "pass": bool(metrics["calibration"]["ece"] <= policy.max_ece),
        },
        "rows": {
            "value": rows,
            "threshold": policy.min_test_rows,
            "comparison": ">=",
            "pass": bool(rows >= policy.min_test_rows),
        },
    }
    return {
        "evaluated_on": "validation",
        "policy": policy.to_dict(),
        "checks": checks,
        "passes_numeric_floor": all(check["pass"] for check in checks.values()),
        "note": (
            "Numeric-floor check on validation only. Promotion additionally "
            "requires quality_gate.py to pass on the frozen test partition."
        ),
    }


def decide(
    families: Sequence[Mapping[str, Any]],
    comparisons: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Apply the pre-registered selection rule (validation metrics only)."""
    ranked = sorted(
        families,
        key=lambda family: (
            -family["aggregate"]["roc_auc_mean"],
            family["aggregate"]["brier_score_mean"],
            COMPLEXITY[family["name"]],
        ),
    )
    leader_auc = ranked[0]["aggregate"]["roc_auc_mean"]
    near = [
        family
        for family in ranked
        if leader_auc - family["aggregate"]["roc_auc_mean"] <= MATERIAL_AUC_MARGIN
    ]
    winner = min(
        near,
        key=lambda family: (
            family["aggregate"]["brier_score_mean"],
            COMPLEXITY[family["name"]],
            -family["aggregate"]["roc_auc_mean"],
        ),
    )
    reference = next(family for family in families if family["name"] == REFERENCE)
    comparison = next(
        (row for row in comparisons if row["candidate"] == winner["name"]), None
    )
    material = False
    if comparison is not None:
        material = bool(
            comparison["auc_diff"]["ci95"][0] > 0.0
            and winner["aggregate"]["brier_score_mean"]
            < reference["aggregate"]["brier_score_mean"] - MATERIAL_BRIER_MARGIN
        )
    return {
        "rule": DECISION_RULE,
        "ranked": [family["name"] for family in ranked],
        "selected": winner["name"],
        "selected_auc_mean": winner["aggregate"]["roc_auc_mean"],
        "selected_brier_mean": winner["aggregate"]["brier_score_mean"],
        "reference": REFERENCE,
        "reference_auc_mean": reference["aggregate"]["roc_auc_mean"],
        "reference_brier_mean": reference["aggregate"]["brier_score_mean"],
        "material_improvement_over_reference": material,
        "rationale": (
            "The selected family is a material improvement."
            if material
            else "Families are statistically indistinguishable; keep the reference."
            if winner["name"] == REFERENCE
            else "Best on validation, but the difference is within noise; no switch justified."
        ),
    }


def _to_float_input(prototype: Any) -> Any:
    """Make the ONNX graph accept `tensor(float)`, as the runtime sends.

    sklearn's histogram GBDT converter asks for float64. The Rust runtime and
    `quality_gate.py` both feed a single float32 tensor, so the graph gets an
    explicit Cast instead of a second input contract.
    """
    import onnx
    from onnx import TensorProto, helper

    graph = prototype.graph
    model_input = graph.input[0]
    if model_input.type.tensor_type.elem_type == TensorProto.FLOAT:
        return prototype
    cast_name = f"{model_input.name}_f64"
    graph.node.insert(
        0, helper.make_node("Cast", [model_input.name], [cast_name], to=TensorProto.DOUBLE)
    )
    for node in graph.node:
        for index, name in enumerate(node.input):
            if name == model_input.name:
                node.input[index] = cast_name
    model_input.type.tensor_type.elem_type = TensorProto.FLOAT
    return onnx.shape_inference.infer_shapes(prototype)


def _rename_outputs(prototype: Any, schema: Mapping[str, Any]) -> Any:
    """Name the probability/label outputs as the schema (and runtime) expect."""
    probability_name = str(schema.get("probability_output", "probabilities"))
    label_name = str(schema.get("label_output", "label"))
    graph = prototype.graph
    probability = None
    label = None
    for output in graph.output:
        shape = output.type.tensor_type.shape
        if len(shape.dim) == 2 and probability is None:
            probability = output
        elif len(shape.dim) == 1 and label is None:
            label = output
    if probability is None:
        probability = graph.output[-1]
    probability.name = probability_name
    remaining = [output for output in graph.output if output is not probability]
    if remaining:
        remaining[0].name = label_name
        for extra in remaining[1:]:
            graph.output.remove(extra)
    return prototype


def _convert_sklearn_model(model: Any, feature_count: int) -> Any:
    """Convert a fitted sklearn model to ONNX with a tensor probability output.

    `zipmap=False` is requested for both the wrapper and (for a pipeline) the
    final estimator, because skl2onnx resolves that option by object id and the
    accepted id differs between a bare estimator and a pipeline.
    """
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType

    initial_types = [("float_input", FloatTensorType([None, feature_count]))]
    option_sets = [{id(model): {"zipmap": False}}]
    steps = getattr(model, "steps", None)
    if steps:
        option_sets.append({id(steps[-1][1]): {"zipmap": False}})
        option_sets.append(
            {id(model): {"zipmap": False}, id(steps[-1][1]): {"zipmap": False}}
        )
    last_error: Optional[Exception] = None
    for options in option_sets:
        try:
            return convert_sklearn(
                model,
                initial_types=initial_types,
                options=options,
                target_opset=15,
            )
        except Exception as error:  # try the next option shape
            last_error = error
    raise RuntimeError(f"skl2onnx conversion failed: {last_error}")


def export_family_onnx(
    name: str,
    model: Any,
    schema: Mapping[str, Any],
    *,
    out_dir: Path,
    stem: str,
    matrices: Mapping[str, np.ndarray],
    write_contract: bool,
) -> Dict[str, Any]:
    """Export one fitted model to ONNX and verify parity with the source model.

    `write_contract` also writes the schema/feature/config files, so a selected
    family can later be passed to `quality_gate.py` unchanged. The bundled model
    is never touched.
    """
    import onnx
    from onnxruntime import InferenceSession

    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / f"{stem}.onnx"
    feature_names = list(schema["feature_names"])
    record: Dict[str, Any] = {
        "family": name,
        "stem": stem,
        "onnx": str(onnx_path),
        "exported": False,
    }

    try:
        if name == "xgboost_combined":
            _export_onnx(model, onnx_path, len(feature_names))
        else:
            prototype = _convert_sklearn_model(model, len(feature_names))
            prototype = _rename_outputs(_to_float_input(prototype), schema)
            onnx.checker.check_model(prototype)
            onnx.save(prototype, onnx_path)
    except Exception as error:  # optional evidence, never a hard failure
        # skl2onnx dumps whole attribute arrays into its failure message, so keep
        # only the leading explanation; the failure itself is what matters.
        message = str(error).split(" with name=")[0].strip()
        record["error"] = f"{type(error).__name__}: {message[:400]}"
        return record

    session = InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    record.update(
        {
            "exported": True,
            "input_names": [item.name for item in inputs],
            "input_shape": [str(dim) for dim in inputs[0].shape],
            "input_type": inputs[0].type,
            "output_names": [item.name for item in outputs],
            "output_shapes": [[str(dim) for dim in item.shape] for item in outputs],
        }
    )
    if len(inputs) != 1 or len(inputs[0].shape) != 2:
        record["error"] = "unexpected ONNX input signature"
        record["exported"] = False
        return record
    if inputs[0].shape[1] not in (None, len(feature_names)):
        record["error"] = "ONNX input feature count does not match the schema"
        record["exported"] = False
        return record

    positive_index = int(schema.get("positive_class_index", 1))
    parity: Dict[str, Any] = {}
    for stage, matrix in matrices.items():
        fed = np.ascontiguousarray(matrix, dtype=np.float32)
        raw = session.run(None, {inputs[0].name: fed})
        candidate = None
        for value in raw:
            array = np.asarray(value)
            if array.ndim == 2 and array.shape[0] == len(fed) and array.shape[1] > positive_index:
                candidate = array[:, positive_index].astype(np.float64)
                break
        if candidate is None:
            record["error"] = "ONNX graph exposes no class-probability output"
            record["exported"] = False
            return record
        reference = _predict(model, matrix)
        diff = float(np.max(np.abs(reference - candidate)))
        parity[stage] = {
            "max_abs_diff": diff,
            "rows": int(len(fed)),
            "within_1e_4": bool(diff < 1e-4),
        }

    record["parity"] = parity
    record["max_abs_diff"] = max(entry["max_abs_diff"] for entry in parity.values())
    record["parity_ok"] = all(entry["within_1e_4"] for entry in parity.values())

    if write_contract:
        (out_dir / f"{stem}_schema.json").write_text(
            json.dumps(schema, indent=2) + "\n", encoding="utf-8"
        )
        (out_dir / f"{stem}_features.json").write_text(
            json.dumps(feature_names, indent=None), encoding="utf-8"
        )
    return record


def run(args: argparse.Namespace) -> int:
    started = time.time()
    data_path = Path(args.data)
    schema_path = Path(args.schema)
    population_label = "all_rows" if args.population == "all" else "no_duplicate_heroes"
    report_path = Path(args.report) if args.report else (
        DEFAULT_MODEL_DIR / f"family_comparison_{population_label}.json"
    )
    seeds = tuple(int(part) for part in str(args.seeds).split(",") if part.strip())
    families = [part.strip() for part in str(args.families).split(",") if part.strip()]
    unknown = sorted(set(families) - set(FAMILY_NAMES))
    if unknown:
        raise SystemExit(f"unknown families {unknown}; known: {list(FAMILY_NAMES)}")
    if REFERENCE not in families:
        raise SystemExit(
            f"{REFERENCE} must be included: it is the reference every comparison is made against"
        )

    base_schema = json.loads(schema_path.read_text(encoding="utf-8"))
    base_names = list(base_schema["feature_names"])
    variant = next(v for v in ablation.VARIANTS if v.name == FEATURE_VARIANT)
    feature_names = ablation.variant_feature_names(base_names, variant)
    schema = ablation.variant_schema(base_schema, feature_names)
    dropped = [name for name in base_names if name not in set(feature_names)]

    dataset = pd.read_parquet(data_path)
    if "radiant_win" not in dataset.columns:
        raise SystemExit(f"{data_path} has no radiant_win label column")
    labels = dataset["radiant_win"]
    if labels.isna().any() or not set(pd.unique(labels.dropna())).issubset({0, 1}):
        raise SystemExit("radiant_win must be binary 0/1 with no missing values")

    data_sha = file_sha256(data_path)
    train, val, test, split_info = ablation.load_frozen_split(
        dataset, Path(args.split_config), data_path=data_path
    )
    overlaps = {
        "train_validation": len(set(train["match_id"]) & set(val["match_id"])),
        "train_test": len(set(train["match_id"]) & set(test["match_id"])),
        "validation_test": len(set(val["match_id"]) & set(test["match_id"])),
    }
    if any(overlaps.values()):
        raise SystemExit(f"match_id overlap between partitions: {overlaps}")

    population: Dict[str, Any] = {
        "label": population_label,
        "rule": None,
        "excluded": {"train": 0, "validation": 0, "test": 0},
    }
    if args.population == "no_duplicates":
        population["rule"] = ablation.DUPLICATE_HERO_RULE
        population["flagged_dataset_rows"] = int(ablation.duplicate_hero_mask(dataset).sum())
        filtered = {}
        for name, part in (("train", train), ("validation", val), ("test", test)):
            mask = ablation.duplicate_hero_mask(part)
            population["excluded"][name] = int(mask.sum())
            filtered[name] = part.loc[~mask]
        train, val, test = filtered["train"], filtered["validation"], filtered["test"]

    print(f"Dataset {len(dataset)} x {dataset.shape[1]}  sha256={data_sha[:12]}...")
    print(f"Split {split_info['sizes']} (source={split_info['source']})")
    print(f"Population {population_label}: excluded {population['excluded']}")
    print(f"Features {len(feature_names)} (dropped {len(dropped)}): {FEATURE_VARIANT}")

    y_train = train["radiant_win"].astype(int).to_numpy()
    y_val = val["radiant_win"].astype(int).to_numpy()
    train_prior = float(y_train.mean())

    X_train = ablation.encode(train, schema, inference_defaults=False)
    X_val_at = ablation.encode(val, schema, inference_defaults=False)
    X_val_inf = ablation.encode(val, schema, inference_defaults=True)
    print(
        f"Train {X_train.shape} positives={int(y_train.sum())} "
        f"({train_prior:.4f}); validation {X_val_inf.shape} "
        f"positives={int(y_val.sum())} ({y_val.mean():.4f})"
    )

    processor = ablation._load_processor(data_path.parent)
    print("Rebuilding partial-draft validation frames...")
    partial_matrices: Dict[int, np.ndarray] = {}
    partial_labels: Dict[int, np.ndarray] = {}
    for level in PROGRESS_LEVELS:
        frame = ablation.build_partial_draft_frame(
            val, level, schema=base_schema, processor=processor
        )
        partial_matrices[level] = ablation.encode(frame, schema, inference_defaults=True)
        partial_labels[level] = frame["radiant_win"].astype(int).to_numpy()
        print(f"  {level} picks/side: {partial_matrices[level].shape}")

    xgb_config = dict(WIDE_CONFIG)
    xgb_config.update(
        {
            "device": "cpu",
            "nthread": int(args.threads),
            "verbosity": 0,
        }
    )
    if args.max_estimators:
        xgb_config["n_estimators"] = int(args.max_estimators)

    logistic_c = None
    logistic_grid: List[Dict[str, Any]] = []
    if "logistic_l2" in families:
        print("\n[logistic_l2] selecting C on validation AUC...")
        logistic_c, logistic_grid = select_logistic_c(
            X_train,
            y_train,
            X_val_inf,
            y_val,
            seed=seeds[0],
            calibration_bins=args.calibration_bins,
        )
        for row in logistic_grid:
            print(
                f"  C={row['C']:<6g} AUC={row['validation_inference_roc_auc']:.4f} "
                f"Brier={row['validation_inference_brier']:.4f} "
                f"({row['fit_seconds']:.1f}s)"
            )
        print(f"  selected C={logistic_c:g}")

    predictions: Dict[Tuple[str, int], Dict[str, np.ndarray]] = {}
    primary_models: Dict[str, Any] = {}
    results: List[Dict[str, Any]] = []

    for name in families:
        print(f"\n[{name}]")
        fits: List[Dict[str, Any]] = []
        for seed in seeds:
            start = time.time()
            info: Dict[str, Any] = {}
            if name == "xgboost_combined":
                model, fit_info = ablation.fit_variant(
                    train, val, schema, xgb_config, seed=seed
                )
                fit_seconds = fit_info.seconds
                info = {
                    "rounds": fit_info.rounds,
                    "probe_best_iteration": fit_info.probe_best_iteration,
                    "probe_rounds_run": fit_info.probe_rounds_run,
                }
            elif name == "hist_gbdt":
                model = fit_hist_gbdt(
                    X_train, y_train, seed=seed, max_iter=args.hist_max_iter
                )
                fit_seconds = time.time() - start
                info = {
                    "iterations": int(model.n_iter_),
                    "max_iter": int(model.max_iter),
                    "early_stopping": True,
                    "stopping_set": "internal 10% split of the training rows",
                }
            elif name == "logistic_l2":
                model = fit_logistic(X_train, y_train, C=float(logistic_c), seed=seed)
                fit_seconds = time.time() - start
                info = {
                    "C": float(logistic_c),
                    "iterations": int(np.max(model.named_steps["clf"].n_iter_)),
                    "scaler": "StandardScaler fitted on training rows only",
                }
            else:  # pragma: no cover - guarded by the CLI validation above
                raise SystemExit(f"unhandled family {name}")

            latency_start = time.perf_counter()
            val_inference = _predict(model, X_val_inf)
            latency_seconds = time.perf_counter() - latency_start

            validation = {
                "as_trained": _metrics(
                    lambda matrix, m=model: _predict(m, matrix),
                    X_val_at,
                    y_val,
                    train_prior=train_prior,
                    calibration_bins=args.calibration_bins,
                ),
                "inference": {
                    "metrics": compute_metrics(
                        y_val, val_inference, calibration_bins=args.calibration_bins
                    ),
                    "baselines": baseline_metrics(y_val, train_prior),
                },
            }
            partial_draft = {
                str(level): {
                    "metrics": compute_metrics(
                        partial_labels[level],
                        _predict(model, partial_matrices[level]),
                        calibration_bins=args.calibration_bins,
                    ),
                    "baselines": baseline_metrics(partial_labels[level], train_prior),
                }
                for level in PROGRESS_LEVELS
            }
            predictions[(name, seed)] = {
                "validation": val_inference,
                **{f"partial_{level}": _predict(model, partial_matrices[level])
                   for level in PROGRESS_LEVELS},
            }
            if seed == seeds[0]:
                primary_models[name] = model

            metrics = validation["inference"]["metrics"]
            print(
                f"  seed {seed}: AUC={metrics['roc_auc']:.4f} "
                f"(CI {metrics['roc_auc_ci95']['lower']:.3f}-{metrics['roc_auc_ci95']['upper']:.3f}) "
                f"Brier={metrics['brier_score']:.4f} ECE={metrics['calibration']['ece']:.4f} "
                f"fit={fit_seconds:.1f}s pred={latency_seconds * 1000:.0f}ms "
                + " ".join(
                    f"{level}={partial_draft[str(level)]['metrics']['roc_auc']:.3f}"
                    for level in PROGRESS_LEVELS
                )
            )
            fits.append(
                {
                    "seed": int(seed),
                    "fit_seconds": float(fit_seconds),
                    "predict_seconds": float(latency_seconds),
                    "predict_ms_per_1k": float(latency_seconds * 1e6 / max(len(y_val), 1)),
                    "fit_info": info,
                    "validation": validation,
                    "partial_draft": partial_draft,
                }
            )
        results.append(
            {
                "name": name,
                "kind": name,
                "complexity_rank": COMPLEXITY[name],
                "config": (
                    dict(xgb_config)
                    if name == "xgboost_combined"
                    else dict(HISTGB_CONFIG, max_iter=args.hist_max_iter or HISTGB_CONFIG["max_iter"])
                    if name == "hist_gbdt"
                    else {"C": float(logistic_c), "solver": "lbfgs", "max_iter": LOGISTIC_MAX_ITER}
                ),
                "fits": fits,
                "aggregate": _aggregate(fits),
            }
        )

    if "logistic_l2" in families:
        # lbfgs is deterministic: the seed must not move the predictions. Prove
        # it rather than assuming it.
        primary = predictions[("logistic_l2", seeds[0])]["validation"]
        drift = max(
            float(np.max(np.abs(predictions[("logistic_l2", seed)]["validation"] - primary)))
            for seed in seeds[1:]
        ) if len(seeds) > 1 else 0.0
        next(family for family in results if family["name"] == "logistic_l2")[
            "seed_invariance_max_abs_diff"
        ] = drift
        print(f"\nlogistic seed invariance: max|diff|={drift:.3e} across seeds")

    comparisons: List[Dict[str, Any]] = []
    reference_predictions = predictions[(REFERENCE, seeds[0])]["validation"]
    for name in families:
        if name == REFERENCE:
            continue
        candidate_predictions = predictions[(name, seeds[0])]["validation"]
        row = {
            "candidate": name,
            "reference": REFERENCE,
            "rows": int(len(y_val)),
            "seed": int(seeds[0]),
            "auc_reference": float(roc_auc_score(y_val, reference_predictions)),
            "auc_candidate": float(roc_auc_score(y_val, candidate_predictions)),
            **paired_bootstrap(
                y_val,
                reference_predictions,
                candidate_predictions,
                replicates=args.bootstrap,
                seed=0,
            ),
        }
        comparisons.append(row)
        print(
            f"\n{name} vs {REFERENCE}: AUC diff {row['auc_diff']['mean']:+.4f} "
            f"(95% CI {row['auc_diff']['ci95'][0]:+.4f}..{row['auc_diff']['ci95'][1]:+.4f}), "
            f"Brier diff {row['brier_diff']['mean']:+.4f} "
            f"(95% CI {row['brier_diff']['ci95'][0]:+.4f}..{row['brier_diff']['ci95'][1]:+.4f})"
        )

    decision = decide(results, comparisons)
    policy = QualityPolicy(calibration_bins=args.calibration_bins)
    policy_by_family = {
        family["name"]: policy_status(family["fits"][0]["validation"]["inference"]["metrics"], policy)
        for family in results
    }
    selected = next(family for family in results if family["name"] == decision["selected"])

    exports: List[Dict[str, Any]] = []
    if not args.no_export:
        print("\nONNX export and parity (bundled model untouched)...")
        for family in results:
            name = family["name"]
            stem = f"family_{name}"
            matrices = {
                "validation_full_draft": X_val_inf,
                "partial_0_picks": partial_matrices[0],
                "partial_1_pick": partial_matrices[1],
                "partial_3_picks": partial_matrices[3],
            }
            record = export_family_onnx(
                name,
                primary_models[name],
                schema,
                out_dir=Path(args.model_dir) / "onnx" / stem,
                stem=stem,
                matrices=matrices,
                write_contract=(name == decision["selected"]),
            )
            exports.append(record)
            if record.get("exported"):
                print(
                    f"  {name}: parity max|diff|={record['max_abs_diff']:.3e} "
                    f"input={record['input_type']} outputs={record['output_names']}"
                )
            else:
                print(f"  {name}: NOT EXPORTED ({record.get('error', 'unknown')})")

    report = {
        "experiment_version": 1,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "purpose": (
            "Is model family a bottleneck for draft-time win prediction? Feature "
            "set, split, population, and evaluation protocol held fixed."
        ),
        "dataset": {
            "path": str(data_path),
            "sha256": data_sha,
            "rows": int(len(dataset)),
            "label_counts": {
                "radiant_win_1": int((dataset["radiant_win"] == 1).sum()),
                "radiant_win_0": int((dataset["radiant_win"] == 0).sum()),
            },
        },
        "split": {
            **split_info,
            "match_id_overlaps": overlaps,
            "test_scored_here": False,
            "train_positive_rate": train_prior,
            "train_rows": int(len(train)),
            "validation_rows": int(len(val)),
        },
        "population": {
            **population,
            "train_label_counts": {
                "radiant_win_1": int((train["radiant_win"] == 1).sum()),
                "radiant_win_0": int((train["radiant_win"] == 0).sum()),
            },
            "validation_label_counts": {
                "radiant_win_1": int((val["radiant_win"] == 1).sum()),
                "radiant_win_0": int((val["radiant_win"] == 0).sum()),
            },
        },
        "feature_set": {
            "variant": FEATURE_VARIANT,
            "description": variant.description,
            "feature_count": len(feature_names),
            "schema_fingerprint": schema_fingerprint(feature_names),
            "dropped_features": dropped,
            "features": feature_names,
        },
        "partial_draft_method": (
            "Each validation row is regenerated by the production feature code from "
            "a draft truncated to k picks per side, then encoded with runtime "
            "defaults; per-hero win rates, aggregates, semantics, synergy and damage "
            "diversity only contain known picks. Bans stay at full-draft values."
        ),
        "protocol": {
            "selection_partition": "validation",
            "test_partition": "not read, not scored, not used for selection",
            "seeds": list(seeds),
            "logistic_c_grid": list(LOGISTIC_C_GRID),
            "logistic_selected_c": logistic_c,
            "logistic_grid": logistic_grid,
            "bootstrap_replicates": int(args.bootstrap),
            "calibration_bins": int(args.calibration_bins),
            "decision_rule": DECISION_RULE,
        },
        "families": results,
        "comparisons": comparisons,
        "decision": decision,
        "policy": {
            "thresholds": policy.to_dict(),
            "by_family": policy_by_family,
            "selected_family": decision["selected"],
            "selected_passes_numeric_floor": policy_by_family[decision["selected"]][
                "passes_numeric_floor"
            ],
            "promotion": False,
        },
        "onnx_export": exports,
        "limitations": list(LIMITATIONS),
        "elapsed_seconds": time.time() - started,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print("\n" + "=" * 72)
    for family in results:
        aggregate = family["aggregate"]
        print(
            f"{family['name']:<20} AUC {aggregate['roc_auc_mean']:.4f} "
            f"(±{aggregate['roc_auc_std']:.4f})  Brier {aggregate['brier_score_mean']:.4f} "
            f"(±{aggregate['brier_score_std']:.4f})  ECE {aggregate['ece_mean']:.4f}"
        )
    print(f"Selected: {decision['selected']}  material={decision['material_improvement_over_reference']}")
    print(
        "Policy numeric floor (validation): "
        + ", ".join(
            f"{name}={'pass' if status['passes_numeric_floor'] else 'fail'}"
            for name, status in policy_by_family.items()
        )
    )
    print(f"Report: {report_path}")
    print(f"Total {time.time() - started:.1f}s")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare model families on the corrected dataset (validation only)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument(
        "--split-config",
        type=Path,
        default=DEFAULT_SPLIT_CONFIG,
        help="frozen split config; required, because a re-cut split is not the protocol",
    )
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument(
        "--population",
        choices=("all", "no_duplicates"),
        default="all",
        help="all rows (primary) or rows without duplicate heroes (sensitivity)",
    )
    parser.add_argument("--families", type=str, default=",".join(FAMILY_NAMES))
    parser.add_argument("--seeds", type=str, default=",".join(str(seed) for seed in SEEDS))
    parser.add_argument("--bootstrap", type=int, default=DEFAULT_BOOTSTRAP)
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    parser.add_argument("--calibration-bins", type=int, default=10)
    parser.add_argument(
        "--max-estimators",
        type=int,
        default=None,
        help="override the XGBoost round cap (smoke runs only)",
    )
    parser.add_argument(
        "--hist-max-iter",
        type=int,
        default=None,
        help="override the histogram GBDT iteration cap (smoke runs only)",
    )
    parser.add_argument(
        "--no-export",
        action="store_true",
        help="skip ONNX export and parity checks",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
