#!/usr/bin/env python3
"""
Train the full feature set and export ONNX with a reproducibility manifest.

Uses every column in the processed parquet as a model input except identity and
label columns (`match_id`, `radiant_win`, `start_time`) and feature families the
Rust runtime cannot reconstruct (`rad_lane_*`, `dire_lane_*`, `emb_*`), keeping
pick-order and match-time columns so the exported model matches the runtime
feature contract (see `ml.rs build_feature_map`).

The defaults target the bundled runtime contract: `draft_features.parquet`
(887 non-embedding features) exported as `draft_model_xgboost_full`. The
embedding dataset is a separate experiment and is never the default.

Validation is used only for early stopping; the test partition is evaluated
exactly once, after model selection, and is never seen during training.

Usage:
    python training/train_full.py
    python training/train_full.py --data ml/datasets/processed/draft_features.parquet
    python training/train_full.py --evaluate-only
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

ML_ROOT = Path(__file__).resolve().parent.parent
if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from training.manifest import build_manifest, manifest_paths, write_manifest
from training.splits import file_sha256, schema_fingerprint, time_split

DEFAULT_DATA = ML_ROOT / "datasets" / "processed" / "draft_features.parquet"
DEFAULT_MODEL_DIR = ML_ROOT / "models"
DEFAULT_MODEL_STEM = "draft_model_xgboost_full"
DEFAULT_PATCH = "7.41"

# Identity, label, and time columns are never model inputs.
NON_FEATURE_COLUMNS = {"match_id", "radiant_win", "start_time", "start_time_utc"}

# Feature families that `ml.rs build_feature_map` never emits. A model trained
# with them would silently fall back to schema defaults at inference time, so
# they are excluded from the exported contract.
RUNTIME_UNSUPPORTED_PREFIXES = ("rad_lane_", "dire_lane_", "emb_")

WIDE_CONFIG = {
    "objective": "binary:logistic",
    "eval_metric": ["auc", "logloss"],
    "n_estimators": 2500,
    "max_depth": 4,
    "learning_rate": 0.02,
    "subsample": 0.75,
    "colsample_bytree": 0.5,
    "min_child_weight": 30,
    "gamma": 0.5,
    "reg_alpha": 2.0,
    "reg_lambda": 5.0,
    "early_stopping_rounds": 200,
    "random_state": 42,
    "tree_method": "hist",
    "verbosity": 1,
}


def select_feature_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Return `(feature_columns, categorical_columns)` for the dataset.

    Columns the Rust runtime cannot reconstruct from a live draft are removed so
    the exported model always matches the runtime feature contract.
    """
    features = [
        c
        for c in df.columns
        if c not in NON_FEATURE_COLUMNS
        and not c.startswith(RUNTIME_UNSUPPORTED_PREFIXES)
    ]
    categoricals = [c for c in features if isinstance(df[c].dtype, pd.CategoricalDtype)]
    return features, categoricals


def encode_features(df: pd.DataFrame, feature_cols: Sequence[str], cat_cols: Sequence[str]):
    """Return the numeric feature matrix, encoding categoricals as integer codes."""
    frame = df[list(feature_cols)].copy()
    for column in cat_cols:
        frame[column] = frame[column].cat.codes
    return frame.fillna(0).to_numpy(dtype=np.float32)


def evaluate(model: "xgb.XGBClassifier", X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    probabilities = model.predict_proba(X)[:, 1]
    return {
        "roc_auc": float(roc_auc_score(y, probabilities)),
        "log_loss": float(log_loss(y, probabilities)),
        "brier_score": float(brier_score_loss(y, probabilities)),
        "accuracy": float(((probabilities > 0.5).astype(int) == y).mean()),
        "rows": int(len(y)),
    }


def _load_config(args: argparse.Namespace) -> Dict:
    config = dict(WIDE_CONFIG)
    if args.config and Path(args.config).exists():
        config.update(json.loads(Path(args.config).read_text(encoding="utf-8")))
    if args.n_estimators is not None:
        config["n_estimators"] = args.n_estimators
    config["random_state"] = args.seed
    return config


def _load_existing_model(path: Path) -> "xgb.XGBClassifier":
    if not path.exists():
        raise SystemExit(f"--skip-train/--evaluate-only requested but {path} does not exist")
    model = xgb.XGBClassifier()
    model.load_model(str(path))
    return model


def run(args: argparse.Namespace) -> int:
    start = time.time()
    data_path = Path(args.data)
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    paths = manifest_paths(model_dir, args.model_stem)

    df = pd.read_parquet(data_path)
    print(f"Loaded {df.shape[0]} samples x {df.shape[1]} columns from {data_path}")
    if "radiant_win" not in df.columns:
        raise SystemExit(f"{data_path} has no radiant_win label column")

    feature_cols, cat_cols = select_feature_columns(df)
    excluded = [
        c
        for c in df.columns
        if c in NON_FEATURE_COLUMNS or c.startswith(RUNTIME_UNSUPPORTED_PREFIXES)
    ]
    if cat_cols:
        print(f"Categorical columns encoded: {cat_cols}")
    print(f"Features: {len(feature_cols)} (excluded {len(excluded)})")

    split = time_split(
        df,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
    )
    if not split.used_start_time:
        print(
            "WARNING: start_time absent; falling back to match_id as the "
            "chronological key (re-export features to add start_time)."
        )
        print(
            "Split by {col}: train={train}, val={validation}, test={test}".format(
                col=split.time_column, **split.sizes()
            )
        )

    y_train = split.train["radiant_win"].astype(int).to_numpy()
    y_val = split.val["radiant_win"].astype(int).to_numpy()
    y_test = split.test["radiant_win"].astype(int).to_numpy()
    X_train = encode_features(split.train, feature_cols, cat_cols)
    X_val = encode_features(split.val, feature_cols, cat_cols)
    X_test = encode_features(split.test, feature_cols, cat_cols)

    config = _load_config(args)
    skip_fit = args.skip_train or args.evaluate_only

    use_gpu = bool(xgb.build_info().get("USE_CUDA", False))
    if use_gpu:
        config.setdefault("device", "cuda")

    if skip_fit:
        print(f"Skipping training; loading existing model {paths['joblib']}")
        model = _load_existing_model(paths["joblib"])
        expected = getattr(model, "n_features_in_", None)
        if expected is not None and int(expected) != len(feature_cols):
            raise SystemExit(
                f"{paths['joblib']} expects {int(expected)} features but the dataset "
                f"selects {len(feature_cols)}; use the matching --data/--model-stem pair "
                "(the bundled contract is draft_features.parquet + draft_model_xgboost_full)."
            )
        best_iteration: Optional[int] = None
    else:
        print("Training on CUDA GPU" if use_gpu else "Training on CPU")
        model = xgb.XGBClassifier(**config)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=100)
        best_iteration = getattr(model, "best_iteration", None)

    val_metrics = evaluate(model, X_val, y_val)
    test_metrics = evaluate(model, X_test, y_test)
    print(
        "Validation: ROC-AUC={roc_auc:.4f} LogLoss={log_loss:.4f} "
        "Brier={brier_score:.4f}".format(**val_metrics)
    )
    print("Test (held out): ROC-AUC={roc_auc:.4f} LogLoss={log_loss:.4f} "
          "Brier={brier_score:.4f}".format(**test_metrics))

    if not args.evaluate_only:
        model.save_model(str(paths["joblib"]))
        print(f"Saved model: {paths['joblib']}")

        _export_onnx(model, paths["onnx"], len(feature_cols))

        paths["features"].write_text(json.dumps(feature_cols), encoding="utf-8")
        print(f"Saved features: {paths['features']}")

        _export_schema(data_path, paths, args.current_patch)
        _write_config(paths["config"], len(feature_cols), config, val_metrics, test_metrics)

    manifest = build_manifest(
        model_stem=args.model_stem,
        model_type="xgboost",
        dataset_path=data_path,
        dataset_sha256=file_sha256(data_path),
        dataset_rows=len(df),
        feature_names=feature_cols,
        schema_fingerprint=schema_fingerprint(feature_cols),
        split_info=split.info(),
        hyperparameters=config,
        metrics={"validation": val_metrics, "test": test_metrics},
        patch_version=args.current_patch,
        best_iteration=best_iteration,
        artifacts={
            "joblib": str(paths["joblib"].name),
            "onnx": str(paths["onnx"].name),
            "features": str(paths["features"].name),
            "schema": f"{args.model_stem}_schema.json",
        },
    )
    write_manifest(paths["manifest"], manifest)
    print(f"Saved manifest: {paths['manifest']}")
    print(f"Done in {time.time() - start:.1f}s")
    return 0


def _export_onnx(model: "xgb.XGBClassifier", onnx_path: Path, feature_count: int) -> None:
    try:
        from onnxmltools import convert_xgboost
        from onnxmltools.convert.common.data_types import FloatTensorType
    except ImportError as error:
        print(f"ONNX export skipped (onnxmltools missing): {error}")
        return

    initial_types = [("float_input", FloatTensorType([None, feature_count]))]
    try:
        onnx_model = convert_xgboost(
            model, name="draft_model_full", initial_types=initial_types, target_opset=14
        )
        onnx_path.write_bytes(onnx_model.SerializeToString())
        print(f"Saved ONNX: {onnx_path}")
    except Exception as error:  # pragma: no cover - depends on onnxmltools internals
        print(f"ONNX export failed: {error}")


def _export_schema(data_path: Path, paths: Dict[str, Path], current_patch: str) -> None:
    try:
        from training.export_model_schema import build_schema
    except Exception as error:  # pragma: no cover - optional dependency path
        print(f"Schema export skipped: {error}")
        return

    schema_path = paths["features"].with_name(paths["features"].stem.removesuffix("_features") + "_schema.json")
    try:
        schema = build_schema(data_path, paths["features"], current_patch)
    except Exception as error:
        print(f"Schema export skipped: {error}")
        return
    schema_path.write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")
    print(f"Saved schema: {schema_path}")


def _write_config(
    config_path: Path,
    feature_count: int,
    config: Dict,
    val_metrics: Dict,
    test_metrics: Dict,
) -> None:
    config_path.write_text(
        json.dumps(
            {
                "model_type": "xgboost",
                "feature_count": feature_count,
                "config": config,
                "metrics": {"validation": val_metrics, "test": test_metrics},
                "training_date": time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Saved config: {config_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the full feature model and export ONNX + manifest",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data", type=str, default=str(DEFAULT_DATA))
    parser.add_argument("--model-dir", type=str, default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--model-stem", type=str, default=DEFAULT_MODEL_STEM)
    parser.add_argument("--current-patch", type=str, default=DEFAULT_PATCH)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--test-frac", type=float, default=0.1)
    parser.add_argument("--n-estimators", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", type=str, default=None, help="JSON hyperparameter overrides")
    parser.add_argument(
        "--skip-train",
        action="store_true",
        default=False,
        help="Reuse an existing model instead of fitting; still re-exports artifacts",
    )
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        default=False,
        help="Reuse an existing model and only report metrics/manifest (no artifact writes)",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.evaluate_only:
        args.skip_train = True
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
