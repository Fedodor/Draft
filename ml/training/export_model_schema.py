#!/usr/bin/env python3
"""Export a runtime feature schema for a trained ONNX model.

The schema records the exact feature order, categorical encodings, and
per-feature fallback values used by the Rust runtime. It is generated from the
same processed parquet used for training; it does not retrain the model.

Usage:
    python training/export_model_schema.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

ML_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = ML_ROOT / "datasets" / "processed" / "draft_features_emb.parquet"
DEFAULT_FEATURES = ML_ROOT / "models" / "draft_model_xgboost_full_features.json"
DEFAULT_OUTPUT = ML_ROOT / "models" / "draft_model_xgboost_full_schema.json"


def _relative(path: Path) -> str:
    try:
        return str(path.relative_to(ML_ROOT))
    except ValueError:
        return str(path)


def build_schema(
    data_path: Path,
    features_path: Path,
    current_patch: str,
) -> dict:
    feature_names = json.loads(features_path.read_text(encoding="utf-8"))
    if not isinstance(feature_names, list) or not all(isinstance(f, str) for f in feature_names):
        raise SystemExit(f"{features_path} must contain a JSON array of feature names")

    df = pd.read_parquet(data_path, columns=feature_names)
    missing = [name for name in feature_names if name not in df.columns]
    if missing:
        raise SystemExit(f"Features missing from {data_path}: {missing[:10]}")

    categoricals = {
        name: [str(value) for value in df[name].cat.categories]
        for name in feature_names
        if isinstance(df[name].dtype, pd.CategoricalDtype)
    }

    encoded = df.copy()
    for name in categoricals:
        encoded[name] = df[name].cat.codes
    encoded = encoded.fillna(0)

    defaults = {
        name: float(encoded[name].median())
        for name in feature_names
    }

    # Live drafts should use the active patch, not the dataset median patch.
    if "patch" in categoricals and current_patch in categoricals["patch"]:
        defaults["patch"] = float(categoricals["patch"].index(current_patch))
    if "patch_version" in categoricals and current_patch in categoricals["patch_version"]:
        defaults["patch_version"] = float(
            categoricals["patch_version"].index(current_patch)
        )

    return {
        "schema_version": 1,
        "model": features_path.stem.removesuffix("_features"),
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "categoricals": categoricals,
        "defaults": defaults,
        "positive_class_index": 1,
        "input_name": "float_input",
        "label_output": "label",
        "probability_output": "probabilities",
        "source": {
            "features_file": _relative(features_path),
            "data_file": _relative(data_path),
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--current-patch", default="7.41")
    args = parser.parse_args()

    schema = build_schema(args.data, args.features, args.current_patch)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(schema, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"Wrote {args.output} ({schema['feature_count']} features, "
        f"{len(schema['categoricals'])} categoricals, {len(schema['defaults'])} defaults)"
    )


if __name__ == "__main__":
    main()
