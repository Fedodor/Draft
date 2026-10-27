#!/usr/bin/env python3
"""Python mirror of the runtime feature-vector contract.

The Rust runtime (`apps/desktop/src-tauri/src/ml.rs`) builds a feature map from
the live draft and then materialises the model input in the order declared by
`<stem>_schema.json`. Anything the live draft cannot provide falls back to the
schema default (then to 0.0 for legacy bundles without a schema).

This module performs exactly the same final step in Python so tests and the
quality gate can exercise the contract without duplicating it in the Rust
tests. It is a *consumer* of the schema, not a second source of truth: feature
order, categorical codes and defaults all come from the exported schema.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Features that describe the finished match rather than the draft. The runtime
# cannot know them at recommendation time and always substitutes the training
# median from the schema. They are also the main train/serve-skew candidates.
CONTEXT_FEATURES: Tuple[str, ...] = (
    "patch",
    "patch_version",
    "duration",
    "avg_mmr",
    "game_mode",
    "lobby_type",
)


@dataclass
class EncodingResult:
    """Encoded matrix plus the diagnostics needed by the quality gate."""

    matrix: np.ndarray
    nan_values: Dict[str, int] = field(default_factory=dict)
    unknown_categories: Dict[str, int] = field(default_factory=dict)
    missing_columns: Tuple[str, ...] = ()
    nonfinite_features: Tuple[str, ...] = ()


def encode_category(schema: Mapping[str, Any], feature: str, value: Any) -> Optional[float]:
    """Encode one categorical value with the schema's category order.

    Mirrors `FeatureSchema::encode_category`: exact match first, then the
    ``unknown`` category when present. Returns ``None`` when the schema has no
    mapping for the feature, so callers can fall back to a default.
    """
    categories = schema.get("categoricals", {}).get(feature)
    if categories is None:
        return None
    text = str(value)
    if text in categories:
        return float(categories.index(text))
    if "unknown" in categories:
        return float(categories.index("unknown"))
    return None


def default_for(schema: Mapping[str, Any], feature: str, fallback: float = 0.0) -> float:
    """Schema default for a feature, then the caller's fallback."""
    defaults = schema.get("defaults", {})
    value = defaults.get(feature, fallback)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(fallback)


def _finite(value: float) -> float:
    if not np.isfinite(value):
        raise ValueError(f"non-finite inference feature value: {value!r}")
    return float(value)


def build_vector(
    feature_map: Mapping[str, Any],
    schema: Mapping[str, Any],
    *,
    inference_defaults: bool = True,
) -> np.ndarray:
    """Build the ordered model input for one draft.

    `inference_defaults=True` reproduces the runtime exactly: post-draft and
    patch context features are always replaced by the schema default, and any
    feature missing from the live draft uses its default. Set it to `False` to
    score the as-logged feature values (used for the gate's as-trained mode).
    """
    names: Sequence[str] = schema["feature_names"]
    vector = np.empty(len(names), dtype=np.float32)
    for index, name in enumerate(names):
        if inference_defaults and name in CONTEXT_FEATURES:
            vector[index] = default_for(schema, name)
            continue

        if name in feature_map:
            value = feature_map[name]
            if name in schema.get("categoricals", {}) and isinstance(value, str):
                encoded = encode_category(schema, name, value)
                value = encoded if encoded is not None else default_for(schema, name)
            try:
                vector[index] = _finite(float(value))
            except (TypeError, ValueError) as error:
                raise ValueError(f"feature {name!r}: {error}") from error
        else:
            vector[index] = default_for(schema, name)
    return vector


def encode_matrix(
    frame: pd.DataFrame,
    schema: Mapping[str, Any],
    *,
    inference_defaults: bool = False,
) -> EncodingResult:
    """Encode a DataFrame into the schema's exact feature order.

    Categoricals are encoded with the schema's category list (not pandas' local
    category order), which keeps evaluation aligned with inference. NaN numeric
    values are filled with 0 exactly like `train_full.encode_features`; the
    count is reported so the gate can make the fill visible. Infinities are
    reported as `nonfinite_features` and must be rejected by the caller.
    """
    names: Sequence[str] = list(schema["feature_names"])
    categoricals: Mapping[str, Sequence[str]] = schema.get("categoricals", {})
    defaults: Mapping[str, float] = schema.get("defaults", {})
    matrix = np.empty((len(frame), len(names)), dtype=np.float32)
    nan_values: Dict[str, int] = {}
    unknown_categories: Dict[str, int] = {}
    missing_columns = tuple(name for name in names if name not in frame.columns)

    for column_index, name in enumerate(names):
        if name in categoricals:
            categories = list(categoricals[name])
            if name not in frame.columns:
                matrix[:, column_index] = default_for(schema, name)
                continue
            mapping = {str(category): index for index, category in enumerate(categories)}
            unknown_index = mapping.get("unknown")
            encoded = np.empty(len(frame), dtype=np.float64)
            unknown_count = 0
            for row_index, value in enumerate(frame[name].to_numpy()):
                if pd.isna(value):
                    encoded[row_index] = default_for(schema, name)
                    continue
                code = mapping.get(str(value))
                if code is None:
                    unknown_count += 1
                    code = (
                        float(unknown_index)
                        if unknown_index is not None
                        else default_for(schema, name)
                    )
                encoded[row_index] = float(code)
            if unknown_count:
                unknown_categories[name] = unknown_count
            matrix[:, column_index] = encoded
            continue

        if inference_defaults and name in CONTEXT_FEATURES:
            matrix[:, column_index] = default_for(schema, name)
            continue

        if name not in frame.columns:
            matrix[:, column_index] = default_for(schema, name)
            continue

        values = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=np.float64)
        nan_count = int(np.isnan(values).sum())
        if nan_count:
            nan_values[name] = nan_count
            values = np.where(np.isnan(values), 0.0, values)
        matrix[:, column_index] = values

    nonfinite = tuple(
        name for index, name in enumerate(names) if not np.isfinite(matrix[:, index]).all()
    )
    return EncodingResult(
        matrix=matrix,
        nan_values=nan_values,
        unknown_categories=unknown_categories,
        missing_columns=missing_columns,
        nonfinite_features=nonfinite,
    )


def contract_diff(generated: Iterable[str], schema: Mapping[str, Any]) -> Dict[str, Tuple[str, ...]]:
    """Return feature names the pipeline emits but the model ignores, and vice versa."""
    produced = set(generated)
    expected = set(schema["feature_names"])
    return {
        "unexpected": tuple(sorted(produced - expected)),
        "missing": tuple(sorted(expected - produced)),
    }
