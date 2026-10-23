#!/usr/bin/env python3
"""Deterministic, time-ordered splitting and dataset fingerprinting.

Training must not let the validation or test partition influence model
selection, and partitions must respect chronology so that later matches are
never used to predict earlier ones. This module centralizes that logic so the
pipeline scripts cannot drift apart.

`start_time` is preferred. Older feature exports predate that column, so
`match_id` (which grows monotonically with time in Dota) is used as a fallback
and the caller is told which column was chosen.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

TIME_COLUMNS: Tuple[str, ...] = ("start_time", "start_time_utc")
FALLBACK_TIME_COLUMN = "match_id"


@dataclass
class SplitResult:
    """Chronologically ordered partitions plus provenance for the manifest."""

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    time_column: str
    used_start_time: bool
    fractions: Dict[str, float]
    boundaries: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def sizes(self) -> Dict[str, int]:
        return {
            "train": int(len(self.train)),
            "validation": int(len(self.val)),
            "test": int(len(self.test)),
        }

    def info(self) -> Dict:
        return {
            "time_column": self.time_column,
            "used_start_time": self.used_start_time,
            "fractions": dict(self.fractions),
            "sizes": self.sizes(),
            "boundaries": self.boundaries,
        }


def resolve_time_column(df: pd.DataFrame, time_col: Optional[str] = None) -> Tuple[str, bool]:
    """Return `(column, used_start_time)` for chronological ordering."""
    if time_col:
        if time_col not in df.columns:
            raise SystemExit(f"Requested time column {time_col!r} is not in the dataset")
        return time_col, time_col in TIME_COLUMNS

    for candidate in TIME_COLUMNS:
        if candidate in df.columns:
            return candidate, True

    if FALLBACK_TIME_COLUMN in df.columns:
        return FALLBACK_TIME_COLUMN, False

    raise SystemExit(
        "Cannot split chronologically: dataset has no start_time or match_id column"
    )


def _validate_fractions(train_frac: float, val_frac: float, test_frac: float) -> Dict[str, float]:
    fractions = {"train": train_frac, "validation": val_frac, "test": test_frac}
    for name, value in fractions.items():
        if value <= 0:
            raise SystemExit(f"{name} fraction must be positive (got {value})")
    total = train_frac + val_frac + test_frac
    if abs(total - 1.0) > 1e-6:
        raise SystemExit(f"Split fractions must sum to 1.0 (got {total})")
    return fractions


def _boundaries(frame: pd.DataFrame, column: str) -> Dict[str, float]:
    if frame.empty:
        return {"min": float("nan"), "max": float("nan")}
    return {"min": float(frame[column].min()), "max": float(frame[column].max())}


def time_split(
    df: pd.DataFrame,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    time_col: Optional[str] = None,
) -> SplitResult:
    """Split `df` into chronological train/validation/test partitions.

    The partitions are contiguous slices of the time-ordered data, so every
    training row precedes every validation row, which in turn precedes every
    test row.
    """
    if len(df) < 3:
        raise SystemExit(f"Need at least 3 rows to split, got {len(df)}")

    fractions = _validate_fractions(train_frac, val_frac, test_frac)
    column, used_start_time = resolve_time_column(df, time_col)

    ordered = df.sort_values(column, kind="mergesort").reset_index(drop=True)
    n = len(ordered)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    if n_train < 1 or n_val < 1 or n - n_train - n_val < 1:
        raise SystemExit(
            f"Split fractions produced an empty partition for n={n}; "
            "use more data or larger fractions"
        )

    train = ordered.iloc[:n_train]
    val = ordered.iloc[n_train : n_train + n_val]
    test = ordered.iloc[n_train + n_val :]

    boundaries = {
        "train": _boundaries(train, column),
        "validation": _boundaries(val, column),
        "test": _boundaries(test, column),
    }

    return SplitResult(
        train=train,
        val=val,
        test=test,
        time_column=column,
        used_start_time=used_start_time,
        fractions=fractions,
        boundaries=boundaries,
    )


def file_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    """Stable SHA-256 of a file, used as the dataset fingerprint."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def schema_fingerprint(feature_names: Sequence[str]) -> str:
    """SHA-256 of an ordered feature-name list."""
    payload = json.dumps(list(feature_names), separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def frame_fingerprint(df: pd.DataFrame, columns: Sequence[str]) -> str:
    """Order-independent fingerprint of the values in `columns`."""
    hashed = pd.util.hash_pandas_object(df[list(columns)], index=False)
    return hashlib.sha256(hashed.values.tobytes()).hexdigest()[:16]
