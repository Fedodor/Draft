#!/usr/bin/env python3
"""Model manifest: the reproducibility record bundled with every export.

A trained artifact is only useful if we can say which data, schema, and
hyperparameters produced it. The manifest captures all of that next to the
ONNX model and its feature schema so a future run can be compared or repeated.
"""

from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

MANIFEST_VERSION = 1


def library_versions() -> Dict[str, str]:
    """Best-effort version map of the training stack."""
    versions: Dict[str, str] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    for module_name in ("numpy", "pandas", "xgboost", "sklearn", "onnx", "onnxmltools"):
        try:
            module = __import__(module_name)
            versions[module_name] = getattr(module, "__version__", "unknown")
        except Exception:
            versions[module_name] = "unavailable"
    return versions


def build_manifest(
    *,
    model_stem: str,
    model_type: str,
    dataset_path: Path,
    dataset_sha256: str,
    dataset_rows: int,
    feature_names: Sequence[str],
    schema_fingerprint: str,
    split_info: Dict,
    hyperparameters: Dict,
    metrics: Dict,
    patch_version: Optional[str] = None,
    best_iteration: Optional[int] = None,
    artifacts: Optional[Dict[str, str]] = None,
) -> Dict:
    """Assemble the manifest dictionary (not yet written)."""
    return {
        "manifest_version": MANIFEST_VERSION,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model_stem": model_stem,
        "model_type": model_type,
        "patch_version": patch_version,
        "best_iteration": best_iteration,
        "dataset": {
            "path": str(dataset_path),
            "sha256": dataset_sha256,
            "rows": int(dataset_rows),
        },
        "feature_schema": {
            "feature_count": len(feature_names),
            "fingerprint": schema_fingerprint,
            "feature_names": list(feature_names),
        },
        "split": split_info,
        "hyperparameters": dict(hyperparameters),
        "metrics": metrics,
        "artifacts": artifacts or {},
        "environment": library_versions(),
    }


def write_manifest(path: Path, manifest: Dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path


def manifest_paths(model_dir: Path, model_stem: str) -> Dict[str, Path]:
    return {
        "joblib": model_dir / f"{model_stem}.joblib",
        "onnx": model_dir / f"{model_stem}.onnx",
        "features": model_dir / f"{model_stem}_features.json",
        "config": model_dir / f"{model_stem}_config.json",
        "manifest": model_dir / f"{model_stem}_manifest.json",
    }
