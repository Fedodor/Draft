#!/usr/bin/env python3
"""Generate a deterministic ONNX golden vector for the bundled model."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd

ML_ROOT = Path(__file__).resolve().parents[2]
DATA = ML_ROOT / "datasets" / "processed" / "draft_features_emb.parquet"
MODEL = ML_ROOT / "models" / "draft_model_xgboost_full.onnx"
FEATURES = ML_ROOT / "models" / "draft_model_xgboost_full_features.json"
OUTPUT = Path(__file__).resolve().parent / "golden_draft_features.json"


def main() -> None:
    feature_names = json.loads(FEATURES.read_text(encoding="utf-8"))
    columns = feature_names + ["radiant_win", "match_id"]
    df = pd.read_parquet(DATA, columns=columns)

    candidates = df.index[
        (df["patch"] == "7.41") & (df["patch_version"] == "7.41")
    ]
    if not len(candidates):
        raise SystemExit("No 7.41 rows found in the processed dataset")
    row_index = int(candidates[0])

    row = df.loc[[row_index]].copy()
    for name in feature_names:
        if isinstance(row[name].dtype, pd.CategoricalDtype):
            row[name] = row[name].cat.codes
    row = row.fillna(0)

    input_vector = row[feature_names].to_numpy(dtype=np.float32)
    session = ort.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    label, probabilities = session.run(
        None, {input_name: input_vector}
    )

    fixture = {
        "feature_names": feature_names,
        "input": [float(value) for value in input_vector[0]],
        "expected": {
            "label": int(label[0]),
            "probabilities": [float(value) for value in probabilities[0]],
            "positive_probability": float(probabilities[0][1]),
        },
        "source": {
            "model": MODEL.name,
            "data": DATA.name,
            "match_id": int(df.loc[row_index, "match_id"]),
            "radiant_win": int(df.loc[row_index, "radiant_win"]),
            "patch": "7.41",
            "patch_version": "7.41",
        },
    }
    OUTPUT.write_text(json.dumps(fixture, indent=2) + "\n", encoding="utf-8")
    print(
        f"Wrote {OUTPUT} from match {fixture['source']['match_id']}: "
        f"label={fixture['expected']['label']} "
        f"positive_probability={fixture['expected']['positive_probability']:.6f}"
    )


if __name__ == "__main__":
    main()
