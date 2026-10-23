#!/usr/bin/env python3
"""
Augment draft_features.parquet with hero-embedding features.

Adds, for each row, the average hero embedding of each team's picks:
    rad_emb_avg_{d}   (d = 0..D-1)
    dire_emb_avg_{d}

Uses the existing rad_pick_0..4 / dire_pick_0..4 columns (hero IDs) and the
precomputed hero_embeddings.parquet table.

Usage:
    python feature-engineering/augment_embeddings.py
"""

import json
from pathlib import Path

ML_ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ML_ROOT / "datasets" / "processed"
EMBED = ML_ROOT / "datasets" / "embeddings"

import numpy as np
import pandas as pd


def main():
    df = pd.read_parquet(PROCESSED / "draft_features.parquet")
    print(f"Loaded {df.shape[0]} x {df.shape[1]}")

    emb = pd.read_parquet(EMBED / "hero_embeddings.parquet")
    dims = [c for c in emb.columns if c.startswith("emb_")]
    D = len(dims)
    emb_map = {int(r.hero_id): r for r in emb.itertuples()}
    print(f"Embedding dims: {D}, heroes: {len(emb_map)}")

    def team_avg(row, prefix):
        out = np.zeros(D)
        count = 0
        for k in range(5):
            col = f"{prefix}_pick_{k}"
            if col not in row or pd.isna(row[col]):
                continue
            hid = int(row[col])
            if hid in emb_map:
                e = emb_map[hid]
                for d in range(D):
                    out[d] += getattr(e, dims[d])
                count += 1
        if count:
            out /= count
        return out

    new_cols = {}
    rad = np.zeros((len(df), D))
    dire = np.zeros((len(df), D))
    for i, row in enumerate(df.itertuples(index=False)):
        rad[i] = team_avg(row._asdict(), "rad")
        dire[i] = team_avg(row._asdict(), "dire")
        if i % 20000 == 0:
            print(f"  row {i}")

    for d in range(D):
        new_cols[f"rad_emb_avg_{d}"] = rad[:, d]
        new_cols[f"dire_emb_avg_{d}"] = dire[:, d]

    out_df = df.copy()
    for k, v in new_cols.items():
        out_df[k] = v

    out_path = PROCESSED / "draft_features_emb.parquet"
    out_df.to_parquet(out_path, index=False)
    print(f"Saved {out_df.shape[0]} x {out_df.shape[1]} -> {out_path}")

    # Record the dims for the Rust runtime
    meta = {"embedding_dims": D, "features": [f"rad_emb_avg_{d}" for d in range(D)] +
            [f"dire_emb_avg_{d}" for d in range(D)]}
    with open(EMBED / "embedding_features.json", "w") as f:
        json.dump(meta, f, indent=2)
    print("Saved embedding feature metadata")


if __name__ == "__main__":
    main()
