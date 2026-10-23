#!/usr/bin/env python3
"""
Compute low-rank hero embeddings from the hero matchup winrate matrix.

Approach:
- Build a hero x hero winrate matrix from hero_matchups.parquet (hero_a's
  winrate vs hero_b), filling missing pairs with the hero's base winrate prior.
- Factorize with TruncatedSVD -> 128-dimensional per-hero embeddings.
- Save to ml/datasets/embeddings/hero_embeddings.parquet (keyed by hero_id,
  with the safe hero name for the Rust runtime).

Usage:
    python feature-engineering/compute_embeddings.py [--dims 128]
"""

import argparse
import json
from pathlib import Path

ML_ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ML_ROOT / "datasets" / "processed"
EMBED_DIR = ML_ROOT / "datasets" / "embeddings"

import numpy as np
import pandas as pd


def build_name_to_id(heroes_file: Path):
    with open(heroes_file) as f:
        roster = json.load(f)
    name_to_id = {}
    for hid_str, hero in roster.items():
        hid = int(hid_str)
        name = hero.get("localized_name", "")
        if not name:
            continue
        safe = name.lower().replace(" ", "_").replace("'", "").replace("-", "_")
        name_to_id[safe] = hid
        # Also map the raw npc name suffix (npc_dota_hero_X -> X)
        npc = hero.get("name", "").replace("npc_dota_hero_", "")
        if npc and npc not in name_to_id:
            name_to_id[npc] = hid
    return name_to_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dims", type=int, default=128)
    args = parser.parse_args()

    heroes_file = ML_ROOT / "datasets" / "heroes.json"
    name_to_id = build_name_to_id(heroes_file)

    # Map hero ids used in matchups/winrates
    with open(heroes_file) as f:
        roster = json.load(f)
    hero_ids = sorted(int(h) for h in roster.keys())

    # Base winrates as priors
    wr_file = PROCESSED / "hero_winrates.parquet"
    wr = pd.read_parquet(wr_file) if wr_file.exists() else None
    base_wr = {int(r.hero_id): float(r.winrate) for r in wr.itertuples()} if wr is not None else {}
    for hid in hero_ids:
        base_wr.setdefault(hid, 0.5)

    # Build matchup matrix
    id_to_idx = {hid: i for i, hid in enumerate(hero_ids)}
    n = len(hero_ids)
    M = np.full((n, n), 0.5, dtype=np.float64)
    for i in range(n):
        M[i, i] = 0.5

    mf = PROCESSED / "hero_matchups.parquet"
    count_used = 0
    if mf.exists():
        mu = pd.read_parquet(mf)
        for r in mu.itertuples():
            h_a_raw = getattr(r, "hero_a", None)
            h_b_raw = getattr(r, "hero_b", None)
            # New parquet stores hero_a/b as ints; old stored as names like "_abaddon"
            if isinstance(h_a_raw, (int, np.integer)) and isinstance(h_b_raw, (int, np.integer)):
                ha, hb = int(h_a_raw), int(h_b_raw)
            else:
                a = str(h_a_raw).lstrip("_")
                b = str(h_b_raw).lstrip("_")
                ha = name_to_id.get(a)
                hb = name_to_id.get(b)
                if ha is None or hb is None:
                    continue
            if ha not in id_to_idx or hb not in id_to_idx:
                continue
            wr = getattr(r, "winrate", None)
            if wr is None:
                wr = getattr(r, "hero_a_winrate", 0.5)
            M[id_to_idx[ha], id_to_idx[hb]] = float(wr)
            M[id_to_idx[hb], id_to_idx[ha]] = 1.0 - float(wr)
            count_used += 1
    print(f"Used {count_used} matchup pairs over {n} heroes")

    # Fill any all-0.5 rows with the hero's base winrate prior
    for hid in hero_ids:
        i = id_to_idx[hid]
        row = M[i]
        default = base_wr.get(hid, 0.5)
        # For missing entries (still exactly 0.5 and diagonal), nudge with prior
        mask = (row == 0.5)
        mask[i] = False
        row[mask] = default

    # SVD -> embeddings (U * sqrt(S))
    from sklearn.decomposition import TruncatedSVD
    k = min(args.dims, n - 1)
    svd = TruncatedSVD(n_components=k, random_state=42)
    U = svd.fit_transform(M)          # (n, k)
    S = np.sqrt(svd.singular_values_)  # scale
    emb = U * S

    # Normalize each embedding to unit-ish scale (stable across runs)
    emb = (emb - emb.mean(0)) / (emb.std(0) + 1e-8)
    emb = np.tanh(emb)  # bounded [-1, 1]

    # Save keyed by hero_id with safe name
    rows = []
    for hid in hero_ids:
        i = id_to_idx[hid]
        rec = {"hero_id": hid}
        for d in range(emb.shape[1]):
            rec[f"emb_{d}"] = float(emb[i, d])
        rows.append(rec)
    out_df = pd.DataFrame(rows)
    EMBED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = EMBED_DIR / "hero_embeddings.parquet"
    out_df.to_parquet(out_path, index=False)
    print(f"Saved {len(out_df)} embeddings ({emb.shape[1]} dims) -> {out_path}")

    # Also dump a JSON (name -> [..]) for the Rust runtime
    safe_of_id = {}
    for hid_str, hero in roster.items():
        name = hero.get("localized_name", "")
        if name:
            safe_of_id[int(hid_str)] = name.lower().replace(" ", "_").replace("'", "").replace("-", "_")
    out_json = []
    for hid in hero_ids:
        i = id_to_idx[hid]
        out_json.append({"id": hid, "name": safe_of_id.get(hid, f"hero_{hid}"),
                         "emb": [float(x) for x in emb[i, :]]})
    json_path = EMBED_DIR / "hero_embeddings.json"
    with open(json_path, "w") as f:
        json.dump(out_json, f)
    print(f"Saved JSON for runtime -> {json_path}")


if __name__ == "__main__":
    main()
