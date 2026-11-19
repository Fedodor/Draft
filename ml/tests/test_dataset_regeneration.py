"""Dataset-regeneration tests: determinism, canonical order, pick/ban integrity.

These tests never touch the production corpus. They build a tiny synthetic raw
directory in both provider shapes and drive the real `FeatureProcessor` through
the streaming `process_all` path, then assert the properties a regenerated
dataset must have before it can be trusted:

* identical inputs -> identical rows, columns, values and schema fingerprint,
  regardless of worker count or chunk size;
* canonical row order (numeric match id) and a documented column order;
* picks and bans stay separate, with exactly five presence flags per side for a
  complete Captain's Mode draft;
* a missing `isPick`/`is_pick` flag is skipped, never guessed as a pick.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ML_ROOT = Path(__file__).resolve().parents[1]
FEATURE_ENGINEERING = ML_ROOT / "feature-engineering"
for path in (str(FEATURE_ENGINEERING), str(ML_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

import processor  # noqa: E402

pytest.importorskip("pyarrow")


def _stratz_match(match_id: int, *, radiant, dire, rad_bans, dire_bans, start: int, patch_id=182):
    entries = []
    order = 0
    for hero in radiant:
        entries.append({"heroId": hero, "isPick": True, "isRadiant": True, "order": order})
        order += 1
    for hero in dire:
        entries.append({"heroId": hero, "isPick": True, "isRadiant": False, "order": order})
        order += 1
    for hero in rad_bans:
        entries.append({"heroId": hero, "isPick": False, "isRadiant": True, "order": order})
        order += 1
    for hero in dire_bans:
        entries.append({"heroId": hero, "isPick": False, "isRadiant": False, "order": order})
        order += 1
    return {
        "id": match_id,
        "match_id": match_id,
        "startDateTime": start,
        "didRadiantWin": match_id % 2 == 0,
        "durationSeconds": 2000 + match_id % 500,
        "gameVersionId": patch_id,
        "pickBans": entries,
    }


def _opendota_match(match_id: int, *, radiant, dire, rad_bans, start: int):
    entries = [
        {"hero_id": hero, "team": 0, "is_pick": True, "order": index}
        for index, hero in enumerate(radiant)
    ]
    entries += [
        {"hero_id": hero, "team": 1, "is_pick": True, "order": 10 + index}
        for index, hero in enumerate(dire)
    ]
    entries += [
        {"hero_id": hero, "team": 0, "is_pick": False, "order": 20 + index}
        for index, hero in enumerate(rad_bans)
    ]
    return {
        "match_id": match_id,
        "start_time": start,
        "radiant_win": match_id % 3 == 0,
        "duration": 1800,
        "game_mode": 22,
        "lobby_type": 0,
        "patch": 60,
        "picks_bans": entries,
    }


def _write_corpus(raw_dir: Path, *, count: int = 12) -> list[int]:
    """Deterministic mix of STRATZ and OpenDota records with stable picks."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for index in range(count):
        match_id = 5000 + index
        # Rotate the roster so several sparse win-rate columns appear.
        base = 1 + (index * 3) % 100
        radiant = [base + step for step in range(5)]
        dire = [base + 40 + step for step in range(5)]
        rad_bans = [base + 70, base + 71]
        dire_bans = [base + 80, base + 81]
        if index % 4 == 0:
            payload = _opendota_match(
                match_id, radiant=radiant, dire=dire, rad_bans=rad_bans, start=1_700_000_000 + index
            )
        else:
            payload = _stratz_match(
                match_id,
                radiant=radiant,
                dire=dire,
                rad_bans=rad_bans,
                dire_bans=dire_bans,
                start=1_700_000_000 + index,
            )
        (raw_dir / f"{match_id}.json").write_text(json.dumps(payload), encoding="utf-8")
        written.append(match_id)
    return written


def _seed_lookup(directory: Path) -> None:
    pd.DataFrame(
        [
            {"hero_id": hero_id, "games": 100, "wins": 50, "winrate": 0.40 + (hero_id % 50) / 100}
            for hero_id in range(1, 140)
        ]
    ).to_parquet(directory / "hero_winrates.parquet", index=False)


def _build(tmp_path: Path, *, workers: int, chunk_size: int, name: str):
    raw = tmp_path / f"raw_{name}"
    out = tmp_path / f"out_{name}"
    out.mkdir(parents=True, exist_ok=True)
    _write_corpus(raw)
    _seed_lookup(out)
    proc = processor.FeatureProcessor(
        raw_dir=raw,
        output_dir=out,
        include_matchups=False,
        include_allpick=False,
        workers=workers,
    )
    proc.process_all(chunk_size=chunk_size, load_into_memory=False, dataset_label=name)
    frame = pd.read_parquet(out / "draft_features.parquet")
    schema = json.loads((out / "dataset_schema.json").read_text(encoding="utf-8"))
    return proc, frame, schema, out


def test_processing_is_deterministic_across_workers_and_chunks(tmp_path):
    _, frame_a, schema_a, out_a = _build(tmp_path, workers=1, chunk_size=7, name="a")
    _, frame_b, schema_b, out_b = _build(tmp_path, workers=4, chunk_size=3, name="b")
    _, frame_c, schema_c, out_c = _build(tmp_path, workers=4, chunk_size=3, name="c")

    # Identical content, row order, column order and dtypes.
    pd.testing.assert_frame_equal(frame_a, frame_b)
    assert list(frame_a.columns) == list(frame_b.columns)
    assert schema_a["feature_fingerprint"] == schema_b["feature_fingerprint"]
    assert schema_a["column_order"] == schema_b["column_order"]
    assert schema_a["output"]["rows"] == schema_b["output"]["rows"]

    # Same configuration must reproduce the exact file bytes.
    assert schema_b["output"]["sha256"] == schema_c["output"]["sha256"]


def test_rows_are_in_canonical_match_order(tmp_path):
    _, frame, _, _ = _build(tmp_path, workers=2, chunk_size=5, name="order")
    assert frame["match_id"].tolist() == sorted(frame["match_id"].tolist())


def test_meta_block_leads_and_winrates_are_sorted(tmp_path):
    _, frame, schema, _ = _build(tmp_path, workers=2, chunk_size=4, name="cols")
    columns = list(frame.columns)

    assert columns[: len(processor.META_COLUMNS)] == list(processor.META_COLUMNS)

    winrates = [name for name in columns if name.startswith(processor.PER_HERO_WINRATE_PREFIXES)]
    assert winrates, "fixture should produce per-hero win-rate columns"
    safe_names = _processor_safe_names(tmp_path)
    ids = {f"rad_wr_{token}": hero_id for hero_id, token in safe_names.items()}
    ids.update({f"dire_wr_{token}": hero_id for hero_id, token in safe_names.items()})
    expected = sorted(
        winrates,
        key=lambda name: (0 if name.startswith("rad_wr_") else 1, ids.get(name, 10 ** 9), name),
    )
    assert winrates == expected

    ban_avg = [name for name in columns if name in processor.BAN_AVG_COLUMNS]
    assert ban_avg == list(processor.BAN_AVG_COLUMNS)
    assert schema["column_order"] == columns


def _processor_safe_names(tmp_path: Path) -> dict:
    return processor.FeatureProcessor(
        raw_dir=tmp_path, output_dir=tmp_path, include_matchups=False
    )._safe_names


def test_presence_vector_is_exactly_five_per_side(tmp_path):
    _, frame, _, _ = _build(tmp_path, workers=2, chunk_size=6, name="presence")
    rad_has = [c for c in frame.columns if c.startswith("rad_has_")]
    dire_has = [c for c in frame.columns if c.startswith("dire_has_")]

    assert len(rad_has) == 139 and len(dire_has) == 139
    assert set(frame[rad_has].sum(axis=1).unique()) == {5.0}
    assert set(frame[dire_has].sum(axis=1).unique()) == {5.0}


def test_bans_are_recorded_separately_from_picks(tmp_path):
    _, frame, _, _ = _build(tmp_path, workers=1, chunk_size=4, name="bans")
    rad_ban = [c for c in frame.columns if c.startswith("rad_ban_")]
    assert rad_ban, "ban one-hots must exist"

    # STRATZ rows carry two Radiant bans; the OpenDota fixture rows (every 4th)
    # carry two as well, so every row has exactly two ban flags total.
    assert set(frame[rad_ban].sum(axis=1).unique()) == {2.0}

    for _, row in frame.iterrows():
        present = {
            name[len("rad_has_"):] for name in frame.columns
            if name.startswith("rad_has_") and row[name] == 1
        }
        banned = {
            name[len("rad_ban_"):] for name in rad_ban if row[name] == 1
        }
        assert present.isdisjoint(banned), "a hero cannot be both picked and banned"


def test_pick_slots_hold_picks_not_bans(tmp_path):
    proc, frame, _, _ = _build(tmp_path, workers=1, chunk_size=4, name="slots")
    slot_columns = [f"rad_pick_{i}" for i in range(5)]
    ban_columns = [c for c in frame.columns if c.startswith("rad_ban_")]
    banned_tokens = {c[len("rad_ban_"):] for c in ban_columns}

    for _, row in frame.iterrows():
        slots = [int(row[column]) for column in slot_columns]
        assert len(set(slots)) == 5
        banned_here = {
            token for token in banned_tokens
            if f"rad_ban_{token}" in frame.columns and row[f"rad_ban_{token}"] == 1
        }
        picked_tokens = {proc.safe_name(hero_id) for hero_id in slots}
        assert picked_tokens.isdisjoint(banned_here)
        for hero_id in slots:
            assert row[f"rad_has_{proc.safe_name(hero_id)}"] == 1


def test_missing_is_pick_is_skipped_never_guessed(tmp_path):
    raw = tmp_path / "raw_missing"
    out = tmp_path / "out_missing"
    raw.mkdir(parents=True)
    out.mkdir(parents=True)
    _seed_lookup(out)

    # One ambiguous entry among otherwise valid picks: the entry is dropped, so
    # the match no longer has five picks and is skipped rather than mislabelled.
    payload = _stratz_match(
        7000,
        radiant=[1, 2, 3, 4, 5],
        dire=[6, 7, 8, 9, 10],
        rad_bans=[],
        dire_bans=[],
        start=1_700_000_000,
    )
    payload["pickBans"][0].pop("isPick")
    (raw / "7000.json").write_text(json.dumps(payload), encoding="utf-8")

    proc = processor.FeatureProcessor(
        raw_dir=raw, output_dir=out, include_matchups=False, include_allpick=False
    )
    with pytest.raises(ValueError, match="No valid matches"):
        proc.process_all(chunk_size=10, load_into_memory=False)

    assert proc.stats["entries_missing_is_pick"] == 1
    assert proc.stats["matches_processed"] == 0
    assert proc.stats["skipped_short_draft"] == 1
    assert not (out / "draft_features.parquet").exists()


def test_null_pick_bans_is_reported_as_malformed(tmp_path):
    raw = tmp_path / "raw_null"
    out = tmp_path / "out_null"
    raw.mkdir(parents=True)
    out.mkdir(parents=True)
    _seed_lookup(out)
    (raw / "8000.json").write_text(json.dumps({"id": 8000, "pickBans": None}), encoding="utf-8")

    proc = processor.FeatureProcessor(
        raw_dir=raw, output_dir=out, include_matchups=False, include_allpick=False
    )
    with pytest.raises(ValueError, match="No valid matches"):
        proc.process_all(chunk_size=10, load_into_memory=False)
    assert proc.stats["skipped_malformed_draft"] == 1


def test_canonical_column_order_is_discovery_order_independent():
    """Sparse blocks must not depend on which match was processed first.

    Structural features are emitted for every match, so their first appearance
    is always the code-defined order. The win-rate and ban-average blocks are
    sparse, and those are what must be ordered explicitly.
    """
    meta = list(processor.META_COLUMNS)
    structural = [
        "rad_has_axe", "dire_has_axe", "rad_ban_axe", "dire_ban_axe",
        "radiant_avg_stun", "dire_avg_stun", "stun_diff",
        "radiant_avg_winrate", "winrate_diff",
    ]
    winrates = [
        "rad_wr_axe", "dire_wr_axe", "rad_wr_zuus", "dire_wr_zuus",
    ]
    ban_avg = list(processor.BAN_AVG_COLUMNS)
    hero_ids = {
        "rad_wr_axe": 2, "dire_wr_axe": 2, "rad_wr_zuus": 22, "dire_wr_zuus": 22,
    }

    order_a = processor.canonical_column_order(
        meta + structural + ban_avg + winrates, hero_ids
    )
    order_b = processor.canonical_column_order(
        meta + structural
        + list(reversed(winrates))
        + list(reversed(ban_avg)),
        hero_ids,
    )
    assert order_a == order_b

    assert order_a[: len(meta)] == meta
    assert order_a[len(meta): len(meta) + len(structural)] == structural
    assert order_a[-4:] == ["rad_wr_axe", "rad_wr_zuus", "dire_wr_axe", "dire_wr_zuus"]
    assert order_a.index("radiant_ban_avg_winrate") < order_a.index("rad_wr_axe")


def test_dataset_schema_records_provenance_and_fingerprint(tmp_path):
    _, frame, schema, out = _build(tmp_path, workers=2, chunk_size=4, name="schema")

    assert schema["output"]["rows"] == len(frame)
    assert schema["output"]["columns"] == len(frame.columns)
    assert schema["feature_count"] == len(frame.columns) - len(processor.META_COLUMNS)
    assert set(schema["feature_columns"]).isdisjoint(processor.META_COLUMNS)
    assert schema["categories"]["patch"], "patch categories must be recorded"
    assert schema["source"]["lookup_files"]["hero_winrates.parquet"]["sha256"]
    assert schema["stats"]["matches_processed"] == len(frame)
    assert (out / "hero_winrates.parquet").is_file(), "lookup table must be carried over"


def test_single_chunk_and_streaming_agree(tmp_path):
    _, one, _, _ = _build(tmp_path, workers=1, chunk_size=1000, name="single")
    _, many, _, _ = _build(tmp_path, workers=3, chunk_size=2, name="many")
    pd.testing.assert_frame_equal(one, many)


SPLIT_CONFIG = ML_ROOT / "configs" / "eval_split_v2.json"
V2_DATASET = ML_ROOT / "datasets" / "processed_v2" / "draft_features.parquet"


def test_frozen_eval_split_is_ordered_and_disclosed():
    """The frozen protocol must be ordered, complete, and honest about leakage."""
    config = json.loads(SPLIT_CONFIG.read_text(encoding="utf-8"))
    parts = config["partitions"]

    assert list(parts) == ["train", "validation", "test"]
    assert parts["train"]["start_time_max"] <= parts["validation"]["start_time_min"]
    assert parts["validation"]["start_time_max"] <= parts["test"]["start_time_min"]

    rows = sum(part["rows"] for part in parts.values())
    ids = sum(part["unique_match_ids"] for part in parts.values())
    assert rows == ids == config["dataset"]["rows"]

    # Selection may never touch the final partition.
    allowed = config["allowed_use"]
    assert "test" not in allowed["feature_selection"]
    assert "test" not in allowed["model_selection"]
    assert "test" in allowed["reference_only"]
    assert allowed["forbidden"]

    # The previous ablation already scored this window, so the protocol must
    # not claim a clean holdout it does not have.
    contamination = config["contamination"]
    assert contamination["untouched"] is False
    assert contamination["clean_holdout_requirement"]
    assert contamination["shared_match_ids_with_previous_test_window"] > 0


@pytest.mark.skipif(
    not V2_DATASET.is_file(),
    reason="corrected dataset is a local, gitignored artifact",
)
def test_frozen_split_reproduces_the_corrected_dataset():
    import hashlib

    config = json.loads(SPLIT_CONFIG.read_text(encoding="utf-8"))
    digest = hashlib.sha256()
    with open(V2_DATASET, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    assert digest.hexdigest() == config["dataset"]["sha256"]

    frame = pd.read_parquet(V2_DATASET, columns=["match_id", "start_time"])
    frame = frame.sort_values(["start_time", "match_id"], kind="mergesort")
    parts = config["partitions"]
    train_max = parts["train"]["start_time_max"]
    val_max = parts["validation"]["start_time_max"]

    assert int((frame["start_time"] <= train_max).sum()) == parts["train"]["rows"]
    assert int(
        ((frame["start_time"] > train_max) & (frame["start_time"] <= val_max)).sum()
    ) == parts["validation"]["rows"]
    assert int((frame["start_time"] > val_max).sum()) == parts["test"]["rows"]
    assert frame["match_id"].is_unique
