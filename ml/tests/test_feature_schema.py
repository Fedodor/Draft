"""Feature-schema reproducibility from the real processor.

A small synthetic fixture drives `FeatureProcessor.process_match` (the same
code that produced the training parquet) and compares what it emits with the
bundled model contract. The fixture is deterministic and needs no downloaded
dataset, so it runs in ordinary CI.

What is pinned:

* the 887-name set is reproduced exactly when matchup data is absent;
* 631 structural features keep a stable, code-defined order;
* the 254 per-hero win-rate features are present as a set, and the two
  ``*_ban_avg_winrate`` features are present too. Their position in the
  shipped parquet is first-seen row history (the builder appends columns in
  dataframe discovery order), not a feature contract, so position is not
  pinned for them;
* adding matchup data appends exactly seven features, which is a schema
  extension that requires retraining and must not be silently shipped against
  the current model;
* a generated row produces a finite, schema-ordered inference vector that the
  bundled ONNX model accepts.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ML_ROOT = Path(__file__).resolve().parents[1]
FEATURE_ENGINEERING = ML_ROOT / "feature-engineering"
if str(FEATURE_ENGINEERING) not in sys.path:
    sys.path.insert(0, str(FEATURE_ENGINEERING))
if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))

import processor  # noqa: E402
from training.inference_contract import build_vector, contract_diff, encode_matrix  # noqa: E402

pytest.importorskip("onnxruntime")

MODEL_DIR = ML_ROOT / "models"
STEM = "draft_model_xgboost_full"
SCHEMA_PATH = MODEL_DIR / f"{STEM}_schema.json"
FEATURES_PATH = MODEL_DIR / f"{STEM}_features.json"
ONNX_PATH = MODEL_DIR / f"{STEM}.onnx"
# The manifest is a local, gitignored build record (it embeds absolute paths and
# a fresh timestamp), so the schema contract is pinned against the tracked
# features/schema pair instead.

IDENTITY_COLUMNS = {"match_id", "start_time", "radiant_win"}
MATCHUP_FEATURES = {
    "rad_matchup_avg",
    "rad_matchup_best",
    "rad_matchup_worst",
    "dire_matchup_avg",
    "dire_matchup_best",
    "dire_matchup_worst",
    "matchup_advantage",
}
# Features whose column position depends on first appearance in the dataset:
# per-hero win rates are emitted only for picked heroes, and the ban averages
# only for matches with bans. The shipped parquet learned their positions from
# row discovery order, so only their membership is contractual.
PER_HERO_WINRATE_PREFIXES = ("rad_wr_", "dire_wr_")
HISTORY_ORDERED_FEATURES = {"radiant_ban_avg_winrate", "dire_ban_avg_winrate"}


def _load_contract():
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    features = json.loads(FEATURES_PATH.read_text(encoding="utf-8"))
    return schema, features


def _roster_ids() -> list[int]:
    roster = json.loads((ML_ROOT / "datasets" / "heroes.json").read_text(encoding="utf-8"))
    return sorted(int(hero_id) for hero_id in roster)


def _winrate(hero_id: int) -> float:
    return 0.40 + (hero_id % 50) / 100.0


def _synthetic_matches(ids: list[int], count: int = 26) -> list[dict]:
    """Deterministic matches that eventually pick every roster hero on each side."""
    matches = []
    for index in range(count):
        radiant = [ids[(index + 26 * step) % len(ids)] for step in range(5)]
        dire = [ids[(index + 63 + 26 * step) % len(ids)] for step in range(5)]
        bans = [
            (ids[(index + 10) % len(ids)], 0),
            (ids[(index + 20) % len(ids)], 1),
        ]
        picks_bans = [
            {"hero_id": hero_id, "team": 0, "is_pick": True, "order": order}
            for order, hero_id in enumerate(radiant)
        ]
        picks_bans += [
            {"hero_id": hero_id, "team": 1, "is_pick": True, "order": 10 + order}
            for order, hero_id in enumerate(dire)
        ]
        picks_bans += [
            {"hero_id": hero_id, "team": team, "is_pick": False, "order": 20 + order}
            for order, (hero_id, team) in enumerate(bans)
        ]
        matches.append(
            {
                "match_id": 1_000 + index,
                "start_time": 1_700_000_000 + index * 60,
                "picks_bans": picks_bans,
                "radiant_win": index % 2 == 0,
                "duration": 2_000 + index,
                "game_mode": 22,
                "lobby_type": 0,
                "patch": 60,  # PATCH_MAP: 60 -> 7.41
            }
        )
    return matches


def _write_auxiliary_data(directory: Path, ids: list[int], *, with_matchups: bool) -> None:
    pd.DataFrame(
        [
            {"hero_id": hero_id, "games": 100, "wins": 50, "winrate": _winrate(hero_id)}
            for hero_id in ids
        ]
    ).to_parquet(directory / "hero_winrates.parquet", index=False)

    if with_matchups:
        rows = []
        for hero_a in ids[:5]:
            for hero_b in ids[63:68]:
                winrate = 0.5 + 0.001 * (hero_a - hero_b)
                rows.append(
                    {
                        "hero_a": hero_a,
                        "hero_b": hero_b,
                        "games": 50,
                        "hero_a_winrate": winrate,
                        "hero_b_winrate": 1.0 - winrate,
                    }
                )
        pd.DataFrame(rows).to_parquet(directory / "hero_matchups.parquet", index=False)


def _generate(directory: Path, *, with_matchups: bool) -> tuple[pd.DataFrame, dict]:
    ids = _roster_ids()
    _write_auxiliary_data(directory, ids, with_matchups=with_matchups)
    feature_processor = processor.FeatureProcessor(raw_dir=directory, output_dir=directory)
    matches = _synthetic_matches(ids)
    rows = [feature_processor.process_match(match) for match in matches]
    assert all(row is not None for row in rows), "the real processor rejected a fixture match"
    # Mirror `FeatureProcessor.process_all`: sparse per-hero columns become 0
    # and object columns become categorical, exactly as in the shipped parquet.
    frame = pd.DataFrame(rows).fillna(0)
    for column in frame.columns:
        if frame[column].dtype == "object":
            frame[column] = frame[column].astype("category")
    return frame, {"processor": feature_processor, "matches": matches, "ids": ids}


def _generated_names(frame: pd.DataFrame) -> list[str]:
    return [name for name in frame.columns if name not in IDENTITY_COLUMNS]


def _is_per_hero_winrate(name: str) -> bool:
    return name.startswith(PER_HERO_WINRATE_PREFIXES)


def test_processor_reproduces_the_bundled_887_feature_set(tmp_path):
    schema, shipped_features = _load_contract()
    frame, _ = _generate(tmp_path, with_matchups=False)
    generated = _generated_names(frame)

    assert len(generated) == 887
    assert set(generated) == set(shipped_features) == set(schema["feature_names"])

    # Structural features whose position is fixed by code (everything except
    # the per-hero win-rate block and the two ban averages) keep the shipped
    # model's exact order.
    def structural(names):
        return [
            name
            for name in names
            if not _is_per_hero_winrate(name) and name not in HISTORY_ORDERED_FEATURES
        ]

    assert structural(generated) == structural(shipped_features)
    assert len(structural(generated)) == 631

    per_hero = [name for name in generated if _is_per_hero_winrate(name)]
    assert len(per_hero) == 254
    assert set(per_hero) == {
        name for name in shipped_features if _is_per_hero_winrate(name)
    }
    # The hard-coded id range keeps the legacy names for 145/155.
    assert "rad_wr_hero_145" in per_hero
    assert "rad_wr_hero_155" in per_hero

    # Documented divergence: the shipped schema lists the ban averages after
    # every win-rate column, but the processor emits them next to the other
    # win-rate aggregates. Membership is what the model contract needs; a
    # regenerated dataset with a different order requires a retrain/re-export.
    assert set(HISTORY_ORDERED_FEATURES).issubset(generated)
    shipped_positions = [shipped_features.index(name) for name in HISTORY_ORDERED_FEATURES]
    generated_positions = [generated.index(name) for name in HISTORY_ORDERED_FEATURES]
    assert shipped_positions != generated_positions


def test_generated_column_order_is_deterministic(tmp_path):
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()

    first, _ = _generate(first_dir, with_matchups=False)
    second, _ = _generate(second_dir, with_matchups=False)

    assert _generated_names(first) == _generated_names(second)


def test_processor_values_come_from_the_real_transformation(tmp_path):
    _, completion = _generate(tmp_path, with_matchups=True)
    feature_processor = completion["processor"]
    ids = completion["ids"]
    match = completion["matches"][0]
    row = feature_processor.process_match(match)

    radiant = [ids[26 * step] for step in range(5)]
    dire = [ids[(63 + 26 * step) % len(ids)] for step in range(5)]

    assert row["rad_pick_0"] == radiant[0]
    assert row["dire_pick_4"] == dire[4]
    for hero_id in radiant:
        assert row[f"rad_has_{feature_processor.safe_name(hero_id)}"] == 1
    for hero_id in dire:
        assert row[f"dire_has_{feature_processor.safe_name(hero_id)}"] == 1
    # A hero on the other side must not be marked present.
    assert row[f"rad_has_{feature_processor.safe_name(dire[0])}"] == 0

    expected_stun = np.mean(
        [feature_processor.get_semantics(hero_id)["stun"] for hero_id in radiant]
    )
    assert row["radiant_avg_stun"] == pytest.approx(expected_stun)
    assert row["stun_diff"] == pytest.approx(
        row["radiant_avg_stun"] - row["dire_avg_stun"]
    )

    expected_winrate = np.mean([_winrate(hero_id) for hero_id in radiant])
    assert row["radiant_avg_winrate"] == pytest.approx(expected_winrate)
    assert row["winrate_diff"] == pytest.approx(
        row["radiant_avg_winrate"] - row["dire_avg_winrate"]
    )

    expected_matchup = np.mean(
        [
            np.mean(
                [
                    feature_processor.hero_matchups[hero_id][opponent]["winrate"]
                    if opponent in feature_processor.hero_matchups.get(hero_id, {})
                    else 0.5
                    for opponent in dire
                ]
            )
            for hero_id in radiant
        ]
    )
    assert row["rad_matchup_avg"] == pytest.approx(expected_matchup)
    assert row["matchup_advantage"] == pytest.approx(
        row["rad_matchup_avg"] - row["dire_matchup_avg"]
    )

    # Every non-categorical value must be finite and numeric.
    for name, value in row.items():
        if name in ("patch", "patch_version"):
            assert isinstance(value, str)
            continue
        assert np.isfinite(float(value)), name


def test_matchup_data_is_a_known_schema_extension(tmp_path):
    schema, shipped_features = _load_contract()
    frame, _ = _generate(tmp_path, with_matchups=True)
    generated = _generated_names(frame)

    assert len(generated) == 894
    assert set(generated) - set(shipped_features) == MATCHUP_FEATURES
    assert not set(shipped_features) - set(generated)
    assert contract_diff(generated, schema)["unexpected"] == tuple(sorted(MATCHUP_FEATURES))
    assert contract_diff(generated, schema)["missing"] == ()


def test_generated_row_builds_a_finite_schema_aligned_inference_vector(tmp_path):
    schema, shipped_features = _load_contract()
    frame, completion = _generate(tmp_path, with_matchups=False)
    row = frame.iloc[0].to_dict()

    vector = build_vector(row, schema, inference_defaults=True)
    assert vector.shape == (887,)
    assert np.isfinite(vector).all()
    assert vector.dtype == np.float32

    names = schema["feature_names"]
    # Runtime semantics: match-only fields use the schema median, never the
    # logged post-draft value.
    assert vector[names.index("duration")] == pytest.approx(schema["defaults"]["duration"])
    assert vector[names.index("avg_mmr")] == pytest.approx(schema["defaults"]["avg_mmr"])
    assert vector[names.index("game_mode")] == pytest.approx(schema["defaults"]["game_mode"])
    assert vector[names.index("lobby_type")] == pytest.approx(schema["defaults"]["lobby_type"])

    encoded = encode_matrix(frame.iloc[[0]], schema, inference_defaults=True)
    assert encoded.nonfinite_features == ()
    assert encoded.missing_columns == ()
    assert np.allclose(encoded.matrix[0], vector, rtol=0, atol=1e-6)

    import onnxruntime as ort

    session = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    _, probabilities = session.run(None, {input_name: vector.reshape(1, -1)})
    assert probabilities.shape == (1, 2)
    assert np.isfinite(probabilities).all()
    assert probabilities[0, 1] == pytest.approx(probabilities[0, 1].item())
    assert 0.0 <= probabilities[0, 1] <= 1.0

    # The fixture does not produce matchup features, so the generated names
    # must line up with the schema exactly.
    assert contract_diff(_generated_names(frame), schema) == {
        "unexpected": (),
        "missing": (),
    }
    assert len(shipped_features) == 887
