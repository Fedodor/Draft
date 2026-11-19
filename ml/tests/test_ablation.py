"""Ablation-harness tests (fast, no production dataset required).

These pin the properties the experiment depends on:

* the ablation groups are derived from the audited feature table, so every
  removed feature has a recorded availability classification and reason;
* feature selection removes exactly the requested groups and keeps the base
  contract order;
* a derived variant schema stays a valid runtime schema (order, defaults,
  categorical subset);
* the partial-draft simulation clears the same fields the runtime cannot fill;
* variant selection uses validation metrics only, with the documented
  parsimony tie-break;
* an end-to-end run exports a gate-compatible bundle and never scores the test
  partition.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ML_ROOT = Path(__file__).resolve().parents[1]
if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))

pytest.importorskip("xgboost")

from training import ablation  # noqa: E402

MINI_FEATURES = [
    "duration",
    "avg_mmr",
    "game_mode",
    "lobby_type",
    "patch",
    "patch_version",
    "rad_pick_0",
    "rad_pick_1",
    "rad_pick_2",
    "rad_pick_3",
    "rad_pick_4",
    "dire_pick_0",
    "dire_pick_1",
    "dire_pick_2",
    "dire_pick_3",
    "dire_pick_4",
    "rad_has_anti_mage",
    "rad_has_axe",
    "dire_has_axe",
    "rad_has_shadow_fiend",
    "rad_has_phantom_lancer",
    "dire_has_puck",
    "dire_has_pudge",
]

# Hero ids used by the mini frame: 1 anti_mage, 2 axe, 11 shadow_fiend,
# 12 phantom_lancer, 13 puck, 14 pudge (the processor's safe-name rule).
HERO_IDS = (1, 2, 11, 12, 13, 14)


def mini_schema() -> dict:
    return {
        "schema_version": 1,
        "model": "mini",
        "feature_count": len(MINI_FEATURES),
        "feature_names": list(MINI_FEATURES),
        "categoricals": {"patch": ["7.40", "7.41"]},
        "defaults": {name: 0.0 for name in MINI_FEATURES} | {"patch": 1.0, "duration": 2163.0},
        "positive_class_index": 1,
        "input_name": "float_input",
        "label_output": "label",
        "probability_output": "probabilities",
        "source": {},
    }


def mini_frame(rows: int = 400, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rad = rng.choice(HERO_IDS[:3], size=(rows, 5))
    dire = rng.choice(HERO_IDS[3:], size=(rows, 5))
    frame = pd.DataFrame(
        {
            "match_id": np.arange(1, rows + 1),
            "start_time": 1_600_000_000 + np.arange(rows) * 60,
            "radiant_win": rng.integers(0, 2, size=rows),
            "duration": rng.integers(1200, 3400, size=rows),
            "avg_mmr": np.zeros(rows),
            "game_mode": np.full(rows, 22),
            "lobby_type": np.zeros(rows),
            "patch": np.where(np.arange(rows) < rows // 2, 0.0, 1.0),
            "patch_version": np.ones(rows),
        }
    )
    for index in range(5):
        frame[f"rad_pick_{index}"] = rad[:, index]
        frame[f"dire_pick_{index}"] = dire[:, index]
    for name in ("rad_has_anti_mage", "rad_has_axe", "dire_has_axe", "rad_has_shadow_fiend",
                 "rad_has_phantom_lancer", "dire_has_puck", "dire_has_pudge"):
        frame[name] = rng.integers(0, 2, size=rows).astype(float)
    return frame


def test_audited_table_covers_groups_and_unique_features():
    names = [row.name for row in ablation.AUDITED_FEATURES]
    assert len(names) == len(set(names))
    assert set(ablation.GROUPS) == {row.group for row in ablation.AUDITED_FEATURES}
    assert len(ablation.group_features("match_context")) == 6
    assert len(ablation.group_features("pick_order")) == 10
    # Every removed feature must state where it comes from.
    for row in ablation.AUDITED_FEATURES:
        assert row.availability
        assert row.reason
        assert row.group in ablation.GROUPS


def test_variant_feature_names_preserves_order_and_drops_group():
    base = list(MINI_FEATURES)
    variant = next(v for v in ablation.VARIANTS if v.name == "no_pick_order")
    names = ablation.variant_feature_names(base, variant)

    assert names == [name for name in base if "_pick_" not in name]
    assert len(names) == len(base) - 10
    assert names == [name for name in base if name in names]


def test_combined_variant_drops_both_groups():
    variant = next(v for v in ablation.VARIANTS if v.name == "no_match_context_no_pick_order")
    names = ablation.variant_feature_names(MINI_FEATURES, variant)
    assert len(names) == len(MINI_FEATURES) - 16
    assert set(ablation.group_features("match_context")) & set(names) == set()
    assert set(ablation.group_features("pick_order")) & set(names) == set()


def test_variant_feature_names_strict_rejects_missing_group_feature():
    truncated = [name for name in MINI_FEATURES if name != "duration"]
    variant = next(v for v in ablation.VARIANTS if v.name == "no_match_context")
    with pytest.raises(ValueError, match="absent from the contract"):
        ablation.variant_feature_names(truncated, variant)
    # Non-strict mode is available for exploratory runs.
    assert len(ablation.variant_feature_names(truncated, variant, strict=False)) == len(truncated) - 5


def test_variant_schema_keeps_order_defaults_and_categorical_subset():
    schema = mini_schema()
    variant = next(v for v in ablation.VARIANTS if v.name == "no_pick_order")
    names = ablation.variant_feature_names(MINI_FEATURES, variant)
    derived = ablation.variant_schema(schema, names)

    assert derived["feature_names"] == names
    assert derived["feature_count"] == len(names)
    assert set(derived["defaults"]) == set(names)
    assert derived["categoricals"]["patch"] == ["7.40", "7.41"]
    assert derived["positive_class_index"] == 1
    assert derived["probability_output"] == "probabilities"


def test_variant_schema_drops_categorical_when_feature_removed():
    names = [name for name in MINI_FEATURES if name != "patch"]
    derived = ablation.variant_schema(mini_schema(), names)
    assert "patch" not in derived["categoricals"]
    assert "patch" not in derived["defaults"]


def test_variant_schema_rejects_feature_without_default():
    schema = mini_schema()
    schema["defaults"].pop("duration")
    with pytest.raises(ValueError, match="no default"):
        ablation.variant_schema(schema, MINI_FEATURES)


def test_variant_schema_rejects_unknown_feature():
    with pytest.raises(ValueError, match="not present in the base schema"):
        ablation.variant_schema(mini_schema(), list(MINI_FEATURES) + ["not_a_feature"])


def test_hero_safe_names_matches_feature_tokens():
    names = ablation.hero_safe_names()
    assert names[1] == "anti_mage"
    assert names[11] == "shadow_fiend"
    assert names[14] == "pudge"


def _processor(directory: Path | None = None):
    return ablation._load_processor(directory or ML_ROOT / "datasets")


def _seed_winrates(directory: Path) -> None:
    pd.DataFrame(
        [
            {"hero_id": hero_id, "games": 100, "wins": 50, "winrate": 0.40 + hero_id / 100}
            for hero_id in range(1, 140)
        ]
    ).to_parquet(directory / "hero_winrates.parquet", index=False)


def test_partial_draft_never_exposes_unmade_picks(tmp_path):
    """Two drafts that agree on the first pick must score identically at that
    stage — otherwise reconstructed features leak picks the runtime cannot see."""
    _seed_winrates(tmp_path)
    processor = _processor(tmp_path)
    schema = mini_schema()
    frame = mini_frame(rows=2)
    rad = [[1, 2, 11, 12, 2], [1, 11, 2, 12, 2]]
    dire = [[13, 14, 13, 14, 13], [13, 13, 14, 14, 13]]
    for row in range(2):
        for slot in range(5):
            frame.at[row, f"rad_pick_{slot}"] = float(rad[row][slot])
            frame.at[row, f"dire_pick_{slot}"] = float(dire[row][slot])

    at_one = ablation.build_partial_draft_frame(frame, 1, schema=schema, processor=processor)
    at_full = ablation.build_partial_draft_frame(frame, 5, schema=schema, processor=processor)

    vector_one = ablation.encode(at_one, schema, inference_defaults=True)
    vector_full = ablation.encode(at_full, schema, inference_defaults=True)
    assert np.array_equal(vector_one[0], vector_one[1])
    assert not np.array_equal(vector_full[0], vector_full[1])

    # Only the known hero is present; heroes that appear later are absent, and
    # their per-hero win rate is never emitted for this stage.
    assert at_one["rad_has_anti_mage"].tolist() == [1.0, 1.0]
    assert at_one["rad_has_shadow_fiend"].tolist() == [0.0, 0.0]
    assert "rad_wr_axe" not in at_one.columns
    assert "rad_wr_anti_mage" in at_one.columns

    # A full draft still reproduces the slots it was given, in order.
    assert at_full["rad_pick_3"].tolist() == [12.0, 12.0]
    assert at_full["dire_pick_0"].tolist() == [13.0, 13.0]


def test_partial_draft_zero_picks_matches_runtime_zeroes():
    processor = _processor()
    schema = mini_schema()
    frame = mini_frame(rows=3)
    empty = ablation.build_partial_draft_frame(frame, 0, schema=schema, processor=processor)

    # `ml.rs` divides semantic sums by picks.len().max(1) and writes 0.0.
    assert (empty["radiant_avg_stun"] == 0.0).all()
    assert (empty["stun_diff"] == 0.0).all()
    # No team has a hero, so every presence flag is clear.
    assert (empty[[c for c in empty.columns if c.startswith("rad_has_")]].sum(axis=1) == 0).all()


def _mask_frame() -> pd.DataFrame:
    tokens = ["anti_mage", "axe", "shadow_fiend", "phantom_lancer", "puck"]
    dire = ["pudge", "puck", "anti_mage", "axe", "shadow_fiend"]
    rows_rad = [[1, 2, 11, 12, 13], [1, 2, 11, 12, 13], [1, 2, 11, 12, 12]]
    rows_dire = [[14, 13, 1, 2, 11]] * 3
    data = {}
    for token in set(tokens) | set(dire):
        data[f"rad_has_{token}"] = [1.0 if token in tokens else 0.0] * 3
        data[f"dire_has_{token}"] = [1.0 if token in dire else 0.0] * 3
        data[f"rad_ban_{token}"] = [0.0, 0.0, 0.0]
        data[f"dire_ban_{token}"] = [0.0, 0.0, 0.0]
    for slot in range(5):
        data[f"rad_pick_{slot}"] = [float(row[slot]) for row in rows_rad]
        data[f"dire_pick_{slot}"] = [float(row[slot]) for row in rows_dire]
    frame = pd.DataFrame(data)
    # Row 1: a hero that is both picked and banned.
    frame.loc[1, "rad_ban_anti_mage"] = 1.0
    # Row 2: anti_mage dropped from presence (a hero the runtime would still
    # have) but the *pick slots* now repeat phantom_lancer.
    return frame


def test_duplicate_hero_mask_flags_collisions_and_short_presence():
    mask = ablation.duplicate_hero_mask(_mask_frame())
    assert mask.tolist() == [False, True, True]
    assert "picked and banned" in ablation.DUPLICATE_HERO_RULE
    # Presence sums are not part of the rule: a picked hero above the one-hot
    # range would otherwise look like a duplicate.
    assert "presence sums are deliberately" in ablation.DUPLICATE_HERO_RULE


def test_duplicate_hero_mask_rejects_a_frame_without_one_hots():
    with pytest.raises(ValueError, match="presence/ban"):
        ablation.duplicate_hero_mask(pd.DataFrame({"match_id": [1, 2]}))


def test_frozen_split_is_applied_and_verified(tmp_path):
    frame = pd.DataFrame(
        {
            "match_id": np.arange(101, 107),
            "start_time": np.arange(1, 7) * 1000,
            "radiant_win": [1, 1, 1, 1, 1, 0],
        }
    )
    config = {
        "dataset": {"sha256": ""},
        "rule": {"fractions": {"train": 0.5, "validation": 0.33, "test": 0.17}},
        "partitions": {
            "train": {"start_time_max": 3000, "rows": 3},
            "validation": {"start_time_max": 5000, "rows": 2},
            "test": {"rows": 1},
        },
        "contamination": {"untouched": False},
    }
    config_path = tmp_path / "split.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    train, val, test, info = ablation.load_frozen_split(frame, config_path)
    assert (len(train), len(val), len(test)) == (3, 2, 1)
    assert info["source"] == "frozen_config"
    assert info["boundaries"]["validation_max"] == 5000
    assert train["start_time"].max() <= val["start_time"].min()
    assert val["start_time"].max() <= test["start_time"].min()
    assert not set(train["match_id"]) & set(test["match_id"])

    # A re-cut dataset must be refused rather than silently re-split.
    config["partitions"]["train"]["rows"] = 4
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(SystemExit, match="frozen split produced"):
        ablation.load_frozen_split(frame, config_path)


V2_DATASET = ML_ROOT / "datasets" / "processed_v2" / "draft_features.parquet"


@pytest.mark.skipif(not V2_DATASET.is_file(), reason="corrected dataset is local-only")
def test_duplicate_mask_matches_the_documented_dataset_count():
    frame = pd.read_parquet(V2_DATASET)
    mask = ablation.duplicate_hero_mask(frame)
    assert int(mask.sum()) == 3_870
    # The rule must not swallow the majority of the corpus.
    assert int(mask.sum()) < 0.05 * len(frame)


def _validation_result(name: str, auc: float, brier: float, feature_count: int) -> dict:
    return {
        "name": name,
        "features": ["f"] * feature_count,
        "validation": {
            "inference": {
                "metrics": {"roc_auc": auc, "brier_score": brier},
            }
        },
    }


def test_select_variant_prefers_higher_auc_beyond_tolerance():
    results = [
        _validation_result("baseline", 0.5122, 0.2495, 887),
        _validation_result("no_match_context", 0.5300, 0.2480, 881),
    ]
    assert ablation.select_variant(results)["name"] == "no_match_context"


def test_select_variant_prefers_lower_brier_within_tolerance():
    results = [
        _validation_result("baseline", 0.5200, 0.2490, 887),
        _validation_result("no_pick_order", 0.5195, 0.2470, 877),
    ]
    winner = ablation.select_variant(results)
    assert winner["name"] == "no_pick_order"


def test_select_variant_prefers_fewer_features_on_full_tie():
    results = [
        _validation_result("baseline", 0.5200, 0.2480, 887),
        _validation_result("no_pick_order", 0.5200, 0.2480, 877),
    ]
    assert ablation.select_variant(results)["name"] == "no_pick_order"


def test_fit_variant_refits_at_selected_rounds():
    frame = mini_frame(rows=300)
    schema = mini_schema()
    config = dict(ablation.WIDE_CONFIG)
    config.update(
        {
            "n_estimators": 12,
            "early_stopping_rounds": 2,
            "device": "cpu",
            "nthread": 2,
            "verbosity": 0,
        }
    )
    model, info = ablation.fit_variant(
        frame.iloc[:200], frame.iloc[200:250], schema, config, seed=3
    )

    assert info.rounds >= 1
    assert model.get_booster().num_boosted_rounds() == info.rounds
    if info.probe_best_iteration is not None:
        assert info.rounds == info.probe_best_iteration + 1

    metrics = ablation.evaluate(
        model,
        ablation.encode(frame.iloc[250:], schema, inference_defaults=True),
        frame.iloc[250:]["radiant_win"].astype(int).to_numpy(),
        train_prior=0.5,
        calibration_bins=5,
    )
    assert metrics["metrics"]["rows"] == len(frame.iloc[250:])
    assert 0.0 <= metrics["metrics"]["roc_auc"] <= 1.0
    assert np.isfinite(metrics["metrics"]["brier_score"])
    assert set(metrics["baselines"]) == {"constant_train_prior", "constant_half"}


def test_run_exports_gate_bundle_and_never_scores_test(tmp_path):
    data_path = tmp_path / "mini.parquet"
    schema_path = tmp_path / "mini_schema.json"
    report_path = tmp_path / "report.json"
    mini_frame().to_parquet(data_path, index=False)
    schema_path.write_text(json.dumps(mini_schema()), encoding="utf-8")

    code = ablation.main(
        [
            "--data",
            str(data_path),
            "--schema",
            str(schema_path),
            "--report",
            str(report_path),
            "--model-dir",
            str(tmp_path / "candidates"),
            "--n-estimators",
            "8",
            "--early-stopping-rounds",
            "2",
            "--threads",
            "2",
            "--calibration-bins",
            "5",
            "--only",
            "baseline,no_match_context",
        ]
    )

    assert code == 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["split"]["test_scored_here"] is False
    assert report["selection"]["selected"] in {"baseline", "no_match_context"}
    assert [variant["name"] for variant in report["variants"]] == [
        "baseline",
        "no_match_context",
    ]
    # Test labels never appear in any reported metric block.
    assert all("test" not in variant for variant in report["variants"])

    export = report["winner_export"]
    candidate_dir = Path(export["model_dir"])
    stem = export["model_stem"]
    assert (candidate_dir / f"{stem}.onnx").is_file()
    assert (candidate_dir / f"{stem}_schema.json").is_file()
    assert (candidate_dir / f"{stem}_features.json").is_file()
    assert (candidate_dir / f"{stem}_manifest.json").is_file()

    manifest = json.loads((candidate_dir / f"{stem}_manifest.json").read_text(encoding="utf-8"))
    assert set(manifest["metrics"]) == {"validation"}
    assert manifest["dataset"]["rows"] == 400
    assert manifest["feature_schema"]["feature_count"] == len(manifest["feature_schema"]["feature_names"])
    assert export["onnx_parity_max_abs_diff"] < 1e-4

    schema = json.loads((candidate_dir / f"{stem}_schema.json").read_text(encoding="utf-8"))
    features = json.loads((candidate_dir / f"{stem}_features.json").read_text(encoding="utf-8"))
    assert schema["feature_names"] == features


def test_run_honours_a_frozen_split_config(tmp_path):
    """The frozen boundaries must drive the split (not a fresh 80/10/10 cut),
    and the export path must work when they do."""
    frame = mini_frame(rows=200)
    data_path = tmp_path / "mini.parquet"
    schema_path = tmp_path / "mini_schema.json"
    report_path = tmp_path / "report.json"
    frame.to_parquet(data_path, index=False)
    schema_path.write_text(json.dumps(mini_schema()), encoding="utf-8")

    ordered = frame.sort_values(["start_time", "match_id"], kind="mergesort")
    train_rows, val_rows = 160, 20
    train_max = int(ordered["start_time"].iloc[train_rows - 1])
    val_max = int(ordered["start_time"].iloc[train_rows + val_rows - 1])
    config = {
        "dataset": {"sha256": ""},
        "rule": {"fractions": {"train": 0.8, "validation": 0.1, "test": 0.1}},
        "partitions": {
            "train": {"start_time_max": train_max, "rows": train_rows},
            "validation": {"start_time_max": val_max, "rows": val_rows},
            "test": {"rows": 200 - train_rows - val_rows},
        },
        "contamination": {"untouched": False},
    }
    config_path = tmp_path / "split.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    code = ablation.main(
        [
            "--data",
            str(data_path),
            "--schema",
            str(schema_path),
            "--split-config",
            str(config_path),
            "--report",
            str(report_path),
            "--model-dir",
            str(tmp_path / "candidates"),
            "--n-estimators",
            "6",
            "--early-stopping-rounds",
            "2",
            "--threads",
            "2",
            "--calibration-bins",
            "5",
            "--only",
            "baseline",
        ]
    )

    assert code == 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["split"]["source"] == "frozen_config"
    assert report["split"]["sizes"] == {
        "train": train_rows,
        "validation": val_rows,
        "test": 200 - train_rows - val_rows,
    }
    assert report["split"]["train_rows"] == train_rows
    assert report["split"]["test_scored_here"] is False
    assert report["population"]["label"] == "all_rows"
