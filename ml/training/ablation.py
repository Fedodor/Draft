#!/usr/bin/env python3
"""Controlled feature-ablation experiment for the draft model.

The bundled model fails the quality gate in `inference` mode. Before assuming a
different model family or more data will help, this script isolates whether
individual *feature groups* hurt draft-time prediction or create train/serve
skew. Every variant keeps the model family, hyperparameters, split, and seed
fixed so the comparison measures features only.

Method (validation-only selection):

1. Reconstruct the chronological split from `training.splits` (identical to the
   gate: 80/10/10 on `start_time`, match-level isolation).
2. Fit each variant on the training partition with early stopping against the
   validation partition, then refit once at the selected round count so the
   exported ONNX contains exactly the trees that were evaluated.
3. Score the validation partition twice for every variant:
   * `as_trained` - logged feature values (what training saw);
   * `inference`  - the runtime contract, where post-draft/match-metadata
     features are replaced by their schema defaults.
4. Select a winner using validation `inference` ROC-AUC (with a parsimony
   tie-break), export it, and stop. **The test partition is never scored here.**
   The only test measurement is `quality_gate.py`, run once against the
   exported candidate.

Usage:
    python ml/training/ablation.py
    python ml/training/ablation.py --report ml/models/ablation/report.json
    python ml/training/ablation.py --only baseline,no_match_context
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

ML_ROOT = Path(__file__).resolve().parent.parent
if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))

import numpy as np
import pandas as pd
import xgboost as xgb

from training.inference_contract import default_for, encode_matrix
from training.manifest import build_manifest, manifest_paths, write_manifest
from training.quality_gate import baseline_metrics, compute_metrics
from training.splits import file_sha256, schema_fingerprint, time_split
from training.train_full import WIDE_CONFIG, _export_onnx

DEFAULT_DATA = ML_ROOT / "datasets" / "processed" / "draft_features.parquet"
DEFAULT_SPLIT_CONFIG = ML_ROOT / "configs" / "eval_split_v2.json"
DEFAULT_MODEL_DIR = ML_ROOT / "models"
DEFAULT_REPORT = DEFAULT_MODEL_DIR / "ablation" / "ablation_report.json"
BASELINE_STEM = "draft_model_xgboost_full"
BASELINE_SCHEMA = DEFAULT_MODEL_DIR / f"{BASELINE_STEM}_schema.json"
DEFAULT_PATCH = "7.41"
EXPERIMENT_VERSION = 1

# Ablation groups are derived from the audited feature table below so the
# experiment and its justification cannot drift apart.
PICK_SLOTS = 5


@dataclass(frozen=True)
class AuditedFeature:
    """One feature that is ablated, with its availability classification.

    `availability` describes *when* the value exists relative to a
    recommendation:

    * `post_match`      - only known after the game ends;
    * `match_metadata`  - known when the lobby loads, but the runtime does not
      receive it (it substitutes a constant);
    * `patch_metadata`  - known live, but encoded inconsistently with training;
    * `partial_draft`   - known for already-made picks, unknown for open slots.
    """

    name: str
    group: str
    availability: str
    reason: str


def _pick_features() -> Tuple[AuditedFeature, ...]:
    rows = []
    for side in ("rad", "dire"):
        for index in range(PICK_SLOTS):
            rows.append(
                AuditedFeature(
                    name=f"{side}_pick_{index}",
                    group="pick_order",
                    availability="partial_draft",
                    reason=(
                        "Raw hero id in final pick slot; the runtime writes it only "
                        "for filled slots and falls back to a real hero id otherwise, "
                        "so an open slot looks like a specific hero that the presence "
                        "vector says is absent."
                    ),
                )
            )
    return tuple(rows)


# Every ablated feature, classified individually. Nothing is dropped merely
# because its name looks suspicious: each entry states where the value comes
# from and why it is (or is not) available when a recommendation is generated.
AUDITED_FEATURES: Tuple[AuditedFeature, ...] = (
    AuditedFeature(
        "duration",
        "match_context",
        "post_match",
        "Match length exists only after the game; the runtime pins it to the "
        "training median (2163 s) for every request.",
    ),
    AuditedFeature(
        "avg_mmr",
        "match_context",
        "match_metadata",
        "Constant 0 across the whole shipped dataset, so it carries no signal; "
        "the runtime also sends 0.",
    ),
    AuditedFeature(
        "game_mode",
        "match_context",
        "match_metadata",
        "Known at lobby load but not delivered to the engine; 89% of rows are "
        "mode 22, which is also the runtime default.",
    ),
    AuditedFeature(
        "lobby_type",
        "match_context",
        "match_metadata",
        "Known at lobby load but not delivered to the engine; 95% of rows are "
        "lobby type 0, which is also the runtime default.",
    ),
    AuditedFeature(
        "patch",
        "match_context",
        "patch_metadata",
        "Known live, but training rows span 23 patches while the runtime always "
        "substitutes the exported default, so the model never sees a real value "
        "at serve time.",
    ),
    AuditedFeature(
        "patch_version",
        "match_context",
        "patch_metadata",
        "Constant 'unknown' for all 81,694 training rows, so it cannot be "
        "learned; the schema default rewrites it to '7.41' at inference.",
    ),
) + _pick_features()


GROUPS: Dict[str, str] = {
    "match_context": (
        "Post-match and match-metadata fields the runtime substitutes with "
        "schema defaults (duration, avg_mmr, game_mode, lobby_type, patch, "
        "patch_version)."
    ),
    "pick_order": (
        "Raw ordinal pick slots rad_pick_0..4 / dire_pick_0..4, whose unfilled "
        "values fall back to real hero ids."
    ),
}


@dataclass(frozen=True)
class AblationVariant:
    name: str
    description: str
    ablated_groups: Tuple[str, ...] = ()

    def dropped_features(self) -> Tuple[str, ...]:
        dropped: List[str] = []
        for row in AUDITED_FEATURES:
            if row.group in self.ablated_groups and row.name not in dropped:
                dropped.append(row.name)
        return tuple(dropped)


VARIANTS: Tuple[AblationVariant, ...] = (
    AblationVariant(
        "baseline",
        "All shipped features; reproduces the bundled contract under the "
        "runtime substitution rules.",
    ),
    AblationVariant(
        "no_match_context",
        "Drops post-match and match-metadata features that the runtime replaces "
        "with constants.",
        ("match_context",),
    ),
    AblationVariant(
        "no_pick_order",
        "Drops the ten raw ordinal pick-slot features.",
        ("pick_order",),
    ),
    AblationVariant(
        "no_match_context_no_pick_order",
        "Combined ablation of both groups.",
        ("match_context", "pick_order"),
    ),
)

SELECTION_RULE = (
    "Highest validation inference ROC-AUC; among variants within 0.002 AUC of "
    "the leader, the lower validation Brier score wins, then the smaller feature "
    "count. Validation only: the test partition is scored exactly once, by "
    "quality_gate.py, after this selection is fixed."
)

# Explicit CPU execution: the shipped pipeline auto-selects CUDA, which is
# faster but makes tree-level reproducibility dependent on the GPU build.
DEFAULT_DEVICE = "cpu"
DEFAULT_THREADS = 8


@dataclass
class FitInfo:
    """How the final (exported) model was fitted."""

    rounds: int
    probe_best_iteration: Optional[int]
    probe_best_score: Optional[float]
    probe_rounds_run: int
    seconds: float


def hero_safe_names(heroes_path: Path = ML_ROOT / "datasets" / "heroes.json") -> Dict[int, str]:
    """Map hero id -> feature-name token using the processor's naming rule."""
    roster = json.loads(Path(heroes_path).read_text(encoding="utf-8"))
    names: Dict[int, str] = {}
    for key, value in roster.items():
        hero_id = int(value.get("id", key))
        label = str(value.get("localized_name") or f"hero_{hero_id}")
        names[hero_id] = (
            label.lower().replace(" ", "_").replace("'", "").replace("-", "_")
        )
    for hero_id in range(1, 140):
        names.setdefault(hero_id, f"hero_{hero_id}")
    return names


def group_features(group: str) -> Tuple[str, ...]:
    """Features belonging to one ablation group, in audit order."""
    if group not in GROUPS:
        raise ValueError(f"unknown ablation group {group!r}; known: {sorted(GROUPS)}")
    return tuple(row.name for row in AUDITED_FEATURES if row.group == group)


def variant_feature_names(
    base_names: Sequence[str],
    variant: AblationVariant,
    *,
    strict: bool = True,
) -> List[str]:
    """Ordered feature list for `variant`, preserving the base contract order.

    With `strict=True` every feature the variant claims to ablate must exist in
    `base_names`; a silent no-op ablation would make the experiment meaningless.
    """
    dropped = set(variant.dropped_features())
    present = set(base_names)
    if strict:
        missing = sorted(dropped - present)
        if missing:
            raise ValueError(
                f"variant {variant.name!r} ablates features absent from the "
                f"contract: {missing[:10]}"
            )
    return [name for name in base_names if name not in dropped]


def variant_schema(
    base_schema: Mapping[str, Any],
    feature_names: Sequence[str],
) -> Dict[str, Any]:
    """Restrict the shipped schema to `feature_names`, keeping order and codes.

    The result is a valid runtime schema: the runtime reads feature order,
    categorical code lists, and per-feature defaults from it. Defaults are kept
    from the base schema (they are training medians of the same dataset), so a
    variant differs only by which features exist.
    """
    names = list(feature_names)
    if len(set(names)) != len(names):
        raise ValueError("feature_names contains duplicates")
    keep = set(names)
    known = set(base_schema.get("feature_names", ()))
    unknown = [name for name in names if name not in known]
    if unknown:
        raise ValueError(f"features not present in the base schema: {unknown[:10]}")

    defaults = {
        name: float(value)
        for name, value in (base_schema.get("defaults") or {}).items()
        if name in keep
    }
    missing_defaults = [name for name in names if name not in defaults]
    if missing_defaults:
        raise ValueError(
            f"base schema has no default for {missing_defaults[:10]}; the runtime "
            "would fall back to 0.0 silently"
        )

    categoricals = {
        name: list(categories)
        for name, categories in (base_schema.get("categoricals") or {}).items()
        if name in keep
    }
    return {
        "schema_version": 1,
        "model": base_schema.get("model"),
        "feature_count": len(names),
        "feature_names": names,
        "categoricals": categoricals,
        "defaults": defaults,
        "positive_class_index": int(base_schema.get("positive_class_index", 1)),
        "input_name": str(base_schema.get("input_name", "float_input")),
        "label_output": str(base_schema.get("label_output", "label")),
        "probability_output": str(base_schema.get("probability_output", "probabilities")),
        "source": {
            "ablation_of": base_schema.get("model"),
            "derived_from": str(BASELINE_SCHEMA.relative_to(ML_ROOT)),
        },
    }


def encode(
    frame: pd.DataFrame,
    schema: Mapping[str, Any],
    *,
    inference_defaults: bool,
) -> np.ndarray:
    """Encode with the schema and refuse non-finite output."""
    result = encode_matrix(frame, schema, inference_defaults=inference_defaults)
    if result.nonfinite_features:
        raise SystemExit(
            "encoded matrix has non-finite features: "
            f"{list(result.nonfinite_features)[:10]}"
        )
    return result.matrix


DUPLICATE_HERO_RULE = (
    "A row is flagged when (a) the same hero has both a presence flag and the "
    "matching ban flag set for one side (picked and banned), or (b) a side's five "
    "pick slots repeat a hero id. Both conditions are recomputable from the stored "
    "features alone. As of the 2026-10-28 corpus, (a) flags 3,870 rows and (b) "
    "flags none; presence sums are deliberately *not* used, because a picked hero "
    "with id > 139 has no one-hot column and would be misread as a duplicate."
)


def duplicate_hero_mask(frame: pd.DataFrame) -> np.ndarray:
    """Boolean mask of rows whose draft contains a hero twice (see the rule).

    Duplicates are a source-data contradiction: the same hero is picked twice by
    one team, or picked after being banned. They are not a parsing artifact, so
    they are reported and optionally excluded rather than silently repaired.
    """
    rad_has = [c for c in frame.columns if c.startswith("rad_has_")]
    dire_has = [c for c in frame.columns if c.startswith("dire_has_")]
    rad_ban = [c for c in frame.columns if c.startswith("rad_ban_") and c != "rad_ban_avg_winrate"]
    dire_ban = [c for c in frame.columns if c.startswith("dire_ban_") and c != "dire_ban_avg_winrate"]
    if not rad_has or not rad_ban or not dire_has or not dire_ban:
        raise ValueError("frame has no presence/ban one-hot columns; wrong dataset?")

    mask = np.zeros(len(frame), dtype=bool)
    rad_picked = frame[rad_has].to_numpy() > 0
    dire_picked = frame[dire_has].to_numpy() > 0
    mask |= (rad_picked & (frame[rad_ban].to_numpy() > 0)).any(axis=1)
    mask |= (dire_picked & (frame[dire_ban].to_numpy() > 0)).any(axis=1)

    for side in ("rad", "dire"):
        slots = [f"{side}_pick_{index}" for index in range(PICK_SLOTS)]
        if not all(column in frame.columns for column in slots):
            continue
        for row_index, row in enumerate(frame[slots].to_numpy(dtype="float64")):
            ids = [int(value) for value in row if np.isfinite(value) and value > 0]
            if len(ids) != len(set(ids)):
                mask[row_index] = True
    return mask


def load_frozen_split(
    dataset: pd.DataFrame,
    config_path: Path,
    *,
    data_path: Optional[Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Apply the frozen boundary timestamps from `ml/configs/eval_split_v2.json`.

    The config is the contract: its dataset SHA-256 must match the file being
    read, and the resulting partition sizes must match the recorded counts. A
    mismatch is a hard error, not a warning, because silently re-cutting the
    split is how a holdout stops being one.
    """
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    expected_sha = config.get("dataset", {}).get("sha256")
    if data_path is not None and expected_sha:
        actual = file_sha256(Path(data_path))
        if actual != expected_sha:
            raise SystemExit(
                f"dataset sha256 {actual[:16]}… does not match the frozen split "
                f"config {expected_sha[:16]}…; refusing to apply it"
            )
    partitions = config["partitions"]
    train_max = float(partitions["train"]["start_time_max"])
    val_max = float(partitions["validation"]["start_time_max"])

    ordered = dataset.sort_values(["start_time", "match_id"], kind="mergesort")
    train = ordered[ordered["start_time"] <= train_max]
    val = ordered[(ordered["start_time"] > train_max) & (ordered["start_time"] <= val_max)]
    test = ordered[ordered["start_time"] > val_max]

    sizes = {"train": len(train), "validation": len(val), "test": len(test)}
    expected = {name: int(partitions[name]["rows"]) for name in sizes}
    if sizes != expected:
        raise SystemExit(
            f"frozen split produced {sizes} but the config records {expected}; "
            "the dataset or the config changed, so neither can be trusted"
        )
    info = {
        "source": "frozen_config",
        "config": str(config_path),
        "dataset_sha256_verified": bool(expected_sha) and data_path is not None,
        "boundaries": {
            "train_max": train_max,
            "validation_max": val_max,
        },
        "fractions": config["rule"]["fractions"],
        "time_column": "start_time",
        "used_start_time": True,
        "sizes": sizes,
        "contamination": config.get("contamination", {}),
    }
    return train, val, test, info


def _load_processor(dataset_dir: Path):
    """Import `feature-engineering/processor.py` and load its hero tables.

    The partial-draft simulation rebuilds feature rows with the *production*
    feature code instead of re-deriving it here, so the simulation cannot drift
    from the dataset it is applied to.
    """
    feature_dir = ML_ROOT / "feature-engineering"
    if str(feature_dir) not in sys.path:
        sys.path.insert(0, str(feature_dir))
    import processor  # noqa: PLC0415 - deliberate late import

    return processor.FeatureProcessor(
        raw_dir=dataset_dir,
        output_dir=dataset_dir,
        include_matchups=False,
        include_allpick=False,
    )


def _pick_and_ban_ids(frame: pd.DataFrame, token_to_id: Mapping[str, int]):
    """Per-row pick slots and ban sets as hero ids (preserves draft order)."""
    rad_picks = frame[[f"rad_pick_{i}" for i in range(PICK_SLOTS)]].to_numpy(dtype="float64")
    dire_picks = frame[[f"dire_pick_{i}" for i in range(PICK_SLOTS)]].to_numpy(dtype="float64")
    rad_bans = []
    dire_bans = []
    rad_ban_columns = [c for c in frame.columns if c.startswith("rad_ban_") and c != "rad_ban_avg_winrate"]
    dire_ban_columns = [c for c in frame.columns if c.startswith("dire_ban_") and c != "dire_ban_avg_winrate"]
    for row in frame[rad_ban_columns].to_numpy() > 0:
        rad_bans.append([token_to_id[c[len("rad_ban_"):]] for c, keep in zip(rad_ban_columns, row) if keep])
    for row in frame[dire_ban_columns].to_numpy() > 0:
        dire_bans.append([token_to_id[c[len("dire_ban_"):]] for c, keep in zip(dire_ban_columns, row) if keep])
    return rad_picks, dire_picks, rad_bans, dire_bans


def _empty_pick_row(
    processor,
    schema: Mapping[str, Any],
    rad_bans: Sequence[int],
    dire_bans: Sequence[int],
) -> Dict[str, Any]:
    """Feature row for a draft with no picks yet, matching `ml.rs` exactly.

    The runtime averages semantics over `picks.len().max(1)`, so with zero picks
    it *writes* 0.0 for every semantic average and every differential, and omits
    the win-rate/synergy aggregates entirely (they fall back to schema defaults).
    """
    row: Dict[str, Any] = {}
    rad_ban_ids = set(rad_bans)
    dire_ban_ids = set(dire_bans)
    for hero_id, token in processor._safe_names.items():
        row[f"rad_has_{token}"] = 0.0
        row[f"dire_has_{token}"] = 0.0
        row[f"rad_ban_{token}"] = 1.0 if hero_id in rad_ban_ids else 0.0
        row[f"dire_ban_{token}"] = 1.0 if hero_id in dire_ban_ids else 0.0
    for key in processor._sem_keys:
        row[f"radiant_avg_{key}"] = 0.0
        row[f"dire_avg_{key}"] = 0.0
    for key in (
        "stun", "initiation", "save", "push", "scaling", "durability",
        "teamfight", "magic_damage", "physical_damage", "disable_count",
    ):
        row[f"{key}_diff"] = 0.0
    return row


def build_partial_draft_frame(
    frame: pd.DataFrame,
    level: int,
    *,
    schema: Mapping[str, Any],
    processor=None,
) -> pd.DataFrame:
    """Rebuild every row as the runtime would see it with `level` picks per side.

    The previous simulation only cleared presence flags and pick slots, leaving
    the per-hero win rates, win-rate aggregates, semantics averages, synergy and
    damage-diversity features computed from the *complete* draft. Those features
    are recomputed by the runtime from the picks it actually knows, so keeping
    them leaked future picks into every partial-draft number. Here the row is
    regenerated by the production feature code from the truncated draft, and any
    feature the runtime would not emit falls back to its schema default.

    Bans are kept at their full-draft values (the simplification): in Captain's
    Mode most bans are already known when the first picks appear, and ban
    timing is not modelled.
    """
    level = max(0, min(PICK_SLOTS, int(level)))
    processor = processor or _load_processor(ML_ROOT / "datasets")
    token_to_id = {token: hero_id for hero_id, token in processor._safe_names.items()}
    rad_picks, dire_picks, rad_bans, dire_bans = _pick_and_ban_ids(frame, token_to_id)

    records: List[Dict[str, Any]] = []
    for index in range(len(frame)):
        if level == 0:
            records.append(_empty_pick_row(processor, schema, rad_bans[index], dire_bans[index]))
            continue
        entries: List[Dict[str, Any]] = []
        for slot in range(level):
            hero = int(rad_picks[index][slot])
            entries.append({"hero_id": hero, "team": 0, "is_pick": True, "order": slot})
        for slot in range(level):
            hero = int(dire_picks[index][slot])
            entries.append({"hero_id": hero, "team": 1, "is_pick": True, "order": PICK_SLOTS + slot})
        for offset, hero in enumerate(rad_bans[index]):
            entries.append({"hero_id": hero, "team": 0, "is_pick": False, "order": 100 + offset})
        for offset, hero in enumerate(dire_bans[index]):
            entries.append({"hero_id": hero, "team": 1, "is_pick": False, "order": 200 + offset})
        payload: Dict[str, Any] = {
            "match_id": int(frame["match_id"].iat[index]),
            "start_time": int(frame["start_time"].iat[index]),
            "radiant_win": bool(frame["radiant_win"].iat[index]),
            "picks_bans": entries,
        }
        for column, key in (
            ("duration", "duration"),
            ("avg_mmr", "avg_mmr"),
            ("game_mode", "game_mode"),
            ("lobby_type", "lobby_type"),
            ("patch", "patch"),
        ):
            if column in frame.columns:
                payload[key] = frame[column].iat[index]
        record = processor.process_match(payload, allow_partial_draft=True)
        if record is None:  # pragma: no cover - guards a truncated-draft bug
            raise SystemExit(
                f"partial draft at level {level} was rejected for match "
                f"{int(frame['match_id'].iat[index])}"
            )
        records.append(record)

    out = pd.DataFrame(records)
    categoricals = set(schema.get("categoricals", {}))
    missing: Dict[str, Any] = {}
    for name in schema["feature_names"]:
        if name in categoricals:
            continue
        default = default_for(schema, name)
        if name not in out.columns:
            missing[name] = default
        elif out[name].isna().any():
            # A feature the runtime would omit must look like its default, not
            # like a zero fill (they coincide today only because every sparse
            # default is 0.0).
            out[name] = out[name].fillna(default)
    if missing:
        out = pd.concat(
            [
                out,
                pd.DataFrame(
                    {name: np.full(len(out), value) for name, value in missing.items()},
                    index=out.index,
                ),
            ],
            axis=1,
        )
    out["match_id"] = frame["match_id"].to_numpy()
    out["radiant_win"] = frame["radiant_win"].to_numpy()
    return out


def _predict(model: "xgb.XGBClassifier", matrix: np.ndarray) -> np.ndarray:
    probabilities = model.predict_proba(matrix)[:, 1].astype(np.float64)
    if not np.isfinite(probabilities).all():
        raise SystemExit("model produced non-finite probabilities")
    if ((probabilities < 0.0) | (probabilities > 1.0)).any():
        raise SystemExit("model produced probabilities outside [0, 1]")
    return probabilities


def evaluate(
    model: "xgb.XGBClassifier",
    matrix: np.ndarray,
    labels: np.ndarray,
    *,
    train_prior: float,
    calibration_bins: int,
) -> Dict[str, Any]:
    """Metrics plus constant baselines on the same rows."""
    return {
        "metrics": compute_metrics(labels, _predict(model, matrix), calibration_bins=calibration_bins),
        "baselines": baseline_metrics(labels, train_prior),
    }


def fit_variant(
    train_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
    schema: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    seed: int,
) -> Tuple["xgb.XGBClassifier", FitInfo]:
    """Fit with early stopping, then refit at the selected round count.

    XGBoost's sklearn wrapper predicts with `best_iteration` after early
    stopping but the ONNX converter exports the whole booster, so exporting the
    probe model would put untrained-on-validation trees into the bundle. The
    refit makes the exported tree count explicit and identical to the evaluated
    one.
    """
    start = time.time()
    X_train = encode(train_frame, schema, inference_defaults=False)
    y_train = train_frame["radiant_win"].astype(int).to_numpy()
    X_val = encode(val_frame, schema, inference_defaults=False)
    y_val = val_frame["radiant_win"].astype(int).to_numpy()

    probe_config = dict(config)
    probe_config["random_state"] = seed
    probe = xgb.XGBClassifier(**probe_config)
    probe.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    best_iteration = getattr(probe, "best_iteration", None)
    rounds = int(best_iteration) + 1 if best_iteration is not None else int(
        probe_config.get("n_estimators", 100)
    )
    rounds = max(1, rounds)
    probe_rounds = 0
    probe_score: Optional[float] = None
    try:
        probe_results = probe.evals_result()
        if probe_results:
            last = list(probe_results.values())[0]
            metric_key = next(iter(last))
            series = last[metric_key]
            probe_rounds = int(len(series))
            if series and best_iteration is not None and int(best_iteration) < len(series):
                probe_score = float(series[int(best_iteration)])
    except Exception:  # pragma: no cover - diagnostics only
        probe_score = None

    final_config = {k: v for k, v in config.items() if k != "early_stopping_rounds"}
    final_config["n_estimators"] = rounds
    final_config["random_state"] = seed
    model = xgb.XGBClassifier(**final_config)
    model.fit(X_train, y_train, verbose=False)

    return model, FitInfo(
        rounds=rounds,
        probe_best_iteration=best_iteration,
        probe_best_score=probe_score,
        probe_rounds_run=probe_rounds,
        seconds=time.time() - start,
    )


def select_variant(results: Sequence[Mapping[str, Any]], tolerance: float = 0.002) -> Mapping[str, Any]:
    """Pick a winner from validation results (never test)."""
    if not results:
        raise ValueError("no variant results to select from")
    ranked = sorted(
        results,
        key=lambda r: (
            -float(r["validation"]["inference"]["metrics"]["roc_auc"]),
            float(r["validation"]["inference"]["metrics"]["brier_score"]),
            len(r["features"]),
        ),
    )
    leader = float(ranked[0]["validation"]["inference"]["metrics"]["roc_auc"])
    near = [
        r
        for r in ranked
        if leader - float(r["validation"]["inference"]["metrics"]["roc_auc"]) <= tolerance
    ]
    return min(
        near,
        key=lambda r: (
            float(r["validation"]["inference"]["metrics"]["brier_score"]),
            len(r["features"]),
            -float(r["validation"]["inference"]["metrics"]["roc_auc"]),
        ),
    )


def export_candidate(
    model: "xgb.XGBClassifier",
    schema: Mapping[str, Any],
    *,
    model_dir: Path,
    model_stem: str,
    data_path: Path,
    data_sha: str,
    data_rows: int,
    split_info: Mapping[str, Any],
    config: Mapping[str, Any],
    fit_info: FitInfo,
    validation: Mapping[str, Any],
    patch: str,
    val_frame: pd.DataFrame,
) -> Dict[str, Any]:
    """Write a gate-compatible artifact bundle for the selected candidate.

    The manifest deliberately carries validation metrics only: the test
    partition must be scored by `quality_gate.py`, once, so that the reported
    test number is not a selection input. The gate records
    `evaluation.as_trained_metric_parity` as "no manifest metric" for this
    candidate, which is visible in its report.
    """
    model_dir.mkdir(parents=True, exist_ok=True)
    paths = manifest_paths(model_dir, model_stem)
    schema_path = model_dir / f"{model_stem}_schema.json"
    feature_names = list(schema["feature_names"])

    model.save_model(str(paths["joblib"]))
    _export_onnx(model, paths["onnx"], len(feature_names))
    if not paths["onnx"].is_file():
        raise SystemExit(
            f"ONNX export did not produce {paths['onnx']}; the gate needs an ONNX bundle"
        )
    paths["features"].write_text(json.dumps(feature_names, indent=None), encoding="utf-8")
    schema_path.write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")
    paths["config"].write_text(
        json.dumps(
            {
                "model_type": "xgboost",
                "feature_count": len(feature_names),
                "config": dict(config),
                "selection": "validation-only ablation",
                "metrics": {"validation": dict(validation)},
                "training_date": time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    # ONNX parity: the exported graph must reproduce the model that was scored.
    from onnxruntime import InferenceSession

    session = InferenceSession(str(paths["onnx"]), providers=["CPUExecutionProvider"])
    matrix = encode(val_frame, schema, inference_defaults=True)
    reference = _predict(model, matrix)
    exported = session.run(
        None, {session.get_inputs()[0].name: np.ascontiguousarray(matrix, dtype=np.float32)}
    )
    exported_probabilities = np.asarray(
        [output for output in exported if getattr(output, "ndim", 0) == 2][0]
    )[:, int(schema.get("positive_class_index", 1))]
    max_abs_diff = float(np.max(np.abs(reference - exported_probabilities)))

    manifest = build_manifest(
        model_stem=model_stem,
        model_type="xgboost",
        dataset_path=data_path,
        dataset_sha256=data_sha,
        dataset_rows=data_rows,
        feature_names=feature_names,
        schema_fingerprint=schema_fingerprint(feature_names),
        split_info=dict(split_info),
        hyperparameters={**dict(config), "seed": int(config.get("random_state", 0)),
                         "selected_rounds": fit_info.rounds,
                         "probe_best_iteration": fit_info.probe_best_iteration},
        metrics={"validation": dict(validation)},
        patch_version=patch,
        best_iteration=fit_info.probe_best_iteration,
        artifacts={
            "joblib": paths["joblib"].name,
            "onnx": paths["onnx"].name,
            "features": paths["features"].name,
            "schema": schema_path.name,
        },
    )
    write_manifest(paths["manifest"], manifest)
    return {
        "model_dir": str(model_dir),
        "model_stem": model_stem,
        "onnx": str(paths["onnx"]),
        "schema": str(schema_path),
        "features": str(paths["features"]),
        "manifest": str(paths["manifest"]),
        "onnx_parity_max_abs_diff": max_abs_diff,
        "gate_command": (
            f"ml/venv/Scripts/python.exe ml/training/quality_gate.py "
            f"--model-dir {model_dir} --model-stem {model_stem}"
        ),
    }


def run(args: argparse.Namespace) -> int:
    start = time.time()
    data_path = Path(args.data)
    schema_path = Path(args.schema)
    report_path = Path(args.report)

    base_schema = json.loads(schema_path.read_text(encoding="utf-8"))
    base_names = list(base_schema["feature_names"])
    dataset = pd.read_parquet(data_path)
    if "radiant_win" not in dataset.columns:
        raise SystemExit(f"{data_path} has no radiant_win label column")

    labels = dataset["radiant_win"]
    if labels.isna().any() or not set(pd.unique(labels.dropna())).issubset({0, 1}):
        raise SystemExit("radiant_win must be binary 0/1 with no missing values")

    data_sha = file_sha256(data_path)
    if args.split_config:
        train, val, test, split_info = load_frozen_split(
            dataset, Path(args.split_config), data_path=data_path
        )
        split = None
    else:
        split = time_split(dataset, time_col=args.time_column)
        train, val, test = split.train, split.val, split.test
        split_info = {
            **split.info(),
            "source": "time_split",
            "test_note": (
                "The test partition is not scored in this report. The single "
                "test measurement is produced by quality_gate.py for the "
                "selected candidate."
            ),
            "test_scored_here": False,
        }
    print(f"Dataset {dataset.shape[0]} x {dataset.shape[1]}  sha256={data_sha[:12]}...")
    print(
        f"Split {split_info['sizes']} on {split_info['time_column']} "
        f"(source={split_info['source']})"
    )

    overlaps = {
        "train_validation": len(set(train["match_id"]) & set(val["match_id"])),
        "train_test": len(set(train["match_id"]) & set(test["match_id"])),
        "validation_test": len(set(val["match_id"]) & set(test["match_id"])),
    }
    if any(overlaps.values()):
        raise SystemExit(f"match_id overlap between partitions: {overlaps}")

    # Population selection is applied per partition so the chronological
    # boundaries (and therefore the frozen split) are unaffected.
    population: Dict[str, Any] = {
        "label": "all_rows",
        "rule": None,
        "excluded": {"train": 0, "validation": 0, "test": 0},
    }
    if args.exclude_duplicates:
        population["label"] = "no_duplicate_heroes"
        population["rule"] = DUPLICATE_HERO_RULE
        flagged_total = int(duplicate_hero_mask(dataset).sum())
        population["flagged_dataset_rows"] = flagged_total
        filtered = {}
        for name, part in (("train", train), ("validation", val), ("test", test)):
            mask = duplicate_hero_mask(part)
            population["excluded"][name] = int(mask.sum())
            filtered[name] = part.loc[~mask]
        train, val, test = filtered["train"], filtered["validation"], filtered["test"]
        print(
            f"Population {population['label']}: excluded "
            f"{population['excluded']} (dataset flagged {flagged_total})"
        )

    train_prior = float(train["radiant_win"].astype(int).mean())
    config = dict(WIDE_CONFIG)
    if args.n_estimators is not None:
        config["n_estimators"] = int(args.n_estimators)
    if args.early_stopping_rounds is not None:
        config["early_stopping_rounds"] = int(args.early_stopping_rounds)
    config.update(
        {
            "device": args.device,
            "nthread": args.threads,
            "random_state": args.seed,
        }
    )

    selected_names = {v.name for v in VARIANTS}
    if args.only:
        requested = [name.strip() for name in args.only.split(",") if name.strip()]
        unknown = sorted(set(requested) - selected_names)
        if unknown:
            raise SystemExit(f"unknown variants {unknown}; known: {sorted(selected_names)}")
        variants = tuple(v for v in VARIANTS if v.name in requested)
    else:
        variants = VARIANTS

    progress_levels = (0, 1, 2, 3, 5)
    processor = _load_processor(data_path.parent)
    print("Rebuilding partial-draft populations with the production feature code...")
    partial_frames: Dict[int, pd.DataFrame] = {}
    for known in progress_levels:
        partial_frames[known] = build_partial_draft_frame(
            val, known, schema=base_schema, processor=processor
        )
        print(f"  {known} picks/side: {partial_frames[known].shape[0]} rows")

    results: List[Dict[str, Any]] = []
    fitted: Dict[str, Tuple["xgb.XGBClassifier", FitInfo]] = {}
    for variant in variants:
        names = variant_feature_names(base_names, variant)
        schema = variant_schema(base_schema, names)
        print(
            f"\n[{variant.name}] {len(names)} features "
            f"(dropped {len(base_names) - len(names)}): {variant.description}"
        )
        model, fit_info = fit_variant(train, val, schema, config, seed=args.seed)
        fitted[variant.name] = (model, fit_info)
        print(
            f"  fit: {fit_info.rounds} rounds "
            f"(probe best_iteration={fit_info.probe_best_iteration}, "
            f"{fit_info.seconds:.1f}s)"
        )

        validation = {
            "as_trained": evaluate(
                model,
                encode(val, schema, inference_defaults=False),
                val["radiant_win"].astype(int).to_numpy(),
                train_prior=train_prior,
                calibration_bins=args.calibration_bins,
            ),
            "inference": evaluate(
                model,
                encode(val, schema, inference_defaults=True),
                val["radiant_win"].astype(int).to_numpy(),
                train_prior=train_prior,
                calibration_bins=args.calibration_bins,
            ),
        }

        partial_draft: Dict[str, Any] = {}
        for known in progress_levels:
            simulated = partial_frames[known]
            metrics = evaluate(
                model,
                encode(simulated, schema, inference_defaults=True),
                simulated["radiant_win"].astype(int).to_numpy(),
                train_prior=train_prior,
                calibration_bins=args.calibration_bins,
            )
            partial_draft[str(known)] = metrics

        as_trained = validation["as_trained"]["metrics"]
        inference = validation["inference"]["metrics"]
        reference = validation["inference"]["baselines"]["constant_train_prior"]
        print(
            f"  as_trained: AUC={as_trained['roc_auc']:.4f} Brier={as_trained['brier_score']:.4f}"
        )
        print(
            f"  inference : AUC={inference['roc_auc']:.4f} "
            f"(CI {inference['roc_auc_ci95']['lower']:.3f}-{inference['roc_auc_ci95']['upper']:.3f}) "
            f"Brier={inference['brier_score']:.4f} LogLoss={inference['log_loss']:.4f} "
            f"ECE={inference['calibration']['ece']:.4f} "
            f"Acc={inference['accuracy_at_0_5']:.4f}"
        )
        print(
            f"  baseline  : prior={reference['probability']:.4f} "
            f"Brier={reference['brier_score']:.4f} LogLoss={reference['log_loss']:.4f}"
        )
        print(
            "  partial draft (rad=dire picks known) AUC: "
            + " ".join(
                f"{k}={partial_draft[k]['metrics']['roc_auc']:.3f}" for k in partial_draft
            )
        )

        results.append(
            {
                "name": variant.name,
                "description": variant.description,
                "ablated_groups": list(variant.ablated_groups),
                "dropped_features": list(variant.dropped_features()),
                "feature_count": len(names),
                "schema_fingerprint": schema_fingerprint(names),
                "features": names,
                "fit": asdict(fit_info),
                "validation": validation,
                "partial_draft": partial_draft,
            }
        )

    winner = select_variant(results)
    winner_spec = next(v for v in VARIANTS if v.name == winner["name"])
    winner_schema = variant_schema(base_schema, winner["features"])
    winner_model, winner_fit = fitted[winner["name"]]
    stem = f"draft_model_ablation_{winner['name']}"
    export = export_candidate(
        winner_model,
        winner_schema,
        model_dir=Path(args.model_dir if args.model_dir else DEFAULT_MODEL_DIR / "ablation")
        / stem,
        model_stem=stem,
        data_path=data_path,
        data_sha=file_sha256(data_path),
        data_rows=len(dataset),
        split_info={
            key: split_info.get(key)
            for key in (
                "source",
                "time_column",
                "used_start_time",
                "fractions",
                "sizes",
                "boundaries",
                "contamination",
            )
        },
        config=config,
        fit_info=winner_fit,
        validation=winner["validation"],
        patch=args.current_patch,
        val_frame=val,
    )

    report = {
        "experiment_version": EXPERIMENT_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "purpose": (
            "Isolate whether post-draft/match-metadata features or raw ordinal "
            "pick slots hurt draft-time prediction; model family, split, and "
            "seed held fixed."
        ),
        "dataset": {
            "path": str(data_path),
            "sha256": file_sha256(data_path),
            "rows": int(len(dataset)),
            "label_counts": {
                "radiant_win_1": int((dataset["radiant_win"] == 1).sum()),
                "radiant_win_0": int((dataset["radiant_win"] == 0).sum()),
            },
        },
        "split": {
            **split_info,
            "match_id_overlaps": overlaps,
            "test_scored_here": False,
            "train_positive_rate": train_prior,
            "train_rows": int(len(train)),
            "validation_rows": int(len(val)),
        },
        "population": {
            **population,
            "train_label_counts": {
                "radiant_win_1": int((train["radiant_win"] == 1).sum()),
                "radiant_win_0": int((train["radiant_win"] == 0).sum()),
            },
            "validation_label_counts": {
                "radiant_win_1": int((val["radiant_win"] == 1).sum()),
                "radiant_win_0": int((val["radiant_win"] == 0).sum()),
            },
        },
        "partial_draft_method": (
            "Each row is regenerated by the production feature code from a draft "
            "truncated to k picks per side, then encoded with runtime defaults, so "
            "per-hero win rates, aggregates, semantics, synergy and damage diversity "
            "only ever contain known picks. Bans stay at full-draft values."
        ),
        "training": {
            "config": config,
            "seed": args.seed,
            "device": args.device,
            "threads": args.threads,
            "early_stopping": "probe fit on validation, then refit at selected rounds",
            "selection_rule": SELECTION_RULE,
            "calibration_bins": args.calibration_bins,
        },
        "groups": {
            name: {
                "description": description,
                "features": list(group_features(name)),
            }
            for name, description in GROUPS.items()
        },
        "audited_features": [asdict(row) for row in AUDITED_FEATURES],
        "variants": results,
        "selection": {
            "rule": SELECTION_RULE,
            "selected": winner["name"],
            "selected_description": winner_spec.description,
            "validation_inference_roc_auc": winner["validation"]["inference"]["metrics"][
                "roc_auc"
            ],
            "validation_inference_brier_score": winner["validation"]["inference"]["metrics"][
                "brier_score"
            ],
        },
        "winner_export": export,
        "elapsed_seconds": time.time() - start,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"\nSelected: {winner['name']} ({winner['feature_count']} features)")
    print(f"  validation inference AUC={winner['validation']['inference']['metrics']['roc_auc']:.4f}")
    print(f"  report: {report_path}")
    print(f"  candidate: {export['model_dir']}")
    print(f"  onnx parity max|diff|={export['onnx_parity_max_abs_diff']:.3e}")
    print(f"\nNext (single test evaluation):\n  {export['gate_command']}")
    print(f"Total {time.time() - start:.1f}s")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Controlled feature-ablation experiment (validation-only selection)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--schema", type=Path, default=BASELINE_SCHEMA)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="parent directory for the exported candidate (default: ml/models/ablation)",
    )
    parser.add_argument("--current-patch", type=str, default=DEFAULT_PATCH)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=None,
        help="override the shipped round cap (tests/smoke runs)",
    )
    parser.add_argument(
        "--early-stopping-rounds",
        type=int,
        default=None,
        help="override the shipped early-stopping patience",
    )
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    parser.add_argument("--calibration-bins", type=int, default=10)
    parser.add_argument(
        "--time-column",
        type=str,
        default=None,
        help="chronological key (default: start_time, then match_id fallback)",
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="comma-separated subset of variants to run",
    )
    parser.add_argument(
        "--split-config",
        type=Path,
        default=None,
        help=(
            "frozen split config (e.g. ml/configs/eval_split_v2.json). Verifies the "
            "dataset SHA-256 and the recorded partition sizes, then applies the "
            "config's boundary timestamps instead of an 80/10/10 re-cut."
        ),
    )
    parser.add_argument(
        "--exclude-duplicates",
        action="store_true",
        help=(
            "drop rows whose draft contains a hero twice (picked twice, or picked "
            "and banned), using the documented recomputable rule"
        ),
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
