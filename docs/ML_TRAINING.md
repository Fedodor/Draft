# ML Training & Model Quality

This document describes the ML pipeline, the artifact that actually ships, and
the checks a model must pass before it can replace that artifact. It replaces
the earlier "0.6710 AUC / 896 features" summary, which described an experiment
that is **not** the bundled model and cannot be reproduced from the tracked
artifacts.

- Bundled model: `ml/models/draft_model_xgboost_full.onnx`
- Training data: `ml/datasets/processed/draft_features.parquet`
- Runtime: ONNX Runtime in `apps/desktop/src-tauri/src/ml.rs`
- ML is **opt-in** (`ml_enabled = false` by default); the app works without it.
- A four-variant feature ablation was trained and evaluated on 2026-10-28. It
  removed the train/serve skew but did **not** produce a usable model, so the
  bundled artifact stands. Details and the dataset defect behind the numbers:
  §3.
- The dataset was then regenerated with correct STRATZ pick/ban parsing into
  `ml/datasets/processed_v2/` (102,118 × 890, deterministic column order). **No
  model has been trained on it yet**, and the shipped artifact/schema are
  unchanged. See §3a for fingerprints, integrity counts, remaining defects and
  the frozen — and explicitly already-inspected — evaluation split.

## 1. Bundled model (measured, 2026-10-28)

| Metric | `as_trained` | `inference` (what the runtime can do) |
| --- | --- | --- |
| ROC-AUC | 0.5388 | **0.5122** |
| 95% CI | 0.5277-0.5500 | 0.5010-0.5234 |
| Brier score | 0.2481 | **0.2495** |
| Log loss | 0.6893 | 0.6922 |
| Accuracy @ 0.5 | 0.5294 | 0.5125 |
| ECE (10 equal-frequency bins) | 0.0124 | 0.0241 |

Reference on the same 10,213 held-out rows: a constant train-prior predictor
(p = 0.5245) scores Brier **0.2493**, log loss 0.6917. The model is slightly
**worse** than that baseline on Brier and barely above chance on AUC. Do not
describe it as accurate, calibrated-to-a-useful-degree, or production-ready.

Dataset and split:

| Item | Value |
| --- | --- |
| Rows | 102,118 |
| Columns | 890 (887 model inputs + `match_id`, `radiant_win`, `start_time`) |
| Class balance | 53,693 Radiant wins / 48,425 Dire wins |
| Dataset SHA-256 | `264a6ad5a0cfdecfb60797212d7878019db26a87cb32a4c836de2b27ce991621` |
| Patch | 7.41 (training export) |
| Split | chronological 80/10/10 on `start_time` |
| Train / validation / test | 81,694 / 10,211 / 10,213 matches |
| Match-id overlap | 0 between all partitions |

`as_trained` scores the logged match rows. `inference` replaces the fields the
live client cannot know with the schema default (training median):
`patch`, `patch_version`, `duration`, `avg_mmr`, `game_mode`, `lobby_type`.
The 0.0266 AUC gap between the two modes is the measured train/serve skew.

## 2. Quality gate

`ml/training/quality_gate.py` is the promotion gate. It never trains and never
rewrites the model or schema.

```bash
# Default: validate + evaluate in runtime-realistic mode
ml/venv/Scripts/python.exe ml/training/quality_gate.py

# Machine-readable report / custom path
ml/venv/Scripts/python.exe ml/training/quality_gate.py --json
ml/venv/Scripts/python.exe ml/training/quality_gate.py --report ml/models/gate.json

# Evaluate against logged values (comparison only; not the promotion default)
ml/venv/Scripts/python.exe ml/training/quality_gate.py --evaluation-mode as_trained

# Copy the bundle only when every mandatory check passes
ml/venv/Scripts/python.exe ml/training/quality_gate.py --promote-to ml/models/promoted
```

Default policy (applied to `inference` mode):

| Check | Threshold |
| --- | --- |
| ROC-AUC | ≥ 0.60 |
| Brier score | ≤ 0.235 |
| ECE | ≤ 0.10 |
| Test rows | ≥ 500 |

Before evaluating, the gate verifies:

- manifest, schema, feature list, and ONNX file exist and agree on feature
  count, order, and schema fingerprint;
- ONNX input is a 2-D `tensor(float)` with the declared feature width, and the
  declared positive-class/probability outputs exist;
- the dataset SHA-256 and row count match the manifest;
- the reconstructed chronological split matches the manifest, has strict time
  boundaries, no duplicated `match_id`, and no `match_id` overlap between
  partitions;
- labels are binary with both classes present;
- predictions are finite probabilities in `[0,1]`;
- every schema feature is encodable (diagnostics: missing columns are reported,
  NaN fills are counted, unknown categories are counted).

Exit codes:

| Code | Meaning | Promotion |
| --- | --- | --- |
| `0` | Passed; all mandatory checks green | Allowed |
| `1` | Quality failure (thresholds, fingerprint mismatch, split problem) | Refused |
| `2` | Evaluation error (missing/malformed artifact, schema drift, bad labels) | Refused |

The report (`<stem>_quality_report.json`) records both modes, the calibration
bins, constant baselines, split boundaries, encoding diagnostics, the leakage
audit, and the failed checks. `ml/models/*` is gitignored, so reports and
candidate artifacts stay local.

The bundled artifact currently exits **1**: `policy.roc_auc` and
`policy.brier_score` fail. That is intentional and visible, not a bug.

## 3. Feature ablation and candidate result (2026-10-28)

Question: do the post-draft / match-metadata fields or the raw ordinal pick
slots hurt draft-time prediction, or create train/serve mismatch? Model family,
hyperparameters, split, and seed were held fixed so only the feature set varied.

Harness: [`ml/training/ablation.py`](../ml/training/ablation.py).

```bash
ml/venv/Scripts/python.exe ml/training/ablation.py
# -> ml/models/ablation/ablation_report.json (gitignored)
# -> ml/models/ablation/draft_model_ablation_<winner>/ (onnx, schema, features, config, manifest)
```

Rules the harness enforces:

- variants are declared in code from a per-feature audit table (name, group,
  availability, reason), so nothing is removed because its name looks wrong;
- the chronological split and `match_id` isolation are re-checked before
  training (`time_split` + an explicit overlap assertion);
- early stopping uses validation, then the model is **refit** at the selected
  round count, because XGBoost predicts with `best_iteration` while the ONNX
  converter exports every tree (exported ONNX parity 1.2e-07);
- selection uses validation `inference` ROC-AUC with a 0.002 parsimony window
  (lower Brier, then fewer features);
- the winner is exported with **validation metrics only** and the test partition
  is scored by `quality_gate.py`. The harness has no test-evaluation path.

### Validation results (10,211 rows, inference mode)

Constant train prior 0.5245 scores Brier 0.2489 / log loss 0.6910 on these rows.

| Variant | Features | ROC-AUC (95% CI) | Brier | Log loss | ECE | `as_trained` AUC | Trees |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline (all shipped) | 887 | 0.5340 (0.5229-0.5452) | 0.2486 | 0.6904 | 0.0245 | 0.5508 | 641 |
| No match context | 881 | **0.5358** (0.5247-0.5470) | **0.2481** | **0.6893** | **0.0151** | 0.5358 | 376 |
| No pick order | 877 | 0.5332 (0.5221-0.5444) | 0.2487 | 0.6905 | 0.0248 | 0.5493 | 608 |
| Combined | 871 | 0.5344 (0.5232-0.5456) | 0.2481 | 0.6894 | 0.0203 | 0.5344 | 420 |

Ablated groups:

| Group | Features | Why it was tested |
| --- | --- | --- |
| Match context | `duration`, `avg_mmr`, `game_mode`, `lobby_type`, `patch`, `patch_version` | post-match (`duration`), not delivered to the engine, or rewritten by the schema default at inference. `avg_mmr` is constant 0; `patch_version` is constant `unknown` in all 81,694 training rows. |
| Pick order | `rad_pick_0..4`, `dire_pick_0..4` | only filled slots exist mid-draft; unfilled slots fall back to schema defaults that are *real hero ids* (66, 64), impersonating a drafted hero while the presence vector says it is absent. |

### Partial-draft simulation (validation AUC)

Unknown slots are cleared and their heroes leave the presence vector, then the
runtime defaults are applied exactly as at inference.

| Variant | 0 known | 1 | 2 | 3 | 5 (full) |
| --- | --- | --- | --- | --- | --- |
| Baseline | 0.5239 | 0.5267 | 0.5285 | 0.5300 | 0.5340 |
| No match context | 0.5209 | 0.5230 | 0.5299 | 0.5316 | 0.5358 |
| No pick order | 0.5293 | 0.5301 | 0.5303 | 0.5309 | 0.5332 |
| Combined | **0.5315** | **0.5326** | **0.5328** | **0.5332** | 0.5344 |

The ordinal slots cost ~0.008 AUC at the first-pick stage and nothing once the
draft is complete. That matches their defaults-collide-with-real-heroes design,
and it is the only place where removing features measurably helped at the point
where a recommendation is actually made.

### Single test result (10,213 rows, `quality_gate.py`, exit 1)

Selected on validation: `no_match_context`, 881 features, 376 trees.

| Metric | Candidate | Bundled model (`inference`) | Constant train prior |
| --- | --- | --- | --- |
| ROC-AUC | 0.5136 (0.5024-0.5248) | 0.5122 (0.5010-0.5234) | 0.5 |
| Brier | 0.2494 | 0.2495 | **0.2493** |
| Log loss | 0.6919 | 0.6922 | **0.6917** |
| Accuracy @ 0.5 | 0.5255 | 0.5125 | **0.5274** |
| ECE | **0.0139** | 0.0241 | — |

`as_trained` and `inference` are identical for the candidate (0.5136 / 0.2494),
which confirms the substitution gap is gone. The candidate still fails
`policy.roc_auc` and `policy.brier_score`, is worse than a constant predictor on
Brier, log loss and accuracy, and lost 0.0222 AUC from validation to test.
**Not promoted; the bundled artifact is unchanged.**

### The dataset behind these numbers is corrupted

The processed parquet was built by `processor.py`, which read
`pb.get('is_pick', True)` while every sampled STRATZ payload uses camelCase
`isPick`. Bans were therefore recorded as picks:

- 82,739 of 102,118 rows have **12** presence flags per side instead of 5;
- `rad_ban_*` is all-zero for 97,402 rows;
- `rad_pick_0..4` mixes bans into the pick slots for those rows.

The runtime builds 5 picks and up to 5 bans per side, so the model was trained
on a different presence distribution than it is served. Parsing is fixed (with a
STRATZ-shape regression test), but the dataset was **not** regenerated, so every
number above — and the bundled model's metrics — reflects the buggy encoding.
No feature ablation can repair this; regeneration is the prerequisite for any
further model claim.

## 3a. Corrected dataset and frozen evaluation protocol (2026-10-28, session 6)

Everything in §1 and §3 was measured on the corrupted v1 parquet
(`ml/datasets/processed/draft_features.parquet`, SHA-256 `264a6ad5…`, still
present and unmodified). A corrected dataset now exists beside it; **nothing has
been retrained on it**, and the bundled ONNX artifact is unchanged.

### Generation

```bash
cd ml && ./venv/Scripts/python.exe feature-engineering/processor.py \
  --output-dir datasets/processed_v2 \
  --lookup-dir datasets/processed \
  --no-allpick --no-matchups --workers 3 --chunk-size 777 \
  --dataset-label candidate_v2_pickban_fix
```

| Item | Value |
| --- | --- |
| Raw files considered | 111,945 (`*.json`, numeric stems sorted ascending) |
| Processed / skipped | 102,118 / 9,827 (32 no draft, 8,929 `pickBans: null`, 866 short draft, 0 errors) |
| Output | `ml/datasets/processed_v2/draft_features.parquet` |
| Rows × columns | 102,118 × 890 (882 non-meta features + 8 meta) |
| SHA-256 / size | `472a5086cf919d8077e43a263cab70642f318b5ddafed1641ca947b19d3b6ce7` / 56,332,285 B |
| Feature fingerprint | `9b04edda99e2d1cebfaf72f3d12cb8cfe802a913add2e66be5090d207c4b2599` |
| Provenance | `ml/datasets/processed_v2/dataset_schema.json` (column order, dtypes, categories, run stats, lookup hashes) |
| Runtime | ~4 min (8 workers, chunk 2,000) / ~14 min (3 workers, chunk 777) on this machine |

`--no-matchups` is deliberate: the seven matchup features are absent from the
shipped contract and `ml.rs` cannot rebuild them from a live draft, so including
them would recreate the train/serve mismatch this exercise is removing. With
them excluded the column set is exactly the shipped one (882 draft features plus
`patch`, `duration`, `avg_mmr`, `game_mode`, `lobby_type`), i.e. the 887 names the
bundled model expects, with corrected values. Those five context columns are
listed in `meta_columns` for ordering and dtype purposes only — they are still
dataset columns and still model inputs of the shipped contract. **No shipped
schema was changed and the 887-name contract is not redefined here.**

### Determinism

`process_all()` processes `canonical_match_files()` (numeric match-id order),
reassembles worker results by input index, streams chunked parts, and orders
columns through `canonical_column_order()` (meta block → code-emission order →
ban averages → per-hero win rates sorted by side then hero id). Two full runs
with **different worker counts (8 vs 3) and chunk sizes (2,000 vs 777)** produced
the same 890 columns in the same order with identical values in every column;
the content fingerprint over all columns is
`e742e24b45572fcafba38bc4ecb06192c22836da8b44af777d1d389cd9171f9f` for both.

Parquet *bytes* still differ (35.7 MB / 56 row groups vs 56.3 MB / 145 row
groups) because chunk size determines row-group layout. Byte identity is only
expected for an identical `--chunk-size`/`--workers` configuration, which is what
`test_processing_is_deterministic_across_workers_and_chunks` asserts at fixture
scale. Quote the content fingerprint, not the file hash, when comparing
datasets built with different chunking.

### Pick/ban integrity, old vs new (recomputed, not carried over)

| Diagnostic | v1 (2026-10-23) | v2 (2026-10-28) |
| --- | --- | --- |
| Rows with 12 `rad_has_*` flags | 82,739 | **0** |
| Rows with exactly 5 `rad_has_*` flags | 15,841 | **100,369** (98.3%) |
| Rows with 12 `dire_has_*` flags | 82,745 | **0** |
| `rad_ban_*` all-zero rows | 97,402 (95.4%) | **8,031** (7.9%) |
| Rows with 7 Radiant bans | 3,698 | **87,406** (85.6%) |
| Rows with 3 Radiant bans | 8 | **3,280** |
| Rows with a hero both picked and banned | 0 | **3,870** (3.8%) |
| Non-finite cells / duplicate `match_id` | 0 / 0 | **0 / 0** |
| Radiant-win rate | 0.5258 | 0.5258 |

Bans were previously invisible because STRATZ `isPick: false` entries were
counted as picks. They are now encoded separately, which is why the ban columns
are populated at all — and why a genuine source contradiction becomes visible.

Validation against the source: 450 matches (every 227th row in canonical order)
were re-parsed straight from their raw JSON by an independent reader and
compared against the four pick/ban one-hot blocks — **0 mismatches**.

### Remaining data defects (not fixed, not hidden)

1. **A hero both picked and banned — 3,870 rows (3.8%).** The same hero carries
   a presence flag and the matching ban flag. This is a source-data
   contradiction (most likely series-level ban data leaking across games), not a
   parser artifact: no row in the corpus repeats a hero inside one side's five
   pick slots. Rows are **kept**; the ablation reports them and runs a second
   population with them excluded. Detection rule (recomputable from the parquet):
   presence flag and matching ban flag both 1 for one side, or a repeated hero
   across a side's five pick slots.
2. **Picked heroes above the one-hot range — 1,748 rows (1.7%).** The presence
   and ban blocks cover hero ids 1-139, but matches contain heroes up to id 155.
   Those heroes have no column, so a row with one can show only 4 presence flags.
   This is a schema coverage gap in both v1 and v2 (and in the runtime, which
   also iterates a fixed roster), *not* a duplicate; the earlier note that those
   rows repeated a hero was wrong and is corrected here. 61 rows additionally
   have more than five picks per side in the source.
2. **Unrevealed bans — 8,031 rows (7.9%)** have all `heroId: null` ban slots in
   the source, so their ban blocks are empty. Ban features are therefore
   missing-not-at-random: a coverage limitation, not corruption.
3. **Short drafts and null drafts — 866 / 8,929 matches** were rejected by the
   documented rules (fewer than 5 picks per side; `pickBans` null or empty).
   They are counted in `dataset_schema.json`, not silently dropped.
4. `patch_version` still resolves to `'unknown'` (STRATZ omits `patch`); it is a
   constant and must not be used as a feature until the source changes.

### Frozen evaluation protocol

`ml/configs/eval_split_v2.json` freezes the split for the corrected dataset
before any model is trained on it: chronological, `start_time` with `match_id`
tie-break, boundaries placed between distinct timestamps, no `match_id` overlap.

| Partition | Rows | Window (UTC) | Allowed use |
| --- | --- | --- | --- |
| train | 81,695 | 2017-05-07 → 2025-11-29 | feature selection, fitting |
| validation | 10,212 | 2025-11-29 → 2026-04-05 | model/variant selection, early stopping |
| test | 10,211 | 2026-04-05 → 2026-08-28 | reference only — **already inspected** |

**The holdout is not untouched.** All 10,211 of its matches fall inside the
session-5 test window whose metrics are reported in §3, so a score on it is a
regression reference, not a fresh held-out result. A clean final number needs
matches newer than `1787903076` (2026-08-28T07:44:36Z), which do not exist in this
repository yet. The config records this explicitly; do not restate it as an
untouched holdout.

### Test-suite pass vs model pass vs promotion

Three different claims are routinely conflated by one line of console output:

| Statement | Evidence | Meaning |
| --- | --- | --- |
| `scripts/verify.ps1` prints `Quality gate PASSED.` | 13/13 steps, exit 0 | The gate **implementation** and its tests pass. Says nothing about any model. |
| `quality_gate.py` exits **1** for the bundled model | ROC-AUC 0.5122 < 0.60, Brier 0.2495 > 0.235 | The bundled model **fails the policy**. |
| `promotion_eligible` | gate report field (`ml/models/session5_gate.json`) | A candidate may replace the artifact only when every mandatory check is green. It is `false`. |

Neither the thresholds nor the bundled artifact changed in this session.

## 3b. Ablation on the corrected dataset (2026-10-28, session 7)

Question: does correct pick/ban encoding give the draft model usable signal, and
do the two suspicious feature groups still hurt?

```bash
cd ml
# population A — every row
./venv/Scripts/python.exe training/ablation.py \
  --data datasets/processed_v2/draft_features.parquet \
  --split-config configs/eval_split_v2.json \
  --report models/ablation_v2/all_rows_report.json \
  --model-dir models/ablation_v2/candidates_all_rows
# population B — duplicate-hero rows excluded
./venv/Scripts/python.exe training/ablation.py ... --exclude-duplicates \
  --report models/ablation_v2/no_duplicates_report.json \
  --model-dir models/ablation_v2/candidates_no_duplicates
```

Fixed for every variant: XGBoost `WIDE_CONFIG` (depth 4, lr 0.02, 2,500 cap,
early stop 200), CPU `hist`, 8 threads, seed 42, schemas derived from the shipped
887-name contract, and early stopping on the validation partition followed by a
refit at the selected round count (exported ONNX parity 2.4e-07 / 3.6e-07).
**The test partition was not scored by either run.**

### Populations

| | A: all rows | B: no duplicate heroes |
| --- | --- | --- |
| Rule | — | hero picked *and* banned, or repeated pick slot |
| Train rows | 81,695 | 79,614 (−2,081) |
| Validation rows | 10,212 (Radiant 53.4%) | 9,145 (Radiant 53.3%) |
| Excluded (train/val/test) | — | 2,081 / 1,067 / 722 of 3,870 |

### Validation results (inference mode, i.e. runtime substitutions applied)

Population A, 10,212 rows:

| Variant | Features | Rounds | ROC-AUC (95% CI) | Brier | Log loss | ECE | Accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 887 | 971 | 0.5454 (0.5342-0.5565) | 0.2478 | 0.6888 | 0.0267 | 0.5370 |
| no match context | 881 | 771 | 0.5444 (0.5333-0.5556) | 0.2475 | 0.6881 | **0.0172** | **0.5463** |
| no pick order | 877 | 1,486 | 0.5471 (0.5360-0.5583) | 0.2478 | 0.6888 | 0.0273 | 0.5388 |
| **combined (selected)** | 871 | 1,110 | **0.5481** (0.5369-0.5592) | **0.2473** | **0.6877** | 0.0177 | 0.5402 |

Population B, 9,145 rows:

| Variant | Features | Rounds | ROC-AUC (95% CI) | Brier | Log loss | ECE | Accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 887 | 1,501 | 0.5466 (0.5348-0.5584) | 0.2480 | 0.6890 | 0.0278 | 0.5347 |
| no match context | 881 | 911 | 0.5485 (0.5367-0.5603) | **0.2473** | 0.6876 | **0.0160** | **0.5447** |
| no pick order | 877 | 1,005 | 0.5472 (0.5355-0.5590) | 0.2477 | 0.6887 | 0.0237 | 0.5372 |
| **combined (selected)** | 871 | 1,319 | **0.5498** (0.5381-0.5616) | **0.2473** | **0.6875** | 0.0188 | 0.5403 |

Baselines on the same rows: population A constant train prior (p = 0.5245) gives
Brier 0.2489 / log loss 0.6910 / accuracy 0.5344; population B gives 0.2490 /
0.6911 / 0.5328. Constant 0.5 gives Brier 0.2500 on both.

### Partial draft (inference mode, AUC by picks known per side)

| Variant | 0 | 1 | 2 | 3 | 5 |
| --- | --- | --- | --- | --- | --- |
| A baseline | 0.5291 | 0.5293 | 0.5308 | 0.5383 | 0.5453 |
| A no match context | 0.5285 | 0.5290 | 0.5331 | 0.5396 | 0.5445 |
| A no pick order | 0.5271 | 0.5214 | 0.5264 | 0.5342 | 0.5472 |
| A combined | 0.5270 | 0.5268 | 0.5322 | 0.5358 | 0.5482 |
| B baseline | 0.5337 | 0.5266 | 0.5317 | 0.5378 | 0.5467 |
| B no match context | 0.5343 | 0.5263 | 0.5344 | 0.5412 | 0.5485 |
| B no pick order | 0.5332 | 0.5269 | 0.5349 | 0.5378 | 0.5473 |
| B combined | 0.5291 | 0.5255 | 0.5326 | 0.5366 | 0.5500 |

### What changed in the harness (and why it mattered)

1. **Frozen split, not a re-cut.** `--split-config` verifies the dataset SHA-256
   and the recorded partition sizes, then applies the config's boundary
   timestamps. A mismatch is a hard error. Before this, the harness silently
   re-derived its own 80/10/10 split, so "the frozen split" was not actually
   being used.
2. **The partial-draft simulation used to leak the full draft.** It cleared only
   presence flags and pick slots; per-hero win rates, win-rate aggregates,
   semantics averages, synergy and damage diversity stayed computed from all five
   picks. The runtime recomputes all of them from the picks it knows, so the old
   partial-draft numbers were optimistic *and* insensitive to the real draft
   state. Rows are now regenerated by the production feature code from a
truncated draft (`allow_partial_draft=True`, a simulation-only opt-out of the
   dataset's five-pick quality gate), and everything the runtime would omit falls
   back to its schema default. Two tests pin this: two drafts that agree on the
   first pick score identically at that stage, and a five-pick rebuild reproduces
   the stored row on 884 of 887 features. The three exceptions are the two
   ban-average aggregates (drift <0.01, because bans above hero id 139 have no
   column to reconstruct from) and `patch_version`, which is re-derived from the
   payload — harmless, since the runtime substitutes both.
3. **Population filtering is explicit and recomputable** (`duplicate_hero_mask`,
   documented rule, count asserted against the dataset).

### Reading the results

- **Correct encoding does add signal, but far too little.** Against the same
  harness on the corrupted v1 parquet (baseline 0.5340, best 0.5358), corrected
  data lifts validation AUC to 0.5454-0.5498 — about +0.011 to +0.014. Every
  variant now edges out the constant train prior on Brier (0.2473-0.2480 vs
  0.2489-0.2490) and on log loss, which v1 never did.
- **Still nowhere near the policy floor.** The 95% CI upper bounds are 0.559
  (A) and 0.562 (B); the 0.60 floor is roughly nine standard errors away. No
  candidate is promotable and none was promoted.
- **Variant differences are not separable from noise.** All CIs overlap by >90%.
  The winner was chosen by the pre-registered rule (highest AUC, then lower
  Brier within 0.002), not because 0.5481 is meaningfully better than 0.5471.
  No paired difference test was computed, which is a limitation.
- **Dropping match context costs ~0.001 AUC** but halves ECE (0.0267 → 0.0172)
  and removes the residual train/serve gap (baseline as-trained 0.5598 vs
  inference 0.5454; the no-context variant is identical in both modes by
  construction).
- **Dropping pick order now hurts partial drafts**, the opposite of v1: at one
  known pick it costs ~0.008 AUC (0.5293 → 0.5214 in A). With correct presence
  encoding the ordinal slots carry real role/order information rather than
  masked bans, so the v1 argument for removing them no longer holds. The
  combined variant still wins on full-draft metrics, which is why selection kept
  it; on a first-pick-only product decision, the baseline's pick slots are
  defensible.
- **Excluding duplicate-hero rows changes nothing material** (+0.0001 to +0.0040
  AUC, same winner, similar calibration). The contradiction is real but too rare
  to drive the result.
- **Partial-draft AUC now rises with information** (≈0.527 at zero picks to
  ≈0.548 with five), which is the signature of a simulation that no longer leaks
  the finished draft.

### Limitations

- The validation partition is the only selection evidence; the test partition
  was deliberately not scored, so this session produces **no new test result**.
  Its window has already been inspected (§3a) and cannot be presented as held
  out.
- Bans are held at their full-draft values in the partial-draft simulation; ban
  timing inside the draft is not modelled.
- Bans with hero id > 139 cannot be reconstructed from the stored one-hots, so
  the rebuilt rows can miss one ban and the two ban-average aggregates drift by
  <0.01.
- One seed (42) and one model family. Single-seed results at this effect size
  should not be read as a ranking.

## 3c. Model-family comparison on the corrected dataset (2026-10-29, session 8)

Question: is the *model family* the bottleneck, or the signal? Everything except
the learner is held fixed to the session-7 winner.

```bash
cd ml
./venv/Scripts/python.exe training/model_families.py --population all
./venv/Scripts/python.exe training/model_families.py --population no_duplicates
```

Fixed for every family: dataset `datasets/processed_v2/draft_features.parquet`
(SHA-256 verified `472a5086…b6ce7`), frozen split from
`configs/eval_split_v2.json` (verified 81,695 / 10,212 / 10,211 and re-applied by
boundary timestamp; a size or hash mismatch is a hard error), feature set
`no_match_context_no_pick_order` (**871 features**, schema fingerprint
`8fd32b2022806eef…`), runtime inference substitutions, partial-draft frames
rebuilt by the production feature code, calibration bins 10, and seeds 42/43/44.
**The test partition is never encoded, scored, or referenced by this harness.**

| Family | Configuration |
| --- | --- |
| `xgboost_combined` (reference) | `WIDE_CONFIG`: `binary:logistic`, depth 4, lr 0.02, 2,500-round cap, subsample 0.75, colsample 0.5, min child weight as shipped; CPU `hist`, 8 threads, early stop 200 on the frozen validation partition, refit at the selected round count |
| `logistic_l2` | `StandardScaler` + `LogisticRegression(C, lbfgs, max_iter 3000, L2)`; scaler and coefficients fitted on **train only**; `C ∈ {0.01, 0.1, 1}` chosen on validation AUC |
| `hist_gbdt` | `HistGradientBoostingClassifier`: lr 0.05, ≤600 iterations, 15 leaves, `min_samples_leaf` 60, L2 1.0, early stopping on a 10% *train-internal* split |

`C` was selected once for both populations (`C = 0.01`; 0.5454 vs 0.5449 at 0.1
and 0.5446 at 1, i.e. the grid is flat and the smallest value wins). Logistic
regression is seed-invariant here — three seeds produced byte-identical metrics
(`max|diff| = 0`), so its spread is reported as 0 rather than as agreement.

### Validation results — population A, all rows (10,212 rows: 5,457 / 4,755)

| Family | ROC-AUC mean (sd) | per-seed ROC-AUC (95% CI, seed 42) | Brier | Log loss | ECE | Accuracy | Fit | Predict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `xgboost_combined` | **0.5479** (0.0004) | 0.5481 / 0.5474 / 0.5483 (0.5369-0.5592) | **0.2473** | **0.6877** | 0.0181 | 0.5428 | 136.9 s | 7 ms / 1k |
| `logistic_l2` | 0.5454 (0.0000) | 0.5454 / 0.5454 / 0.5454 (0.5343-0.5565) | 0.2488 | 0.6907 | 0.0346 | 0.5325 | **4.3 s** | 9 ms / 1k |
| `hist_gbdt` | 0.5380 (0.0026) | 0.5392 / 0.5344 / 0.5405 (0.5281-0.5504) | 0.2479 | 0.6889 | **0.0158** | 0.5401 | 21.9 s | 15 ms / 1k |
| constant train prior (0.5245) | 0.5000 | — | 0.2489 | 0.6910 | — | 0.5344 | — | — |
| constant 0.5 | 0.5000 | — | 0.2500 | 0.6931 | — | 0.4656 | — | — |

`as_trained` and `inference` modes are identical for all three (0.5481 / 0.5454 /
0.5392 AUC): the 871-feature set contains no feature the runtime substitutes.

### Validation results — population B, duplicate heroes excluded (9,145 rows: 4,872 / 4,273)

| Family | ROC-AUC mean (sd) | per-seed ROC-AUC | Brier | Log loss | ECE | Accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| `xgboost_combined` | **0.5481** (0.0012) | 0.5498 / 0.5470 / 0.5474 | **0.2473** | **0.6878** | 0.0173 | 0.5397 |
| `logistic_l2` | 0.5441 (0.0000) | 0.5441 ×3 | 0.2491 | 0.6915 | 0.0406 | 0.5328 |
| `hist_gbdt` | 0.5370 (0.0009) | 0.5375 / 0.5378 / 0.5357 | 0.2481 | 0.6893 | 0.0165 | 0.5360 |
| constant train prior (0.5245) | 0.5000 | — | 0.2490 | 0.6911 | — | 0.5328 |

Excluded rows: 2,081 train / 1,067 validation / 722 test of 3,870 flagged. The
ranking, the winner, and the *direction* of every difference are unchanged from
A, so the filtering is not driving anything. Both populations are reported; A is
the primary result.

### Partial draft — AUC by picks known per side (mean over seeds)

| Population | Family | 0 | 1 | 2 | 3 | 5 |
| --- | --- | --- | --- | --- | --- | --- |
| A | `xgboost_combined` | **0.5292** | 0.5262 | 0.5317 | 0.5373 | **0.5481** |
| A | `logistic_l2` | 0.5280 | **0.5372** | **0.5365** | **0.5417** | 0.5454 |
| A | `hist_gbdt` | 0.5261 | 0.5161 | 0.5240 | 0.5324 | 0.5380 |
| B | `xgboost_combined` | **0.5294** | 0.5277 | 0.5327 | 0.5376 | **0.5482** |
| B | `logistic_l2` | 0.5295 | **0.5387** | **0.5376** | **0.5413** | 0.5441 |
| B | `hist_gbdt` | 0.5186 | 0.5191 | 0.5261 | 0.5333 | 0.5371 |

This is the product-relevant table. **From one to three known picks per side the
linear model is better than XGBoost** (A: +0.0110 at one pick, +0.0048 at two,
+0.0059 at three; B: +0.0110 / +0.0049 / +0.0037). Only at five known picks per
side does XGBoost pull ahead (+0.0027 A, +0.0041 B), by less than the early-draft
gap it loses. `hist_gbdt` is worse at every stage.

Partial-draft Brier (A, seed 42) moves the other way: XGBoost
0.2491/0.2494/0.2487/0.2484/0.2473 versus logistic
0.2543/0.2501/0.2510/0.2497/0.2488. The linear model ranks better early but is
less calibrated on truncated drafts, because truncation shifts the base rate away
from the train prior and nothing re-fits the intercept per stage.

### Paired bootstrap (2,000 resamples of the same validation rows, seed 42)

| Population | Candidate − XGBoost | AUC difference (95% CI) | Brier difference (95% CI) |
| --- | --- | --- | --- |
| A | `logistic_l2` | −0.00263 (−0.00981 … +0.00424) | +0.00148 (+0.00062 … +0.00237) |
| A | `hist_gbdt` | −0.00884 (−0.01547 … −0.00190) | +0.00048 (−0.00006 … +0.00103) |
| B | `logistic_l2` | −0.00577 (−0.01310 … +0.00128) | +0.00184 (+0.00092 … +0.00273) |
| B | `hist_gbdt` | −0.01252 (−0.01968 … −0.00544) | +0.00082 (+0.00019 … +0.00144) |

Positive Brier difference = candidate worse. Read strictly: the linear model is
**not** distinguishable from XGBoost on AUC (the CI crosses zero in both
populations) but is reliably ~0.0015 worse on Brier. `hist_gbdt` is reliably
worse on AUC and equal-to-worse on Brier. No family beats the reference.

Pre-registered decision rule (`DECISION_RULE`, written before the runs): a family
replaces the reference only if the paired AUC CI excludes zero **and** mean Brier
improves by more than 0.002. Nothing qualified, so the reference is retained and
`material_improvement_over_reference = false` in both reports.

### Quality policy (unchanged, evaluated on validation only)

| Threshold | XGBoost (A) | Logistic (A) | Hist GBDT (A) |
| --- | --- | --- | --- |
| ROC-AUC ≥ 0.60 | 0.5481 ✗ | 0.5454 ✗ | 0.5392 ✗ |
| Brier ≤ 0.235 | 0.2473 ✗ | 0.2488 ✗ | 0.2479 ✗ |
| ECE ≤ 0.10 | 0.0177 ✓ | 0.0346 ✓ | 0.0140 ✓ |
| rows ≥ 500 | 10,212 ✓ | 10,212 ✓ | 10,212 ✓ |

Every family fails the numeric floor. The best candidate is a failing candidate
on validation; nothing was promoted and `quality_gate.py` was not re-run, because
it would score the already-inspected test window.

### ONNX export and parity (candidate artifacts only)

Written to the gitignored `ml/models/families/onnx/`. The bundled
`draft_model_xgboost_full.onnx` and schema were not touched.

| Family | Input | Outputs | Parity (max abs diff) | Stages checked |
| --- | --- | --- | --- | --- |
| `xgboost_combined` | `float_input` `[None, 871]` `tensor(float)` | `label`, `probabilities` `[None, 2]` | 2.98e-07 | full draft, 0/1/3 picks |
| `logistic_l2` | `float_input` `[None, 871]` `tensor(float)` | `label`, `probabilities` `[None, 2]` | 2.47e-07 | full draft, 0/1/3 picks |
| `hist_gbdt` | — | — | **not exportable** | — |

`hist_gbdt` fails skl2onnx conversion (`TreeEnsembleClassifier` with these
attributes is rejected by the op builder). Exporting it would need a different
converter or a hand-written graph; since it is also the worst family at every
stage, that work was not done. This is a tooling limitation, not a runtime
contract change, and it did not block the comparison.

### Reading the results

- **Model family is not the bottleneck.** Three materially different learners
  land inside 0.01 AUC of each other, all barely above the constant baselines,
  and none moves toward the 0.60 floor. Swapping the learner cannot fix this.
- **Interactions stop paying off exactly where the product needs them.** The
  boosted trees only win once most of the draft is revealed; with one pick known
  per side a logistic regression on the same 871 features is ~0.011 AUC better.
  Draft-time recommendations are generated at that early stage.
- **The linear model is a legitimate cheap baseline** (`C = 0.01`, 4.3 s fit,
  9 ms/1k predictions) and is exported with parity, so it is available as a
  reference floor for future candidates — not as a replacement.
- **Brier is the one place the reference is reliably better** (+0.0015 for the
  linear model, CI excludes zero). If a future candidate is calibrated on
  truncated drafts, that gap is the thing to close first.
- `hist_gbdt` has the best ECE (0.0158) and the worst ranking: calibration alone
  is not evidence of useful signal.

### Limitations

- Validation-only. No test number exists or may be claimed; the test partition
  was not read (each report records `test_scored_here: false`), and its window
  has already been inspected (§3a).
- One feature set (the session-7 winner). A linear model might prefer a different
  representation; testing that would be a new experiment, and re-selecting
  features after seeing these results would break the frozen protocol.
- `hist_gbdt` uses its own 10% train-internal early-stopping split, not the frozen
  validation partition, because scikit-learn exposes no external-validation
  early stopping for that class. Its round count is therefore not chosen on the
  same signal as XGBoost's.
- Partial-draft frames hold bans at full-draft values (ban timing is not in the
  parquet), so early-stage numbers are optimistic relative to a real draft where
  bans arrive gradually. §3d measures the size of that optimism: -0.0044 AUC at
  zero picks and -0.0029 at one pick, on 9,507 rows with raw ban order.
- Per-stage differences (0.004-0.011 AUC) were not bootstrapped individually;
  only the full-draft headline was. Treat the stage ranking as a consistent
  pattern across two populations and three seeds, not as a tested effect.

## 3d. Objective feasibility study (2026-10-29, session 9)

Research only: **no model was fitted for selection, nothing was promoted, and the
test partition was not read** (`test_scored_here: false` in the report). The
question is no longer "which learner" or "which features" but "is full-match
win probability the right target?".

Command and artefact:

```
ml/venv/Scripts/python.exe ml/training/objective_diagnostics.py
  # exit 0, 40-60 s (mode scan is cached after the first run)
  # -> ml/models/objective_study/diagnostics.json
```

The candidate scored throughout is the session-8 combined XGBoost
(871 features, fingerprint `8fd32b20…c8c52`, seed 42, 1,110 rounds). It was
re-fitted once to restore a *population-matched* export: the ONNX left on disk by
session 8 came from the duplicate-excluded run (trained on 79,614 rows), so
scoring it against the full validation partition reported 0.5501 instead of the
recorded 0.5481. The refit reproduced the recorded metrics exactly — AUC
0.5480650, Brier 0.2472861, 1,110 rounds — which is also a determinism check.

### What the audit confirmed

| Check | Result |
| --- | --- |
| All families scored on the same validation rows/labels | Yes — `X_val_inf`, `y_val` built once and shared |
| Partial-draft stages use the same eligible rows in every comparison | Yes — 10,212 rows per stage per family; the timing study below uses a paired 9,507-row subset for both arms |
| Later picks cannot leak into an earlier stage | Yes — rebuild uses the production feature code from truncated picks; existing test asserts two drafts agreeing on pick 1 score identically |
| Post-match fields cannot leak | Yes — `inference` mode substitutes schema defaults; the candidate drops them outright |
| Bans cannot leak | **No** — the harness holds all bans at full-draft values. Quantified below |
| Constant-prior baseline uses training labels only | Yes — `train_prior = mean(y_train)` |
| Learned preprocessing fitted on train only | Yes — `StandardScaler` inside the logistic pipeline |
| Hyperparameter selection separated from reported metrics | Partly — logistic `C` and XGBoost rounds are both chosen on the same validation rows they are reported on (validation reuse, ~0.0005-0.002 effect) |
| Labels / finiteness | `radiant_win` binary, no missing, `match_id` unique; 0 non-finite features in the 871-column matrix |

### New defect: `game_mode` is mis-encoded

`processor.py` maps the STRATZ string enum with
`gm = 22 if 'CAPTAIN' in gm.upper() else 0`, so every Captains Mode match is
stored as **22 (All Pick)** and every non-captain mode as **0**. True modes, read
from the raw payloads for all 102,118 rows:

| True mode | Rows | Stored `game_mode` |
| --- | --- | --- |
| `CAPTAINS_MODE` | 87,746 | 22 |
| OpenDota Captains Mode | 4,475 | 2 (correct) |
| `CAPTAINS_DRAFT` | 3,172 | 22 |
| `RANDOM_DRAFT` | 3,285 | 0 |
| `SINGLE_DRAFT` | 1,422 | 0 |
| Turbo (OpenDota / STRATZ) | 918 / 606 | 23 / 0 |
| `ALL_PICK_RANKED` / `ALL_RANDOM` / `LEAST_PLAYED` / `ALL_PICK` | 133 / 86 / 23 / 11 | 0 |

Impact: the column is semantically wrong and conflates `CAPTAINS_MODE` with
`CAPTAINS_DRAFT`, so **any analysis that segments by `game_mode` is invalid**.
The 871-feature candidate drops the group that contains it, so the session-7/8
metrics are unaffected; the bundled 887-feature model still trains on it and the
runtime still sends 22. Fixing the mapping requires a dataset regeneration, so it
is recorded rather than patched (see ROADMAP §8).

Two population facts follow, and they are *good* news for product alignment:
**90.3% of the corpus is Captains Mode** (92,221 rows), not All Pick. What is
missing is not draft data but *decision* data.

### Per-mode validation metrics (same candidate, same rows)

| Segment | Rows | ROC-AUC (95% CI) | Brier | Accuracy |
| --- | --- | --- | --- | --- |
| All validation | 10,212 | 0.5481 (0.5369-0.5592) | 0.2473 | 0.5402 |
| Captains Mode | 8,162 | 0.5480 (0.5356-0.5605) | 0.2474 | 0.5375 |
| Captains Draft | 969 | 0.5666 (0.5306-0.6026) | 0.2456 | 0.5655 |
| OpenDota Captains Mode | 705 | 0.5336 (0.4909-0.5764) | 0.2480 | 0.5362 |
| Random/Single/other | 376 | 0.5356 (0.4773-0.5939) | 0.2481 | 0.5426 |

No mode is materially different, and none is close to the 0.60 floor. The
dominant mode — the one the product operates in — is exactly the overall number.

### Ban timing: how much of the early-draft score was future information

Each stage was rebuilt twice on the same 9,507 validation rows that have a raw
`pickBans` order: with **all** bans (the harness convention) and with only the
bans that had already happened at that stage. The dominant CM order is
`BBBBBBBPPBBBPPPPPPBBBBPP`, so 3 bans per side are known at the first pick, 5 by
the second/third, and all 7 only at the end.

| Picks/side | Bans/side known | AUC all bans | AUC timed bans | Delta | Brier (timed) |
| --- | --- | --- | --- | --- | --- |
| 0 | 2.82 | 0.5271 | **0.5228** | -0.0044 | 0.2498 |
| 1 | 2.93 | 0.5248 | **0.5219** | -0.0029 | 0.2499 |
| 2 | 4.42 | 0.5332 | 0.5332 | 0.0000 | 0.2487 |
| 3 | 4.50 | 0.5381 | 0.5376 | -0.0005 | 0.2483 |

So the harness convention overstated the two earliest stages by 0.003-0.004 AUC
— about a third of the distance those stages had above chance. The previously
reported 0.527 at zero picks becomes **0.523**, and the model is still barely
above chance on bans alone. From two picks on, the convention was harmless
because the real draft has already revealed those bans.

### How much can the score separate two candidate heroes?

Predicted probabilities within a stage (timed rebuild):

| Picks/side | SD | p05 | p50 | p95 | p95 - p05 |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.0184 | 0.5354 | 0.5687 | 0.5945 | 0.0591 |
| 1 | 0.0514 | 0.4491 | 0.5417 | 0.6189 | 0.1698 |
| 3 | 0.0502 | 0.4384 | 0.5225 | 0.6032 | 0.1648 |
| 5 (stored row) | — | — | — | — | 0.2473 Brier |

At the first pick the *entire* spread of the model across 9,507 different drafts
is 5.9 percentage points. That is the practical resolution of a "win probability"
recommendation at the moment it is generated.

### Can the corpus support the other targets?

| Probe | Result | Implication |
| --- | --- | --- |
| Distinct full 10-hero drafts | 102,080 of 102,118 | Contexts are essentially unique |
| Full drafts seen more than once | 38 (max 2) | No repeated full-draft outcomes to compare |
| Distinct 2-a-side contexts | 101,400 | — |
| 2-a-side contexts with >= 50 matches | **0** | No context has enough followers to rank heroes |
| Heroes with >= 50 radiant picks | 125 (median 3,249) | Hero-level rates are estimable |
| Per-hero win-rate CI half-width | 1.7 pp | Usable precision |
| Observed vs binomial per-hero SD | 2.29 pp vs 1.04 pp (SNR 2.0) | Real between-hero variation exists, but it is observational and confounded with player/team skill and patch |
| Validation AUC CI half-width | 0.0111 | Differences below ~1.1 AUC points are unresolvable here |

### Feasibility of the four objectives

| | Target and unit | Inputs at inference | Label construction | Leverage / selection risk | Metric | Data sufficient? |
| --- | --- | --- | --- | --- | --- | --- |
| **A. Full-match win** | P(radiant wins); one per match | Partial state, runtime defaults | Observed outcome | None new; already near ceiling | AUC/Brier/ECE | Yes, but the ceiling is ~0.55 |
| **B. Conditional draft-stage win** | P(win \| state at decision point); one per match x stage | Only the picks/bans known then | Same outcome, restricted features | Needs ban timing (now measurable); repeated-match weighting | AUC/Brier per stage | Yes; 0.523-0.538 measured |
| **C. Next-pick ranking** | Order over legal heroes; unit = (state, hero) | Full legal set at that moment | **Requires counterfactual outcomes**; observed picks only label one alternative | Strong: winners pick differently; no repeated contexts; hero x player skill | Top-k imitation accuracy, decision value on held-out drafts | **No** for counterfactual; yes for imitation |
| **D. Pick-level marginal** | Win-rate delta attributable to one pick | Same as B | One match label shared by 10 picks — not identified | Unidentifiable without a design assumption or intervention | — | **No** |

### Recommendation

1. **A is finished as the headline metric.** Three model families, two
   populations, corrected data: 0.538-0.550 AUC. The CI half-width is 0.011, so
   the remaining gap to 0.60 is ~5 standard errors. More of the same data moves the
   interval, not the point estimate.
2. **B is the only objective this corpus supports today**, and it is already what
   the partial-draft evaluation measures. It should be reported as a *relative*
   comparison (which of two states looks better) rather than as a calibrated win
   probability, because the absolute spread at the first pick is 5.9 pp.
3. **C is the product-right objective but is not identifiable here.** No 2-a-side
   context occurs 50 times, and 38 full drafts out of 102,080 repeat at all. Any
   "this pick is worth +X%" claim from this data would be model extrapolation, not
   an estimated effect. **Do not ship counterfactual win-delta claims.**
4. **D requires a labelling strategy that does not exist yet** — 102,118 labels
   for 1,021,180 pick instances, and no way to attribute an outcome to one pick.
   The hero-level rate (SNR 2.0, +/-1.7 pp) is the closest identifiable object, and
   it is confounded by who picks the hero.
5. **Minimum next experiment (C, done honestly):** train a next-pick *imitation*
   model (which hero did a strong captain pick in this state, among the legal set)
   and evaluate top-1/top-3 accuracy and log loss on held-out drafts. It is
   identifiable, needs no counterfactuals, and tests whether the state carries
   decision-relevant signal at all. A decision-value check can follow only where
   contexts repeat — or from a new data source that logs rejected alternatives.

### Limitations

- No fresh final evaluation exists. The frozen test window was inspected in an
  earlier session and stays out of selection; a clean number needs matches after
  `2026-08-28T07:44:36Z`, which the repository does not have.
- "True mode" comes from the raw STRATZ payloads; the 5,634 OpenDota rows keep
  their numeric mode and are labelled `OPENDOTA_NUMERIC_*`. Those rows carry no
  ban order, so they are excluded from the paired timing study (9,507 of 10,212).
- The ban-timing rebuild is the harness's own producer, not the Rust runtime; a
  real client also knows *which* team is on the clock, which is still not modelled.
- Per-hero rates are observational. `avg_mmr` is identically 0, so skill cannot be
  controlled even in principle from this corpus.
- `CAPTAINS_DRAFT` (3 bans/side) and Turbo rows are pooled into the same training
  population; a mode-specific model would have less data, not more signal.

## 3e. Next-pick imitation experiment (2026-10-29, session 10)

Research only: no production inference change, no promotion, test partition never
parsed. Question: **does the draft state predict the observed next pick better
than a hero-frequency table?**

```
ml/venv/Scripts/python.exe ml/training/imitation.py
  # exit 0, 486 s (44 s of it re-parsing raw payloads; cached afterwards)
  # -> ml/models/imitation/imitation_report.json
```

This is an **imitation** target. It measures whether the state carries
decision-relevant information, not whether the imitated pick was good.

### Chronology: is the target trustworthy?

The parquet stores picks in five slots per side with no interleaving, so order was
rebuilt from the raw payloads. `order` is dense and unique in every eligible CM
payload (0..23), which makes the sequence unambiguous. Action types are stable:

| Template | Matches | Share |
| --- | --- | --- |
| `BBBBBBBPPBBBPPPPPPBBBBPP` | 2,577 of 2,735 sampled | 94% |
| `BBBBPPPPBBBBBBPPPPBBBBPP` | 124 | older CM order |
| 22-23 actions | 34 | short/incomplete |

A cross-check of 60 random CM matches found **0** mismatches between the ordered
raw picks and the parquet's `rad_pick_0..4` / `dire_pick_0..4` slots, and 0 label
mismatches, so the frozen split applies to the same rows.

Payloads that cannot establish order are **excluded, never repaired**. From the
91,907 train+validation CM candidates:

| Exclusion | Matches |
| --- | --- |
| `order` missing (null) | 3,022 |
| malformed side/type flags | 2,005 |
| hero outside the 1..139 contract | 1,557 |
| unsupported mode (`CAPTAINS_DRAFT` etc., from raw `gameMode`) | 2,791 |
| no STRATZ `gameMode` (OpenDota rows) | 705 |
| more than 24 actions | 126 |
| **Eligible** | **81,701** (74,593 train / 7,108 validation) |

Every eligible match contributes exactly 10 states: **745,930 train** and
**71,080 validation**. Mean legal candidate set 124.5 (116-135); 125 distinct
target heroes.

### Leakage controls

A state contains only the actions before its pick: the six bias and 556 removal
features are built from that prefix alone, the match result is never read, and a
test asserts that changing every later pick, later ban and the outcome leaves
state 0 and 1 unchanged. The test partition is not parsed at all — the action
scan is restricted to train+validation match ids, and the run aborts if a test
match appears in the action table.

### Model and baselines

One model: multinomial logistic regression over 139 hero classes, features
1,668 → **562** columns after the capacity fix described below. Legal candidates
are masked and the distribution renormalised; ties break to the lower hero id.

| Method | What it is |
| --- | --- |
| `global_frequency` | training-count frequency, Laplace 1.0 |
| `stage_side_frequency` | frequency within (draft phase, acting side) |
| `model_bucket_only` | the same model with all removal columns zeroed — its per-phase hero intercept alone |
| `model` | full model |

`C` was chosen on an internal chronological holdout **inside the training
partition** (last 15% of training matches), never on the frozen validation set:
C=0.03 → 0.0585 top-1, C=0.1 → 0.0590, C=0.3 → 0.0586; C=0.1 selected.

Two design corrections were needed, both measured rather than assumed:

1. **No per-phase hero intercept.** The first version had only a global class
   bias, so it could not represent the frequency baseline it was compared
   against. Six always-on `(phase, side)` indicators fix this: because the model
   gives each class its own coefficient per feature, one column carries a full
   per-phase hero intercept.
2. **Phase-specific removal coefficients overfit.** With removals split by phase
   (1,668 columns) a 4,000-match smoke run put first-pick top-1 at 0.0509 against
   a 0.0990 baseline. Pooling removal effects across phases and keeping preference
   phase-specific (562 columns) raised that smoke figure to 0.0688 and improved
   every other stage; the full-data run of the same configuration reports 0.0744
   below. Both corrections were kept for the final run.

### Results (validation: 71,080 states from 7,108 matches)

| Method | Top-1 | Top-3 | MRR | Log loss |
| --- | --- | --- | --- | --- |
| `global_frequency` | 0.0324 | 0.0733 | 0.0927 | 4.5652 |
| `stage_side_frequency` | 0.0405 | 0.0959 | 0.1107 | 4.4637 |
| `model_bucket_only` | 0.0302 | 0.0671 | 0.0864 | 4.8153 |
| **`model`** | **0.0551** | **0.1312** | **0.1412** | **4.3381** |

Match-level cluster bootstrap (1,000 resamples of the 7,108 matches, so the ten
decisions of one draft stay together):

| Difference | Top-1 | Top-3 | MRR |
| --- | --- | --- | --- |
| model − stage/side frequency | **+0.0146** (+0.0128..+0.0165) | +0.0352 (+0.0322..+0.0380) | +0.0304 (+0.0285..+0.0322) |
| model − bucket-only | **+0.0249** (+0.0230..+0.0267) | +0.0640 (+0.0610..+0.0667) | +0.0548 (+0.0529..+0.0566) |
| bucket-only − stage/side frequency | −0.0103 (−0.0117..−0.0089) | −0.0288 | −0.0243 |

All three intervals exclude zero with 100% of resamples on the same side.

### Per-pick breakdown — where the signal actually lives

| Pick | Top-1 base | Top-1 model | Top-3 base | Top-3 model | Bucket-only top-1 |
| --- | --- | --- | --- | --- | --- |
| 0 | **0.0990** | 0.0744 | 0.1546 | 0.1543 | 0.1003 |
| 1 | **0.0689** | 0.0457 | 0.1178 | 0.1196 | 0.0716 |
| 2 | 0.0319 | **0.0490** | 0.1103 | 0.1217 | 0.0302 |
| 3 | 0.0405 | **0.0519** | 0.1140 | 0.1235 | 0.0369 |
| 4 | 0.0238 | **0.0476** | 0.0926 | 0.1239 | 0.0272 |
| 5 | 0.0208 | **0.0468** | 0.0698 | 0.1152 | 0.0132 |
| 6 | 0.0160 | **0.0436** | 0.0499 | 0.1106 | 0.0096 |
| 7 | 0.0113 | **0.0550** | 0.0318 | 0.1234 | 0.0051 |
| 8 | 0.0506 | **0.0741** | 0.1140 | 0.1653 | 0.0056 |
| 9 | 0.0416 | **0.0629** | 0.1048 | 0.1542 | 0.0021 |

By phase: early (picks 0-1) base 0.0840 vs model **0.0601**; mid (2-7) 0.0241 vs
**0.0490**; late (8-9) 0.0461 vs **0.0685**.

### Reading the results

- **The state carries real information — after the first pick.** From pick 2 on
the model beats the frequency table at every stage, and the gap widens as heroes
leave the pool (pick 7: 0.0550 vs 0.0113, ~5x). Pooled removal effects are what
drive it: removing state (`model_bucket_only`) costs 0.0249 top-1.
- **At the opening pick it is worse than a plain frequency table** (0.0744 vs
  0.0990), and its own intercept in isolation does fine there (0.1003). The ban
  features actively mislead: ban→pick substitution patterns learned from later
  phases are applied to a state that has no picks yet. Early-stage usefulness —
  the product's main concern — is therefore *not* demonstrated.
- **The model's intercept alone is not a good frequency estimate** (−0.0103 vs the
  explicit table): joint fitting with 556 removal columns and L2 shrinkage costs
  accuracy on a statistic the table computes directly. That is why the comparison
  of record is model vs `stage_side_frequency`, not model vs its own intercept.
- **Absolute accuracy is low.** 5.5% top-1 among ~125 legal heroes; top-3 13%.
  Non-trivial relative to the 0.8% chance baseline, but far from a recommendation
  engine.

### What this does and does not establish

- It establishes that φ(draft state) is informative about **what a strong captain
tends to do next**, especially mid-draft, and that a frequency table is not the
ceiling.
- It says nothing about win rate, strategic quality, or counterfactual value. A
pick can be perfectly imitated and still be wrong. No "+X% win" claim follows.- The early-stage deficit is a model-capacity finding, not proof that the
  information is absent: a stage-aware model (or an explicit early-stage fallback)
  might match the frequency table at pick 0. That is future work, not a result.

## 3f. Phase-aware imitation experiment (2026-10-29, session 11)

Research only: no production inference change, nothing promoted, test partition
never parsed. Question: **can the opening picks match or beat the stage/side
frequency table without giving up the pooled model's mid/late gains?**

```
ml/venv/Scripts/python.exe ml/training/imitation_phase.py --bootstrap 1000
  # exit 0, 356 s
  # -> ml/models/imitation/phase/report.json
  # -> ml/models/imitation/phase/decision_rule.json   (written before any metric)
```

### Protocol

Reuses the session-10 action cache, states, splits and legal-candidate masking, so
the populations are identical: 74,593 train / 7,108 validation matches, 745,930 /
71,080 states. The pooled model is refitted from the same configuration and
**reproduced session 10 exactly** (overall top-1 delta 0.0e+00, pick-0 delta
0.0e+00, 71,080 states), which is the check that the harness and cache still agree.

Four strategies, all scored on the same rows and candidate sets:

| Strategy | What it is |
| --- | --- |
| `stage_side_frequency` (A) | training-only frequency within (phase, acting side), Laplace 1.0; the primary comparator |
| `pooled_model` (B) | session-10 model, 562 features, C=0.1 |
| `hybrid_freq_then_pooled` (C) | A at picks 0-1, B at picks 2-9 |
| `early_pick0` / `early_pick1` / `early_shared` (D) | separate model on that stage's states only: 280 features (bans + side bias) for pick 0, 558 (adds the one known pick) for pick 1, 558 for the shared 0-1 model |
| `pooled_intercept_only` | diagnostic: B with all 556 removal columns zeroed, i.e. its per-phase hero intercept alone |

`C` was selected on the same internal chronological holdout inside training (last
15% of training matches) for every model: pooled {0.03, 0.1, 0.3} → 0.0585 / 0.0590
/ 0.0586; early grids {0.01, 0.03, 0.1} → pick 0: 0.0916 / 0.0990 / **0.0994**,
pick 1: 0.0456 / 0.0526 / **0.0610**, shared: 0.0677 / 0.0744 / **0.0764**. No
validation row influenced any choice.

### Decision rule (pre-registered, `decision_rule.json`, 2026-10-29T11:13:58)

Written before the first validation metric existed and refused on later runs if
edited: the hybrid is acceptable if it does not regress against the table at picks
0-1 and keeps the pooled model's gains at 2-9; an early model is promising only if
its top-1 at the target stage beats the table **and** the paired, match-clustered
95% interval for the difference excludes zero on the positive side; beating the
intercept-only diagnostic does not qualify anything; otherwise the recommendation
is the frequency fallback, not a wider search.

### Results (validation, top-1)

| Strategy | pick 0 | pick 1 | picks 2-7 | picks 8-9 | all 71,080 |
| --- | --- | --- | --- | --- | --- |
| `stage_side_frequency` | 0.0990 | 0.0689 | 0.0241 | 0.0461 | 0.0405 |
| `pooled_model` | 0.0744 | 0.0457 | 0.0490 | 0.0685 | 0.0551 |
| **`hybrid_freq_then_pooled`** | **0.0990** | **0.0689** | **0.0490** | **0.0685** | **0.0599** |
| `pooled_intercept_only` | 0.1003 | 0.0716 | 0.0204 | 0.0039 | 0.0302 |
| `early_pick0` | 0.0575 | 0.0457 | 0.0490 | 0.0685 | 0.0534 |
| `early_pick1` | 0.0744 | 0.0629 | 0.0490 | 0.0685 | 0.0568 |
| `early_shared` | 0.0640 | 0.0563 | 0.0490 | 0.0685 | 0.0551 |

`early_*` and the hybrid keep the pooled model outside their own stage, so identical
cells are routing, not new fits. Secondary metrics (all states): baseline top-3
0.0959 / MRR 0.1107 / log loss 4.4637; pooled 0.1312 / 0.1412 / 4.3381; hybrid
0.1310 / 0.1427 / 4.3446; `early_pick1` 0.1332 / 0.1432 / 4.3326.

Top-1 differences vs the stage/side table, paired match-level cluster bootstrap
(1,000 resamples; all ten decisions of a draft move together):

| Stage | Strategies | Δ top-1 (95% CI) |
| --- | --- | --- |
| pick 0 | `early_pick0` | −0.0416 (−0.0492..−0.0339) |
| pick 0 | `early_shared` | −0.0351 (−0.0426..−0.0277) |
| pick 0 | `early_pick1`, `pooled` | −0.0246 (−0.0307..−0.0189) |
| pick 0 | `pooled_intercept_only` | +0.0012 (−0.0018..+0.0041) |
| pick 1 | `early_pick1` | −0.0061 (−0.0124..+0.0007) |
| pick 1 | `pooled_intercept_only` | +0.0027 (−0.0011..+0.0066) |
| picks 0-1 | `early_shared` | −0.0239 (−0.0291..−0.0187) |
| picks 0-1 | `pooled` | −0.0239 (−0.0281..−0.0199) |
| picks 2-7 | `pooled`, hybrid, `early_*` | **+0.0249** (+0.0225..+0.0270) |
| picks 8-9 | `pooled`, hybrid, `early_*` | **+0.0223** (+0.0175..+0.0274) |
| all | hybrid | **+0.0194** (+0.0178..+0.0211) |
| all | `pooled` | +0.0146 (+0.0128..+0.0165) |
| all | `pooled_intercept_only` | −0.0103 (−0.0117..−0.0089) |

### Reading the results

- **No trained early-stage model beats the table.** All three lose at their target
  stage, two of them decisively: −0.042 (pick 0), −0.024 (picks 0-1). The best
  (`early_pick1`, −0.006 at pick 1) still has an interval that includes zero, so it
  does not qualify under the rule.
- **The early information is the prior, not the ban state.** The intercept-only
  diagnostic — the same phase-and-side hero preferences with every removal column
  zeroed — is statistically indistinguishable from the table at picks 0-1 (+0.0012,
  CI −0.0018..+0.0041) and collapses later (0.0039 at picks 8-9). Adding ban/removal
  columns to that intercept costs 2-4 points at pick 0. Bans reveal what a captain
  is *avoiding*; they do not predict the hero they will take first.
- **A transfer failure, not a tuning miss.** `early_pick0` scored 0.0994 on the
  internal chronological holdout (≈ the table's 0.0990) and 0.0575 on validation.
  A 4-point drop on 7,108 states is far outside sampling noise, so the apparent
  holdout parity was not a real capacity for the future — a warning that the
  last-15%-of-training holdout is an optimistic proxy for the next time window.
- **The hybrid is the best single configuration** (0.0599 overall vs 0.0551 pooled
  and 0.0405 table) and satisfies the pre-registered acceptance condition exactly:
  it equals the table at picks 0-1 (Δ 0.0000) and equals the pooled model at picks
  2-7 and 8-9 (Δ 0.0000), so its overall advantage over the pooled model is exactly
  the pooled model's opening deficit (+0.0239 at picks 0-1, CI +0.0199..+0.0281 on
  the pooled side).
- **Decision: `frequency_fallback_only`**, i.e. strategy C — table at picks 0-1,
  pooled model from pick 2. No prototype was built and no model was promoted.

### Disclosure

The first execution reported `stop` because the acceptance predicate for strategy
C also compared the *aggregate* top-1 against the pooled model, which the hybrid
must change by construction. That comparison was removed before the reported run;
the pre-registered rule text, the early-model criterion, the grids and the seeds
were not touched. Both reports are kept (`report.json` and
`report_first_run_acceptance_bug.json`); every metric is identical between them,
so nothing downstream depends on the fix.

### Limitations

- Validation only, and the same validation partition has now been used by the
  session-10 and session-11 experiments. This is exploratory reuse, not a fresh
  final test; the contaminated test window was not scored.
- One learner, one feature representation, one small grid per model. The early-stage
  conclusion could differ with a richer state (production win rates, hero roles,
  player pools) — but nothing in these results suggests an early-stack ranking
  signal exists to recover.
- Unrevealed ban slots are simply absent from a state, so some early states are
  weaker than the real draft was at that moment.
- Imitation accuracy is a behavioural target: it never measures win probability,
  strategic quality, or counterfactual value, and no threshold was changed.

### Limitations

- Validation only; the frozen test window stays untouched and already inspected.
- One model family, deliberately. No sweep, no ensembling, no per-phase models.
- The model sees a 562-column summary of the draft (who removed whom), not the
  production feature set (win rates, semantics, synergy). A richer state
  representation could change the conclusion in either direction.
- Bans are only as complete as the source: unrevealed ban slots are simply absent
  from the state, which is realistic for the client but makes some states weaker
  than the real draft was at that moment.
- Imitation accuracy is bounded by captain behaviour, not by optimality: heroes
  are picked for reasons (comfort, team plan, role) the features do not encode.

## 4. Feature contract

The model has **887 inputs**. The current processed parquet has no matchup
family; the processor would add seven matchup features if matchup data were
present, producing **894** names. The bundled parquet has no matchup columns,
so shipping them requires a regenerated dataset, a retrain, and a new export.

| Block | Count | Notes |
| --- | --- | --- |
| Hero presence (`rad_has_*`, `dire_has_*`) | 278 | 139 per side; ids `1..139` |
| Hero bans (`rad_ban_*`, `dire_ban_*`) | 279 | Includes legacy `hero_*` names for ids absent from the roster |
| Per-hero win rates (`rad_wr_*`, `dire_wr_*`) | 254 | 127 per side |
| Win-rate aggregates | 8 | Team avg/max/min, `winrate_diff`, ban averages |
| Synergy | 6 | Rad/Dire avg/max/min |
| Semantic team averages | 34 | 17 attributes × 2 sides |
| Semantic differentials | 10 | `stun_diff`, `initiation_diff`, ... |
| Damage diversity | 2 | `radiant_*`, `dire_*` |
| Match/meta | 6 | `patch`, `patch_version`, `duration`, `avg_mmr`, `game_mode`, `lobby_type` |
| Pick order | 10 | `rad_pick_0..4`, `dire_pick_0..4` |
| **Total** | **887** | |

Feature order is part of the contract. `draft_model_xgboost_full_schema.json`
is the runtime source of truth for order, categorical codes, and defaults.

Reproducibility notes (pinned by `ml/tests/test_feature_schema.py`):

- The real `FeatureProcessor` reproduces the exact 887-name set on a fixture
  with no matchup data.
- 631 structural features keep a code-defined order. The 254 per-hero win-rate
  columns and the two `*_ban_avg_winrate` columns are ordered by first
  appearance in the shipped dataset, not by code; the test pins their
  membership, not their position.
- With matchup data the processor emits exactly the 887 names plus the seven
  `*_matchup_*` / `matchup_advantage` features.
- `process_all()` collects thread results in completion order, so a regenerated
  dataset can have a different column order. Canonicalising that order is open
  work (P1-5).

### Historical experiment (not the bundled model)

Earlier notes in this repository reported **0.6710 ROC-AUC / 62.5% accuracy /
0.2273 Brier / 896 features / 103,162 samples**, with matchup features
described as the largest gain. Those numbers belong to a different dataset and
feature set (matchup-enriched, five-fold CV) and are **not reproducible from
the tracked 887-input artifacts**. They are retained here only as historical
context. Do not quote them as the current model.

## 5. Training pipeline

Prerequisites:

```bash
cd ml
python -m venv venv
# Windows
venv\Scripts\activate
pip install -r requirements.lock
```

`requirements.txt` remains the direct-dependency list; `requirements.lock` pins
the full verified Python 3.10.11 environment and is what CI installs.

### Reference training run

`train_full.py` is the script that produced the bundled artifact. It uses the
chronological split, early stopping on validation, and touches the test
partition once.

```bash
ml/venv/Scripts/python.exe ml/training/train_full.py
```

Re-evaluate without retraining:

```bash
ml/venv/Scripts/python.exe ml/training/train_full.py --evaluate-only
```

Export the runtime schema after a retrain (order, categorical codes, defaults):

```bash
ml/venv/Scripts/python.exe ml/training/export_model_schema.py
```

Then run the gate. A candidate may replace the bundled model only after:

1. `quality_gate.py` exits `0` in `inference` mode;
2. the golden ONNX fixture and schema tests still pass;
3. `scripts/verify.ps1` passes;
4. the change is reviewed as a data/model change, not a docs-only edit.

### One-command collection pipeline

`train_pipeline.py` collects and processes data before training:

```bash
python training/train_pipeline.py \
  --matches 100000 \
  --patches 7.41 7.40 7.39 7.38 7.37 \
  --include-stratz \
  --stratz-key YOUR_STRATZ_KEY \
  --api-key YOUR_OPENDOTA_KEY \
  --force
```

This is expensive (network + the full 1.9 GB dataset) and is not part of CI.

### GPU training

```bash
python -c "import xgboost as xgb; print(xgb.__version__)"
python training/trainer.py --device cuda --tree-method hist
```

The bundled `train_full.py` defaults to CPU unless configured otherwise; the
measured wall time in this environment was seconds for evaluation, not a
training benchmark.

## 6. Data sources

| Source | Scale | Format | Access |
| --- | --- | --- | --- |
| OpenDota proMatches | ~6K | JSON match detail | API key optional |
| OpenDota publicMatches | ~5K | JSON match detail | API key optional |
| STRATZ leagues | ~106K | GraphQL full match | Bearer token required |
| STRATZ pro players | ~1.4K | GraphQL full match | Bearer token required |
| All Pick | ~9.8K on disk | JSON | Win-rate/matchup baselines |

STRATZ setup:

1. Get a key at https://stratz.com/api.
2. Set `STRATZ_API_KEY` in `ml/.env` or pass `--stratz-key`.
3. Never commit `ml/.env`; the repository ignores it and runs a secret scan.

## 7. Improvement directions

Ranked by expected value, not novelty:

1. **Regenerate the dataset with correct pick/ban parsing (done in code, not in data).**
   The ablation (§3) already answered the leakage question: dropping the
   context group removes the substitution gap and improves ECE, and dropping
   ordinal pick slots helps the first-pick stage, but neither produces
   discrimination worth shipping. The blocker is now data quality: bans were
   encoded as picks for 81% of rows. Fix `process_all()` ordering, regenerate,
   then repeat the ablation on clean features before tuning anything.
2. **Beat the constant baseline first.** A candidate that cannot beat Brier
   0.2493 and AUC 0.5 on the chronological holdout is not worth integrating,
   regardless of feature count. The 2026-10-28 candidate did not, so it was not
   integrated.
3. **Fresh evaluation window.** The 2026-10-28 test partition has been consumed
   once (one candidate, selected on validation). Any further model claim needs a
   new untouched time window; do not re-score that partition.
4. **Matchup data at training time.** The processor can emit the seven matchup
   features, but the bundled dataset does not contain them. Regenerate with
   matchup data and retrain; do not bolt them onto the existing schema without
   training.
5. **Lane/role features.** The processor has lane extraction paths, but lane
   features are disabled in the bundled contract and need lane-enriched raw
   data.
6. **Calibration.** ECE is low (0.024 bundled, 0.0139 candidate), but that is
   because the predictions sit within a few points of the base rate. Re-check
   calibration only after a model improves discrimination.
6. **Ensembles/embeddings.** Historical experiments did not beat the baseline
   they were compared against; treat them as experiments until a gated
   candidate shows a real gain.

## 8. Runtime integration

- `apps/desktop/src-tauri/src/ml.rs` loads the ONNX model and
  `_schema.json`, builds the feature map from the live draft, and reads
  `probabilities[class=1]` (`P(Radiant wins)`).
- Missing model/runtime or failed inference returns neutral `0.5`; the
  deterministic engine keeps working.
- Feature families the runtime cannot reconstruct (`rad_lane_*`,
  `dire_lane_*`, `emb_*`) are excluded by `train_full.py`.
- Regenerate the schema after any retrain; the Rust loader validates the
  feature list against the schema and refuses a mismatched bundle.

Useful commands:

```bash
cd apps/desktop/src-tauri
cargo run --example inference_check   # loads the bundled artifact
```

## 9. File locations

```
ml/
├── datasets/
│   ├── raw/                       # collected match JSON (not tracked)
│   ├── processed/
│   │   ├── draft_features.parquet # 102,118 × 890 (bundled training data, v1)
│   │   ├── hero_matchups.parquet
│   │   └── hero_winrates.parquet
│   ├── processed_v2/
│   │   ├── draft_features.parquet # 102,118 × 890 (corrected pick/ban, v2)
│   │   └── dataset_schema.json    # column order, stats, fingerprints
│   └── heroes.json                # tracked roster (127 entries)
├── feature-engineering/
│   └── processor.py               # raw match JSON -> features
├── training/
│   ├── train_full.py              # bundled-model training/evaluation/export
│   ├── ablation.py                # feature-group ablation + candidate export
│   ├── model_families.py          # validation-only family comparison
│   ├── objective_diagnostics.py   # objective study: modes, ban timing, identifiability
│   ├── imitation.py               # next-pick imitation: states, baselines, model
│   ├── imitation_phase.py         # phase-aware routing: early model vs frequency
│   ├── export_model_schema.py     # runtime schema export
│   ├── quality_gate.py            # promotion gate (new)
│   ├── inference_contract.py      # schema-order encoding (new)
│   ├── splits.py                  # chronological split + fingerprints
│   ├── manifest.py                # manifest construction
│   ├── train_pipeline.py          # collect -> process -> train
│   └── data_collector.py          # OpenDota + STRATZ
├── tests/
│   ├── test_feature_schema.py     # processor -> 887-name contract (new)
│   ├── test_quality_gate.py       # gate/promotion behaviour (new)
│   ├── test_ablation.py           # ablation config/selection/export (new)
│   ├── test_model_families.py     # family config, preprocessing, export parity
│   ├── test_objective_diagnostics.py  # ban-timing rule + timed stage rebuild
│   ├── test_imitation.py          # chronology, leakage, legality, metrics
│   ├── test_imitation_phase.py    # pre-registration, early matrices, routing
│   ├── test_onnx_contract.py      # golden vector + schema
│   └── fixtures/                  # golden feature vector
└── models/                        # gitignored except the 4 bundled artifacts
    ├── ablation/                  # session-4 candidate + reports (gitignored)
    ├── ablation_v2/               # session-7 corrected-data ablation (gitignored)
    ├── families/                  # session-8 family comparison + ONNX (gitignored)
    ├── objective_study/           # session-9 diagnostics + cached modes (gitignored)
    └── imitation/                 # session-10 cache + report, phase/ (gitignored)
```

## 10. Troubleshooting

| Issue | Solution |
| --- | --- |
| OpenDota 429 | Lower `--rate-limit`, add an API key |
| STRATZ 401 | Check the bearer token in `ml/.env` |
| Gate exits 2 | Read `evaluation.error` in the report: artifact, fingerprint, split, labels, or ONNX contract |
| Gate exits 1 | The model failed a threshold; improve/retrain it. Do not lower the policy to make it pass |
| `Duration`/`avg_mmr` warnings | Those fields are post-draft/constant; the runtime substitutes schema defaults |
| XGBoost UBJSON warning | Pre-existing artifact format warning; harmless |
| Low AUC | Check the split and baseline first, then features; adding columns rarely fixes a near-baseline model |

## 11. Next steps

1. Repeat the ablation (§3) on `processed_v2` using the frozen split in
   `ml/configs/eval_split_v2.json`. Select on train/validation only, exclude the
   3,870 duplicate-hero rows explicitly, and keep ML opt-in until a candidate
   passes the gate and a live review.
2. Treat the v2 test partition as a regression reference, not a fresh holdout:
   collect matches newer than 2026-08-28 and freeze them before any final claim.
3. Regenerate with matchup data only as part of a gated retrain, after `ml.rs`
   can rebuild those features from a live draft.
4. Keep the bundled artifact and the 887-feature schema unchanged until then.
5. Optionally collect more history: the corpus has no matches between
   2026-08-28 and today, so every current window is >2 months stale.
6. If the imitation line is continued, route picks 0-1 to the stage/side frequency
   table (§3f) rather than a trained early model, and stop expanding the model
   search: three families and three early-stage configurations have all failed to
   beat a frequency table where the product needs them most.
