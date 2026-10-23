# Smart Draft Assistant — Production ML Architecture & Training Pipeline

## Project

AI-powered drafting assistant for a competitive drafting environment.

## Goal

Create a production-quality drafting intelligence system with:

- large-scale datasets
- real-time inference
- contextual recommendations
- explainable ML
- long-term scalability

This document describes:

- final ML architecture
- ideal training pipeline
- dataset strategy
- feature engineering
- inference design
- problems in the current implementation
- how to evolve the system into a real drafting AI

---

# 1. High-Level ML Philosophy

## CRITICAL RULE

The ML system should **NOT** be:

- a raw win predictor
- a black-box neural network
- a pure statistical memorizer

The ML system **SHOULD** be:

- a contextual draft evaluator
- a confidence estimator
- an augmentation layer for the draft engine

---

# Final System Architecture

```text
                    ┌────────────────────┐
                    │ Draft State Engine │
                    └─────────┬──────────┘
                              │
                              ▼
                    ┌────────────────────┐
                    │ Feature Generator  │
                    └─────────┬──────────┘
                              ▼
                ┌──────────────────────────┐
                │ Rule-Based Draft Engine  │
                └─────────┬────────────────┘
                          ▼
                ┌──────────────────────────┐
                │ ML Recommendation Layer  │
                └─────────┬────────────────┘
                          ▼
                ┌──────────────────────────┐
                │ Final Recommendation API │
                └──────────────────────────┘
```

---

# Important Design Principle

The draft engine ALWAYS works.

ML improves:

- confidence
- contextual weighting
- hidden synergies
- personalization

If ML crashes:

the app still functions.

---

# 2. Problems In Current Implementation

Your current implementation is a strong prototype,
but not production-ready.

Main issues:

---

## PROBLEM 1 — Raw Hero IDs As Features

### Current code

```python
row[f"{team_prefix}_hero_{pb['order']}"] = pb['hero_id']
```

This is mathematically incorrect.

The model sees:

```text
Axe = 2
Invoker = 74
```

It incorrectly assumes:

- `74 > 2`
- numerical relationships exist

But hero IDs are categorical values.

### FIX

Use:

- one-hot encoding
- OR embeddings

### Correct Representation

#### BAD

```text
radiant_pick_1 = 74
```

#### GOOD

```text
radiant_has_invoker = 1
radiant_has_axe = 0
```

---

## PROBLEM 2 — Pick Order Dependency

Current code:

```text
rad_hero_0
rad_hero_1
```

This creates unstable features because:

- draft order changes
- patches change
- pick positions vary

The model learns noise.

### FIX

Represent:

- team composition
- NOT pick slots

---

## PROBLEM 3 — No Ban Features

Bans are strategically important.

Without bans:

- draft context is incomplete
- strategy inference becomes weak

### FIX

Add:

```text
radiant_ban_invoker
dire_ban_pudge
```

---

## PROBLEM 4 — No Patch Awareness

Dota is highly patch-dependent.

A hero can go from:

- S-tier
- to unplayable

in one patch.

Your current model mixes:

- incompatible metas
- outdated hero relationships

### FIX

Add:

- patch version
- patch timestamp
- patch meta features

---

## PROBLEM 5 — No Hero Semantics

Your current model does NOT understand:

- stun
- push
- scaling
- initiation
- greed

It only memorizes hero combinations.

### FIX

Add semantic hero features.

### Example

```json
{
  "hero": "Mars",
  "stun": 0.8,
  "teamfight": 0.95,
  "mobility": 0.4,
  "scaling": 0.5,
  "push": 0.2
}
```

---

## PROBLEM 6 — Accuracy Is A Weak Metric

Current metric:

```python
accuracy_score()
```

This is insufficient.

### Why

- win prediction is probabilistic
- calibration matters more
- confidence matters more

### FIX

Use:

- ROC-AUC
- LogLoss
- Brier Score
- Calibration Error

---

## PROBLEM 7 — Data Leakage

Current:

```python
train_test_split()
```

This mixes:

- old matches
- future matches

The model accidentally learns future meta trends.

### FIX

Use:

- time-based split

### Example

```text
Train:
7.41a → 7.41d

Validation:
7.41e

Test:
7.41f
```

---

## PROBLEM 8 — Dataset Too Small

Current:

```text
~1000 matches
```

This is nowhere near enough.

You need:

- millions of matches
- multiple MMR brackets
- multiple drafts
- multiple patches

---

## PROBLEM 9 — Runtime API Fetching

Current:
fetching match data directly during execution.

### Problems

- slow
- unreliable
- rate-limited

### FIX

Create:

- offline data pipeline
- local datasets
- scheduled updates

---

## PROBLEM 10 — No Feature Engineering

Current model:
mostly memorization.

Production ML:
depends heavily on feature engineering.

---

# 3. Final Dataset Strategy

## Goal

Build:

- massive structured historical dataset.

---

# Recommended Match Sources

## OpenDota

Primary source.

## STRATZ

Advanced analytics.

## Steam API

Player metadata.

---

# Dataset Scale Target

## MINIMUM

```text
500,000 matches
```

## GOOD

```text
5,000,000+ matches
```

---

# Recommended Scope

Start from:

- all ranked matches
- all pro matches
- patch 7.41+

---

# Recommended Filters

Use:

- ranked only
- non-abandoned matches
- valid drafts only

Optional:

- high MMR priority
- immortal bracket weighting

---

# 4. Dataset Architecture

## Raw Dataset

Store original API responses.

```text
datasets/raw/
```

Never overwrite raw data.

---

## Processed Dataset

Normalized features.

```text
datasets/processed/
```

---

## Feature Store

Precomputed ML features.

```text
datasets/features/
```

---

# Suggested Storage Format

Use:

```text
parquet
```

NOT:

```text
CSV
```

### Why

- smaller
- faster
- typed
- columnar
- scalable

---

# Recommended Structure

```text
ml/

├── datasets/
│   ├── raw/
│   ├── processed/
│   ├── features/
│   └── embeddings/
│
├── pipelines/
│   ├── fetch/
│   ├── clean/
│   ├── transform/
│   └── train/
│
├── models/
│
└── experiments/
```

---

# 5. Feature Engineering

## MOST IMPORTANT SECTION

Feature engineering matters more than model complexity.

---

# Final Feature Categories

## A. Hero Presence Features

```text
radiant_has_axe
dire_has_invoker
```

One-hot encoded.

---

## B. Hero Semantic Features

### Team totals

```text
radiant_total_stun
radiant_total_push
radiant_total_teamfight
radiant_total_mobility
```

---

## C. Draft Balance Features

```text
radiant_magic_damage
radiant_physical_damage
radiant_disable_count
radiant_save_count
```

---

## D. Synergy Features

```text
phoenix_mars_combo
magnus_void_combo
```

---

## E. Matchup Features

```text
anti_carry_score
lane_pressure_score
```

---

## F. Meta Features

```text
hero_meta_strength
patch_winrate
pickrate_trend
```

---

## G. Player Features

```text
comfort_score
hero_mastery
preferred_role
hero_games
```

---

## H. Draft Tempo Features

```text
early_game_strength
late_game_scaling
snowball_potential
```

---

## I. Lane Features

```text
safe_lane_strength
offlane_synergy
mid_lane_advantage
```

---

# 6. Final ML Architecture

## Phase 1 — Gradient Boosting

Use:

- XGBoost
- LightGBM

### Reason

- best tabular performance
- explainable
- fast
- low-latency inference

---

## Phase 2 — Hero Embeddings

Train embeddings from:

- co-picks
- matchups
- draft contexts

### Purpose

Learn hidden hero relationships.

---

## Phase 3 — Neural Draft Understanding

Optional future stage.

### Potential

- transformers
- graph neural networks

ONLY after:

- huge dataset
- stable pipeline
- proven baseline

---

# 7. Training Pipeline

## Step 1 — Fetch Matches

Continuously ingest:

- new ranked matches
- pro matches
- high MMR matches

---

## Step 2 — Normalize Data

Convert:

```text
raw API JSON
→ structured schema
```

---

## Step 3 — Generate Features

Build:

- one-hot vectors
- semantic scores
- synergy scores
- meta scores

---

## Step 4 — Split By Time

IMPORTANT.

Never random split.

Use:

- chronological split

---

## Step 5 — Train Models

Train:

- baseline model
- meta-aware model
- personalized model

---

## Step 6 — Evaluate

### Metrics

- ROC-AUC
- calibration
- prediction confidence

---

## Step 7 — Export Model

Convert:

```text
XGBoost
→ ONNX
```

### Reason

Fast desktop inference.

---

## Step 8 — Runtime Inference

The desktop app:

- loads local model
- runs inference locally
- never depends on cloud latency

---

# 8. Recommended Training Targets

NOT ONLY:

```text
predict radiant win
```

That is too simplistic.

---

# Better Targets

## A. Draft Strength

How strong is this draft?

---

## B. Pick Value

How valuable is this hero pick in current context?

---

## C. Lane Advantage

Expected lane dominance.

---

## D. Comfort Prediction

Will this player perform well on this hero?

---

## E. Teamfight Strength

Expected teamfight performance.

---

# 9. Real Production Inference

## IMPORTANT

Do NOT run:

- heavy ML pipelines during draft.

Inference must be:

- tiny
- fast
- cached

---

# Best Runtime Design

Precompute:

- synergy matrices
- hero embeddings
- matchup tables

At runtime:
only:

- lightweight scoring
- tiny ML inference

---

# 10. Explainability System

Critical feature.

The model must explain:
WHY a hero is recommended.

---

# Example Output

```text
Mars recommended because:
+ strong against Sniper
+ lane synergy with Phoenix
+ team lacks initiation
+ high comfort hero score
+ favorable meta
```

Use SHAP Values.

Recommended:

- SHAP explanations for XGBoost

### Purpose

Show feature contributions.

---

# 11. Final Production Architecture

## Best Long-Term Design

```text
RULE ENGINE
+
ML LAYER
+
PERSONALIZATION
+
META ANALYZER
```

### Example

```text
Rule engine score: 7.4
ML confidence boost: +0.8
Comfort bonus: +0.5

Final score: 8.7
```

This is FAR better than:

```text
pure neural prediction
```

---

# 12. Best Development Order

## Stage 1

Build:

- massive dataset pipeline
- hero semantic system
- one-hot features

---

## Stage 2

Build:

- rule-based draft engine
- explainability

---

## Stage 3

Train:

- XGBoost recommendation model

---

## Stage 4

Add:

- personalization
- comfort modeling
- lane prediction

---

## Stage 5

Add:

- embeddings
- advanced ML

---

# Final Advice

Your current implementation is:

- a VERY strong prototype.

But to become:

- a real drafting intelligence platform

You must move from:

```text
"hero ID memorization"
```

To:

```text
"contextual draft understanding"
```

That transition is the most important step in the entire project.

---

# 13. Advanced Production-Level Problems

These are the next-generation architectural problems that appear after the initial ML system becomes functional.

---

## PROBLEM 1 — HERO FEATURE COLLISION

Current implementation pattern:

```python
for hero_id in range(1, 150):
```

This is dangerous.

### Why

Hero IDs:

- are not continuous
- may change
- may contain gaps

### Correct Approach

Generate features dynamically:

```python
all_heroes = hero_metadata.keys()
```

This avoids hardcoded assumptions.

---

## PROBLEM 2 — Massive Sparse Matrix

Current feature expansion:

```text
130 heroes
× pick
× ban
× radiant/dire
```

Already creates:

- hundreds of sparse features

Later additions:

- synergies
- combos
- lane interactions

will explode feature dimensionality.

### Current State

This is acceptable for early development.

### Future Requirement

Sparse optimization becomes mandatory.

### Recommended Solutions

Use:

- scipy sparse matrices

OR:

- LightGBM categorical optimization

---

## PROBLEM 3 — Semantic Features Are Too Naive

Current feature style:

```python
rad_total_stun += ...
```

This is a good starting point.

But:

### Why It Is Insufficient

Two stun heroes do NOT automatically create a strong draft.

Example:

```text
Lion + Shadow Shaman
```

May have:

- huge disable potential

BUT ALSO:

- terrible mobility
- weak lanes
- low tempo

### Required Improvement

Add interaction features.

### Examples

```text
teamfight_balance
frontline_presence
save_dependency
tempo_alignment
```

---

## PROBLEM 4 — No Lane Understanding

Current model:

- does NOT understand lane structure.

This is critical for drafting intelligence.

### Required Addition

Lane simulation features.

### Examples

```text
expected_safe_lane_strength
mid_matchup_score
offlane_pressure
```

---

## PROBLEM 5 — No Role Assignment

Very large issue.

### Example

```text
Pudge:
pos1
pos4
pos5
```

These are effectively different heroes for ML.

### Required Solution

Probabilistic role inference.

### Example

```json
{
  "Pudge": {
    "pos1": 0.05,
    "pos2": 0.12,
    "pos4": 0.71
  }
}
```

Without this:

- the model becomes confused
- feature meaning becomes unstable

---

## PROBLEM 6 — No Draft State Modeling

Current model:

- sees only the final draft.

But a real assistant must work during the draft.

### Required Training Separation

Train separately for:

- first pick situations
- second phase situations
- last pick situations

### Why This Matters

The best hero for:

- first pick
- last pick

is a completely different problem.

### Correct Production Architecture

Separate draft-phase models.

### Example

```text
model_fp.pkl
model_second_phase.pkl
model_last_pick.pkl
```

---

## PROBLEM 7 — Your Label Is Still Weak

Current label:

```text
radiant_win
```

This is extremely noisy.

### Why

Victory depends on:

- execution
- smurfs
- throws
- mistakes
- disconnects

Not only the draft.

### Better Labels

Use draft-quality targets.

### Examples

```text
gold advantage at 15
lane win probability
teamfight dominance
expected win contribution
```

This is a huge improvement.

---

## PROBLEM 8 — No Personalization

Current model:

- universal

But the application should become:

- a personal drafting assistant.

### Required Layer

Personal scoring layer.

### Examples

```text
comfort_bonus
historical_performance
hero_mastery
tilt_risk
```

---

## PROBLEM 9 — No Meta Segmentation

Very important.

Do NOT combine:

```text
all 7.41 patches together
```

Instead:

```text
7.41a
7.41b
7.41c
```

### Why

Letter patches:

- change balance
- shift priorities
- alter draft viability

Even micro patches can completely change the meta.

---

## PROBLEM 10 — Inference Size

Large XGBoost models:

```text
500 trees
```

Can become:

- huge
- memory-heavy
- slow

### Required Optimization

Use:

- ONNX export

OR:

- tree pruning

---

## PROBLEM 11 — Dataset Bias

If the dataset contains:

- only professional matches

Then:

- recommendations become terrible for public games.

### Better Dataset Mix

Use:

- pro matches
- immortal matches
- divine matches
- ranked pubs

This improves generalization.

---

## PROBLEM 12 — No Negative Draft Sampling

Very advanced issue.

Current training:

- only learns from played drafts.

The model does NOT understand:

- what was NOT picked.

### Future Improvement

Candidate pick evaluation.

### Example

```text
simulate adding hero
evaluate draft delta
```

This transforms the model from:

- passive prediction

into:

- active recommendation reasoning.

---

# Most Important Next Improvement

## HERO EMBEDDINGS

This is the next massive step.

### What It Enables

The model starts understanding:

- similar heroes
- hidden synergies
- latent draft structures

### Example Relationships

```text
Invoker ~ Storm Spirit
Axe ~ Centaur
```

In embedding space.

### Training Sources

Train embeddings using:

- co-picks
- win contexts
- lane pairings

---

# Next Huge Step

## Draft Simulation Engine

Instead of asking:

```text
"who wins?"
```

The system asks:

```text
"what happens if we add Mars?"
```

This transforms the assistant into:

- a real drafting AI.

---

# Recommended Next Priorities

## PRIORITY 1

Add:

- patch segmentation
- role inference
- lane features

---

## PRIORITY 2

Move toward:

- candidate-based recommendation training

---

## PRIORITY 3

Add:

- hero embeddings

---

## PRIORITY 4

Build:

- draft phase models

---

## PRIORITY 5

Add:

- personalization layer


