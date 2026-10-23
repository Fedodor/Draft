Development Phases
PHASE 1 — Foundation
Goals

Build stable architecture.

Tasks
Desktop App Setup
Tauri
React
TypeScript
Data Layer
SQLite
hero datasets
Draft State System
draft representation
state management
Manual Draft UI
select heroes manually
PHASE 2 — Core Draft Intelligence
Goals

Functional recommendation engine.

Tasks
Matchup Engine
Synergy Engine
Team Composition Analysis
Hero Scoring
Explainability System
PHASE 3 — Automatic Live Draft
Goals

Real-time automation.

Tasks
Game State Integration
Draft Detection
Auto-updating picks
Overlay Window
Low-latency event pipeline
PHASE 4 — ML Infrastructure
Goals

Smart recommendations.

Tasks
Dataset Collection Pipeline
Feature Engineering
Model Training
Offline Evaluation
Inference Optimization
PHASE 5 — Advanced Intelligence
Goals

Near-pro-level analysis.

Tasks
Enemy Pick Prediction
Lane Simulation
Draft Outcome Prediction
Meta Forecasting
Adaptive Recommendations
PHASE 6 — AI Coach Ecosystem
Future Expansion

After matches:

replay analysis
decision review
improvement suggestions
hero training recommendations
Performance Requirements
Critical

Draft phase is time-sensitive.

Target Metrics
Recommendation Speed

<300ms

UI Refresh

<16ms

Startup Time

<3 seconds

Memory Usage

<300MB

Important Engineering Decisions
Avoid
Electron

Too heavy.

Python runtime in overlay

Potential lag.

OCR-based detection

Too unreliable.

Prefer
Rust backend
Cached local data
Async architecture
Incremental updates
Precomputed matrices
Optimization Strategy
Precompute Everything

Before draft:

matchup matrices
synergy matrices
hero embeddings
lane strength tables

At runtime:
only lightweight scoring.

Suggested Repository Structure
smart-draft-assistant/

├── apps/
│   ├── desktop/
│   └── overlay/
│
├── services/
│   ├── draft-engine/
│   ├── ml-engine/
│   ├── live-match/
│   ├── meta-service/
│   └── recommendation-service/
│
├── packages/
│   ├── ui/
│   ├── hero-data/
│   ├── shared-types/
│   └── utils/
│
├── ml/
│   ├── training/
│   ├── datasets/
│   ├── feature-engineering/
│   └── models/
│
├── infrastructure/
│   ├── scripts/
│   ├── migrations/
│   └── seeders/
│
└── docs/
Long-Term Vision

Eventually this can evolve into:

draft intelligence platform
replay coaching system
esports analytics tool
team drafting assistant
ranked improvement platform
Recommended First Milestone

Build this first:

Working Desktop Draft Assistant

Features:

live draft sync
auto recommendations
hero scoring
explainable picks
comfort hero weighting

This alone is already a very strong project.

Final Recommendation

The most important architectural rule:

DO NOT build:
"an AI app"

Build:
"a fast draft engine with AI augmentation"

This distinction is critical.

The draft engine must always work even if ML fails.
ML improves confidence and prediction quality