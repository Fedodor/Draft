# Smart Draft Assistant - AI Agent Rules

## Project Context
You are helping build a production-quality desktop AI assistant for Dota 2 draft analysis.

The application:
- works as a desktop overlay companion
- analyzes live drafts in real time
- recommends optimal hero picks
- uses ML + heuristic systems
- prioritizes speed and explainability
- is designed for long-term extensibility

The stack:
- Tauri v2
- React + TypeScript + Vite
- Rust (backend)
- SQLite (local data)
- Python (ML training only)
- ONNX Runtime (inference)

---

# Core Philosophy

## IMPORTANT
Do NOT optimize for:
- quick hacks
- temporary solutions
- MVP shortcuts
- overengineering

Optimize for:
- maintainability
- low latency
- modular architecture
- scalability
- clean abstractions
- production-quality code

---

# Main Architectural Rule
The system must ALWAYS function without ML.
ML is an enhancement layer.
Core draft logic must remain:
- deterministic
- explainable
- debuggable

---

# Performance Requirements
Draft recommendations are time-sensitive.
Target:
- recommendation latency under 300ms
- minimal RAM usage (<300MB)
- zero UI freezes
- async architecture

Avoid:
- blocking operations
- unnecessary rerenders
- runtime-heavy calculations
- repeated API calls

Prefer:
- caching
- precomputed matrices
- incremental updates
- memoization

---

# Karpathy LLM Wiki Research Method
> **Source**: https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f

When researching for this project, use the **LLM Wiki pattern**:

## Core Idea
Instead of just retrieving from raw documents at query time, **incrementally build and maintain a persistent wiki** — a structured, interlinked collection of markdown files that sits between you and the raw sources.

When you add a new source (paper, article, API doc, match analysis):
1. **Read** the source completely
2. **Extract** key information
3. **Integrate** into existing wiki — updating entity pages, revising topic summaries, noting contradictions
4. **Cross-reference** with existing knowledge
5. **Log** the ingestion

The wiki is a **persistent, compounding artifact**. Cross-references are already there. Contradictions already flagged. Synthesis reflects everything you've read.

## Three Layers
1. **Raw sources** — immutable original documents (API responses, patch notes, research papers)
2. **The wiki** — LLM-generated markdown files (summaries, entity pages, concept pages, comparisons)
3. **The schema** — configuration (this file) telling the LLM how the wiki is structured

## Operations
### Ingest
Drop a new source → LLM reads → discusses key takeaways → writes summary page → updates index → updates entity/concept pages → appends to log

### Query
Ask questions against the wiki → LLM searches relevant pages → synthesizes answer with citations → **good answers get filed back into wiki**

### Lint
Periodically health-check: contradictions, stale claims, orphan pages, missing cross-references, data gaps

## Index & Log Files
- **index.md** — catalog of everything in wiki (page, summary, metadata)
- **log.md** — chronological record (ingests, queries, lint passes) with format: `## [YYYY-MM-DD] ingest | Source Title`

## Application to Smart Draft Assistant
Use this method for:
- **Patch analysis**: Ingest patch notes → update hero meta pages, synergies, counters
- **Pro match study**: Ingest match replays/analyses → update hero performance, draft patterns
- **Meta research**: Ingest tier lists, pro player guides → update hero semantics, role probabilities
- **ML experimentation**: Ingest training logs → update model performance wiki, feature importance

---

# Technology Rules

## Frontend
Use: React, TypeScript, Zustand, Tailwind, Framer Motion
Avoid: Redux unless absolutely needed, unnecessary abstractions, giant component files
Components: small, reusable, composable, strongly typed

## Backend (Rust)
Use: Rust for performance-critical systems
Design: modular, isolated, testable services
Avoid: monolithic services, hidden global state, tightly coupled systems

## ML Rules
Use ML ONLY when:
- heuristics are insufficient
- prediction uncertainty exists
- statistical learning provides measurable value

Avoid:
- neural networks too early
- black-box recommendations
- unexplainable outputs

Prefer: XGBoost, LightGBM, interpretable feature systems
Every ML prediction exposes: confidence score, contributing factors, fallback logic

---

# Draft Recommendation Rules
Recommendations must NEVER be:
- pure winrate-based
- purely meta-based
- random
- unexplained

Recommendations should consider:
- team composition
- lane balance
- hero synergy
- counters
- player comfort
- draft tempo
- scaling
- patch meta

Each recommendation includes:
- score
- explanation
- risk level
- role fit

---

# Explainability Rule
Every recommendation MUST answer: "Why is this hero suggested?"

Example:
```
Mars recommended because:
- strong against Sniper
- synergizes with Phoenix
- team lacks initiation
- comfort hero bonus
```

Never output: "Pick Mars" without explanation.

---

# UI/UX Rules
Overlay used during active drafting.
UI must be: fast to read, minimal, low distraction, compact, responsive
Avoid: clutter, excessive animations, large text blocks, unnecessary clicks
Prioritize: readability, keyboard navigation, instant feedback

---

# Code Quality Rules
Always:
- use TypeScript strict mode
- write explicit types
- separate business logic from UI
- prefer composition over inheritance

Functions:
- do one thing
- be testable
- avoid side effects

Avoid:
- giant utility files
- duplicated logic
- magic numbers
- deeply nested conditions

---

# State Management Rules
Global state only contains:
- active draft state
- app configuration
- user session data

Avoid:
- storing derived values globally
- duplicated state
- mutation-heavy patterns

Prefer:
- selectors
- computed values
- immutable updates

---

# Data Rules
External APIs: OpenDota, STRATZ, Steam API
Never rely entirely on external APIs at runtime.
Cache: hero data, matchup data, meta statistics
Degrade gracefully if APIs fail.

---

# Live Match Integration Rules
Prefer: Dota Game State Integration (GSI)
Avoid: OCR, screenshot parsing, fragile visual detection
Draft state pipeline: update incrementally, avoid full recomputation, debounce rapid updates

---

# Hero Intelligence Rules
Hero evaluations multi-dimensional.
Do NOT represent heroes with only: winrate, role, popularity
Each hero contains: strengths, weaknesses, timings, tempo profile, lane power, utility profile

---

# Draft Engine Rules
Heart of the application.
Must: work offline, remain deterministic, produce explainable scores, support simulation
Avoid: hardcoded hero exceptions, giant switch statements, fragile scoring hacks
Prefer: weighted systems, modular evaluators, configurable scoring

---

# Folder Structure
```
smart-draft-assistant/
├── apps/
│   ├── desktop/           # Tauri + React app
│   │   ├── src/           # React frontend
│   │   └── src-tauri/     # Rust backend
│   └── overlay/           # Future: standalone overlay
├── packages/
│   ├── shared-types/      # TypeScript/Rust shared types
│   ├── ui/                # Shared UI components
│   ├── hero-data/         # Hero database
│   └── utils/
├── ml/
│   ├── datasets/
│   │   ├── raw/           # Original API responses
│   │   ├── processed/     # Normalized features (parquet)
│   │   ├── features/      # Precomputed ML features
│   │   └── embeddings/    # Hero embeddings
│   ├── pipelines/
│   │   ├── fetch/         # Data collection
│   │   ├── clean/         # Normalization
│   │   ├── transform/     # Feature engineering
│   │   └── train/         # Model training
│   ├── models/            # Trained models (joblib, ONNX)
│   └── experiments/       # Experiment tracking
├── infrastructure/
│   ├── scripts/
│   ├── migrations/
│   └── seeders/
└── docs/
```

---

# Scalability Rules
Architecture supports future:
- replay coaching
- lane simulation
- enemy pick prediction
- AI coaching
- match analytics
- cloud sync

Avoid decisions that block future expansion.

---

# Communication Rules
When proposing implementations:
- explain tradeoffs
- prioritize performance
- prioritize maintainability
- avoid unnecessary complexity

When uncertain:
- prefer simpler architecture
- prefer deterministic systems
- explain assumptions clearly

---

# Anti-Patterns To Avoid
Never:
- tightly couple UI to business logic
- put ML directly into UI components
- perform heavy calculations during rendering
- fetch APIs repeatedly during draft
- block the main thread
- use massive singleton systems

---

# Final Principle
Build: **"a fast intelligent drafting engine"**
NOT: **"a flashy AI demo"**

Reliability, speed, and explainability are more important than complexity.