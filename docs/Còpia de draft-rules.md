---
trigger: always_on
---

# AI Agent Rules — Smart Draft Assistant Project

## Project Context

You are helping build a production-quality desktop AI assistant for
:contentReference[oaicite:0]{index=0} draft analysis.

The application:
- works as a desktop overlay companion
- analyzes live drafts in real time
- recommends optimal hero picks
- uses ML + heuristic systems
- prioritizes speed and explainability
- is designed for long-term extensibility

The stack:
- Tauri
- React
- TypeScript
- Rust
- SQLite
- Python (ML training only)

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

The system should target:
- recommendation latency under 300ms
- minimal RAM usage
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

# Technology Rules

# Frontend

Use:
- React
- TypeScript
- Zustand
- Tailwind

Avoid:
- Redux unless absolutely needed
- unnecessary abstractions
- giant component files

Components should:
- be small
- reusable
- composable
- strongly typed

---

# Backend

Use:
- Rust for performance-critical systems

Design backend services as:
- modular
- isolated
- testable

Avoid:
- monolithic services
- hidden global state
- tightly coupled systems

---

# ML Rules

Use ML ONLY when:
- heuristics are insufficient
- prediction uncertainty exists
- statistical learning provides measurable value

Avoid:
- using neural networks too early
- black-box recommendations
- unexplainable outputs

Prefer:
- XGBoost
- LightGBM
- interpretable feature systems

Every ML prediction should expose:
- confidence score
- contributing factors
- fallback logic

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

Each recommendation should include:
- score
- explanation
- risk level
- role fit

---

# Explainability Rule

Every recommendation MUST answer:
"Why is this hero suggested?"

Example:

Mars recommended because:
- strong against Sniper
- synergizes with Phoenix
- team lacks initiation
- comfort hero bonus

Never output:
"Pick Mars"
without explanation.

---

# UI/UX Rules

The overlay is used during active drafting.

UI must be:
- fast to read
- minimal
- low distraction
- compact
- responsive

Avoid:
- clutter
- excessive animations
- large text blocks
- unnecessary clicks

Prioritize:
- readability
- keyboard navigation
- instant feedback

---

# Code Quality Rules

Always:
- use TypeScript strict mode
- write explicit types
- separate business logic from UI
- prefer composition over inheritance

Functions should:
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

Global state should only contain:
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

External APIs:
- OpenDota
- STRATZ
- Steam API

Never rely entirely on external APIs at runtime.

Cache:
- hero data
- matchup data
- meta statistics

The application should degrade gracefully if APIs fail.

---

# Live Match Integration Rules

Prefer:
- Dota Game State Integration

Avoid:
- OCR
- screenshot parsing
- fragile visual detection

The draft state pipeline must:
- update incrementally
- avoid full recomputation
- debounce rapid updates

---

# Hero Intelligence Rules

Hero evaluations should be multi-dimensional.

Do NOT represent heroes with only:
- winrate
- role
- popularity

Each hero should contain:
- strengths
- weaknesses
- timings
- tempo profile
- lane power
- utility profile

---

# Draft Engine Rules

The draft engine is the heart of the application.

It must:
- work offline
- remain deterministic
- produce explainable scores
- support simulation

Avoid:
- hardcoded hero exceptions
- giant switch statements
- fragile scoring hacks

Prefer:
- weighted systems
- modular evaluators
- configurable scoring

---

# Recommended Folder Structure

```text
src/

├── core/
│   ├── draft-engine/
│   ├── recommendation-engine/
│   ├── hero-system/
│   └── scoring/
│
├── features/
│   ├── overlay/
│   ├── draft/
│   ├── recommendations/
│   └── settings/
│
├── services/
│   ├── opendota/
│   ├── stratz/
│   ├── live-match/
│   └── cache/
│
├── ml/
│   ├── inference/
│   ├── features/
│   └── models/
│
├── shared/
│   ├── types/
│   ├── utils/
│   └── constants/
│
└── ui/
Scalability Rules

The architecture should support future features:

replay coaching
lane simulation
enemy pick prediction
AI coaching
match analytics
cloud sync

Avoid decisions that block future expansion.

Communication Rules

When proposing implementations:

explain tradeoffs
prioritize performance
prioritize maintainability
avoid unnecessary complexity

When uncertain:

prefer simpler architecture
prefer deterministic systems
explain assumptions clearly
Anti-Patterns To Avoid

Never:

tightly couple UI to business logic
put ML directly into UI components
perform heavy calculations during rendering
fetch APIs repeatedly during draft
block the main thread
use massive singleton systems
Final Principle

Build:
"a fast intelligent drafting engine"

NOT:
"a flashy AI demo"

Reliability, speed, and explainability are more important than complexity.