# Smart Draft Assistant — Architecture & Development Plan

## Vision

Desktop AI assistant for Dota 2 draft phase.

The application:
- automatically detects live draft
- analyzes ally/enemy picks in real time
- predicts optimal hero picks
- adapts to player's personal hero pool
- uses ML + heuristics + matchup analytics
- gives explainable recommendations
- works as low-latency desktop overlay

The project goal:
Create a production-quality intelligent drafting system.

---

# Core Product Goals

## Main Objectives

1. Real-time draft recommendations
2. Extremely fast response time (<300ms)
3. Personalized recommendations
4. Explainable AI
5. Auto-sync with active Dota match
6. Meta-aware analysis
7. Long-term learning system

---

# Product Philosophy

This is NOT:
- another stat tracker
- simple counter picker
- static winrate app

This IS:
- contextual draft intelligence system

The app should understand:
- draft structure
- lane balance
- power spikes
- scaling
- teamfight composition
- comfort heroes
- player tendencies
- current patch meta

---

# High-Level Architecture

                ┌────────────────────┐
                │   Dota 2 Client    │
                └─────────┬──────────┘
                          │
              Game State Integration
                          │
                          ▼
                ┌────────────────────┐
                │ Live Match Service │
                └─────────┬──────────┘
                          │
          ┌───────────────┼────────────────┐
          ▼               ▼                ▼
 ┌────────────────┐ ┌──────────────┐ ┌───────────────┐
 │ Draft Engine   │ │ ML Engine    │ │ Meta Analyzer │
 └────────────────┘ └──────────────┘ └───────────────┘
          │               │                │
          └───────────────┼────────────────┘
                          ▼
                ┌────────────────────┐
                │ Recommendation API │
                └─────────┬──────────┘
                          ▼
                ┌────────────────────┐
                │ Desktop Overlay UI │
                └────────────────────┘

---

# Tech Stack

# Desktop Application

## Preferred Stack

### Frontend/UI
- React
- TypeScript
- Tailwind
- Zustand
- Framer Motion

### Desktop Shell
- Tauri

Why Tauri:
- lightweight
- low RAM usage
- better performance than Electron
- Rust backend support
- ideal for overlay tools

---

# Backend Layer

## Main Runtime
Rust

Why:
- speed
- low latency
- native desktop performance
- concurrency
- future ML inference support

---

# ML Layer

## Training
Python

Libraries:
- pandas
- scikit-learn
- XGBoost
- LightGBM
- PyTorch (later)

## Inference
ONNX Runtime or Rust inference

Reason:
Python inference in real-time overlay is risky for latency.

---

# Database

## Local
SQLite

Stores:
- hero data
- matchup data
- cached matches
- user profiles
- embeddings
- ML features

---

# External APIs

## OpenDota
Main public statistics.

## STRATZ
Advanced analytics and live data.

## Steam API
User integration.

---

# Core Modules

# 1. Live Match Integration

## Goal
Automatically detect:
- match start
- draft phase
- current picks
- bans
- player information

---

## Best Solution

### Dota Game State Integration

Avoid:
- OCR
- screenshots
- image recognition

Use:
- direct game telemetry

---

# Responsibilities

## Live Match Service
- parse game state JSON
- detect draft updates
- stream changes internally
- maintain current draft state

---

# 2. Hero Intelligence System

## Purpose

Central hero knowledge base.

---

## Hero Features

Each hero should contain:

```ts
HeroProfile {
  heroId: number
  name: string

  roles: string[]

  attributes: {
    initiation: number
    stun: number
    save: number
    push: number
    scaling: number
    mobility: number
    durability: number
    teamfight: number
    splitpush: number
    vision: number
  }

  laneStrength: {
    early: number
    mid: number
    late: number
  }

  matchupScores: {}
  synergyScores: {}

  metaScore: number
}

3. Draft Engine
Core Brain

Responsible for:

evaluating draft quality
scoring heroes
understanding composition
Draft Understanding

The engine should analyze:

Team Structure
frontline
initiation
crowd control
scaling
push
save
damage types
Lane Balance
offlane strength
support synergy
safe lane survival
Draft Tempo
early game
mid game
late game
Win Conditions
teamfight
split push
pickoff
snowball
Draft Scoring System
Final Hero Score
finalScore =
  matchupScore
+ synergyScore
+ comfortScore
+ metaScore
+ laneScore
+ teamNeedScore
+ confidenceScore
- riskPenalty
4. ML Recommendation Engine
Important

ML should NOT replace draft logic.

ML should ENHANCE draft logic.

ML Goals
Predict:
pick success probability
lane outcome
synergy efficiency
likely enemy picks
player comfort probability
Recommended ML Approach
Phase 1

Gradient boosting models.

Use:

XGBoost
LightGBM

Reason:

fast
interpretable
strong tabular performance
Training Data
Input Features
ally heroes
enemy heroes
patch
player MMR
hero pool
role
lane matchup
meta statistics
Output Labels
win/loss
lane dominance
KDA impact
player performance score
Future ML Systems
Neural Draft Embeddings

Learn:

hero relationships
hidden synergies
meta clusters
5. Recommendation Engine
Responsibilities

Generate:

Top 5 picks
safe picks
aggressive picks
comfort picks
anti-meta picks
Explainability Layer

Critical feature.

Every recommendation must explain:

Mars recommended because:
+ strong vs Sniper
+ combo with Phoenix
+ team lacks initiation
+ comfort hero
6. Overlay UI System
Goal

Minimal distraction.
Maximum clarity.

UI Principles
ultra fast readability
compact layout
keyboard navigable
no clutter
always-on-top
transparent background support
Main UI Sections
Current Draft
ALLY PICKS
ENEMY PICKS
BANS
Recommendations
TOP PICKS
SAFE PICKS
HIGH IMPACT PICKS
Team Analysis
Missing:
- stun
- frontline

Strong:
- teamfight
- late game
7. Meta Analyzer
Purpose

Track:

patch changes
hero popularity
emerging builds
hidden OP heroes
Meta Score Formula
metaScore =
winrate
+ pickrate trend
+ high MMR success
+ pro scene weight
8. User Personalization
Player Profile

Analyze:

favorite heroes
winrates
preferred tempo
role tendencies
aggressive/passive behavior
Comfort Score

Very important.

A weaker meta hero may still be optimal if:

player has 300 games
high historical performance