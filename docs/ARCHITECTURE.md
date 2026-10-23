# Project Architecture: Smart Draft Assistant

The Smart Draft Assistant is a production-quality desktop overlay designed to provide real-time draft analysis for Dota 2. It leverages a hybrid approach combining deterministic heuristics with machine learning enhancements.

## Technology Stack

- **Framework**: [Tauri](https://tauri.app/) (v2)
- **Frontend**: React + TypeScript + Vite
- **Backend**: Rust (Performance-critical logic)
- **State Management**: Zustand (Frontend), Arc<RwLock<T>> (Backend)
- **Styling**: Tailwind CSS
- **Data Integration**: Dota 2 Game State Integration (GSI)

## System Overview

```mermaid
graph TD
    Dota2[Dota 2 Game] -- GSI (HTTP/JSON) --> GSI_Server[GSI Server (Rust/Warp)]
    GSI_Server -- Updates --> DraftEngine[Draft Engine (Rust)]
    DraftEngine -- Recommendations --> TauriRPC[Tauri Commands]
    TauriRPC -- Invoke --> ReactApp[React Frontend]
    ReactApp -- Display --> Overlay[Desktop Overlay]
    
    ML_Model[ML Inference (Python/Rust)] -.-> DraftEngine
```

### Core Components

1.  **Draft Engine (Rust)**:
    - Located in `apps/desktop/src-tauri/src/engine`.
    - Responsible for maintaining the current draft state and generating hero recommendations.
    - Uses a modular `Scorer` system to evaluate heroes based on matchups, synergy, and team composition.
    - Reads hero data from the SQLite database (126 heroes with full stats and semantic attributes).

2.  **GSI Server (Rust)**:
    - Located in `apps/desktop/src-tauri/src/gsi`.
    - Listens for HTTP POST requests from the Dota 2 client.
    - Decodes the game state and notifies the `DraftEngine` of changes (picks, bans, game phase).

3.  **React Frontend**:
    - Located in `apps/desktop/src`.
    - A minimal, high-performance overlay UI.
    - Communicates with the Rust backend via Tauri's IPC (`invoke`) and events (`listen`).

4.  **Shared Types**:
    - Located in `packages/shared-types`.
    - Ensures type safety across the TypeScript frontend and (via Serde) the Rust backend.

5.  **Database Layer (SQLite)**:
    - Located in `apps/desktop/src-tauri/src/database.rs` (public module).
    - Stores the full hero roster (126 heroes) with base stats, roles, attack types, and image paths.
    - Stores hero semantic attributes (initiation, stun, save, lane strength, tempo profiles, etc.).
    - Stores hero matchup and synergy data.
    - Seeded via `infrastructure/seeders/seed_heroes` from JSON files.
    - Supports `PrimaryAttribute::Str`, `Agi`, `Int`, and `All` (Universal heroes).

## Data Flow

1.  **Input**: Dota 2 sends a JSON payload to the GSI server whenever the game state changes.
2.  **Processing**: The GSI server parses the payload and updates the `DraftEngine`'s state.
3.  **Analysis**: The `DraftEngine` runs its scoring algorithms on the updated state.
4.  **Output**: The frontend receives a `gsi-update` event and fetches the latest recommendations from the engine.
