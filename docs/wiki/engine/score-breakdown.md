# Score Breakdown Panel

`get_score_breakdown` has existed since Step 3, but no UI called it. Users saw the final recommendation percentage and a few explanation lines, not the six weighted components behind it.

- Each recommendation card has a lazy **Score breakdown** disclosure; opening it calls `get_score_breakdown` for that hero and the selected side.
- Each row shows the component value, its weight, and the weighted contribution (`62% × 25% = 15.5 pts`), plus a value bar.
- The panel reconstructs the weighted total from the contributions.
- It refetches when the selected side changes and displays the typed command error on failure.
- Weights mirror `W_*` in `apps/desktop/src-tauri/src/engine/scorer.rs`: matchup 0.25, synergy 0.20, composition 0.15, meta 0.10, comfort 0.05, ML 0.25. The display copy lives in `apps/desktop/src/services/scoreBreakdown.ts`.
- A pure-helper test asserts the weights sum to one and that the contributions reconstruct the backend total.

Canonical source: [`../../STEP_10_SCORE_BREAKDOWN.md`](../../STEP_10_SCORE_BREAKDOWN.md). Related: [Scoring data flow](scoring.md), [Patch meta data](meta.md).

## Open follow-ups

- The weights are duplicated between Rust and TypeScript; a Rust weight change must update `scoreBreakdown.ts`.
- Component-level UI tests need a DOM/testing-library setup; only the pure helper is covered today.
