# Step 10: Score Breakdown Panel

- Date: 2026-10-27
- Status: Implemented
- Depends on: Step 3 (scorer wiring and `get_score_breakdown`), Step 7 (UI state), Step 9 (meta component)
- Wiki summary: [`wiki/engine/score-breakdown.md`](wiki/engine/score-breakdown.md)

## Problem

The engine has exposed `get_score_breakdown` since Step 3, and every
recommendation carries a weighted `score`, but the UI only showed the final
percentage and a few explanation lines. Users could not see which components
produced the score, how much each component was weighted, or whether a pick was
driven by matchup, synergy, composition, meta, comfort, or ML. The documented
follow-up ("add a debug panel; the IPC command already exists") was still open.

## Outcome

Each recommendation card now has a lazy **Score breakdown** disclosure. Opening
it calls `get_score_breakdown` for that hero and the currently selected side and
renders:

- one row per component with its value, weight, and weighted contribution
  (`62% × 25% = 15.5 pts`);
- a value bar for each component;
- the reconstructed weighted total.

The panel refetches when the selected side changes and shows the typed command
error instead of failing silently.

## Scoring contract

The recommendation total is the weighted sum of six components:

| Component | Key | Weight |
| --- | --- | --- |
| Matchup | `matchup` | 0.25 |
| Synergy | `synergy` | 0.20 |
| Composition | `composition` | 0.15 |
| Meta | `meta` | 0.10 |
| Comfort | `comfort` | 0.05 |
| ML | `ml` | 0.25 |

All component values are in `[0, 1]` with `0.5` neutral. The weights mirror the
`W_*` constants in `apps/desktop/src-tauri/src/engine/scorer.rs`; the frontend
keeps the display copy in `apps/desktop/src/services/scoreBreakdown.ts`.

## Files

- `apps/desktop/src/services/scoreBreakdown.ts` — component labels/weights,
  `breakdownRows`, and `breakdownTotal`.
- `apps/desktop/src/services/scoreBreakdown.test.ts` — weight-sum, row, and
  total-reconstruction tests.
- `apps/desktop/src/components/RecommendationCard.tsx` — disclosure, lazy IPC
  fetch, loading/error states, contribution display.

## Acceptance

- Every recommendation card can reveal the six components without leaving the
  panel.
- Each row shows the component value, weight, and weighted contribution.
- The displayed total equals the sum of the contributions.
- The panel uses the selected side and recovers from command errors.
- Existing recommendations, categories, and explanations are unchanged.

## Verification

```powershell
cd apps\desktop
npm run lint
npm run test
npm run build
```

Full gate: `powershell -NoProfile -ExecutionPolicy Bypass -File scripts\verify.ps1`.

## Residual risks

- The weights are duplicated between Rust and TypeScript; a backend change must
  update `scoreBreakdown.ts`. A test asserts the display weights sum to one, but
  it cannot detect a changed Rust constant.
- Component tests are limited to the pure helper because the frontend test
  environment has no DOM/testing-library setup.
- The breakdown is fetched on demand per card, so opening many cards issues one
  IPC call each; the command is a cheap in-memory lookup.
