//! Payload -> engine lifecycle tests for GSI draft ingestion.
//!
//! These tests drive [`apply_gsi_payload`], which is the exact function the
//! warp handler calls for every POST. Only the Tauri event emission and the
//! HTTP framing are skipped; draft state is mutated through the real
//! `reconcile_draft` / `DraftEngine` path.
//!
//! The payloads are hand-written fixtures shaped like real GSI traffic
//! (`map.game_state`, `player.team_name`, `draft.team2`/`team3` with
//! `pickN_id`/`banN_id` slots). They are **not** captured from a live client.
//! Assumptions about GSI behaviour are called out where they matter, and the
//! limitations that remain are documented in `docs/GSI.md` and
//! `docs/AUDIT_2026-10-27.md` (P0-2).

use super::*;
use crate::models::{DraftAction, DraftBan, DraftPick, DraftState, HeroId, PlayerId, TeamSide};
use serde_json::{json, Value};

const INIT: &str = "DOTA_GAMERULES_STATE_INIT";
const IN_GAME: &str = "DOTA_GAMERULES_STATE_GAME_IN_PROGRESS";
const POST_GAME: &str = "DOTA_GAMERULES_STATE_POST_GAME";

/// Heroes 1..=HERO_COUNT exist in the fixture hero database. Anything outside
/// that range (including the empty slot id `0`) is unknown to the engine.
const HERO_COUNT: i32 = 40;

// ---------------------------------------------------------------------------
// Fixture
// ---------------------------------------------------------------------------

struct Harness {
    engine: Arc<std::sync::Mutex<DraftEngine>>,
    server: GsiServerState,
}

impl Harness {
    fn new() -> Self {
        let mut engine = DraftEngine::new();
        engine.hero_db = (1..=HERO_COUNT)
            .map(|id| {
                let mut hero = DraftEngine::fallback_heroes().remove(0);
                hero.id = HeroId(id);
                hero.cm_enabled = true;
                hero
            })
            .collect();
        engine.state.write().available_heroes =
            engine.hero_db.iter().map(|hero| hero.id.clone()).collect();
        Self {
            engine: Arc::new(std::sync::Mutex::new(engine)),
            server: GsiServerState::default(),
        }
    }

    /// Feed one payload through the same core the HTTP handler uses.
    fn send(&mut self, game_state: &str, draft: Option<Value>) -> GsiOutcome {
        self.send_payload(payload(game_state, draft, None), None)
    }

    fn send_payload(&mut self, payload: Value, expected_token: Option<&str>) -> GsiOutcome {
        apply_gsi_payload(&self.engine, &self.server, &payload, expected_token).outcome
    }

    fn state(&self) -> DraftState {
        self.engine.lock().unwrap().get_state()
    }

    fn engine(&self) -> std::sync::MutexGuard<'_, DraftEngine> {
        self.engine.lock().unwrap()
    }
}

/// Start a fresh match: a non-draft state followed by the first hero-selection
/// update, which resets the engine and is deliberately not ingested.
fn start_match(harness: &mut Harness) {
    assert_eq!(harness.send(INIT, None), GsiOutcome::Ignored);
    assert_eq!(
        harness.send(HERO_SELECTION, Some(empty_draft())),
        GsiOutcome::DraftReset
    );
}

fn payload(game_state: &str, draft: Option<Value>, token: Option<&str>) -> Value {
    let mut value = json!({
        "provider": { "name": "Dota 2", "appid": 570 },
        "map": { "game_state": game_state, "game_time": 0.0, "clock_time": 0.0 },
        "player": { "team_name": "radiant" },
    });
    if let Some(token) = token {
        value["auth"] = json!({ "token": token });
    }
    if let Some(draft) = draft {
        value["draft"] = draft;
    }
    value
}

/// One team's slot block, mirroring the fixed `pickN_id`/`banN_id` GSI fields.
fn slots(team_id: Option<i32>, picks: &[i32], bans: &[i32]) -> Value {
    let mut object = serde_json::Map::new();
    if let Some(team_id) = team_id {
        object.insert("team_id".to_string(), json!(team_id));
    }
    for (slot, hero_id) in picks.iter().enumerate() {
        object.insert(format!("pick{slot}_id"), json!(hero_id));
    }
    for (slot, hero_id) in bans.iter().enumerate() {
        object.insert(format!("ban{slot}_id"), json!(hero_id));
    }
    Value::Object(object)
}

/// `active` is the optional `(activeteam, pick)` pair GSI sends while a team is
/// on the clock. Omitting it exercises the CM-order fallback.
fn draft(active: Option<(i32, bool)>, team2: Value, team3: Value) -> Value {
    let mut object = serde_json::Map::new();
    if let Some((team, is_pick)) = active {
        object.insert("activeteam".to_string(), json!(team));
        object.insert("pick".to_string(), json!(is_pick));
    }
    object.insert("team2".to_string(), team2);
    object.insert("team3".to_string(), team3);
    Value::Object(object)
}

fn empty_draft() -> Value {
    draft(None, slots(None, &[], &[]), slots(None, &[], &[]))
}

fn pick_heroes(picks: &[DraftPick]) -> Vec<i32> {
    picks.iter().map(|pick| pick.hero_id.0).collect()
}

fn ban_heroes(bans: &[DraftBan]) -> Vec<i32> {
    bans.iter().map(|ban| ban.hero_id.0).collect()
}

fn state_json(state: &DraftState) -> Value {
    serde_json::to_value(state).expect("draft state must serialize")
}

// ---------------------------------------------------------------------------
// Invariants (Phase 2)
// ---------------------------------------------------------------------------

/// Invariants that must hold after *every* payload, whatever the input.
fn assert_invariants(state: &DraftState) {
    let mut seen = std::collections::HashSet::new();
    let mut accepted = std::collections::HashSet::new();

    for pick in state.radiant_picks.iter().chain(state.dire_picks.iter()) {
        assert!(
            (1..=HERO_COUNT).contains(&pick.hero_id.0),
            "accepted pick {} is not in the hero database",
            pick.hero_id.0
        );
        assert!(
            seen.insert(pick.hero_id.0),
            "hero {} was assigned twice",
            pick.hero_id.0
        );
        assert!(pick.order >= 1, "pick order must be >= 1");
        accepted.insert(pick.hero_id.0);
    }

    for ban in &state.bans {
        assert!(
            (1..=HERO_COUNT).contains(&ban.hero_id.0),
            "accepted ban {} is not in the hero database",
            ban.hero_id.0
        );
        assert!(
            seen.insert(ban.hero_id.0),
            "hero {} is both picked and banned (or banned twice)",
            ban.hero_id.0
        );
        assert!(ban.order >= 1, "ban order must be >= 1");
        accepted.insert(ban.hero_id.0);
    }

    assert!(
        state.radiant_picks.len() <= 5,
        "Radiant has more than 5 picks"
    );
    assert!(state.dire_picks.len() <= 5, "Dire has more than 5 picks");
    assert!(
        state
            .bans
            .iter()
            .filter(|ban| ban.team == TeamSide::Radiant)
            .count()
            <= 5,
        "Radiant has more than 5 bans"
    );
    assert!(
        state
            .bans
            .iter()
            .filter(|ban| ban.team == TeamSide::Dire)
            .count()
            <= 5,
        "Dire has more than 5 bans"
    );

    let expected_available: Vec<i32> = (1..=HERO_COUNT)
        .filter(|id| !accepted.contains(id))
        .collect();
    assert_eq!(
        state
            .available_heroes
            .iter()
            .map(|hero_id| hero_id.0)
            .collect::<Vec<_>>(),
        expected_available,
        "available heroes must be the hero pool minus accepted actions"
    );

    assert_eq!(
        state.pick_number as usize,
        state.radiant_picks.len() + state.dire_picks.len(),
        "pick_number must equal the number of accepted picks"
    );
}

// ---------------------------------------------------------------------------
// A. Consecutive matches
// ---------------------------------------------------------------------------

#[test]
fn lifecycle_consecutive_matches_do_not_leak_state() {
    let mut harness = Harness::new();

    // 1. App startup, no draft yet.
    assert_eq!(harness.send(INIT, None), GsiOutcome::Ignored);
    assert_invariants(&harness.state());

    // 2. Entering hero selection resets. The triggering payload is assumed to
    //    still describe the previous game (the reason the reset exists), so its
    //    contents must NOT be ingested.
    let stale_first_game = draft(
        None,
        slots(Some(2), &[1], &[11]),
        slots(Some(3), &[2], &[12]),
    );
    assert_eq!(
        harness.send(HERO_SELECTION, Some(stale_first_game)),
        GsiOutcome::DraftReset
    );
    let state = harness.state();
    assert!(state.radiant_picks.is_empty());
    assert!(state.dire_picks.is_empty());
    assert!(state.bans.is_empty());
    assert_eq!(state.available_heroes.len(), HERO_COUNT as usize);
    assert_eq!((state.current_pick, state.is_picking_phase), (None, false));
    assert_invariants(&state);

    // 3. Match 1 actions arrive as growing full snapshots.
    let first_draft = draft(
        Some((2, false)),
        slots(Some(2), &[1], &[11, 13]),
        slots(Some(3), &[2], &[12]),
    );
    assert_eq!(
        harness.send(HERO_SELECTION, Some(first_draft)),
        GsiOutcome::DraftApplied { picks: 2, bans: 3 }
    );
    let first_draft = draft(
        Some((3, true)),
        slots(Some(2), &[1, 3], &[11, 13]),
        slots(Some(3), &[2, 4], &[12]),
    );
    assert_eq!(
        harness.send(HERO_SELECTION, Some(first_draft)),
        GsiOutcome::DraftApplied { picks: 2, bans: 0 }
    );

    let state = harness.state();
    assert_eq!(pick_heroes(&state.radiant_picks), vec![1, 3]);
    assert_eq!(pick_heroes(&state.dire_picks), vec![2, 4]);
    assert_eq!(ban_heroes(&state.bans), vec![11, 13, 12]);
    assert_eq!(
        (state.current_pick, state.is_picking_phase),
        (Some(TeamSide::Dire), true)
    );
    assert_invariants(&state);

    // 4. The game leaves the draft. A post-game payload that still contains the
    //    finished draft must not mutate or duplicate anything.
    let finished = draft(
        None,
        slots(Some(2), &[1, 3], &[11, 13]),
        slots(Some(3), &[2, 4], &[12]),
    );
    let before = state_json(&harness.state());
    assert_eq!(
        harness.send(POST_GAME, Some(finished.clone())),
        GsiOutcome::Ignored
    );
    assert_eq!(
        harness.send(IN_GAME, Some(finished.clone())),
        GsiOutcome::Ignored
    );
    assert_eq!(state_json(&harness.state()), before);

    // 5. Match 2 starts. The first hero-selection payload is the reset trigger
    //    and still carries match 1's slots; it must be skipped, not re-ingested.
    assert_eq!(
        harness.send(HERO_SELECTION, Some(finished)),
        GsiOutcome::DraftReset
    );
    let state = harness.state();
    assert!(
        state.radiant_picks.is_empty(),
        "match 2 inherited a Radiant pick"
    );
    assert!(state.dire_picks.is_empty(), "match 2 inherited a Dire pick");
    assert!(state.bans.is_empty(), "match 2 inherited a ban");
    assert_eq!(state.pick_number, 0);
    assert_eq!(
        state.available_heroes.len(),
        HERO_COUNT as usize,
        "the hero pool was not restored"
    );
    assert_invariants(&state);

    // 6. New actions with different heroes are accepted normally. Team
    //    attribution and the turn state reflect the new payload, not match 1.
    let second_draft = draft(
        Some((2, false)),
        slots(Some(2), &[21, 23], &[31]),
        slots(Some(3), &[22, 24], &[32]),
    );
    assert_eq!(
        harness.send(HERO_SELECTION, Some(second_draft.clone())),
        GsiOutcome::DraftApplied { picks: 4, bans: 2 }
    );
    let state = harness.state();
    assert_eq!(pick_heroes(&state.radiant_picks), vec![21, 23]);
    assert_eq!(pick_heroes(&state.dire_picks), vec![22, 24]);
    assert_eq!(ban_heroes(&state.bans), vec![31, 32]);
    assert_eq!(
        (state.current_pick, state.is_picking_phase),
        (Some(TeamSide::Radiant), false)
    );
    assert_invariants(&state);

    // Match 1's heroes are back in the pool; match 2's are not.
    for hero_id in [1, 2, 3, 4, 11, 12, 13] {
        assert!(
            state.available_heroes.iter().any(|hero| hero.0 == hero_id),
            "hero {hero_id} from match 1 should be available in match 2"
        );
    }
    for hero_id in [21, 22, 23, 24, 31, 32] {
        assert!(
            !state.available_heroes.iter().any(|hero| hero.0 == hero_id),
            "hero {hero_id} from match 2 should not be available"
        );
    }

    // 7. Replaying the identical match-2 snapshot is a no-op.
    assert_eq!(
        harness.send(HERO_SELECTION, Some(second_draft)),
        GsiOutcome::DraftApplied { picks: 0, bans: 0 }
    );
    assert_eq!(pick_heroes(&harness.state().radiant_picks), vec![21, 23]);
    assert_invariants(&harness.state());
}

#[test]
fn lifecycle_strategy_time_after_draft_keeps_state() {
    let mut harness = Harness::new();
    start_match(&mut harness);

    let drafted = draft(
        None,
        slots(None, &[1, 2, 3, 4, 5], &[11, 12, 13, 14, 15]),
        slots(None, &[6, 7, 8, 9, 10], &[16, 17, 18, 19, 20]),
    );
    assert_eq!(
        harness.send(HERO_SELECTION, Some(drafted.clone())),
        GsiOutcome::DraftApplied {
            picks: 10,
            bans: 10
        }
    );
    let before = state_json(&harness.state());

    // Strategy and pre-game are still part of the same draft session.
    assert_eq!(
        harness.send(STRATEGY_TIME, Some(drafted.clone())),
        GsiOutcome::DraftApplied { picks: 0, bans: 0 }
    );
    assert_eq!(
        harness.send(PRE_GAME, Some(drafted)),
        GsiOutcome::DraftApplied { picks: 0, bans: 0 }
    );
    assert_eq!(state_json(&harness.state()), before);
    assert_invariants(&harness.state());
}

// ---------------------------------------------------------------------------
// B. Duplicate and invalid actions
// ---------------------------------------------------------------------------

#[test]
fn lifecycle_duplicate_and_conflicting_actions_are_deduped() {
    let mut harness = Harness::new();
    start_match(&mut harness);

    // The same hero twice in Radiant's slots, and also as a Dire ban.
    // Picks are reconciled before bans, so the pick wins and the rest is
    // skipped. This precedence is a deliberate contract, not GSI ordering.
    let duplicate = draft(None, slots(None, &[5, 5], &[]), slots(None, &[], &[5]));
    assert_eq!(
        harness.send(HERO_SELECTION, Some(duplicate.clone())),
        GsiOutcome::DraftApplied { picks: 1, bans: 0 }
    );
    let state = harness.state();
    assert_eq!(pick_heroes(&state.radiant_picks), vec![5]);
    assert!(state.bans.is_empty());
    assert_invariants(&state);

    // Replay is still a no-op, not a second copy.
    assert_eq!(
        harness.send(HERO_SELECTION, Some(duplicate)),
        GsiOutcome::DraftApplied { picks: 0, bans: 0 }
    );
    assert_eq!(pick_heroes(&harness.state().radiant_picks), vec![5]);
    assert_invariants(&harness.state());
}

#[test]
fn lifecycle_unknown_heroes_are_ignored_without_dropping_valid_actions() {
    let mut harness = Harness::new();
    start_match(&mut harness);

    // 999 / 1000 do not exist in the fixture pool; 0 is an empty slot and is
    // dropped while reading the slots. Valid hero 7 must still be accepted.
    let snapshot = draft(
        None,
        slots(None, &[999, 7], &[0]),
        slots(None, &[], &[1000]),
    );
    assert_eq!(
        harness.send(HERO_SELECTION, Some(snapshot)),
        GsiOutcome::DraftApplied { picks: 1, bans: 0 }
    );
    let state = harness.state();
    assert_eq!(pick_heroes(&state.radiant_picks), vec![7]);
    assert!(state.bans.is_empty());
    assert_invariants(&state);
}

#[test]
fn lifecycle_stale_snapshots_never_remove_or_duplicate_actions() {
    let mut harness = Harness::new();
    start_match(&mut harness);

    let older = draft(None, slots(None, &[1], &[11]), slots(None, &[], &[]));
    assert_eq!(
        harness.send(HERO_SELECTION, Some(older.clone())),
        GsiOutcome::DraftApplied { picks: 1, bans: 1 }
    );

    let newer = draft(None, slots(None, &[1, 2], &[11]), slots(None, &[], &[12]));
    assert_eq!(
        harness.send(HERO_SELECTION, Some(newer)),
        GsiOutcome::DraftApplied { picks: 1, bans: 1 }
    );
    let before = state_json(&harness.state());

    // An older full snapshot arriving late must not remove the newer actions
    // and must not re-add the ones already present. Ingestion is append-only
    // by design: it can never express "un-pick" from a stale payload.
    assert_eq!(
        harness.send(HERO_SELECTION, Some(older)),
        GsiOutcome::DraftApplied { picks: 0, bans: 0 }
    );
    assert_eq!(state_json(&harness.state()), before);
    assert_eq!(pick_heroes(&harness.state().radiant_picks), vec![1, 2]);
    assert_eq!(ban_heroes(&harness.state().bans), vec![11, 12]);
    assert_invariants(&harness.state());
}

#[test]
fn lifecycle_malformed_payloads_are_rejected_without_mutating_state() {
    let mut harness = Harness::new();
    start_match(&mut harness);
    assert_eq!(
        harness.send(
            HERO_SELECTION,
            Some(draft(None, slots(None, &[1], &[]), slots(None, &[], &[])))
        ),
        GsiOutcome::DraftApplied { picks: 1, bans: 0 }
    );
    let before = state_json(&harness.state());

    // Missing `map` entirely.
    assert_eq!(
        harness.send_payload(json!({ "player": { "team_name": "radiant" } }), None),
        GsiOutcome::ParseError
    );
    // `draft` has the wrong shape (a scalar instead of an object).
    assert_eq!(
        harness.send_payload(
            json!({
                "map": { "game_state": HERO_SELECTION, "game_time": 0.0, "clock_time": 0.0 },
                "draft": 5,
            }),
            None
        ),
        GsiOutcome::ParseError
    );
    // A slot contains a non-numeric hero id.
    assert_eq!(
        harness.send_payload(
            json!({
                "map": { "game_state": HERO_SELECTION, "game_time": 0.0, "clock_time": 0.0 },
                "draft": { "team2": { "pick0_id": "five" } },
            }),
            None
        ),
        GsiOutcome::ParseError
    );

    assert_eq!(state_json(&harness.state()), before);
    assert_invariants(&harness.state());
}

#[test]
fn lifecycle_auth_is_enforced_before_any_state_change() {
    let mut harness = Harness::new();

    // A rejected payload must not even advance the lifecycle tracker.
    let wrong = payload(HERO_SELECTION, Some(empty_draft()), Some("wrong"));
    assert_eq!(
        harness.send_payload(wrong, Some("secret")),
        GsiOutcome::Unauthorized
    );
    assert_eq!(harness.state().available_heroes.len(), HERO_COUNT as usize);

    // Authenticated traffic behaves normally: init, then hero selection resets.
    let init = payload(INIT, None, Some("secret"));
    assert_eq!(
        harness.send_payload(init, Some("secret")),
        GsiOutcome::Ignored
    );

    // A correctly authenticated hero-selection payload still resets normally.
    let first = payload(HERO_SELECTION, Some(empty_draft()), Some("secret"));
    assert_eq!(
        harness.send_payload(first, Some("secret")),
        GsiOutcome::DraftReset
    );

    let draft = payload(
        HERO_SELECTION,
        Some(draft(None, slots(None, &[1], &[]), slots(None, &[], &[]))),
        Some("secret"),
    );
    assert_eq!(
        harness.send_payload(draft, Some("secret")),
        GsiOutcome::DraftApplied { picks: 1, bans: 0 }
    );
    assert_eq!(pick_heroes(&harness.state().radiant_picks), vec![1]);
    assert_invariants(&harness.state());
}

// ---------------------------------------------------------------------------
// C. Manual undo
// ---------------------------------------------------------------------------

#[test]
fn lifecycle_manual_undo_is_not_reverted_by_repeated_snapshots() {
    let mut harness = Harness::new();
    start_match(&mut harness);

    let picked = draft(None, slots(None, &[5], &[]), slots(None, &[], &[]));
    assert_eq!(
        harness.send(HERO_SELECTION, Some(picked.clone())),
        GsiOutcome::DraftApplied { picks: 1, bans: 0 }
    );

    // What the `undo_last_action` command does: remove the pick, return the
    // hero to the pool, and remember the manual correction.
    let undone = harness
        .engine()
        .undo_last_action()
        .expect("the GSI pick should be undoable");
    match undone {
        DraftAction::Pick { pick } => assert_eq!(pick.hero_id.0, 5),
        other => panic!("expected the pick to be undone, got {other:?}"),
    }
    assert!(harness.engine().is_suppressed(&HeroId(5)));
    let state = harness.state();
    assert!(state.radiant_picks.is_empty());
    assert_eq!(state.pick_number, 0);
    assert!(state.available_heroes.iter().any(|hero| hero.0 == 5));
    assert_invariants(&state);

    // Dota keeps resending the snapshot that still contains hero 5. It must not
    // resurrect the undone pick while suppression is active.
    for attempt in 0..3 {
        assert_eq!(
            harness.send(HERO_SELECTION, Some(picked.clone())),
            GsiOutcome::DraftApplied { picks: 0, bans: 0 },
            "repeat {attempt} re-added the undone pick"
        );
        let state = harness.state();
        assert!(state.radiant_picks.is_empty());
        assert_eq!(state.pick_number, 0);
        assert!(state.available_heroes.iter().any(|hero| hero.0 == 5));
        assert_invariants(&state);
    }

    // A genuinely new action is accepted even though hero 5 is suppressed.
    let with_new_action = draft(None, slots(None, &[5, 6], &[]), slots(None, &[], &[]));
    assert_eq!(
        harness.send(HERO_SELECTION, Some(with_new_action)),
        GsiOutcome::DraftApplied { picks: 1, bans: 0 }
    );
    let state = harness.state();
    assert_eq!(
        pick_heroes(&state.radiant_picks),
        vec![6],
        "only the new pick should be present"
    );
    assert_eq!(state.pick_number, 1);
    assert_invariants(&state);

    // DOCUMENTED LIMITATION: suppression is keyed by hero id for the whole
    // match, so it cannot tell "the same stale action again" apart from "the
    // user really wants that hero now on the same side". Both are ignored
    // until the next reset. Changing this needs a payload-level action
    // identity that GSI does not provide (slots are renumbered).
}

#[test]
fn lifecycle_undo_suppression_clears_on_the_next_match() {
    let mut harness = Harness::new();
    start_match(&mut harness);

    let picked = draft(None, slots(None, &[5], &[]), slots(None, &[], &[]));
    harness.send(HERO_SELECTION, Some(picked.clone()));
    harness.engine().undo_last_action().expect("pick to undo");
    assert!(harness.engine().is_suppressed(&HeroId(5)));

    // Post-game does not clear suppression: the match is still the same.
    assert_eq!(
        harness.send(POST_GAME, Some(picked.clone())),
        GsiOutcome::Ignored
    );
    assert!(harness.engine().is_suppressed(&HeroId(5)));

    // The next hero selection is a new match: reset clears suppression and
    // restores the full pool.
    assert_eq!(
        harness.send(HERO_SELECTION, Some(picked.clone())),
        GsiOutcome::DraftReset
    );
    assert!(!harness.engine().is_suppressed(&HeroId(5)));
    let state = harness.state();
    assert_eq!(state.available_heroes.len(), HERO_COUNT as usize);
    assert_invariants(&state);

    // The same hero can now be picked normally in match 2.
    assert_eq!(
        harness.send(HERO_SELECTION, Some(picked)),
        GsiOutcome::DraftApplied { picks: 1, bans: 0 }
    );
    assert_eq!(pick_heroes(&harness.state().radiant_picks), vec![5]);
    assert_invariants(&harness.state());
}

#[test]
fn lifecycle_explicit_manual_pick_lifts_undo_suppression() {
    let mut harness = Harness::new();
    start_match(&mut harness);

    let picked = draft(None, slots(None, &[5], &[]), slots(None, &[], &[]));
    harness.send(HERO_SELECTION, Some(picked.clone()));
    harness.engine().undo_last_action().expect("pick to undo");
    assert!(harness.engine().is_suppressed(&HeroId(5)));

    // The manual_pick command path: validate -> lift suppression -> add.
    {
        let engine = harness.engine();
        let pick = DraftPick {
            hero_id: HeroId(5),
            player_id: PlayerId(0),
            order: 1,
            team: TeamSide::Radiant,
        };
        engine
            .validate_pick(&pick)
            .expect("the undone hero must be available again");
        engine.lift_suppression(&pick.hero_id);
        engine.add_pick(pick);
    }
    assert!(!harness.engine().is_suppressed(&HeroId(5)));
    assert_eq!(pick_heroes(&harness.state().radiant_picks), vec![5]);

    // A later GSI snapshot that still contains the hero must not double-add it.
    assert_eq!(
        harness.send(HERO_SELECTION, Some(picked)),
        GsiOutcome::DraftApplied { picks: 0, bans: 0 }
    );
    assert_eq!(pick_heroes(&harness.state().radiant_picks), vec![5]);
    assert_invariants(&harness.state());
}

// ---------------------------------------------------------------------------
// D. Team attribution and draft order
// ---------------------------------------------------------------------------

#[test]
fn lifecycle_team_attribution_survives_repeated_snapshots() {
    let mut harness = Harness::new();
    start_match(&mut harness);

    // `team_id` wins over slot position: team2 is Dire, team3 is Radiant.
    let swapped = draft(
        Some((3, true)),
        slots(Some(3), &[5], &[]),
        slots(Some(2), &[6], &[]),
    );
    assert_eq!(
        harness.send(HERO_SELECTION, Some(swapped.clone())),
        GsiOutcome::DraftApplied { picks: 2, bans: 0 }
    );

    for _ in 0..3 {
        assert_eq!(
            harness.send(HERO_SELECTION, Some(swapped.clone())),
            GsiOutcome::DraftApplied { picks: 0, bans: 0 }
        );
        let state = harness.state();
        assert_eq!(pick_heroes(&state.dire_picks), vec![5]);
        assert_eq!(pick_heroes(&state.radiant_picks), vec![6]);
        assert_eq!(
            (state.current_pick, state.is_picking_phase),
            (Some(TeamSide::Dire), true)
        );
        assert_invariants(&state);
    }
}

#[test]
fn lifecycle_team_attribution_falls_back_to_slots() {
    let mut harness = Harness::new();
    start_match(&mut harness);

    // No `team_id` anywhere: team2 is Radiant, team3 is Dire.
    let snapshot = draft(None, slots(None, &[7], &[9]), slots(None, &[8], &[10]));
    assert_eq!(
        harness.send(HERO_SELECTION, Some(snapshot)),
        GsiOutcome::DraftApplied { picks: 2, bans: 2 }
    );
    let state = harness.state();
    assert_eq!(pick_heroes(&state.radiant_picks), vec![7]);
    assert_eq!(pick_heroes(&state.dire_picks), vec![8]);
    let radiant_bans: Vec<i32> = state
        .bans
        .iter()
        .filter(|ban| ban.team == TeamSide::Radiant)
        .map(|ban| ban.hero_id.0)
        .collect();
    let dire_bans: Vec<i32> = state
        .bans
        .iter()
        .filter(|ban| ban.team == TeamSide::Dire)
        .map(|ban| ban.hero_id.0)
        .collect();
    assert_eq!(radiant_bans, vec![9]);
    assert_eq!(dire_bans, vec![10]);
    assert_invariants(&state);
}

#[test]
fn lifecycle_unrecognized_team_ids_fall_back_to_the_slot() {
    let mut harness = Harness::new();
    start_match(&mut harness);

    let snapshot = draft(None, slots(Some(7), &[7], &[]), slots(Some(99), &[8], &[]));
    assert_eq!(
        harness.send(HERO_SELECTION, Some(snapshot)),
        GsiOutcome::DraftApplied { picks: 2, bans: 0 }
    );
    let state = harness.state();
    assert_eq!(pick_heroes(&state.radiant_picks), vec![7]);
    assert_eq!(pick_heroes(&state.dire_picks), vec![8]);
    assert_invariants(&state);
}

#[test]
fn lifecycle_duplicate_team_ids_cannot_overfill_a_side() {
    let mut harness = Harness::new();
    start_match(&mut harness);

    // Both slots claim to be Radiant. The engine accepts at most five picks per
    // side, so the second block is rejected rather than corrupting the draft.
    let snapshot = draft(
        None,
        slots(Some(2), &[1, 2, 3, 4, 5], &[]),
        slots(Some(2), &[6, 7, 8, 9, 10], &[]),
    );
    assert_eq!(
        harness.send(HERO_SELECTION, Some(snapshot)),
        GsiOutcome::DraftApplied { picks: 5, bans: 0 }
    );
    let state = harness.state();
    assert_eq!(pick_heroes(&state.radiant_picks), vec![1, 2, 3, 4, 5]);
    assert!(state.dire_picks.is_empty());
    assert_invariants(&state);
}

#[test]
fn lifecycle_bans_are_capped_per_team() {
    let mut harness = Harness::new();
    start_match(&mut harness);

    let full_ban_phase = draft(
        None,
        slots(None, &[], &[11, 12, 13, 14, 15]),
        slots(None, &[], &[21, 22, 23, 24, 25]),
    );
    assert_eq!(
        harness.send(HERO_SELECTION, Some(full_ban_phase)),
        GsiOutcome::DraftApplied { picks: 0, bans: 10 }
    );

    // A later snapshot renumbers both sides' ban slots and introduces heroes
    // that were never banned. Captain's Mode allows five bans per side, so the
    // extra bans are rejected instead of being appended to the draft.
    let sixth_ban = draft(
        None,
        slots(None, &[], &[11, 12, 13, 14, 16]),
        slots(None, &[], &[21, 22, 23, 24, 26]),
    );
    assert_eq!(
        harness.send(HERO_SELECTION, Some(sixth_ban)),
        GsiOutcome::DraftApplied { picks: 0, bans: 0 }
    );

    let state = harness.state();
    assert_eq!(
        ban_heroes(&state.bans),
        vec![11, 12, 13, 14, 15, 21, 22, 23, 24, 25]
    );
    // Rejected bans stay in the pool.
    assert!(state.available_heroes.iter().any(|hero| hero.0 == 16));
    assert!(state.available_heroes.iter().any(|hero| hero.0 == 26));
    assert_invariants(&state);
}

#[test]
fn lifecycle_cm_order_fallback_walks_the_full_draft() {
    let mut harness = Harness::new();
    start_match(&mut harness);

    // No `activeteam`/`pick` on any payload: the engine must follow CM_ORDER
    // for turn state. Existing unit tests pin CM_ORDER itself; this test pins
    // the wiring from accepted-action counts to turn state.
    let mut radiant_picks: Vec<i32> = Vec::new();
    let mut dire_picks: Vec<i32> = Vec::new();
    let mut radiant_bans: Vec<i32> = Vec::new();
    let mut dire_bans: Vec<i32> = Vec::new();

    for (index, (side, is_pick)) in CM_ORDER.iter().copied().enumerate() {
        let hero_id = index as i32 + 1;
        if is_pick {
            match side {
                TeamSide::Radiant => radiant_picks.push(hero_id),
                TeamSide::Dire => dire_picks.push(hero_id),
            }
        } else {
            match side {
                TeamSide::Radiant => radiant_bans.push(hero_id),
                TeamSide::Dire => dire_bans.push(hero_id),
            }
        }

        let snapshot = draft(
            None,
            slots(None, &radiant_picks, &radiant_bans),
            slots(None, &dire_picks, &dire_bans),
        );
        let expected = if is_pick {
            GsiOutcome::DraftApplied { picks: 1, bans: 0 }
        } else {
            GsiOutcome::DraftApplied { picks: 0, bans: 1 }
        };
        assert_eq!(
            harness.send(HERO_SELECTION, Some(snapshot)),
            expected,
            "action {index} was not applied exactly once"
        );

        let state = harness.state();
        assert_eq!(
            pick_heroes(&state.radiant_picks),
            radiant_picks,
            "Radiant attribution broke at action {index}"
        );
        assert_eq!(
            pick_heroes(&state.dire_picks),
            dire_picks,
            "Dire attribution broke at action {index}"
        );
        assert_invariants(&state);

        match CM_ORDER.get(index + 1) {
            Some((next_side, next_is_pick)) => assert_eq!(
                (state.current_pick, state.is_picking_phase),
                (Some(*next_side), *next_is_pick),
                "turn after action {index} should come from CM_ORDER"
            ),
            None => {
                assert_eq!(state.current_pick, None);
                assert!(!state.is_picking_phase);
            }
        }
    }

    assert_eq!(radiant_picks.len(), 5);
    assert_eq!(dire_picks.len(), 5);
    assert_eq!(radiant_bans.len(), 5);
    assert_eq!(dire_bans.len(), 5);
}

#[test]
fn lifecycle_ambiguous_turn_payloads_are_handled_safely() {
    let mut harness = Harness::new();
    start_match(&mut harness);

    // `activeteam` without `pick` is ambiguous; the documented behaviour is to
    // ignore both and fall back to the CM order (action 0 = Radiant ban).
    let mut ambiguous = draft(None, slots(None, &[], &[]), slots(None, &[], &[]));
    ambiguous["activeteam"] = json!(3);
    assert_eq!(
        harness.send(HERO_SELECTION, Some(ambiguous)),
        GsiOutcome::DraftApplied { picks: 0, bans: 0 }
    );
    let state = harness.state();
    assert_eq!(
        (state.current_pick, state.is_picking_phase),
        (Some(TeamSide::Radiant), false),
        "missing `pick` must fall back to CM order"
    );

    // An unknown `activeteam` value also falls back instead of panicking.
    let mut unknown = draft(None, slots(None, &[], &[]), slots(None, &[], &[]));
    unknown["activeteam"] = json!(42);
    unknown["pick"] = json!(true);
    harness.send(HERO_SELECTION, Some(unknown));

    // A hero-selection payload without any `draft` component is a no-op.
    assert_eq!(
        harness.send(HERO_SELECTION, None),
        GsiOutcome::DraftApplied { picks: 0, bans: 0 }
    );
    assert_invariants(&harness.state());
}

// ---------------------------------------------------------------------------
// Property-style coverage
// ---------------------------------------------------------------------------

fn next_u32(seed: &mut u64) -> u32 {
    *seed = seed
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    (*seed >> 33) as u32
}

fn random_hero(seed: &mut u64) -> i32 {
    // Includes 0 (empty slot), valid ids, and ids outside the hero database.
    (next_u32(seed) % (HERO_COUNT as u32 + 4)) as i32
}

fn random_team_id(seed: &mut u64) -> Option<i32> {
    match next_u32(seed) % 5 {
        0 => None,
        1 => Some(2),
        2 => Some(3),
        3 => Some(7),
        _ => Some(0),
    }
}

#[test]
fn lifecycle_invariants_hold_for_adversarial_payload_sequences() {
    let mut harness = Harness::new();
    let mut seed: u64 = 0x5EED_1234_ABCD_0001;
    let phases = [
        INIT,
        HERO_SELECTION,
        HERO_SELECTION,
        STRATEGY_TIME,
        PRE_GAME,
        IN_GAME,
        POST_GAME,
    ];

    for step in 0..300 {
        let phase = phases[(next_u32(&mut seed) as usize) % phases.len()];

        let build = |seed: &mut u64| {
            let picks: Vec<i32> = (0..(next_u32(seed) % 6))
                .map(|_| random_hero(seed))
                .collect();
            let bans: Vec<i32> = (0..(next_u32(seed) % 6))
                .map(|_| random_hero(seed))
                .collect();
            slots(random_team_id(seed), &picks, &bans)
        };
        let team2 = build(&mut seed);
        let team3 = build(&mut seed);
        let active = if next_u32(&mut seed).is_multiple_of(2) {
            Some((
                2 + (next_u32(&mut seed) % 2) as i32,
                next_u32(&mut seed).is_multiple_of(2),
            ))
        } else {
            None
        };
        let snapshot = draft(active, team2, team3);

        let outcome = harness.send(phase, Some(snapshot.clone()));
        assert_invariants(&harness.state());

        // A snapshot that was ingested must be idempotent: replaying the exact
        // same bytes adds nothing and leaves the state (including timestamps)
        // unchanged. A reset trigger is exempt because the first replay is the
        // payload that repopulates the new draft by design.
        if matches!(outcome, GsiOutcome::DraftApplied { .. }) {
            let before = state_json(&harness.state());
            let replay = harness.send(phase, Some(snapshot));
            assert_eq!(
                replay,
                GsiOutcome::DraftApplied { picks: 0, bans: 0 },
                "step {step}: replaying an ingested snapshot added actions"
            );
            assert_eq!(
                state_json(&harness.state()),
                before,
                "step {step}: replaying an ingested snapshot changed state"
            );
        }
    }
}
