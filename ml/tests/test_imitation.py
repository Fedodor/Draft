"""Tests for the next-pick imitation experiment.

The tests that matter are about *time* and *legality*, not about accuracy:

* chronology: `order` defines the sequence, and a payload with duplicate or
  non-contiguous orders is rejected rather than repaired;
* visibility: a state contains exactly the actions before its pick — no future
  pick, future ban, or match result can reach the feature vector;
* candidate filtering: removed heroes are never legal, and a hero with no
  training observations stays legal (zero prior, not exclusion);
* baselines count training states only, and unsupported cells are reported;
* the metrics and the cluster bootstrap behave on hand-checkable inputs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ML_ROOT = Path(__file__).resolve().parents[1]
if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))

from training import imitation as im  # noqa: E402

# Canonical Captain's Mode shape: 7 bans, 2 picks, 3 bans, 6 picks, 4 bans, 2 picks.
BAN_SLOTS = (0, 1, 2, 3, 4, 5, 6, 9, 10, 11, 18, 19, 20, 21)
PICK_SLOTS = (7, 8, 12, 13, 14, 15, 16, 17, 22, 23)
RADIANT_SLOTS = (0, 2, 4, 6, 7, 9, 11, 13, 15, 17, 19, 21, 22)


def _payload(
    *,
    radiant_win: bool = True,
    hero_shift: int = 0,
    mutate_after: int | None = None,
    actions: int = 24,
) -> dict:
    """A synthetic STRATZ payload with a valid CM draft.

    Hero ids are `slot + 1` (so 1..24, all inside the roster); bans and picks are
    therefore unique. `mutate_after` changes every hero strictly after that slot,
    which is what the leakage test compares against.
    """
    entries = []
    for slot in range(actions):
        hero = slot + 1 + hero_shift
        if mutate_after is not None and slot > mutate_after:
            hero += 50
        entries.append(
            {
                "order": slot,
                "heroId": hero,
                "isPick": slot in PICK_SLOTS,
                "isRadiant": slot in RADIANT_SLOTS,
            }
        )
    return {"gameMode": "CAPTAINS_MODE", "pickBans": entries, "didRadiantWin": radiant_win}


def _states(payload: dict, partition: str = "train", match_id: int = 1):
    mode, actions, reason = im.parse_payload(payload)
    assert reason is None, reason
    assert mode == "CAPTAINS_MODE"
    assert im.draft_exclusion(actions) is None
    return im.states_for_match(match_id, partition, actions)


def test_parse_payload_sorts_by_order():
    payload = _payload()
    payload["pickBans"] = list(reversed(payload["pickBans"]))
    _, actions, reason = im.parse_payload(payload)
    assert reason is None
    assert [action.order for action in actions] == list(range(24))
    assert actions[7].is_pick is True


def test_parse_payload_rejects_non_dense_or_duplicate_orders():
    payload = _payload()
    payload["pickBans"][3]["order"] = 2
    _, _, reason = im.parse_payload(payload)
    assert reason == "non_dense_order"

    payload = _payload()
    payload["pickBans"][5]["order"] = 99
    _, _, reason = im.parse_payload(payload)
    assert reason == "non_dense_order"


def test_parse_payload_rejects_missing_side_or_type():
    payload = _payload()
    payload["pickBans"][0].pop("isPick")
    _, _, reason = im.parse_payload(payload)
    assert reason == "malformed_side_or_type"


def test_parse_payload_rejects_missing_stratz_mode():
    payload = _payload()
    payload.pop("gameMode")
    _, _, reason = im.parse_payload(payload)
    assert reason == "no_stratz_game_mode"


def test_state_before_first_pick_knows_only_earlier_bans():
    states = _states(_payload())
    first = states[0]
    assert first.pick_index == 0
    assert first.rad_pick == () and first.dire_pick == ()
    # Slots 0..6 are the first ban phase; the first pick is slot 7.
    expected_rad = tuple(slot + 1 for slot in BAN_SLOTS if slot < 7 and slot in RADIANT_SLOTS)
    expected_dire = tuple(slot + 1 for slot in BAN_SLOTS if slot < 7 and slot not in RADIANT_SLOTS)
    assert first.rad_ban == expected_rad
    assert first.dire_ban == expected_dire
    assert len(first.rad_ban) + len(first.dire_ban) == 7
    # No second-phase ban is visible yet.
    assert set(first.rad_ban) | set(first.dire_ban) == {1, 2, 3, 4, 5, 6, 7}


def test_second_state_contains_the_first_pick():
    states = _states(_payload())
    assert states[1].rad_pick == (8,) or states[1].dire_pick == (8,)
    assert states[1].pick_index == 1


def test_every_pick_produces_one_state_in_order():
    states = _states(_payload())
    assert len(states) == 10
    assert [state.pick_index for state in states] == list(range(10))
    assert [state.target for state in states] == [slot + 1 for slot in PICK_SLOTS]


def test_future_actions_do_not_change_earlier_states():
    """Changing every later pick, ban and the match result must not move state 0/1."""
    base = _states(_payload())
    # State 1's target is slot 8, so mutate strictly after slot 8: states 0 and 1
    # must be untouched while later states must change.
    mutated = _states(_payload(mutate_after=8, radiant_win=False))
    for index in (0, 1):
        assert base[index] == mutated[index]

    X_base = im.build_matrix(base)
    X_mutated = im.build_matrix(mutated)
    assert np.array_equal(X_base[0].toarray(), X_mutated[0].toarray())

    # And the later states do differ, proving the mutation is observable there.
    assert not np.array_equal(X_base[9].toarray(), X_mutated[9].toarray())


def test_draft_exclusion_rejects_short_or_duplicate_drafts():
    short = _payload(actions=22)
    _, actions, _ = im.parse_payload(short)
    assert im.draft_exclusion(actions) == "pick_count"

    duplicate = _payload()
    duplicate["pickBans"][23]["heroId"] = 1
    _, actions, _ = im.parse_payload(duplicate)
    assert im.draft_exclusion(actions) == "duplicate_hero"

    outside = _payload()
    outside["pickBans"][7]["heroId"] = 200
    _, actions, _ = im.parse_payload(outside)
    assert im.draft_exclusion(actions) == "pick_hero_outside_roster"


def test_legal_mask_excludes_removed_heroes_only():
    states = _states(_payload())
    legal = im.legal_matrix(states)
    first = states[0]
    removed = set(first.rad_ban) | set(first.dire_ban)
    assert legal.shape == (10, im.N_HEROES)
    for hero_id in removed:
        assert not legal[0, im.HERO_INDEX[hero_id]]
    # A hero never seen in the draft is still a legal candidate.
    unseen = max(removed) + 1
    assert legal[0, im.HERO_INDEX[unseen]]


def test_matrix_marks_the_removal_block_of_the_right_side():
    states = _states(_payload())
    X = im.build_matrix(states)
    assert X.shape == (10, im.N_FEATURES)
    first = states[0]
    hero = first.rad_ban[0]
    # Block 2 is "radiant ban"; block 3 is "dire ban".
    assert X[0, 2 * im.N_HEROES + im.HERO_INDEX[hero]] == 1.0
    assert X[0, 3 * im.N_HEROES + im.HERO_INDEX[hero]] == 0.0
    # A hero nobody removed is marked in no block at all.
    untouched = max(first.removed()) + 1
    assert X[0, : im.N_REMOVAL_FEATURES].toarray()[0, :].sum() == len(first.removed())


def test_matrix_carries_exactly_one_bucket_bias():
    """The bias block is what lets the model represent the stage/side baseline."""
    states = _states(_payload())
    X = im.build_matrix(states)
    bias = X[:, im.N_REMOVAL_FEATURES :].toarray()
    assert bias.shape == (len(states), im.N_BIAS_FEATURES)
    assert np.array_equal(bias.sum(axis=1), np.ones(len(states)))
    for row, state in enumerate(states):
        expected = im.PICK_STAGE_GROUP[state.pick_index] * 2 + (1 if state.acting_radiant else 0)
        assert bias[row, expected] == 1.0


def test_frequency_scores_count_training_states_only():
    train = _states(_payload())
    scores = im.frequency_scores(train, by_stage_and_side=False)
    assert scores["global"].sum() == len(train)
    # Every target hero in this synthetic draft appears exactly once per stage group
    # it is picked in (each hero is picked once overall), so counts are 0/1.
    assert scores["global"].max() == 1.0


def test_frequency_baseline_prefers_the_training_hero_and_stays_legal():
    train = _states(_payload())
    # A validation target never seen in training (120 is outside this draft).
    val = [im.State(1, "validation", 0, True, 120, (), (), (), ())]
    scores = im.frequency_scores(train, by_stage_and_side=False)
    probabilities, _ = im.frequency_probabilities(scores, val, by_stage_and_side=False)
    # Hero 120 has zero training evidence; a training hero has more mass.
    assert probabilities[0, im.HERO_INDEX[120]] < probabilities[0, im.HERO_INDEX[8]]
    assert probabilities[0].sum() == pytest.approx(1.0)
    # An illegal candidate gets exactly zero.
    illegal = im.State(1, "validation", 0, True, 120, (8,), (), (), ())
    probabilities, _ = im.frequency_probabilities(scores, [illegal], by_stage_and_side=False)
    assert probabilities[0, im.HERO_INDEX[8]] == 0.0


def test_unsupported_stage_side_cell_is_reported_not_invented():
    train = _states(_payload())  # all states are in stage groups 0..2, radiant or dire
    scores = im.frequency_scores(train, by_stage_and_side=True)
    # Ask for a cell training never saw by blanking the table.
    emptied = {key: value for key, value in scores.items() if key != (2, True)}
    val = [im.State(2, "validation", 8, True, 30, (), (), (), ())]
    probabilities, unsupported = im.frequency_probabilities(
        emptied, val, by_stage_and_side=True
    )
    assert (2, True) in unsupported
    # Uniform over legal candidates, not a frequency claim.
    legal_count = int(im.legal_matrix(val)[0].sum())
    assert probabilities[0].sum() == pytest.approx(1.0)
    assert np.allclose(probabilities[0][probabilities[0] > 0], 1.0 / legal_count)


def test_rank_and_logp_values():
    probabilities = np.zeros((1, im.N_HEROES))
    probabilities[0, 0] = 0.5
    probabilities[0, 1] = 0.3
    probabilities[0, 2] = 0.2
    rank, logp = im.rank_and_logp(probabilities, np.array([0]))
    assert rank[0] == 0 and logp[0] == pytest.approx(-np.log(0.5))
    rank, logp = im.rank_and_logp(probabilities, np.array([2]))
    assert rank[0] == 2


def test_rank_tie_break_prefers_lower_hero_id():
    probabilities = np.zeros((2, im.N_HEROES))
    probabilities[:, 10] = 0.3
    probabilities[:, 20] = 0.3
    probabilities[:, 30] = 0.4
    # Hero 30 (0.4) is ahead of both; between the tied pair the lower id wins.
    rank, _ = im.rank_and_logp(probabilities, np.array([20]))
    assert rank[0] == 2
    rank, _ = im.rank_and_logp(probabilities, np.array([10]))
    assert rank[0] == 1


def test_summarise_metrics():
    rank = np.array([0, 0, 1, 3])
    logp = np.array([0.1, 0.2, 0.3, 0.4])
    summary = im.summarise(rank, logp)
    assert summary["states"] == 4
    assert summary["top1"] == pytest.approx(0.5)
    assert summary["top3"] == pytest.approx(0.75)
    assert summary["mrr"] == pytest.approx((1 + 1 + 0.5 + 0.25) / 4)
    assert summary["log_loss"] == pytest.approx(0.25)


def test_per_match_means_groups_states_by_match():
    match_ids = np.array([1, 1, 2])
    values = np.array([1.0, 0.0, 2.0])
    means = im.per_match_means(match_ids, values)
    assert means.tolist() == [0.5, 2.0]


def test_cluster_bootstrap_rejects_unknown_comparison():
    per_match = {"model": np.array([0.5, 0.5]), "stage_side_frequency": np.array([0.4, 0.4])}
    with pytest.raises(ValueError):
        im.cluster_bootstrap(
            np.array([1, 2]),
            per_match,
            replicates=10,
            seed=0,
            comparisons=(("model", "missing_method"),),
        )


def test_cluster_bootstrap_reports_a_clear_difference():
    match_ids = np.array([1, 1, 2, 2, 3, 3, 4, 4])
    per_match = {
        "stage_side_frequency": np.array([0.4, 0.4, 0.4, 0.4]),
        "model": np.array([0.6, 0.6, 0.6, 0.6]),
    }
    result = im.cluster_bootstrap(
        match_ids,
        per_match,
        replicates=200,
        seed=0,
        comparisons=(("model", "stage_side_frequency"),),
    )
    delta = result["differences"]["model_minus_stage_side_frequency"]
    assert result["matches"] == 4
    assert delta["mean"] == pytest.approx(0.2)
    assert delta["ci95"] == pytest.approx([0.2, 0.2])
    assert delta["fraction_positive"] == 1.0


def test_cluster_bootstrap_intervals_bracket_the_point_estimate():
    rng = np.random.default_rng(1)
    per_match = {
        "stage_side_frequency": rng.uniform(0.0, 0.2, size=200),
        "model": rng.uniform(0.2, 0.4, size=200),
    }
    result = im.cluster_bootstrap(
        np.arange(200),
        per_match,
        replicates=300,
        seed=2,
        comparisons=(("model", "stage_side_frequency"),),
    )
    block = result["metrics"]["model"]["ci95"]
    assert block[0] < per_match["model"].mean() < block[1]
    delta = result["differences"]["model_minus_stage_side_frequency"]
    assert delta["ci95"][0] > 0.0


def test_states_for_match_raises_on_reused_target():
    _, actions, _ = im.parse_payload(_payload())
    duplicated = list(actions)
    duplicated[12] = im.Action(12, True, False, duplicated[7].hero)
    with pytest.raises(ValueError):
        im.states_for_match(1, "train", duplicated)


def test_partition_is_carried_on_every_state():
    train_states = _states(_payload(), partition="train", match_id=11)
    val_states = _states(_payload(hero_shift=60), partition="validation", match_id=12)
    assert {state.partition for state in train_states} == {"train"}
    assert {state.partition for state in val_states} == {"validation"}
    assert {state.match_id for state in train_states}.isdisjoint(
        {state.match_id for state in val_states}
    )
