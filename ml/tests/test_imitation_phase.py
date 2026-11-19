"""Tests for the phase-aware imitation experiment.

The load-bearing properties are:

* the decision rule is written before any validation metric and cannot change
  afterwards;
* the early-stage matrix contains only blocks that can exist at that stage, and
  only heroes that were already removed;
* routing replaces exactly the rows of the stage it is meant to replace, and the
  result stays a legal, normalised distribution;
* stage masks partition the ten picks without overlap or gaps.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

ML_ROOT = Path(__file__).resolve().parents[1]
if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))

from training import imitation as im  # noqa: E402
from training import imitation_phase as phase  # noqa: E402

pytest.importorskip("sklearn")


def _state(
    *,
    pick_index: int = 0,
    radiant: bool = True,
    target: int = 30,
    rad_pick: tuple = (),
    dire_pick: tuple = (),
    rad_ban: tuple = (1, 2, 3, 4),
    dire_ban: tuple = (5, 6, 7),
) -> im.State:
    return im.State(
        match_id=1,
        partition="validation",
        pick_index=pick_index,
        acting_radiant=radiant,
        target=target,
        rad_pick=rad_pick,
        dire_pick=dire_pick,
        rad_ban=rad_ban,
        dire_ban=dire_ban,
    )


def test_early_widths_reflect_available_blocks():
    assert phase.early_width(include_picks=False) == 2 * im.N_HEROES + 2 == 280
    assert phase.early_width(include_picks=True) == 4 * im.N_HEROES + 2 == 558


def test_pick0_matrix_has_no_pick_columns_and_a_side_bias():
    state = _state()
    X = phase.build_early_matrix([state], include_picks=False)
    assert X.shape == (1, 280)
    row = X.toarray()[0]
    assert row.sum() == 1 + len(state.rad_ban) + len(state.dire_ban)
    # Bans land in their block.
    assert row[im.HERO_INDEX[1]] == 1.0
    assert row[im.N_HEROES + im.HERO_INDEX[5]] == 1.0
    # Acting-side bias: radiant is column width-1.
    assert row[278] == 0.0 and row[279] == 1.0


def test_shared_matrix_marks_the_known_pick():
    state = _state(pick_index=1, radiant=False, rad_pick=(9,), target=30)
    X = phase.build_early_matrix([state], include_picks=True)
    row = X.toarray()[0]
    assert row[im.HERO_INDEX[9]] == 1.0  # radiant pick block comes first
    assert row[556] == 1.0 and row[557] == 0.0  # dire acts


def test_early_matrix_ignores_future_actions():
    """A pick-0 state must be identical whatever happens later."""
    a = _state()
    b = _state(rad_pick=(11, 12), dire_pick=(13, 14), target=31)
    assert np.array_equal(
        phase.build_early_matrix([a], include_picks=False).toarray(),
        phase.build_early_matrix([b], include_picks=False).toarray(),
    )


def test_route_replaces_only_selected_rows():
    base = np.full((4, im.N_HEROES), 0.01)
    replacement = np.full((4, im.N_HEROES), 0.02)
    mask = np.array([True, False, True, False])
    out = phase.route(base, replacement, mask)
    assert np.allclose(out[0], 0.02) and np.allclose(out[2], 0.02)
    assert np.allclose(out[1], 0.01) and np.allclose(out[3], 0.01)
    # The input is not mutated.
    assert np.allclose(base[0], 0.01)


def test_stage_masks_partition_the_ten_picks():
    pick_of = np.arange(10)
    masks = phase.stage_masks(pick_of)
    covered = np.zeros(10, dtype=int)
    for name in ("pick0", "pick1", "picks27", "picks89"):
        covered += masks[name].astype(int)
    # The exclusive stages tile the ten picks exactly once; the combined views
    # (picks01, all) overlap them on purpose.
    assert covered.tolist() == [1] * 10
    assert masks["all"].all()
    assert masks["picks01"].tolist() == [True, True] + [False] * 8
    assert masks["picks27"].sum() == 6
    assert masks["picks89"].sum() == 2


def test_evaluate_stage_uses_only_its_rows():
    probabilities = np.zeros((4, im.N_HEROES))
    probabilities[:, 5] = 1.0
    truth = np.array([5, 5, 7, 7])
    mask = np.array([True, True, False, False])
    result = phase.evaluate_stage(probabilities, truth, mask, np.array([1, 2, 3, 4]))
    assert result["states"] == 2
    assert result["matches"] == 2
    assert result["top1"] == 1.0
    # Two rows of match 1/2 -> both counted separately at match level.
    assert result["log_loss"] == pytest.approx(0.0)


def test_aligned_probabilities_masks_and_renormalises():
    class Stub:
        classes_ = np.array([0, 1])

        def predict_proba(self, X):  # noqa: N803 - sklearn signature
            return np.array([[0.7, 0.3], [0.2, 0.8]])

    legal = np.zeros((2, im.N_HEROES), dtype=bool)
    legal[:, 0] = True
    legal[:, 1] = True
    legal[0, 1] = False  # class 1 illegal for row 0
    probabilities = phase.aligned_probabilities(Stub(), sparse.csr_matrix((2, 2)), legal)
    assert probabilities[0, 0] == pytest.approx(1.0)
    assert probabilities[0, 1] == 0.0
    assert probabilities[1, 0] == pytest.approx(0.2)
    assert probabilities[1].sum() == pytest.approx(1.0)


def test_pre_registration_is_written_once_and_immutable(tmp_path):
    rule = phase.pre_register(tmp_path)
    path = tmp_path / "decision_rule.json"
    assert path.is_file()
    stored = json.loads(path.read_text(encoding="utf-8"))
    assert stored["rule"] == phase.DECISION_RULE
    assert stored["declared_before_validation_metrics"] is True
    assert stored["declared_at"] == rule["declared_at"]

    # A second call returns the same record.
    again = phase.pre_register(tmp_path)
    assert again["declared_at"] == rule["declared_at"]

    # Tampering after the fact is refused rather than silently accepted.
    stored["rule"] = "something easier"
    path.write_text(json.dumps(stored), encoding="utf-8")
    with pytest.raises(SystemExit):
        phase.pre_register(tmp_path)


def test_decision_rule_states_the_required_conditions():
    text = phase.DECISION_RULE
    assert "excludes zero" in text
    assert "top-1" in text
    assert "intercept-only" in text
    assert "frequency fallback" in text
    assert "never establishes win-rate" in text
