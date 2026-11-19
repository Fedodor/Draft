"""Tests for the phase-aware ranking presentation spike.

These cover the properties the mockup depends on: the routing boundary, ranking
and tie-breaking, churn measurement on a synthetic draft, and — most importantly —
that the generated document makes no claim the data cannot support.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pytest

ML_ROOT = Path(__file__).resolve().parents[1]
if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))

from training import hybrid_presentation as hp  # noqa: E402
from training import imitation as im  # noqa: E402

pytest.importorskip("sklearn")


def test_routing_boundary_is_frequency_then_contextual():
    assert hp.routing_source(0) == "observed-frequency"
    assert hp.routing_source(1) == "observed-frequency"
    assert hp.routing_source(2) == "contextual-model"
    assert hp.routing_source(9) == "contextual-model"
    assert "frequenc" in hp.routing_label(0).lower()
    assert "contextual" in hp.routing_label(2).lower()


def test_hybrid_probabilities_route_exactly_by_pick_index():
    baseline = np.full((10, im.N_HEROES), 0.01)
    pooled = np.full((10, im.N_HEROES), 0.02)
    pick_of = np.arange(10)
    hybrid = hp.hybrid_probabilities(baseline, pooled, pick_of)
    assert np.allclose(hybrid[0], 0.01) and np.allclose(hybrid[1], 0.01)
    assert np.allclose(hybrid[2], 0.02) and np.allclose(hybrid[9], 0.02)
    # The inputs are not mutated.
    assert np.allclose(baseline[5], 0.01)


def test_top_k_breaks_ties_toward_the_lower_hero():
    scores = np.zeros((1, 5))
    scores[0, 3] = 0.5
    # Heroes 0,1,2,4 tie at 0.0 -> ascending order, then the 0.5 hero.
    assert hp.top_k(scores, 3)[0].tolist() == [3, 0, 1]


def test_survivor_ranks_ignore_removed_heroes():
    scores = np.zeros(4)
    scores[3] = 1.0  # hero id 4
    survivors = np.array([True, True, False, True])  # hero id 3 is removed
    rank = hp.survivor_ranks(scores, survivors, im.ROSTER[:4])
    assert rank.tolist() == [1, 2, 0]  # hero 4 leads; heroes 1 and 2 tie


def test_kendall_tau_extremes_and_guard():
    assert hp.kendall_tau(np.array([0, 1, 2]), np.array([0, 1, 2])) == pytest.approx(1.0)
    assert hp.kendall_tau(np.array([0, 1, 2]), np.array([2, 1, 0])) == pytest.approx(-1.0)
    assert hp.kendall_tau(np.array([0]), np.array([0])) == 1.0
    with pytest.raises(ValueError):
        hp.kendall_tau(np.array([0, 1]), np.array([0]))


def test_top1_survived_detects_leadership_change():
    previous = np.array([0.9, 0.2, 0.1, 0.05])
    current = np.array([0.9, 0.2, 0.1, 0.05])
    survivors = np.array([True, True, True, True])
    heroes = im.ROSTER[:4]
    assert hp.top1_survived(previous, current, survivors, heroes) is True

    # The previous leader is removed; the second hero takes over.
    survivors = np.array([False, True, True, True])
    assert hp.top1_survived(previous, current, survivors, heroes) is True

    # Among survivors (heroes 2, 3, 4) the lead moves from hero 2 to hero 3.
    reordered = np.array([0.1, 0.2, 0.95, 0.05])
    assert hp.top1_survived(previous, reordered, survivors, heroes) is False


def test_transition_metrics_flags_the_removed_leader():
    n_matches = 2
    probs = np.full((n_matches, 10, im.N_HEROES), 0.1 / (im.N_HEROES - 1))
    probs[:, :, 0] = 0.9
    legal = np.ones((n_matches, 10, im.N_HEROES), dtype=bool)
    legal[:, 1:, 0] = False  # hero 0 is banned before pick 1

    transitions = hp.transition_metrics(
        probs.reshape(-1, im.N_HEROES),
        legal.reshape(-1, im.N_HEROES),
        n_matches,
        tau_matches=np.array([0, 1]),
        hero_ids=im.ROSTER,
    )
    first = transitions[0]
    assert first["top1_removed_share"] == 1.0
    assert first["removed_mean"] == 1.0
    assert first["top1_retained_share"] == 1.0
    assert first["top5_overlap_share"] == 1.0
    assert first["kendall_tau_mean"] == pytest.approx(1.0)
    assert len(transitions) == 9


def test_sample_match_ids_is_deterministic_and_uniform():
    ids = list(range(1000, 2000))
    first = hp.sample_match_ids(ids, 6, 11)
    assert first == hp.sample_match_ids(ids, 6, 11)
    assert len(first) == 6 and len(set(first)) == 6
    assert first == sorted(first)
    assert set(first) <= set(ids)


# --------------------------------------------------------------------------- #
# The document itself
# --------------------------------------------------------------------------- #
def _snapshot(draft_index: int = 1, pick_index: int = 2, previous_head: str | None = None,
              previous_legal: bool = True) -> dict:
    candidates = [
        {"rank": position + 1, "hero_id": 10 + position, "hero": f"Hero {position + 1}",
         "attr": "agi", "tier": position + 1}
        for position in range(5)
    ]
    return {
        "draft_index": draft_index,
        "match_id": 42,
        "pick_index": pick_index,
        "acting_side": "Radiant",
        "source": hp.routing_source(pick_index),
        "source_label": hp.routing_label(pick_index),
        "legal_candidates": 120,
        "candidates": candidates,
        "previous_head": previous_head,
        "previous_head_still_legal": previous_legal,
        "observed_target": "Hero 2",
        "timeline": [
            {"kind": "ban", "side": "Radiant", "label": "ban", "hero": "Hero A", "hero_id": 1},
            {"kind": "pick", "side": "Dire", "label": "pick", "hero": "Hero B", "hero_id": 2},
        ],
        "head_is_observed": True,
    }


def _render(snapshots: list[dict]) -> str:
    return hp.render_mockup(
        snapshots,
        stability={
            "top1_removed_share": "44%",
            "top1_retained_share": "52%",
            "top5_overlap_share": "61%",
            "model_vs_baseline_top1": "37%",
        },
        coverage={"matches": 7108, "states": 71080},
    )


def test_mockup_makes_no_forbidden_claim():
    document = _render([_snapshot(), _snapshot(pick_index=0)])
    # Collapse the source line breaks so prose assertions are not fooled by wrapping.
    flattened = re.sub(r"\s+", " ", document.lower())
    for phrase in hp.FORBIDDEN_LABELS:
        assert phrase not in flattened, phrase
    assert "not a prediction of who wins" in flattened
    assert "not shipped" in flattened


def test_mockup_shortlist_is_ordinal_only():
    document = _render([_snapshot()])
    block = re.search(r'<ol class="shortlist">(.*?)</ol>', document, re.S)
    assert block is not None
    inner = block.group(1)
    assert inner.count('<li class="cand">') == 5
    assert "%" not in inner
    assert re.search(r"\d+\.\d", inner) is None


def test_mockup_labels_the_source_and_warns_about_a_removed_leader():
    document = _render([_snapshot(pick_index=0)])
    assert "Observed pick frequencies (first two picks)" in document
    contextual = _render([_snapshot(pick_index=4, previous_head="Hero 1", previous_legal=False)])
    assert "Contextual model (from the third pick on)" in contextual
    assert "no longer available" in contextual


def test_mockup_renders_one_section_per_snapshot():
    document = _render([_snapshot(draft_index=1), _snapshot(draft_index=2, pick_index=5)])
    assert document.count('<section class="draft">') == 2
    assert "Example draft 1" in document and "Example draft 2" in document
    # Default snapshot pick_index is 2, so the headings read "pick 3" and "pick 6".
    assert "pick 3 of 10" in document and "pick 6 of 10" in document


def test_hero_names_use_player_facing_labels():
    heroes = hp.load_heroes(ML_ROOT / "datasets" / "heroes.json")
    assert heroes[11]["name"] == "Shadow Fiend"
    assert heroes[1]["name"] == "Anti-Mage"
    assert heroes[20]["name"] == "Vengeful Spirit"
    assert heroes[53]["name"] == "Nature's Prophet"
    assert heroes[7]["name"] == "Earthshaker"
    assert hp.hero_name(heroes, 9999) == "Hero 9999"
