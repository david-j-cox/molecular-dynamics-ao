"""Cue competition: blocking and overshadowing require shared (competitive) prediction error.

These lock in the engine mechanism demonstrated by experiments/exp039: the Rescorla-Wagner rule with
``credit_assignment="rw_competitive"`` produces Kamin blocking and overshadowing, and
``rw_independent`` does not.
"""

from __future__ import annotations

from behavioral_md.atoms import default_atom_set
from behavioral_md.config import SimulationConfig
from behavioral_md.learning import EligibilityTrace, make_learning_rule

A, B = "light", "cue"     # two conditioned stimuli; the US is the appetitive signal
AF = 0                    # approach_food drive atom (valence +1)


def _intens(a: float, b: float) -> dict[str, float]:
    return {A: a, B: b, "food": 0.0, "danger": 0.0}


def _train(scheme: str, phases: list[tuple[int, dict[str, float]]]) -> tuple[float, float]:
    cfg = SimulationConfig(credit_assignment=scheme, learning_rate=0.3, reinforcement_asymptote=1.0)
    atoms = default_atom_set()
    rule = make_learning_rule(cfg)
    elig = EligibilityTrace(len(atoms), 0.95)
    for trials, intens in phases:
        for _ in range(trials):
            elig.trace[:] = 0.0
            elig.trace[AF] = 1.0
            rule.update(atoms, elig, intens, appetitive=1.0, aversive=0.0,
                        appetitive_exposure=True)
    w = atoms[AF].history_weights
    return w[A], w[B]


def test_blocking_requires_competitive_credit():
    # pretrain A alone, then reinforce the compound A+B
    phases = [(60, _intens(1.0, 0.0)), (60, _intens(1.0, 1.0))]
    _, w_b_comp = _train("rw_competitive", phases)
    _, w_b_indep = _train("rw_independent", phases)
    assert w_b_comp < 0.1, f"B should be blocked under competitive credit, got {w_b_comp}"
    assert w_b_indep > 0.9, f"B should NOT be blocked under independent credit, got {w_b_indep}"


def test_overshadowing_requires_competitive_credit():
    # reinforce the compound A+B from the start
    phases = [(60, _intens(1.0, 1.0))]
    _, w_b_comp = _train("rw_competitive", phases)
    _, w_b_indep = _train("rw_independent", phases)
    # competitive: the two cues share the associative strength (each ~ half of asymptote)
    assert 0.3 < w_b_comp < 0.7, f"B should be overshadowed (shared), got {w_b_comp}"
    assert w_b_indep > 0.9, f"B should reach full strength under independent, got {w_b_indep}"


def test_cue_trained_alone_reaches_asymptote():
    _, w_b = _train("rw_competitive", [(60, _intens(0.0, 1.0))])
    assert w_b > 0.9, f"B trained alone should reach asymptote, got {w_b}"
