"""Robustness of the new zoo phenomena across parameters (experiments/exp042).

Cheap checks that the signatures are not knife-edge. Blocking is tabular (instant) so we sweep
learning rate; the contrast monotonicity is covered by exp042 (heavier) and the point tests.
"""

from __future__ import annotations

from behavioral_md.atoms import default_atom_set
from behavioral_md.config import SimulationConfig
from behavioral_md.learning import EligibilityTrace, make_learning_rule

A, B, AF = "light", "cue", 0


def _w_b(scheme, lr, phases):
    cfg = SimulationConfig(credit_assignment=scheme, learning_rate=lr, reinforcement_asymptote=1.0)
    atoms = default_atom_set()
    rule = make_learning_rule(cfg)
    elig = EligibilityTrace(len(atoms), 0.95)
    for trials, (ia, ib) in phases:
        intens = {A: ia, B: ib, "food": 0.0, "danger": 0.0}
        for _ in range(trials):
            elig.trace[:] = 0.0
            elig.trace[AF] = 1.0
            rule.update(atoms, elig, intens, appetitive=1.0, aversive=0.0, appetitive_exposure=True)
    return atoms[AF].history_weights[B]


def test_blocking_robust_across_learning_rate():
    phases = [(60, (1.0, 0.0)), (60, (1.0, 1.0))]
    for lr in (0.1, 0.3, 0.5):
        comp = _w_b("rw_competitive", lr, phases)
        indep = _w_b("rw_independent", lr, phases)
        assert comp < 0.1, f"blocking should hold at lr={lr}, got w_B={comp:.3f}"
        assert indep > 0.9, f"no blocking under independent at lr={lr}, got w_B={indep:.3f}"


def test_overshadowing_robust_across_learning_rate():
    phases = [(60, (1.0, 1.0))]
    for lr in (0.1, 0.3, 0.5):
        comp = _w_b("rw_competitive", lr, phases)
        assert 0.3 < comp < 0.7, f"overshadowing should hold at lr={lr}, got w_B={comp:.3f}"
