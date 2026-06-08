"""Tests for the death-pattern / survival metrics (metrics.py)."""

import numpy as np

from behavioral_md.metrics import (
    cause_breakdown,
    mortality_by_life,
    survival_curve,
    time_to_death,
)


def test_time_to_death_keeps_only_the_dead():
    survived = np.array([[10, 30, 30], [5, 30, 12]])     # 30 = censored (survived n_steps)
    cause = np.array([[1, 0, 0], [2, 0, 1]])             # 0 alive, 1 starve, 2 danger
    dt = time_to_death(survived, cause)
    assert sorted(dt.tolist()) == [5, 10, 12]            # the three that died, survivors dropped


def test_cause_breakdown_fractions_sum_to_one():
    cause = np.array([0, 0, 1, 2, 1, 0])
    b = cause_breakdown(cause)
    assert set(b) == {"alive", "starvation", "danger"}
    assert np.isclose(sum(b.values()), 1.0)
    assert np.isclose(b["alive"], 3 / 6) and np.isclose(b["starvation"], 2 / 6)
    assert cause_breakdown(np.array([], dtype=int)) == {"alive": 0.0, "starvation": 0.0,
                                                        "danger": 0.0}


def test_survival_curve_starts_full_and_is_monotone():
    survived = np.array([2, 5, 10, 10])                  # n_steps = 10
    steps, frac = survival_curve(survived, 10)
    assert steps.shape == frac.shape == (11,)
    assert frac[0] == 1.0                                 # all alive at step 0
    assert np.all(np.diff(frac) <= 1e-9)                 # non-increasing
    assert np.isclose(frac[-1], 0.0)                     # > n_steps never true -> 0 at the end


def test_mortality_by_life_shapes_and_values():
    survived = np.array([[10, 20, 20], [5, 20, 20]])     # 2 lives x 3 organisms, n_steps=20
    cause = np.array([[1, 0, 0], [2, 0, 0]])
    m = mortality_by_life(survived, cause)
    assert m["death_rate"].shape == (2,)
    assert np.allclose(m["death_rate"], [1 / 3, 1 / 3])
    assert np.allclose(m["starvation_rate"], [1 / 3, 0.0])
    assert np.allclose(m["danger_rate"], [0.0, 1 / 3])
    assert np.allclose(m["mean_lifespan"], [50 / 3, 45 / 3])
