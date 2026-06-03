"""Tests for the shared experiment helpers."""

import numpy as np

from behavioral_md.experiment_utils import compute_mean_ci, fit_matching_law, weak_innate_atoms


def test_compute_mean_ci_basic():
    mean, ci = compute_mean_ci([1.0, 2.0, 3.0])
    assert mean == 2.0
    assert ci > 0.0
    # Empty -> NaN mean, zero CI; single value -> zero CI.
    m_empty, ci_empty = compute_mean_ci([])
    assert np.isnan(m_empty) and ci_empty == 0.0
    assert compute_mean_ci([5.0]) == (5.0, 0.0)


def test_fit_matching_law_recovers_line():
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    y = 0.7 * x + 0.1                       # known slope/intercept
    a, log_b, r2 = fit_matching_law(x, y)
    assert abs(a - 0.7) < 1e-9
    assert abs(log_b - 0.1) < 1e-9
    assert abs(r2 - 1.0) < 1e-9


def test_weak_innate_atoms_sets_only_approach_food():
    atoms = weak_innate_atoms(0.2)
    by_name = {a.name: a for a in atoms}
    assert by_name["approach_food"].sensitivity["food"] == 0.2
