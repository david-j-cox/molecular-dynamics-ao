"""Tests for sequence-based parameter recovery and identifiability (roadmap 2.1; exp055/exp056)."""

import numpy as np

from behavioral_md import recovery as rc


def test_planner_sequence_recovers_R():
    """Maximum-likelihood recovery from choice sequences recovers the true requirement R (and the
    other latent parameters) of the survival planner -- the core identifiability claim."""
    true = {"m_day": 0.04, "R": 0.45, "beta": 10.0}
    data = rc.simulate_choice_sequences(true["m_day"], true["R"], true["beta"],
                                        n_org=100, n_cycles=12, seed=1)
    fit = rc.fit_sequence(data)
    assert abs(fit["R"] - true["R"]) < 0.04
    assert abs(fit["m_day"] - true["m_day"]) < 0.01
    assert abs(fit["beta"] - true["beta"]) < 3.0


def test_sequence_identifies_R_better_than_aggregate():
    """R is sharply identified from the sequence (narrow profile-likelihood) but not from the
    aggregate proportion + occupancy (a flat profile across a wide R range) -- the devil is in the
    sequence."""
    true = {"m_day": 0.04, "R": 0.45, "beta": 10.0}
    data = rc.simulate_choice_sequences(true["m_day"], true["R"], true["beta"],
                                        n_org=100, n_cycles=12, seed=2)
    R_grid = np.linspace(0.32, 0.60, 15)

    def width(prof):
        ok = R_grid[prof > -1.92]
        return float(ok.max() - ok.min()) if ok.size else 0.0

    w_seq = width(rc.profile_loglik_R(data, R_grid, true, aggregate=False))
    w_agg = width(rc.profile_loglik_R(data, R_grid, true, aggregate=True))
    assert w_seq < 0.08                      # sequence pins R to ~a grid step
    assert w_agg > 3 * max(w_seq, R_grid[1] - R_grid[0])   # aggregate leaves R wide open


def test_rl_value_learner_has_a_degenerate_ridge():
    """The value-learner's (alpha, beta) likelihood is an anticorrelated valley: along the best-fit
    ridge, a faster learning rate trades off against a flatter softmax temperature."""
    d = rc.rl_simulate(0.2, 6.0, n_trials=400, seed=3)
    alphas = np.linspace(0.05, 0.7, 30)
    betas = np.linspace(1.0, 16.0, 30)
    surf = rc.rl_loglik_surface(d["choices"], d["rewards"], alphas, betas)
    ridge_beta = betas[np.argmin(surf, axis=1)]
    valley = surf.min(axis=1) < 2.0
    corr = np.corrcoef(alphas[valley], ridge_beta[valley])[0, 1]
    assert corr < -0.7                       # strongly anticorrelated trade-off


def test_rl_fit_recovers_truth_with_enough_trials():
    """With a long enough sequence the value-learner's parameters are recovered (the degeneracy is a
    precision/trade-off problem at realistic lengths, not a bias)."""
    d = rc.rl_simulate(0.2, 6.0, n_trials=800, seed=4)
    fit = rc.rl_fit(d["choices"], d["rewards"])
    assert abs(fit["alpha"] - 0.2) < 0.1
    assert abs(fit["beta"] - 6.0) < 4.0
