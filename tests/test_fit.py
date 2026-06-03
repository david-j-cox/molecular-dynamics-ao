"""Tests for the derivative-free sensitivity fit (fit).

The fit must reduce the target loss and move parameters in the expected direction
(higher targets -> larger beta). Skipped automatically if JAX is not installed.

Small n_org/n_steps/maxiter keep these fast; the qualitative checks hold at reduced
settings.
"""

import pytest

pytest.importorskip("jax")

from behavioral_md.fit import fit, fit_dims  # noqa: E402
from behavioral_md.matching import MatchConfig  # noqa: E402

_KW = dict(n_org=48, n_steps=600, maxiter=40)


def test_fit_reduces_loss():
    # Target both sensitivities upward from the (reduced-setting) baseline.
    _fitted, hist = fit((0.50, 0.65), **_KW)
    first = hist[0]["loss"]
    best = min(h["loss"] for h in hist)
    assert best < first


def test_higher_targets_give_larger_beta():
    # beta is the dominant lever: aiming for higher sensitivities should land on a
    # larger beta than aiming for lower ones.
    up, _ = fit((0.50, 0.66), **_KW)
    down, _ = fit((0.38, 0.56), **_KW)
    assert up.beta > down.beta


def test_fit_only_touches_free_params():
    # lr_cue (and other config) must be left untouched -- only the free params move.
    # The default free set excludes amount_exponent, so it stays at its default.
    base = MatchConfig()
    fitted, _ = fit((0.45, 0.62), **_KW)
    assert fitted.lr_cue == base.lr_cue
    assert fitted.damping == base.damping
    assert fitted.grid_size == base.grid_size
    assert fitted.amount_exponent == base.amount_exponent


def test_amount_exponent_fits_low_amount_target():
    # With amount_exponent in the free set, aiming for a LOW amount sensitivity should
    # drive rho below 1 (concave magnitude utility) -- the decoupling lever in action.
    free = ("temperature", "approach_gain", "beta", "amount_exponent")
    fitted, _ = fit((0.45, 0.35), free=free, **_KW)
    assert fitted.amount_exponent < 1.0


def test_fit_dims_orders_sigma_by_prob_target():
    # probability_exponent (sigma) is the a_prob lever: a higher probability-sensitivity
    # target should land on a larger sigma. Search sigma alone so the lever must do the
    # work (with beta free the shared anchor can absorb the target at tiny settings).
    lo, _ = fit_dims({"prob": 0.15}, free=("probability_exponent",), **_KW)
    hi, _ = fit_dims({"prob": 0.45}, free=("probability_exponent",), **_KW)
    assert hi.probability_exponent > lo.probability_exponent
    # (delay_k as an a_delay lever is covered by test_delay_k_is_orthogonal in
    # test_matching_diff; the delay signal is too weak to resolve fine a_delay targets
    # at these fast test settings, and exp025 exercises the delay fit end-to-end.)
