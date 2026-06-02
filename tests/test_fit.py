"""Tests for the derivative-free sensitivity fit (fit).

The fit must reduce the target loss and move parameters in the expected direction
(higher targets -> larger beta). Skipped automatically if JAX is not installed.

Small n_org/n_steps/maxiter keep these fast; the qualitative checks hold at reduced
settings.
"""

import pytest

pytest.importorskip("jax")

from behavioral_md.fit import fit  # noqa: E402
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
