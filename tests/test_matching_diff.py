"""Tests for the differentiable matching surrogate (matching_diff).

The surrogate must land in the right regime (undermatching, both sensitivities in a
plausible band) and respond to its levers the same direction as the stochastic engine,
so a fit on it transfers. Skipped automatically if JAX is not installed.

Small n_org/n_steps keep these fast; the qualitative checks (sign, ordering, range)
hold at reduced settings even though the exact sensitivity values shift a little.
"""

import pytest

pytest.importorskip("jax")
import jax  # noqa: E402

from behavioral_md.matching import MatchConfig  # noqa: E402
from behavioral_md.matching_diff import (  # noqa: E402
    default_params,
    soft_sensitivities,
)

_N_ORG, _N_STEPS = 48, 600


def _sens(params):
    return soft_sensitivities(params, MatchConfig(), _N_STEPS, _N_ORG,
                              jax.random.key(0))


def test_default_sensitivities_in_undermatching_band():
    a_rate, a_amt = _sens(default_params(MatchConfig()))
    a_rate, a_amt = float(a_rate), float(a_amt)
    # Undermatching: both slopes strictly between 0 and 1.
    assert 0.0 < a_rate < 1.0
    assert 0.0 < a_amt < 1.0


def test_beta_raises_rate_sensitivity():
    # beta is the lever the fit relies on for a_rate; in both the surrogate and the
    # stochastic engine a larger beta yields a larger rate sensitivity (transfers).
    base = default_params(MatchConfig())
    low = float(_sens({**base, "beta": 4.0})[0])
    high = float(_sens({**base, "beta": 9.0})[0])
    assert high > low


def test_sensitivities_deterministic_under_fixed_key():
    # Common random numbers -> the forward model is a deterministic function of params
    # (this is what makes the derivative-free search well-posed).
    p = default_params(MatchConfig())
    a1 = tuple(float(x) for x in _sens(p))
    a2 = tuple(float(x) for x in _sens(p))
    assert a1 == a2


def test_amount_exponent_is_orthogonal_to_rate():
    # rho (amount_exponent) is the decoupling lever: it scales amount sensitivity while
    # leaving rate sensitivity EXACTLY untouched -- the rate sweep runs at equal amounts
    # (amount=1), so amount**rho=1 for any rho and that rollout is bit-identical.
    base = default_params(MatchConfig())
    lo = _sens({**base, "amount_exponent": 0.5})
    hi = _sens({**base, "amount_exponent": 1.5})
    assert float(lo[0]) == float(hi[0])      # a_rate identical across rho
    assert float(hi[1]) > float(lo[1])       # a_amt rises with rho
