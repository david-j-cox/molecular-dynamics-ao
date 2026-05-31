"""Tests for the matching preparation: matching emerges, bias ~0 on the odd grid.

Skipped automatically if JAX is not installed.
"""

import numpy as np
import pytest

pytest.importorskip("jax")
import jax  # noqa: E402

from behavioral_md.matching import MatchConfig, make_matching_sim  # noqa: E402

PATCH_POS = np.array([[2.0, 5.0], [8.0, 5.0]])
PATCH_CUE = np.array([0.2, 0.8])
START = [5.0, 5.0]


def _run(arm, n_org=300, n_steps=3000, seed=0):
    mcfg = MatchConfig()
    sim, initial_state = make_matching_sim(mcfg, PATCH_POS, PATCH_CUE, START)
    import jax.numpy as jnp
    keys = jax.random.split(jax.random.key(0), n_steps)
    state0 = initial_state(n_org, jax.random.key(seed))
    time_at, count, _amt = sim(state0, keys, jnp.array(arm))
    return np.asarray(time_at)


def test_allocation_follows_richer_schedule():
    # Left patch much richer -> more time on the left.
    B = _run([0.18, 0.02])
    bL, bR = B[:, 0].sum(), B[:, 1].sum()
    assert bL > bR


def test_equal_schedule_no_side_bias():
    # 1:1 schedule on the symmetric odd grid -> near-equal allocation (bias ~0).
    B = _run([0.10, 0.10])
    bL, bR = B[:, 0], B[:, 1]
    ok = (bL > 0) & (bR > 0)
    median_log_ratio = float(np.median(np.log(bL[ok] / bR[ok])))
    assert abs(median_log_ratio) < 0.25


def test_matching_slope_positive_and_subunity():
    # Sweep ratios, fit GML slope; expect undermatching (0 < a < 1) at this COD.
    import jax.numpy as jnp  # noqa: F401
    pairs = [(0.18, 0.02), (0.14, 0.06), (0.10, 0.10), (0.06, 0.14), (0.02, 0.18)]
    x, y = [], []
    for arm in pairs:
        B = _run(list(arm))
        bL, bR = B[:, 0], B[:, 1]
        ok = (bL > 0) & (bR > 0)
        y.append(np.log(bL[ok] / bR[ok]))
        x.append(np.full(ok.sum(), np.log(arm[0] / arm[1])))
    a, _ = np.polyfit(np.concatenate(x), np.concatenate(y), 1)
    assert 0.05 < a < 1.0
