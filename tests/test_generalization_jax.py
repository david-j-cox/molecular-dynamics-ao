"""Generalization and peak shift on the JAX fast path (experiments/exp044).

Locks in that the JAX cue mechanism (Shepard tuning + summed-error Rescorla-Wagner, one jitted scan)
reproduces a generalization gradient peaked at the trained value and a peak shift away from S-.
Self-contained (mirrors exp044's math so it depends only on behavioral_md + jax); skipped if JAX is
not installed.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from behavioral_md.config import SimulationConfig  # noqa: E402

_CFG = SimulationConfig()
K = _CFG.n_cue_receptors
BETA = _CFG.cue_generalization_beta
LO, HI = _CFG.history_weight_min, _CFG.history_weight_max
CENTERS = jnp.linspace(0.0, 1.0, K)
RATE, LAM, N_AGENTS, NOISE = 0.02, 1.0, 200, 0.02
V_PLUS, V_MINUS = 0.40, 0.55
PROBES = np.linspace(0.0, 1.0, 41)


@jax.jit
def _train(key, values, mags):
    noise = jax.random.normal(key, (values.shape[0], N_AGENTS)) * NOISE

    def step(w, x):
        v, mag, ns = x
        sensed = jnp.clip(v + ns, 0.0, 1.0)
        act = jnp.exp(-BETA * jnp.abs(sensed[:, None] - CENTERS[None, :]))
        err = LAM * mag - jnp.sum(w * act, axis=1)
        return jnp.clip(w + RATE * act * err[:, None], LO, HI), None

    w, _ = jax.lax.scan(step, jnp.zeros((N_AGENTS, K)), (values, mags, noise))
    return w


def _peak(values, mags):
    w = _train(jax.random.key(0), jnp.asarray(values), jnp.asarray(mags))
    act = jnp.exp(-BETA * jnp.abs(jnp.asarray(PROBES)[:, None] - CENTERS[None, :]))
    resp = np.asarray(w @ act.T)
    return PROBES[np.argmax(resp.mean(0))]


def test_generalization_peaks_at_trained_value():
    peak = _peak(np.full(400, V_PLUS), np.ones(400))
    assert abs(peak - V_PLUS) <= 0.03, f"gradient should peak at S+={V_PLUS}, got {peak}"


def test_peak_shifts_away_from_s_minus():
    peak = _peak(np.tile([V_PLUS, V_MINUS], 400), np.tile([1.0, 0.0], 400))
    assert peak < V_PLUS - 0.02, f"peak should shift below S+={V_PLUS} away from S-, got {peak}"
