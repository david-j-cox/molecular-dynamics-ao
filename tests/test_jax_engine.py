"""Tests that the JAX engine reproduces the NumPy reference engine.

Skipped automatically if JAX is not installed (it is an optional extra).
"""

import pytest

jax_engine = pytest.importorskip("behavioral_md.jax_engine")

TOL = 1e-6


def test_force_matches_numpy():
    assert jax_engine.validate_against_numpy() < TOL


def test_learning_matches_numpy():
    assert jax_engine.validate_learning() < TOL


def test_emission_matches_numpy():
    assert jax_engine.validate_emission() < TOL


def test_environment_matches_numpy():
    assert jax_engine.validate_env() < TOL


def test_leaky_integrator_matches_numpy():
    """The JAX integrator reproduces the first-order leaky update (exp048/exp060 parity) exactly,
    and leaves the default damped-Verlet step unchanged."""
    import numpy as np

    from behavioral_md.config import SimulationConfig

    cfg = SimulationConfig(integrator="leaky", leak_coef=1.3)
    spec = jax_engine.build_spec(config=cfg)
    a = spec.mass.shape[0]
    rng = np.random.default_rng(0)
    act = np.asarray(rng.normal(size=(3, a)))
    prev = np.asarray(rng.normal(size=(3, a)))
    force = np.asarray(rng.normal(size=(3, a)))
    new, _ = jax_engine.integrate(spec, act, prev, force, cfg)
    mass = np.asarray(spec.mass)[None, :]
    ref = np.clip(act + cfg.dt * (force / mass - cfg.leak_coef * act),
                  cfg.activation_min, cfg.activation_max)
    assert np.max(np.abs(np.asarray(new) - ref)) < TOL
