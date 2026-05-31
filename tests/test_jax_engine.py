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
