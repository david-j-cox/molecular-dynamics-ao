"""Tests for the multi-patch foraging engine.

Skipped automatically if JAX is not installed (it is an optional extra).
"""

import pytest

forage = pytest.importorskip("behavioral_md.forage")

TOL = 1e-6


def test_single_patch_reduces_to_make_simulate():
    """With P == 1 the multi-patch sim must reproduce jax_engine.make_simulate."""
    assert forage.validate_against_single_patch() < TOL
