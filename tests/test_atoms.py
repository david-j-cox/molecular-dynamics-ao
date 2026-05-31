"""Tests for behavioral atoms, Verlet integration, and behavioral momentum."""

import numpy as np

from behavioral_md.atoms import BehavioralAtom, default_atom_set, verlet_update


def test_verlet_update_matches_hand_calculation():
    # x=2, x_prev=1, F=4, m=2, dt=0.5 -> 2*2 - 1 + (4/2)*0.25 = 3.5
    got = verlet_update(np.array([2.0]), np.array([1.0]), np.array([4.0]), 2.0, 0.5)
    assert np.isclose(got[0], 3.5)


def test_integrate_from_rest_and_rolls_state():
    a = BehavioralAtom("t", np.array([0.0]), np.array([0.0]), mass=2.0)
    a.integrate(force=4.0, dt=0.1, activation_min=-10, activation_max=10)
    # from rest: x1 = (F/m) dt^2 = (4/2)*0.01 = 0.02; previous rolls to old current (0)
    assert np.isclose(a.activation, 0.02)
    assert np.isclose(a.previous_state[0], 0.0)


def test_integrate_clips_to_bounds():
    a = BehavioralAtom("c", np.array([9.5]), np.array([9.0]), mass=1.0)
    a.integrate(force=100.0, dt=0.5, activation_min=-10, activation_max=10)
    assert a.activation == 10.0


def test_behavioral_momentum_mass_resists_change():
    """Higher mass -> smaller activation change for the same force (inertia)."""
    light = BehavioralAtom("light", np.array([0.0]), np.array([0.0]), mass=1.0)
    heavy = BehavioralAtom("heavy", np.array([0.0]), np.array([0.0]), mass=5.0)
    for atom in (light, heavy):
        atom.integrate(force=3.0, dt=0.1, activation_min=-10, activation_max=10)
    assert abs(heavy.activation) < abs(light.activation)
    # change scales as 1/mass
    assert np.isclose(heavy.activation * 5.0, light.activation * 1.0)


def test_default_atom_set_two_tier_structure():
    atoms = default_atom_set()
    names = {a.name for a in atoms}
    assert {"approach_food", "avoid_danger", "move_up", "consume"} <= names
    drives = [a for a in atoms if a.valence != 0.0 and a.direction is None]
    movers = [a for a in atoms if a.direction is not None]
    assert drives and movers
    # drive atoms carry a stimulus + valence; movement atoms are topographies
    assert all(a.stimulus is not None for a in drives)
    assert all(a.valence == 0.0 for a in movers)
