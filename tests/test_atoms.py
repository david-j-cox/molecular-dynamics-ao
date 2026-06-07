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


def test_default_atoms_are_scalar_and_uncoupled():
    """Backward compatibility: every default atom is scalar (dims=1) with no internal coupling."""
    atoms = default_atom_set()
    assert all(a.dims == 1 for a in atoms)
    assert all(a.internal_coupling is None for a in atoms)


def test_scalar_atom_with_constant_drive_ramps():
    """A scalar atom (no restoring term) integrates -> ramps under a constant drive (exp060)."""
    a = BehavioralAtom("s", np.array([0.0]), np.array([0.0]))
    for _ in range(40):
        a.integrate(0.3, 0.1, -10, 10)
    assert a.activation > 1.0                       # has ramped well above baseline


def test_internal_coupling_makes_an_oscillator():
    """A multi-dimensional atom with a spring internal coupling oscillates (a CPG), where the bare
    scalar integrator cannot: the activation changes sign repeatedly and stays bounded."""
    from behavioral_md.atoms import oscillator_atom
    a = oscillator_atom("cpg", period=40.0, dt=0.1, amplitude=1.0)
    assert a.dims == 2
    out = []
    for _ in range(120):
        a.integrate(np.zeros(2), 0.1, -10, 10)
        out.append(a.activation)
    out = np.array(out)
    crossings = int((np.diff(np.sign(out)) != 0).sum())
    assert crossings >= 4                            # multiple cycles -> a rhythm
    assert np.abs(out).max() < 1.5                   # bounded (restoring force), not ramping


def test_oscillator_muscles_are_anti_phase():
    """The 2-D oscillator's two muscle dimensions run anti-phase (a stepping gait)."""
    from behavioral_md.atoms import oscillator_atom
    a = oscillator_atom("gait", period=40.0, dt=0.1, rel_phase=np.pi)
    m0, m1 = [], []
    for _ in range(120):
        a.integrate(np.zeros(2), 0.1, -10, 10)
        m0.append(a.state[0])
        m1.append(a.state[1])
    assert np.corrcoef(m0, m1)[0, 1] < -0.8          # anti-phase


def test_reset_restores_oscillator_initial_condition():
    """reset() restores an oscillator atom to its CONSTRUCTED initial state/velocity (so a pacemaker
    resumes its designed phase), preserving dimensionality -- not collapsing it to baseline rest."""
    from behavioral_md.atoms import oscillator_atom
    a = oscillator_atom("cpg", period=40.0, amplitude=1.0)
    init = a.state.copy()
    for _ in range(10):
        a.integrate(np.zeros(2), 0.1, -10, 10)
    a.reset()
    assert a.state.shape == (2,)
    assert np.allclose(a.state, init)          # restored to its initial (oscillating) condition
    assert not np.allclose(a.state, a.baseline_activation)   # NOT collapsed to baseline rest
