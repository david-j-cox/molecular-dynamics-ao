"""Tests for the two-tier force calculator."""

import numpy as np

from behavioral_md.atoms import default_atom_set
from behavioral_md.config import SimulationConfig
from behavioral_md.forces import ForceCalculator, SensoryField


def _sensory(food_dir=(0.0, 0.0), food_int=0.0, danger_dir=(0.0, 0.0), danger_int=0.0):
    z = SensoryField(np.array([0.0, 0.0]), 0.0)
    return {
        "food": SensoryField(np.array(food_dir), food_int),
        "danger": SensoryField(np.array(danger_dir), danger_int),
        "light": z,
        "cue": z,
    }


def test_force_shape_matches_atoms():
    atoms = default_atom_set()
    fc = ForceCalculator(atoms, config=SimulationConfig())
    force, comp = fc.compute(_sensory(food_dir=(0.0, 1.0), food_int=0.5), energy=0.5)
    assert force.shape == (len(atoms),)
    assert np.allclose(force, comp.total)


def _activate(atoms, name, value):
    """Set a drive atom's current activation (movement expresses drive activations)."""
    for a in atoms:
        if a.name == name:
            a.state[:] = value


def test_directional_expression_food_up_drives_move_up():
    # An active approach_food drive, with food up, should express toward move_up.
    atoms = default_atom_set()
    names = [a.name for a in atoms]
    _activate(atoms, "approach_food", 1.0)
    fc = ForceCalculator(atoms, config=SimulationConfig())
    force, _ = fc.compute(_sensory(food_dir=(0.0, 1.0), food_int=1.0), energy=0.5)
    assert force[names.index("move_up")] > force[names.index("move_down")]


def test_danger_avoidance_pushes_movement_away():
    # An active avoid_danger drive (valence -1), danger up -> move_up < move_down.
    atoms = default_atom_set()
    names = [a.name for a in atoms]
    _activate(atoms, "avoid_danger", 1.0)
    fc = ForceCalculator(atoms, config=SimulationConfig())
    force, _ = fc.compute(_sensory(danger_dir=(0.0, 1.0), danger_int=1.0), energy=1.0)
    assert force[names.index("move_up")] < force[names.index("move_down")]


def test_energy_deficit_increases_food_drive():
    # The energy-deficit (motivational) term acts on the approach_food DRIVE atom.
    atoms = default_atom_set()
    names = [a.name for a in atoms]
    fc = ForceCalculator(atoms, config=SimulationConfig())
    s = _sensory(food_dir=(0.0, 1.0), food_int=1.0)
    _, sated = fc.compute(s, energy=1.0)
    _, starving = fc.compute(s, energy=0.05)
    af = names.index("approach_food")
    assert starving.motivational[af] > sated.motivational[af]
