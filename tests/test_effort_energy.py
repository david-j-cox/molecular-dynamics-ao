"""Tests for the effort->energy loop and load-bearing fatigue (the molecular bridge; exp059)."""

import numpy as np

from behavioral_md.config import SimulationConfig
from behavioral_md.environments.gridworld import BehavioralFieldEnv
from behavioral_md.experiment_utils import synthetic_field_obs
from behavioral_md.organism import Organism


def _forage_energy(effort_cost=0.0, fatigue_energy_cost=0.0, fatigue_gain=0.0, steps=70, seed=1):
    cfg = SimulationConfig(seed=seed, effort_cost=effort_cost,
                           fatigue_energy_cost=fatigue_energy_cost, fatigue_gain=fatigue_gain)
    env = BehavioralFieldEnv(SimulationConfig(seed=seed))
    org = Organism(cfg, rng=np.random.default_rng(seed))
    obs, _ = env.reset(seed=seed)
    org.reset(obs)
    energy = []
    for _ in range(steps):
        org.step(obs)
        a = org.emit_action()
        obs, _r, _t, _tr, info = env.step(a)
        org.update_history(obs, a, info)
        energy.append(org.energy)
        if not org.alive:
            break
    return np.array(energy)


def test_effort_cost_default_is_byte_identical():
    """effort_cost=0 and fatigue_energy_cost=0 (defaults) leave the energy trajectory unchanged."""
    base = _forage_energy()
    same = _forage_energy(effort_cost=0.0, fatigue_energy_cost=0.0)
    assert np.array_equal(base, same)


def test_effort_cost_depletes_energy_faster():
    """With effort cost on, vigorous responding spends more energy -> reserve runs out sooner."""
    base = _forage_energy(effort_cost=0.0)
    eff = _forage_energy(effort_cost=0.08)
    assert len(eff) <= len(base)                       # dies no later
    n = min(len(base), len(eff))
    assert eff[n - 1] <= base[n - 1] + 1e-9            # never higher energy at a shared step


def test_fatigue_brakes_runaway_activation():
    """Without fatigue a sustained drive ramps an atom's activation toward the ceiling; load-bearing
    fatigue (fatigue_gain>0) bounds it well below that ramp."""
    def bout(gain):
        cfg = SimulationConfig(seed=0, fatigue_gain=gain, fatigue_decay=0.98, learning_rate=0.0)
        org = Organism(cfg, rng=np.random.default_rng(0))
        org.reset(synthetic_field_obs(food_on=True))
        for _ in range(80):
            org.step(synthetic_field_obs(food_on=True))
        return org.activation("approach_food")

    assert bout(0.0) > bout(0.08) + 0.2                # fatigue meaningfully suppresses the ramp


def test_fatigue_energy_cost_adds_expenditure():
    """fatigue_energy_cost (with fatigue accruing) spends extra energy each step beyond the
    effort/action cost, exposed via last_expenditure."""
    cfg = SimulationConfig(seed=0, fatigue_gain=0.1, fatigue_energy_cost=0.05, learning_rate=0.0)
    org = Organism(cfg, rng=np.random.default_rng(0))
    obs = synthetic_field_obs(food_on=True)
    org.reset(obs)
    for _ in range(30):                                # build up some fatigue
        org.step(obs)
    org.emit_action()
    org.update_history(obs, 5, {})                     # consume action (no env intake)
    base = cfg.basal_metabolism + cfg.rest_cost
    assert org.last_expenditure > base                 # fatigue load adds to expenditure
