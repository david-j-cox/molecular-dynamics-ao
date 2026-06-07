"""Tests for the effort->energy loop and load-bearing fatigue (the molecular bridge; exp059)."""

import numpy as np

from behavioral_md.config import SimulationConfig
from behavioral_md.environments.gridworld import BehavioralFieldEnv
from behavioral_md.organism import Organism


def _food_obs(on: bool) -> dict:
    z = np.zeros(2)
    o = {f"{s}_vector": z.copy() for s in ("food", "danger", "light", "cue")}
    o.update({f"{s}_intensity": np.array([0.0]) for s in ("food", "danger", "light", "cue")})
    o["food_intensity"] = np.array([1.0 if on else 0.0])
    o["food_vector"] = np.array([1.0, 0.0])
    o["food_contact"] = np.array([1.0 if on else 0.0])
    o["cue_value"] = np.array([0.0])
    o["context"] = np.array([0.0])
    return o


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
        org.reset(_food_obs(True))
        for _ in range(80):
            org.step(_food_obs(True))
        return org.activation("approach_food")

    assert bout(0.0) > bout(0.08) + 0.2                # fatigue meaningfully suppresses the ramp


def test_fatigue_energy_cost_adds_expenditure():
    """fatigue_energy_cost (with fatigue accruing) spends extra energy each step beyond the
    effort/action cost, exposed via last_expenditure."""
    cfg = SimulationConfig(seed=0, fatigue_gain=0.1, fatigue_energy_cost=0.05, learning_rate=0.0)
    org = Organism(cfg, rng=np.random.default_rng(0))
    obs = _food_obs(True)
    org.reset(obs)
    for _ in range(30):                                # build up some fatigue
        org.step(obs)
    org.emit_action()
    org.update_history(obs, 5, {})                     # consume action (no env intake)
    base = cfg.basal_metabolism + cfg.rest_cost
    assert org.last_expenditure > base                 # fatigue load adds to expenditure
