"""Tests for the Organism: action emission, energy bookkeeping, death."""

from behavioral_md.config import SimulationConfig
from behavioral_md.environments import ACTIONS, BehavioralFieldEnv
from behavioral_md.organism import Organism


def _fresh(**cfg_kw):
    cfg = SimulationConfig(**cfg_kw)
    env = BehavioralFieldEnv(cfg)
    org = Organism(cfg)
    obs, _ = env.reset(seed=0)
    org.reset(obs)
    return cfg, env, org, obs


def test_emits_valid_action():
    cfg, env, org, obs = _fresh(grid_size=8)
    org.step(obs)
    action = org.emit_action()
    assert action in ACTIONS


def test_energy_decreases_without_food():
    # Place food and danger far away; metabolism + movement should drain energy.
    cfg = SimulationConfig(grid_size=10, max_steps=30)
    env = BehavioralFieldEnv(cfg)
    org = Organism(cfg)
    layout = {"position": [5, 5], "food": [9, 9], "danger": [0, 0], "light": [0, 9], "cue": [9, 0]}
    obs, _ = env.reset(seed=0, options={"layout": layout})
    org.reset(obs)
    start = org.energy
    for _ in range(20):
        org.step(obs)
        a = org.emit_action()
        obs, _, _, _, info = env.step(a)
        org.update_history(obs, a, info)
    assert org.energy < start


def test_starvation_causes_death():
    cfg = SimulationConfig(grid_size=10, max_steps=500, energy_init=0.05, basal_metabolism=0.02)
    env = BehavioralFieldEnv(cfg)
    org = Organism(cfg)
    layout = {"position": [5, 5], "food": [9, 9], "danger": [0, 0], "light": [0, 9], "cue": [9, 0]}
    obs, _ = env.reset(seed=0, options={"layout": layout})
    org.reset(obs)
    for _ in range(cfg.max_steps):
        org.step(obs)
        a = org.emit_action()
        obs, _, _, _, info = env.step(a)
        org.update_history(obs, a, info)
        if not org.alive:
            break
    assert not org.alive
    assert org.cause_of_death == "starvation"
    assert org.energy <= 0.0
