"""Tests for the BehavioralFieldEnv (Gymnasium compliance + dynamics)."""


from behavioral_md.config import SimulationConfig
from behavioral_md.environments import ACTIONS, BehavioralFieldEnv


def test_reset_returns_valid_observation():
    env = BehavioralFieldEnv(SimulationConfig(grid_size=8))
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    assert "position" in obs and "food_contact" in obs and "cue_value" in obs


def test_step_returns_gymnasium_five_tuple():
    env = BehavioralFieldEnv(SimulationConfig(grid_size=8, max_steps=50))
    env.reset(seed=0)
    out = env.step(1)
    assert len(out) == 5
    obs, reward, terminated, truncated, info = out
    assert isinstance(reward, float)
    assert isinstance(terminated, bool) and isinstance(truncated, bool)


def test_truncates_at_max_steps():
    cfg = SimulationConfig(grid_size=8, max_steps=10)
    env = BehavioralFieldEnv(cfg)
    env.reset(seed=1)
    truncated = False
    for _ in range(cfg.max_steps):
        _, _, _, truncated, _ = env.step(0)
    assert truncated


def test_food_biomass_depletes_and_regrows():
    cfg = SimulationConfig(grid_size=8)
    env = BehavioralFieldEnv(cfg)
    layout = {"position": [4, 4], "food": [4, 4], "danger": [0, 7], "light": [7, 0], "cue": [7, 7]}
    env.reset(seed=0, options={"layout": layout})
    # Organism is on the food cell: a consume/no-op step should draw biomass down.
    full = env.food_biomass
    _, _, _, _, info = env.step(0)
    assert info["food_intake"] > 0.0
    assert env.food_biomass < full
    assert env.food_biomass >= cfg.food_min_biomass


def test_actions_cover_seven_discrete():
    assert set(ACTIONS) == set(range(7))
