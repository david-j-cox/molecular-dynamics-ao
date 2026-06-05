"""Tests for the BehavioralFieldEnv (Gymnasium compliance + dynamics)."""


from behavioral_md.config import SimulationConfig
from behavioral_md.environments import ACTIONS, BehavioralFieldEnv
from behavioral_md.environments.gridworld import ambient_light


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


def test_ambient_light_cycle():
    """L(t) = 0.5*(1-cos(2pi*phase)): 0 at midnight (t=0), 1 at noon (t=spd/2)."""
    assert ambient_light(0, 96) == 0.0
    assert abs(ambient_light(48, 96) - 1.0) < 1e-9
    assert ambient_light(96, 96) == 0.0          # wraps to next midnight
    assert 0.0 <= ambient_light(13, 96) <= 1.0


_LAY = {"position": [4, 4], "food": [4, 8], "danger": [4, 4], "light": [0, 0], "cue": [7, 7]}


def test_day_night_off_is_unmodulated():
    """With day_night off, ambient_light is 1.0 and intensities are unscaled."""
    env = BehavioralFieldEnv(SimulationConfig(day_night=False))
    obs, _ = env.reset(seed=0, options={"layout": _LAY})
    assert obs["ambient_light"][0] == 1.0
    raw = float(env._intensity(env.danger_pos))
    assert abs(float(obs["danger_intensity"][0]) - raw) < 1e-9


def test_day_night_grades_danger_detectability():
    """With day_night on, sensed danger = true * (floor + (1-floor)*L): low at night,
    full by day."""
    cfg = SimulationConfig(day_night=True, steps_per_day=8, danger_detect_floor=0.2)
    env = BehavioralFieldEnv(cfg)
    obs, _ = env.reset(seed=0, options={"layout": _LAY})  # t=0, midnight, L=0
    raw = float(env._intensity(env.danger_pos))
    assert abs(float(obs["danger_intensity"][0]) - raw * 0.2) < 1e-6   # floor at night
    seen = []
    for _ in range(8):
        obs, _r, _te, _tr, info = env.step(0)
        seen.append(float(obs["danger_intensity"][0]))
    # Detectability peaks at noon (L=1 -> raw) and is lowest at night (-> raw*floor).
    assert max(seen) > min(seen)
    assert max(seen) <= raw + 1e-6
