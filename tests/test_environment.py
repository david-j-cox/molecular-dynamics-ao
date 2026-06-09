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


# --- temporal stimulus control: sun as a learnable cue + time-locked food (exp064) -------------

def test_temporal_cue_feeds_ambient_light_as_cue():
    """With temporal_cue on, the cue value the organism receives IS the sun L(t), at full presence;
    off, it is the usual spatial cue (byte-identical), so behavior is unchanged by default."""
    cfg = SimulationConfig(temporal_cue=True, steps_per_day=8)
    env = BehavioralFieldEnv(cfg)
    env.reset(seed=0, options={"layout": _LAY})
    seen = []
    for _ in range(8):
        obs, *_ = env.step(0)
        seen.append((float(obs["cue_value"][0]), float(obs["cue_intensity"][0])))
    # cue_value tracks L(t) (peaks at noon), intensity is a constant 1.0 (ambient presence).
    assert abs(seen[3][0] - ambient_light(4, 8)) < 1e-6 and abs(seen[3][0] - 1.0) < 1e-6
    assert all(abs(it - 1.0) < 1e-9 for _v, it in seen)
    # Off: cue value is the (static) spatial cue, not the light.
    env_off = BehavioralFieldEnv(SimulationConfig(), cue_value=0.7)
    obs_off, _ = env_off.reset(seed=0, options={"layout": _LAY})
    assert abs(float(obs_off["cue_value"][0]) - 0.7) < 1e-6


def test_food_phase_window_time_locks_food():
    """A food_phase_window makes food appear (visible + edible) only within the phase window; with
    no window food is available at every phase (byte-identical)."""
    cfg = SimulationConfig(steps_per_day=8, food_phase_window=(0.4, 0.6))
    env = BehavioralFieldEnv(cfg)
    env.reset(seed=0, options={"layout": {"position": [4, 8], "food": [4, 8], "danger": [0, 0],
                                          "light": [0, 0], "cue": [0, 0]}})
    intakes = []
    for _ in range(8):
        obs, _r, _te, _tr, info = env.step(0)        # sit on the food cell
        intakes.append(info["food_intake"])
    # Food reinforces only while the pre-step phase i/8 is in [0.4, 0.6) -- i.e. at noon (i=4).
    expected_open = [0.4 <= (i / 8) < 0.6 for i in range(8)]
    assert any(intakes) and not all(i > 0 for i in intakes)
    assert all((intake > 0) == op for intake, op in zip(intakes, expected_open, strict=True))
    # No window => food available every step (the un-gated default).
    env2 = BehavioralFieldEnv(SimulationConfig(steps_per_day=8))
    env2.reset(seed=0, options={"layout": {"position": [4, 8], "food": [4, 8], "danger": [0, 0],
                                           "light": [0, 0], "cue": [0, 0]}})
    assert all(env2.step(0)[4]["food_intake"] > 0 for _ in range(8))
