"""BehavioralFieldEnv: a 2D grid arena that presents sensory force-fields.

The arena holds an organism plus food, danger, light, and a neutral cue. The
observation exposes only *sensory* information (direction + intensity to each
source, plus the last consequence) rather than omniscient state. Rewards are
returned for Gymnasium compatibility but are intended to be consumed as
*consequence events* that update learning history, not as an RL training signal.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from behavioral_md.config import SimulationConfig

# Discrete action set (see CLAUDE.md spec).
ACTIONS: dict[int, str] = {
    0: "no-op",
    1: "move_up",
    2: "move_down",
    3: "move_left",
    4: "move_right",
    5: "consume",
    6: "pause",
}

# Movement deltas in (row, col) grid coordinates for the directional actions.
_MOVES: dict[int, np.ndarray] = {
    1: np.array([0.0, 1.0]),   # up   -> +y
    2: np.array([0.0, -1.0]),  # down -> -y
    3: np.array([-1.0, 0.0]),  # left -> -x
    4: np.array([1.0, 0.0]),   # right-> +x
}


def ambient_light(t: int, steps_per_day: int) -> float:
    """Global day/night light L(t) in [0, 1]: 0 at midnight (t=0), 1 at noon.

    A raised cosine, L(t) = 0.5*(1 - cos(2*pi*(t mod steps_per_day)/steps_per_day)).
    """
    phase = (t % steps_per_day) / steps_per_day
    return float(0.5 * (1.0 - np.cos(2.0 * np.pi * phase)))


class BehavioralFieldEnv(gym.Env):
    """Grid arena presenting food/danger/light/cue stimulus fields.

    Parameters
    ----------
    config:
        Simulation configuration controlling grid size, sensor falloff, etc.
    food_reinforces:
        If False, consuming food yields no positive consequence (used by the
        extinction demo). Food remains visible.
    cue_value:
        Scalar position of the neutral cue in an abstract stimulus space. Used
        by the generalization demo to vary cue similarity across runs.
    """

    metadata = {"render_modes": ["human", "rgb_array", "none"], "render_fps": 8}

    # Consequence magnitudes.
    FOOD_REWARD = 1.0
    DANGER_REWARD = -1.0

    def __init__(
        self,
        config: SimulationConfig | None = None,
        *,
        food_reinforces: bool = True,
        food_present: bool = True,
        cue_value: float = 0.0,
        context: float = 0.0,
    ) -> None:
        super().__init__()
        self.config = config or SimulationConfig()
        self.food_reinforces = food_reinforces
        # Whether food exists in the arena at all. False = food absent (no signal, no
        # contact, no intake) -- a true food-free interval, distinct from
        # food_reinforces=False (food present but unrewarded = extinction).
        self.food_present = food_present
        self.cue_value = cue_value
        # Scalar context signal (A/B environments). Surfaced in the observation and
        # used by the dual learning rule to make inhibition context-specific (renewal).
        self.context = context
        # Time-locked food window (exp064); per-life overridable via reset options so a
        # control condition can randomize it. None = food available at all phases.
        self.food_phase_window = self.config.food_phase_window

        self.grid_size = self.config.grid_size
        self.sensor_range = self.config.sensor_range
        self.consume_radius = self.config.consume_radius

        self.render_mode = (
            None if self.config.render_mode == "none" else self.config.render_mode
        )

        self.action_space = spaces.Discrete(len(ACTIONS))
        g = float(self.grid_size)
        self.observation_space = spaces.Dict(
            {
                "position": spaces.Box(0.0, g, shape=(2,), dtype=np.float32),
                "food_vector": spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32),
                "danger_vector": spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32),
                "light_vector": spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32),
                "cue_vector": spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32),
                "food_intensity": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
                "danger_intensity": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
                "light_intensity": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
                "cue_intensity": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
                # Abstract scalar cue dimension (for stimulus generalization).
                "cue_value": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
                # Scalar context (A/B environments; gates inhibition for renewal).
                "context": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
                # Global ambient light L(t) in [0,1] (day/night sun; 0 = midnight).
                "ambient_light": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
                "food_contact": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
                "last_consequence": spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32),
            }
        )

        # State populated in reset().
        self.position = np.zeros(2, dtype=np.float64)
        self.food_pos = np.zeros(2, dtype=np.float64)
        self.danger_pos = np.zeros(2, dtype=np.float64)
        self.light_pos = np.zeros(2, dtype=np.float64)
        self.cue_pos = np.zeros(2, dtype=np.float64)
        self.barriers: set[tuple[int, int]] = set()
        self.food_biomass = self.config.food_carrying_capacity
        self.t = 0
        self.last_consequence = 0.0
        self._window = None  # pygame surface, lazily created

    # ------------------------------------------------------------------ #
    # Gymnasium API
    # ------------------------------------------------------------------ #
    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        options = options or {}

        # Allow per-episode overrides used by the demos.
        if "food_reinforces" in options:
            self.food_reinforces = bool(options["food_reinforces"])
        if "food_present" in options:
            self.food_present = bool(options["food_present"])
        if "cue_value" in options:
            self.cue_value = float(options["cue_value"])
        if "context" in options:
            self.context = float(options["context"])
        if "food_phase_window" in options:
            self.food_phase_window = options["food_phase_window"]

        rng = self.np_random
        g = self.grid_size

        def rand_cell() -> np.ndarray:
            return rng.integers(0, g, size=2).astype(np.float64)

        # Place sources and organism on distinct cells.
        taken: set[tuple[int, int]] = set()

        def place() -> np.ndarray:
            while True:
                c = rand_cell()
                key = (int(c[0]), int(c[1]))
                if key not in taken:
                    taken.add(key)
                    return c

        explicit = options.get("layout", {})
        self.position = np.asarray(explicit.get("position", place()), dtype=np.float64)
        self.food_pos = np.asarray(explicit.get("food", place()), dtype=np.float64)
        self.danger_pos = np.asarray(explicit.get("danger", place()), dtype=np.float64)
        self.light_pos = np.asarray(explicit.get("light", place()), dtype=np.float64)
        self.cue_pos = np.asarray(explicit.get("cue", place()), dtype=np.float64)
        self.barriers = set(explicit.get("barriers", set()))

        self.food_biomass = self.config.food_carrying_capacity
        self.t = 0
        self.last_consequence = 0.0

        obs = self._build_observation()
        info = self._build_info()
        if self.render_mode == "human":
            self.render()
        return obs, info

    def step(
        self, action: int
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        action = int(action)

        if action in _MOVES:
            self._attempt_move(_MOVES[action])
        # action 0 (no-op), 5 (consume), 6 (pause) do not move the organism.

        # Foraging world: ingestion happens on CONTACT with food (time-at-patch
        # feeding), not via a consume action. Food is a renewable resource: it
        # depletes when eaten and regrows logistically, so a spent patch becomes
        # unavailable until it regrows (interval-like / VI). The world is
        # continuous -- eating does NOT end it.
        cfg = self.config
        in_range = (self.food_present
                    and self._distance(self.position, self.food_pos) <= self.consume_radius)
        food_intake = 0.0
        if in_range and self.food_reinforces and self._food_phase_open():
            food_intake = min(cfg.food_intake_rate, self.food_biomass - cfg.food_min_biomass)
            food_intake = max(0.0, food_intake)
            self.food_biomass -= food_intake
        # Logistic regrowth toward carrying capacity, scaled by daylight (food grows
        # faster by day; the factor is 1.0 when day/night is off).
        _light, _danger_f, food_f = self._light_factors()
        k = cfg.food_carrying_capacity
        self.food_biomass += food_f * cfg.food_regrowth_rate * self.food_biomass * (
            1.0 - self.food_biomass / k
        )
        self.food_biomass = float(np.clip(self.food_biomass, cfg.food_min_biomass, k))
        at_food = food_intake > 0.0

        # Danger contact: binary for now (Phase 5 will gate detection by light).
        danger_contact = (
            1.0 if self._distance(self.position, self.danger_pos) <= self.consume_radius else 0.0
        )

        self.last_consequence = float(at_food) - danger_contact
        self.t += 1

        # No environmental terminal state: the world is endless. Death (energy
        # depletion) is the organism's terminal condition, handled by the loop.
        terminated = False
        truncated = self.t >= self.config.max_steps

        obs = self._build_observation()
        info = self._build_info()
        info["at_food"] = at_food
        info["food_consumed"] = at_food  # fed this step (used for latency)
        info["food_intake"] = food_intake  # actual energy ingested (biomass-limited)
        info["food_biomass"] = self.food_biomass
        info["danger_contact"] = danger_contact
        info["action_name"] = ACTIONS[action]

        if self.render_mode == "human":
            self.render()
        return obs, self.last_consequence, terminated, truncated, info

    def render(self) -> np.ndarray | None:
        if self.render_mode in (None, "none"):
            return None
        return self._render_pygame()

    def close(self) -> None:
        if self._window is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self._window = None

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _attempt_move(self, delta: np.ndarray) -> None:
        target = self.position + delta
        # Clamp to grid bounds.
        target = np.clip(target, 0, self.grid_size - 1)
        if (int(target[0]), int(target[1])) in self.barriers:
            return  # blocked
        self.position = target

    @staticmethod
    def _distance(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(a - b))

    def _unit_vector(self, target: np.ndarray) -> np.ndarray:
        """Unit direction from organism toward target (zero if coincident)."""
        diff = target - self.position
        dist = np.linalg.norm(diff)
        if dist < 1e-9:
            return np.zeros(2, dtype=np.float32)
        return (diff / dist).astype(np.float32)

    def _intensity(self, target: np.ndarray) -> float:
        """Exponential distance falloff in [0, 1]; 1.0 when coincident."""
        dist = self._distance(self.position, target)
        return float(np.exp(-dist / self.sensor_range))

    def _day_phase(self) -> float:
        """Phase of the day in [0, 1): 0 = midnight, 0.5 = noon."""
        return (self.t % self.config.steps_per_day) / self.config.steps_per_day

    def _food_phase_open(self) -> bool:
        """Whether food reinforces at the current phase (time-locked food window, exp064).

        ``None`` window = always open (byte-identical to the un-windowed world).
        """
        w = self.food_phase_window
        if w is None:
            return True
        return w[0] <= self._day_phase() < w[1]

    def _light_factors(self) -> tuple[float, float, float]:
        """(ambient light L, danger-detectability factor, food visibility/growth factor).

        With ``day_night`` off, returns (1.0, 1.0, 1.0): L is full and there is no
        perceptual modulation, so behavior is unchanged. With it on, danger and food
        signals are scaled by ``floor + (1-floor)*L`` (each with its own floor), so both
        are weakest at night (L -> 0) and full by day (L = 1).
        """
        cfg = self.config
        if not cfg.day_night:
            return 1.0, 1.0, 1.0
        light = ambient_light(self.t, cfg.steps_per_day)
        danger_factor = cfg.danger_detect_floor + (1.0 - cfg.danger_detect_floor) * light
        food_factor = cfg.food_light_floor + (1.0 - cfg.food_light_floor) * light
        return light, danger_factor, food_factor

    def _build_observation(self) -> dict[str, np.ndarray]:
        # Food intensity and contact scale with biomass: a depleted patch is less
        # visible (weaker approach) and less edible (weaker consummatory drive).
        biomass_frac = self.food_biomass / self.config.food_carrying_capacity
        in_range = self._distance(self.position, self.food_pos) <= self.consume_radius
        # When food is absent there is no food signal/contact at all (a food-free
        # interval), independent of the (renewable) biomass state.
        present = 1.0 if self.food_present else 0.0
        # Time-locked food (exp064): outside the food window food is neither visible nor
        # edible, so it APPEARS at a feeding time. Any approach before the window is then
        # unambiguously anticipatory. 1.0 (no gating) when food_phase_window is None.
        food_open = 1.0 if self._food_phase_open() else 0.0
        # Day/night perceptual gating: danger is harder to detect and food harder to see
        # at night (factors are 1.0 when day/night is off).
        light, danger_f, food_f = self._light_factors()
        # Temporal stimulus control (exp064): the cue the organism conditions on IS the
        # day phase, sensed as the ambient light L(t), present everywhere (intensity 1).
        # Off => the usual spatial cue (value + distance falloff), byte-identical.
        if self.config.temporal_cue:
            cue_value = ambient_light(self.t, self.config.steps_per_day)
            cue_intensity = 1.0
        else:
            cue_value = self.cue_value
            cue_intensity = self._intensity(self.cue_pos)
        return {
            "position": self.position.astype(np.float32),
            "food_vector": self._unit_vector(self.food_pos),
            "danger_vector": self._unit_vector(self.danger_pos),
            "light_vector": self._unit_vector(self.light_pos),
            "cue_vector": self._unit_vector(self.cue_pos),
            "food_intensity": np.array(
                [self._intensity(self.food_pos) * biomass_frac * present * food_f * food_open],
                dtype=np.float32,
            ),
            "danger_intensity": np.array(
                [self._intensity(self.danger_pos) * danger_f], dtype=np.float32
            ),
            "light_intensity": np.array(
                [self._intensity(self.light_pos)], dtype=np.float32
            ),
            "cue_intensity": np.array([cue_intensity], dtype=np.float32),
            "cue_value": np.array([cue_value], dtype=np.float32),
            "context": np.array([self.context], dtype=np.float32),
            "ambient_light": np.array([light], dtype=np.float32),
            # Contact signal within consume_radius, scaled by remaining biomass.
            "food_contact": np.array(
                [biomass_frac if (in_range and self.food_present and food_open) else 0.0],
                dtype=np.float32,
            ),
            "last_consequence": np.array([self.last_consequence], dtype=np.float32),
        }

    def _build_info(self) -> dict[str, Any]:
        return {
            "timestep": self.t,
            "position": self.position.copy(),
            "food_pos": self.food_pos.copy(),
            "danger_pos": self.danger_pos.copy(),
            "light_pos": self.light_pos.copy(),
            "cue_pos": self.cue_pos.copy(),
            "cue_value": self.cue_value,
            "distance_to_food": self._distance(self.position, self.food_pos),
            "food_reinforces": self.food_reinforces,
            "ambient_light": self._light_factors()[0],
        }

    def _render_pygame(self) -> np.ndarray | None:
        import pygame

        cell = 48
        size = self.grid_size * cell
        if self._window is None:
            pygame.init()
            if self.render_mode == "human":
                self._window = pygame.display.set_mode((size, size))
                pygame.display.set_caption("BehavioralFieldEnv")
            else:
                self._window = pygame.Surface((size, size))

        surf = pygame.Surface((size, size))
        surf.fill((20, 20, 24))

        # Grid lines.
        for i in range(self.grid_size + 1):
            pygame.draw.line(surf, (40, 40, 48), (0, i * cell), (size, i * cell))
            pygame.draw.line(surf, (40, 40, 48), (i * cell, 0), (i * cell, size))

        def to_px(pos: np.ndarray) -> tuple[int, int]:
            # Flip y so +y renders upward.
            x = int(pos[0] * cell + cell / 2)
            y = int(size - (pos[1] * cell + cell / 2))
            return x, y

        for cell_xy in self.barriers:
            pygame.draw.rect(
                surf,
                (90, 90, 100),
                pygame.Rect(cell_xy[0] * cell, size - (cell_xy[1] + 1) * cell, cell, cell),
            )

        markers = [
            (self.food_pos, (60, 200, 90)),    # food: green
            (self.danger_pos, (220, 60, 60)),  # danger: red
            (self.light_pos, (240, 220, 80)),  # light: yellow
            (self.cue_pos, (90, 140, 230)),    # cue: blue
        ]
        for pos, color in markers:
            pygame.draw.circle(surf, color, to_px(pos), cell // 4)

        # Organism: white circle.
        pygame.draw.circle(surf, (240, 240, 240), to_px(self.position), cell // 3)

        if self.render_mode == "human":
            self._window.blit(surf, (0, 0))
            pygame.event.pump()
            pygame.display.flip()
            return None

        # rgb_array
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(surf)), axes=(1, 0, 2)
        )
