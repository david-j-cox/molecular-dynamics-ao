"""Simulation loop: couple an Organism to a BehavioralFieldEnv and log everything.

Each timestep the organism senses, integrates its atoms, emits an action, and
updates its learning history from the consequence. Every timestep is logged in
long format (one row per atom) to a pandas DataFrame for later analysis.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from behavioral_md.atoms import STIMULI
from behavioral_md.config import SimulationConfig
from behavioral_md.environments import BehavioralFieldEnv
from behavioral_md.organism import Organism

# Per-episode reset options as a function of episode index (e.g. toggle
# reinforcement for extinction, or set the cue value for generalization).
OptionsFn = Callable[[int], dict[str, Any]]


@dataclass
class DataLogger:
    """Accumulates long-format timestep rows and writes them to disk."""

    rows: list[dict[str, Any]] = field(default_factory=list)

    def log_step(
        self,
        organism: Organism,
        episode: int,
        timestep: int,
        position: np.ndarray,
        action: int,
        reward: float,
        terminated: bool,
        truncated: bool,
        intensities: dict[str, float],
        food_biomass: float = 0.0,
        sources: dict[str, np.ndarray] | None = None,
    ) -> None:
        comp = organism.last_components
        base = {
            "episode": episode,
            "timestep": timestep,
            "x": float(position[0]),
            "y": float(position[1]),
            "action": action,
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "food_intensity": intensities["food"],
            "danger_intensity": intensities["danger"],
            "light_intensity": intensities["light"],
            "cue_intensity": intensities["cue"],
            "energy": organism.energy,
            "alive": organism.alive,
            "food_biomass": food_biomass,
        }
        # Source positions (constant within an episode) for trajectory plots.
        if sources is not None:
            for name, pos in sources.items():
                base[f"{name}_x"] = float(pos[0])
                base[f"{name}_y"] = float(pos[1])
        for i, atom in enumerate(organism.atoms):
            row = dict(base)
            row.update(
                {
                    "atom_name": atom.name,
                    "atom_activation": atom.activation,
                    "atom_force": float(organism.last_force[i]),
                    "atom_fatigue": atom.fatigue,
                    # spec's single history-weight column = the atom's primary channel
                    "atom_history_weight": _primary_weight(atom),
                    # richer per-channel breakdown for analysis
                    **{f"hw_{s}": atom.history_weights.get(s, 0.0) for s in STIMULI},
                }
            )
            if comp is not None:
                row.update(
                    {
                        "force_sensory": float(comp.sensory[i]),
                        "force_history": float(comp.history[i]),
                        "force_motivational": float(comp.motivational[i]),
                        "force_coupling": float(comp.coupling[i]),
                    }
                )
            self.rows.append(row)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df = self.to_dataframe()
        if path.suffix == ".parquet":
            df.to_parquet(path, index=False)
        else:
            df.to_csv(path, index=False)
        return path


def _primary_weight(atom) -> float:
    """History weight for the atom's most-sensitive channel (food fallback)."""
    if atom.sensitivity:
        primary = max(atom.sensitivity, key=lambda s: abs(atom.sensitivity[s]))
        if abs(atom.sensitivity[primary]) > 0:
            return atom.history_weights.get(primary, 0.0)
    return atom.history_weights.get("food", 0.0)


def run_episode(
    env: BehavioralFieldEnv,
    organism: Organism,
    config: SimulationConfig,
    episode: int,
    logger: DataLogger | None = None,
    reset_options: dict[str, Any] | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Run one life; return a summary dict.

    The world is continuous: the episode ends only on **death** (energy
    depletion) or truncation at ``max_steps``. ``latency`` is the step of first
    food consumption (censored at ``max_steps`` if it never eats).
    """
    obs, info = env.reset(seed=seed, options=reset_options)
    organism.reset(obs)

    # Source positions are fixed within a life; capture once for trajectory plots.
    sources = {
        "food": info["food_pos"],
        "danger": info["danger_pos"],
        "light": info["light_pos"],
        "cue": info["cue_pos"],
    }

    latency = config.max_steps  # censored if food never consumed
    first_consumed = False
    n_consumed = 0
    steps = 0

    for t in range(config.max_steps):
        organism.step(obs)
        action = organism.emit_action()
        next_obs, reward, terminated, truncated, info = env.step(action)
        organism.update_history(next_obs, action, info)
        # Death is the organism's terminal condition (env has no terminal state).
        terminated = terminated or (not organism.alive)

        if logger is not None:
            logger.log_step(
                organism,
                episode=episode,
                timestep=t,
                position=obs["position"],
                action=action,
                reward=reward,
                terminated=terminated,
                truncated=truncated,
                intensities=organism.last_intensities,
                food_biomass=float(info.get("food_biomass", 0.0)),
                sources=sources,
            )

        steps = t + 1
        if info.get("food_consumed", False):
            n_consumed += 1
            if not first_consumed:
                first_consumed = True
                latency = steps
        obs = next_obs
        if terminated or truncated:
            break

    return {
        "episode": episode,
        "steps": steps,
        "consumed": first_consumed,
        "latency": latency,
        "n_consumed": n_consumed,
        "final_energy": organism.energy,
        "alive": organism.alive,
        "cause_of_death": organism.cause_of_death,
    }


def run_simulation(
    config: SimulationConfig,
    env: BehavioralFieldEnv | None = None,
    organism: Organism | None = None,
    options_fn: OptionsFn | None = None,
    logger: DataLogger | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run ``config.n_episodes`` episodes; return ``(timestep_log, episode_summary)``.

    A single organism persists across episodes so that learning history carries
    over (acquisition/extinction). ``options_fn(episode)`` supplies per-episode
    env reset options.
    """
    env = env or BehavioralFieldEnv(config)
    organism = organism or Organism(config)
    logger = logger if logger is not None else DataLogger()

    summaries: list[dict[str, Any]] = []
    for ep in range(config.n_episodes):
        reset_options = options_fn(ep) if options_fn is not None else None
        # Vary the layout per episode but reproducibly from the master seed.
        summary = run_episode(
            env, organism, config, ep, logger, reset_options, seed=config.seed + ep
        )
        summaries.append(summary)

    return logger.to_dataframe(), pd.DataFrame(summaries)
