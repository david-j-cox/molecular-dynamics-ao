"""Tests for the episode runner and the data-logging schema (simulation.py)."""

import numpy as np

from behavioral_md.config import SimulationConfig
from behavioral_md.environments.gridworld import BehavioralFieldEnv
from behavioral_md.organism import Organism
from behavioral_md.simulation import DataLogger, run_episode


def _setup(steps=40, seed=0):
    cfg = SimulationConfig(max_steps=steps, seed=seed)
    return BehavioralFieldEnv(cfg), Organism(cfg, rng=np.random.default_rng(seed)), cfg


def test_run_episode_returns_well_formed_summary():
    env, org, cfg = _setup()
    summary = run_episode(env, org, cfg, episode=0, seed=0)
    assert set(summary) >= {"episode", "steps", "consumed", "latency", "n_consumed",
                            "final_energy", "alive", "cause_of_death"}
    assert 1 <= summary["steps"] <= cfg.max_steps
    assert summary["latency"] <= cfg.max_steps
    assert isinstance(summary["alive"], bool)
    assert 0.0 <= summary["final_energy"] <= cfg.energy_capacity


def test_datalogger_long_format_schema():
    """One row per atom per timestep, with the spec's logged columns present."""
    env, org, cfg = _setup(steps=10)
    logger = DataLogger()
    run_episode(env, org, cfg, episode=0, logger=logger, seed=0)
    df = logger.to_dataframe()
    required = {"episode", "timestep", "x", "y", "action", "reward", "terminated", "truncated",
                "food_intensity", "danger_intensity", "light_intensity", "cue_intensity", "energy",
                "atom_name", "atom_activation", "atom_force", "atom_history_weight", "atom_fatigue"}
    assert required <= set(df.columns)
    n_atoms = len(org.atoms)
    # long format: every logged timestep has exactly one row per atom
    per_step = df.groupby("timestep")["atom_name"].nunique()
    assert (per_step == n_atoms).all()
    assert df["timestep"].tolist() == sorted(df["timestep"].tolist())   # rows in time order
