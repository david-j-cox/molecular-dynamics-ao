"""Generate the standard figure set from foraging runs.

Aggregate plots (learning curve, weight acquisition) run a POPULATION of agents
in parallel across CPU cores so they show mean +/- 95% CI across agents. The
single-life plots (trajectory, energy, biomass, atom time series) use one
representative agent's full log.

Run:  python scripts/make_figures.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from behavioral_md.config import SimulationConfig
from behavioral_md.environments import BehavioralFieldEnv
from behavioral_md.organism import Organism
from behavioral_md.parallel import run_sweep
from behavioral_md.simulation import DataLogger, run_episode
from behavioral_md.visualization import (
    plot_atom_series,
    plot_energy,
    plot_food_biomass,
    plot_force_decomposition_grid,
    plot_learning_curve,
    plot_occupancy_landscape,
    plot_weight_acquisition,
)

FIG_DIR = Path("outputs/figures")
N_AGENTS = 200
N_EPISODES = 40
STEPS = 600
REP_SEED = 1  # agent whose full log feeds the single-life plots


def agent_worker(cell: dict[str, Any]) -> dict[str, Any]:
    """Run one agent for N_EPISODES lives; return per-life summaries + weights."""
    seed = cell["seed"]
    cfg = SimulationConfig(n_episodes=N_EPISODES, max_steps=STEPS, seed=seed)
    env = BehavioralFieldEnv(cfg)
    org = Organism(cfg)  # persists across lives -> learning accumulates
    afi, adi = org.index["approach_food"], org.index["avoid_danger"]
    summaries, weights = [], []
    for ep in range(N_EPISODES):
        r = run_episode(env, org, cfg, ep, None, None, seed=seed * 1000 + ep)
        summaries.append(
            {"seed": seed, "episode": ep, "n_consumed": r["n_consumed"],
             "steps": r["steps"], "latency": r["latency"]}
        )
        weights.append(
            {"seed": seed, "episode": ep,
             "approach_food.hw_food": org.atoms[afi].history_weights["food"],
             "avoid_danger.hw_danger": org.atoms[adi].history_weights["danger"]}
        )
    return {"summaries": summaries, "weights": weights}


def representative_log() -> tuple[pd.DataFrame, int]:
    """Run the representative agent with full logging; return (log, best episode)."""
    cfg = SimulationConfig(n_episodes=N_EPISODES, max_steps=STEPS, seed=REP_SEED)
    env = BehavioralFieldEnv(cfg)
    org = Organism(cfg)
    logger = DataLogger()
    steps_per_ep = []
    for ep in range(N_EPISODES):
        r = run_episode(env, org, cfg, ep, logger, None, seed=REP_SEED * 1000 + ep)
        steps_per_ep.append((ep, r["steps"]))
    best = max(steps_per_ep, key=lambda t: t[1])[0]  # longest-surviving life
    return logger.to_dataframe(), best


def main() -> None:
    print(f"Running {N_AGENTS} agents x {N_EPISODES} lives across cores...")
    results = run_sweep(agent_worker, [{"seed": s} for s in range(N_AGENTS)],
                        progress_every=50)
    summaries_df = pd.DataFrame([r for res in results for r in res["summaries"]])
    weights_df = pd.DataFrame([r for res in results for r in res["weights"]])

    rep_log, best = representative_log()
    # A mature, long-lived life for the force decomposition (learning accumulated).
    rep_steps = rep_log.groupby("episode")["timestep"].max()
    mature = rep_steps[rep_steps.index >= N_EPISODES // 2]
    mature_ep = int(mature.idxmax()) if len(mature) else best

    drive_atoms = ["approach_food", "avoid_danger", "approach_light", "orient_to_cue"]
    action_atoms = ["move_up", "move_down", "move_left", "move_right", "consume", "pause"]

    figs = [
        plot_learning_curve(summaries_df, FIG_DIR / "learning_curve.png"),
        plot_weight_acquisition(weights_df, FIG_DIR / "weight_acquisition.png"),
        plot_occupancy_landscape(rep_log, best, FIG_DIR / "occupancy_landscape.png"),
        plot_energy(rep_log, best, FIG_DIR / "energy.png"),
        plot_food_biomass(rep_log, best, FIG_DIR / "food_biomass.png"),
        plot_atom_series(rep_log, best, "atom_activation", action_atoms,
                         FIG_DIR / "activation.png", ylabel="Activation"),
        plot_atom_series(rep_log, best, "atom_force", drive_atoms,
                         FIG_DIR / "drive_force.png", ylabel="Force"),
        plot_force_decomposition_grid(
            rep_log, mature_ep,
            drive_atoms + ["consume", "pause", "explore"] + action_atoms[:4],
            FIG_DIR / "force_decomposition_grid.png",
        ),
    ]
    print(f"Population: {N_AGENTS} agents (95% CI). Representative agent seed {REP_SEED}.")
    print("Wrote figures:")
    for f in figs:
        print(f"  {f}")


if __name__ == "__main__":
    main()
