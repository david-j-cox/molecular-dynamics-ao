"""Generate the standard figure set from foraging runs.

Aggregate plots (learning curve, weight acquisition) run many organisms (seeds)
so they can show mean +/- 95% CI across seeds. Single-life plots (trajectory,
energy, atom time series) use one representative organism's full log.

Run:  python scripts/make_figures.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from behavioral_md.config import SimulationConfig
from behavioral_md.environments import BehavioralFieldEnv
from behavioral_md.organism import Organism
from behavioral_md.simulation import DataLogger, run_episode
from behavioral_md.visualization import (
    plot_atom_series,
    plot_energy,
    plot_learning_curve,
    plot_trajectory,
    plot_weight_acquisition,
)

FIG_DIR = Path("outputs/figures")
N_SEEDS = 12
N_EPISODES = 40
STEPS = 600
REP_SEED = 1  # organism whose full log feeds the single-life plots


def main() -> None:
    summaries: list[dict] = []
    weights: list[dict] = []
    rep_log: pd.DataFrame | None = None

    for seed in range(N_SEEDS):
        cfg = SimulationConfig(n_episodes=N_EPISODES, max_steps=STEPS, seed=seed)
        env = BehavioralFieldEnv(cfg)
        org = Organism(cfg)  # persists across lives -> learning accumulates
        logger = DataLogger() if seed == REP_SEED else None
        afi, adi = org.index["approach_food"], org.index["avoid_danger"]

        for ep in range(N_EPISODES):
            r = run_episode(env, org, cfg, ep, logger, None, seed=seed * 1000 + ep)
            summaries.append(
                {"seed": seed, "episode": ep, "n_consumed": r["n_consumed"],
                 "steps": r["steps"], "latency": r["latency"]}
            )
            weights.append(
                {"seed": seed, "episode": ep,
                 "approach_food.hw_food": org.atoms[afi].history_weights["food"],
                 "avoid_danger.hw_danger": org.atoms[adi].history_weights["danger"]}
            )
        if seed == REP_SEED:
            rep_log = logger.to_dataframe()

    summaries_df = pd.DataFrame(summaries)
    weights_df = pd.DataFrame(weights)

    rep = summaries_df[summaries_df["seed"] == REP_SEED]
    best = int(rep.sort_values("steps").iloc[-1]["episode"])  # longest-surviving life

    drive_atoms = ["approach_food", "avoid_danger", "approach_light", "orient_to_cue"]
    action_atoms = ["move_up", "move_down", "move_left", "move_right", "consume", "pause"]

    figs = [
        plot_learning_curve(summaries_df, FIG_DIR / "learning_curve.png"),
        plot_weight_acquisition(weights_df, FIG_DIR / "weight_acquisition.png"),
        plot_trajectory(rep_log, best, FIG_DIR / "trajectory.png"),
        plot_energy(rep_log, best, FIG_DIR / "energy.png"),
        plot_atom_series(rep_log, best, "atom_activation", action_atoms,
                         FIG_DIR / "activation.png", ylabel="Activation"),
        plot_atom_series(rep_log, best, "atom_force", drive_atoms,
                         FIG_DIR / "drive_force.png", ylabel="Force"),
    ]

    survived = int((rep["steps"] == STEPS).sum())
    print(f"Ran {N_SEEDS} organisms x {N_EPISODES} lives (rep seed {REP_SEED}: "
          f"{survived}/{N_EPISODES} lives survived the full {STEPS} steps).")
    print("Wrote figures:")
    for f in figs:
        print(f"  {f}")


if __name__ == "__main__":
    main()
