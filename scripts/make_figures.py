"""Generate the standard figure set from a foraging run.

Runs a single organism across many lives (history carries over), logs every
step, and writes trajectory / energy / activation / acquisition figures to
outputs/figures/.

Run:  python scripts/make_figures.py
"""

from __future__ import annotations

from pathlib import Path

from behavioral_md.config import SimulationConfig
from behavioral_md.simulation import DataLogger, run_simulation
from behavioral_md.visualization import (
    plot_atom_series,
    plot_energy,
    plot_learning_curve,
    plot_trajectory,
    plot_weight_acquisition,
)

FIG_DIR = Path("outputs/figures")


def main() -> None:
    cfg = SimulationConfig(n_episodes=40, max_steps=600, seed=1)
    logger = DataLogger()
    log, summary = run_simulation(cfg, logger=logger)
    # Use the longest-surviving life for the single-life figures (most illustrative).
    best = int(summary.sort_values("steps").iloc[-1]["episode"])

    drive_atoms = ["approach_food", "avoid_danger", "approach_light", "orient_to_cue"]
    action_atoms = ["move_up", "move_down", "move_left", "move_right", "consume", "pause"]

    figs = [
        plot_learning_curve(summary, FIG_DIR / "learning_curve.png"),
        plot_weight_acquisition(log, FIG_DIR / "weight_acquisition.png"),
        plot_trajectory(log, best, FIG_DIR / "trajectory.png"),
        plot_energy(log, best, FIG_DIR / "energy.png"),
        plot_atom_series(
            log, best, "atom_activation", action_atoms,
            FIG_DIR / "activation.png", ylabel="activation",
        ),
        plot_atom_series(
            log, best, "atom_force", drive_atoms,
            FIG_DIR / "drive_force.png", ylabel="force",
        ),
    ]

    survived = int((summary["steps"] == cfg.max_steps).sum())
    print(f"Ran {cfg.n_episodes} lives (seed {cfg.seed}); "
          f"{survived} survived the full {cfg.max_steps} steps.")
    print("Wrote figures:")
    for f in figs:
        print(f"  {f}")


if __name__ == "__main__":
    main()
