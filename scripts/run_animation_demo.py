"""Animate one organism's life as a GIF, so people can *watch* what the engine does.

Trains an organism on the controlled acquisition layout (so it has learned to approach food), then
runs one more life with full per-step logging and renders it to an animated GIF: the organism moving
through the arena to food, its recent trail, the food/danger/light/cue sources, the current action
and most-active atom, and the energy reserve over time.

The patch uses a slightly richer regrowth rate than the package default (food_regrowth_rate 0.2 vs
0.1) so the patch is self-sustaining and the organism SURVIVES the episode (a friendly showcase).
The harsher default world (where the patch is eventually exhausted and the organism starves) is the
scientifically interesting case shown elsewhere; this script is the friendly demo.

Run:  python scripts/run_animation_demo.py
Output: docs/media/organism_life.gif (committed, embedded in the README)
"""

from __future__ import annotations

from pathlib import Path

from behavioral_md.config import SimulationConfig
from behavioral_md.environments import BehavioralFieldEnv
from behavioral_md.experiment_utils import weak_innate_atoms
from behavioral_md.organism import Organism
from behavioral_md.simulation import DataLogger, run_episode
from behavioral_md.visualization import animate_life

OUT = Path("docs/media/organism_life.gif")
TRAIN_LIVES = 25
STEPS = 220
SENSOR_RANGE = 8.0
INNATE_FOOD = 0.2
REINF_ASYMPTOTE = 2.0
FOOD_REGROWTH = 0.2  # self-sustaining patch so the showcase organism survives (default is 0.1)
LAYOUT = {"position": [4, 4], "food": [4, 8], "danger": [9, 0], "light": [0, 9], "cue": [8, 4]}


def main() -> None:
    cfg = SimulationConfig(n_episodes=TRAIN_LIVES + 1, max_steps=STEPS, seed=7,
                           sensor_range=SENSOR_RANGE, reinforcement_asymptote=REINF_ASYMPTOTE,
                           food_regrowth_rate=FOOD_REGROWTH)
    env = BehavioralFieldEnv(cfg)
    org = Organism(cfg, atoms=weak_innate_atoms(INNATE_FOOD))

    print(f"Training {TRAIN_LIVES} lives on the fixed layout...")
    for ep in range(TRAIN_LIVES):
        run_episode(env, org, cfg, ep, None, {"layout": LAYOUT}, seed=7000 + ep)

    print("Recording one trained life...")
    logger = DataLogger()
    summary = run_episode(env, org, cfg, TRAIN_LIVES, logger, {"layout": LAYOUT}, seed=0)
    print(f"  latency to food: {summary['latency']}  steps: {summary['steps']}  "
          f"alive at end: {summary['alive']}  consumed: {summary['consumed']}")
    if not summary["alive"]:
        print("  (note: organism did not survive; tune FOOD_REGROWTH for a survivor demo)")

    print("Rendering GIF...")
    out = animate_life(logger.to_dataframe(), OUT, grid_size=cfg.grid_size, fps=12)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
