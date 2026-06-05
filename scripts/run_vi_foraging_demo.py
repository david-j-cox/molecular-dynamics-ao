"""Animate VI-schedule foraging: the patch depletes when eaten and regrows on interval timing.

The food patch follows logistic depletion/regrowth (it is eaten down, then regrows toward carrying
capacity over time), which behaves like a variable-interval (VI) schedule: reinforcement (a full
patch) becomes available again only after enough time has passed, so over-visiting a spent patch
does not pay. With a slow-enough regrowth rate and a low biomass floor, the patch visibly empties
and refills, and the trained organism settles into the VI foraging pattern: harvest the patch, leave
as it depletes and its food signal fades, return once it has regrown.

The animation shows the patch state directly: the food marker shrinks as biomass is consumed and
grows back as it regrows, and the right panel overlays food biomass (green) on the energy reserve
(purple).

Run:  python scripts/run_vi_foraging_demo.py
Output: docs/media/organism_vi_foraging.gif
"""

from __future__ import annotations

from pathlib import Path

from behavioral_md.config import SimulationConfig
from behavioral_md.environments import BehavioralFieldEnv
from behavioral_md.experiment_utils import weak_innate_atoms
from behavioral_md.organism import Organism
from behavioral_md.simulation import DataLogger, run_episode
from behavioral_md.visualization import animate_life

OUT = Path("docs/media/organism_vi_foraging.gif")
TRAIN_LIVES = 25
STEPS = 260
SENSOR_RANGE = 8.0
INNATE_FOOD = 0.2
REINF_ASYMPTOTE = 2.0
# VI patch: slow logistic regrowth + low floor so the patch visibly empties and refills on interval
# timing, while staying rich enough for the organism to survive by spacing its visits.
FOOD_REGROWTH = 0.14
FOOD_MIN_BIOMASS = 0.05
FOOD_INTAKE_RATE = 0.06
LAYOUT = {"position": [4, 4], "food": [4, 8], "danger": [9, 0], "light": [0, 9], "cue": [8, 4]}


def main() -> None:
    cfg = SimulationConfig(n_episodes=TRAIN_LIVES + 1, max_steps=STEPS, seed=7,
                           sensor_range=SENSOR_RANGE, reinforcement_asymptote=REINF_ASYMPTOTE,
                           food_regrowth_rate=FOOD_REGROWTH, food_min_biomass=FOOD_MIN_BIOMASS,
                           food_intake_rate=FOOD_INTAKE_RATE)
    env = BehavioralFieldEnv(cfg)
    org = Organism(cfg, atoms=weak_innate_atoms(INNATE_FOOD))

    print(f"Training {TRAIN_LIVES} lives on the VI patch...")
    for ep in range(TRAIN_LIVES):
        run_episode(env, org, cfg, ep, None, {"layout": LAYOUT}, seed=7000 + ep)

    print("Recording one trained life...")
    logger = DataLogger()
    summary = run_episode(env, org, cfg, TRAIN_LIVES, logger, {"layout": LAYOUT}, seed=0)
    df = logger.to_dataframe()
    bm = df.groupby("timestep")["food_biomass"].first()
    print(f"  latency to food: {summary['latency']}  steps: {summary['steps']}  "
          f"alive at end: {summary['alive']}  biomass range: {bm.min():.2f}-{bm.max():.2f}")

    print("Rendering GIF...")
    out = animate_life(df, OUT, grid_size=cfg.grid_size, fps=12)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
