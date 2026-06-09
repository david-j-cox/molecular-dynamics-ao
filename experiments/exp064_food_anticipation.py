"""exp064 -- temporal stimulus control: food anticipation emerges from learning, not a clock.

The last single-organism coverage item (see ToDO.txt "DEFINITION OF DONE"). We wanted the organism
to ANTICIPATE food at a fixed time of day. The first-principles way to get anticipation is NOT to
add an internal hunger drive or an imposed clock; it is to let the organism learn that an OBSERVABLE
cue it already senses -- the day/night sun L(t) -- predicts food, via the EXISTING Rescorla-Wagner
receptors. Nothing here is a new module: the sun phase is fed in as the scalar cue value (config
``temporal_cue``), food is made available only in a phase window (``food_phase_window``) so it
appears at a feeding time, and anticipation (approach BEFORE the food appears) emerges or it does
not.

Setup. One organism lives many short lives in a small arena; food becomes visible and edible only
during a midday window (high light), so any food-seeking before the window is anticipatory (food is
not yet visible). The sun L(t) is the cue. Two conditions, same organism architecture, same total
food, differing only in whether the cue PREDICTS the food:

  PAIRED   -- the feeding window is fixed at midday every life, so L(t) reliably predicts food.
  UNPAIRED -- the feeding window is re-randomized every day, so L(t) predicts nothing (the
              explicitly-unpaired control). Same cue, same reinforcement, no contingency.

Readouts (all observable): (A) the conditioned cue drive on approach_food across the day, late in
training -- in PAIRED it should ramp up BEFORE the window (anticipation); (B) that anticipatory
drive across lives (does it grow with learning?); (C) the learned cue-receptor weights vs light
level -- in PAIRED they should peak at the high-light (midday) receptors, the learned "food time".

Caveat recorded: the sun L(t) is symmetric about noon (morning rise and evening fall pass through
the same values), so the organism conditions on light LEVEL, not a directed clock; anticipation here
is the rising-light limb driving approach before the window. Distinguishing anticipation from
post-window perseveration would need a monotonic phase signal (an internal circadian oscillator),
which is a separate thread -- this demo deliberately uses only the sensed light.

Run:  python experiments/exp064_food_anticipation.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from behavioral_md.config import SimulationConfig  # noqa: E402
from behavioral_md.environments.gridworld import BehavioralFieldEnv  # noqa: E402
from behavioral_md.organism import Organism  # noqa: E402

FIG = Path("outputs/figures")

STEPS_PER_DAY = 40
DAYS_PER_LIFE = 5
N_LIVES = 160
GRID = 5
WINDOW = (0.4, 0.6)                       # midday feeding window (paired)
WINDOW_HALF = 0.1                         # half-width for the randomized unpaired window
PRE = (0.30, 0.40)                        # pre-window band (food not yet visible) where the noon
#                                           association reaches via generalization = anticipation
LAYOUT = {"position": [2, 2], "food": [2, 2], "danger": [0, 0], "light": [4, 4], "cue": [4, 0]}


def make_config() -> SimulationConfig:
    return SimulationConfig(
        grid_size=GRID,
        max_steps=STEPS_PER_DAY * DAYS_PER_LIFE,
        temporal_cue=True,                # the sun L(t) is the learnable cue
        steps_per_day=STEPS_PER_DAY,
        food_phase_window=WINDOW,
        cue_learning_rate=0.01,           # demo amplifier (default 0.0005 learns the same, slower)
        energy_capacity=4.0,              # benign economy so lives run their full length
        move_cost=0.002,
        seed=0,
    )


def _band(phase: float, lohi) -> bool:
    return lohi[0] <= phase < lohi[1]


def run_condition(paired: bool):
    """Run one organism over N_LIVES.

    The conditioned drive carries an overall offset (all weights drift positive), so the temporal
    SIGNAL is the drive at a phase relative to the drive at an OFF time (near midnight, far from the
    feeding window). We track, per life, the anticipation index = mean(pre-window drive) - mean(off
    drive). It is positive only if the cue evokes EXTRA approach as the feeding time nears.
    Returns (per-life anticipation, late-training day-profile, cue weights, receptor centers).
    """
    cfg = make_config()
    env = BehavioralFieldEnv(cfg)
    org = Organism(cfg, rng=np.random.default_rng(0))
    rng = np.random.default_rng(1 if paired else 2)
    off = ((0.0, 0.12), (0.88, 1.0))                 # near midnight: far from the midday window

    nbin = 20
    profile_sum = np.zeros(nbin)
    profile_cnt = np.zeros(nbin)
    anticip_by_life = []

    for life in range(N_LIVES):
        window = WINDOW if paired else _rand_window(rng)
        obs, _ = env.reset(seed=life, options={"layout": LAYOUT, "food_phase_window": window})
        org.reset(obs)
        pre_vals, off_vals = [], []
        for t in range(cfg.max_steps):
            if not paired and t % STEPS_PER_DAY == 0:
                env.food_phase_window = _rand_window(rng)   # re-randomize each day (no contingency)
            org.step(obs)
            drive = org.last_cue_drive                       # conditioned cue->approach drive
            phase = (env.t % STEPS_PER_DAY) / STEPS_PER_DAY
            action = org.emit_action()
            obs, _r, _term, _trunc, info = env.step(action)
            org.update_history(obs, action, info)
            if life >= N_LIVES - 40:                         # day-profile from post-learning lives
                profile_sum[min(int(phase * nbin), nbin - 1)] += drive
                profile_cnt[min(int(phase * nbin), nbin - 1)] += 1
            if _band(phase, PRE):
                pre_vals.append(drive)
            elif _band(phase, off[0]) or _band(phase, off[1]):
                off_vals.append(drive)
            if not org.alive:
                break
        base = float(np.mean(off_vals)) if off_vals else 0.0
        anticip_by_life.append((float(np.mean(pre_vals)) if pre_vals else 0.0) - base)

    profile = np.divide(profile_sum, profile_cnt, out=np.zeros(nbin), where=profile_cnt > 0)
    return np.array(anticip_by_life), profile, org.cue_field.weights.copy(), org.cue_field.centers


def _rand_window(rng):
    # Centre spans the FULL light range (phase 0.12..0.88 maps L from ~0.13 up to 1.0 and back), so
    # across days food occurs at every light level and L stops predicting it -- a true unpaired
    # control. (Drawing only from the bright half would leave "bright = food" intact.)
    c = float(rng.uniform(0.12, 0.88))
    return (c - WINDOW_HALF, c + WINDOW_HALF)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    ant_p, prof_p, w_p, centers = run_condition(paired=True)
    ant_u, prof_u, w_u, _ = run_condition(paired=False)

    # Baseline-subtract the day profile by its off-time (near-midnight) bins, so the panel shows the
    # temporal STRUCTURE (the offset from overall positive weights is removed for both conditions).
    off_bins = [0, 1, 18, 19]
    prof_p = prof_p - prof_p[off_bins].mean()
    prof_u = prof_u - prof_u[off_bins].mean()
    print(f"PAIRED:   anticipation index (late) = {ant_p[-40:].mean():+.3f}; "
          f"peak-light weight = {w_p[-1]:+.3f}")
    print(f"UNPAIRED: anticipation index (late) = {ant_u[-40:].mean():+.3f}; "
          f"peak-light weight = {w_u[-1]:+.3f}")

    phase_centers = (np.arange(20) + 0.5) / 20
    fig, ax = plt.subplots(1, 3, figsize=(16, 5.0))

    # A. within-day conditioned response (off-time baselined), paired vs unpaired
    ax[0].axhline(0, color="0.7", lw=0.8)
    ax[0].axvspan(WINDOW[0], WINDOW[1], color="0.85", label="feeding window")
    ax[0].axvspan(PRE[0], PRE[1], color="tab:orange", alpha=0.15, label="pre-window (anticipation)")
    ax[0].plot(phase_centers, prof_p, "o-", color="tab:red", lw=2, label="paired (L predicts food)")
    ax[0].plot(phase_centers, prof_u, "s--", color="tab:gray", lw=2, label="unpaired (control)")
    ax[0].set_xlabel("phase of day (0=midnight, 0.5=noon)")
    ax[0].set_ylabel("conditioned drive vs off-time")
    ax[0].set_title("A. Approach ramps up THROUGH the pre-window\nto the feeding time -- only when "
                    "L predicts food")
    ax[0].legend(fontsize=7.5, loc="upper left")

    # B. anticipation across lives (learning curve)
    def smooth(x, k=12):
        return np.convolve(x, np.ones(k) / k, mode="valid")
    ax[1].plot(smooth(ant_p), color="tab:red", lw=2, label="paired")
    ax[1].plot(smooth(ant_u), color="tab:gray", lw=2, ls="--", label="unpaired")
    ax[1].axhline(0, color="0.7", lw=0.8)
    ax[1].set_xlabel("life (12-life moving average)")
    ax[1].set_ylabel("anticipation index (pre-window − off-time)")
    ax[1].set_title("B. Anticipation EMERGES over lives\n(paired grows, unpaired stays flat)")
    ax[1].legend(fontsize=8)

    # C. learned cue-receptor weights vs light level
    ax[2].axhline(0, color="0.7", lw=0.8)
    ax[2].plot(centers, w_p, "o-", color="tab:red", lw=2, label="paired")
    ax[2].plot(centers, w_u, "s--", color="tab:gray", lw=2, label="unpaired")
    ax[2].set_xlabel("receptor centre = light level L (1.0 = noon)")
    ax[2].set_ylabel("learned association weight")
    ax[2].set_title("C. The learned weights peak at the\nhigh-light (midday) feeding time")
    ax[2].legend(fontsize=8)

    fig.tight_layout()
    out = FIG / "exp064_food_anticipation.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
