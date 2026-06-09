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
DAYS_PER_LIFE = 4
N_LIVES = 90
N_AGENTS = 16                             # independent organisms per condition (for error bars)
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
        cue_learning_rate=0.02,           # demo amplifier (default 0.0005 learns the same, slower)
        energy_capacity=4.0,              # benign economy so lives run their full length
        move_cost=0.002,
        seed=0,
    )


def _band(phase: float, lohi) -> bool:
    return lohi[0] <= phase < lohi[1]


NBIN = 20
OFF = ((0.0, 0.12), (0.88, 1.0))                     # near midnight: far from the midday window


def run_agent(paired: bool, agent: int):
    """Run ONE independent organism over N_LIVES (its own seed for organism, env, control draws).

    The conditioned drive carries an overall offset (all weights drift positive), so the temporal
    SIGNAL is the drive at a phase relative to an OFF time (near midnight). Per life we track the
    anticipation index = mean(pre-window drive) - mean(off drive): positive only if the cue evokes
    EXTRA approach as the feeding time nears. Returns this agent's (per-life anticipation,
    off-baselined late day-profile, learned cue weights, receptor centers).
    """
    cfg = make_config()
    env = BehavioralFieldEnv(cfg)
    org = Organism(cfg, rng=np.random.default_rng(agent))
    rng = np.random.default_rng(1000 + agent)        # control-window draws, independent per agent
    profile_sum = np.zeros(NBIN)
    profile_cnt = np.zeros(NBIN)
    anticip_by_life = []

    for life in range(N_LIVES):
        window = WINDOW if paired else _rand_window(rng)
        obs, _ = env.reset(seed=agent * 1000 + life,
                           options={"layout": LAYOUT, "food_phase_window": window})
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
            if life >= N_LIVES - 30:                          # day-profile from post-learning lives
                profile_sum[min(int(phase * NBIN), NBIN - 1)] += drive
                profile_cnt[min(int(phase * NBIN), NBIN - 1)] += 1
            if _band(phase, PRE):
                pre_vals.append(drive)
            elif _band(phase, OFF[0]) or _band(phase, OFF[1]):
                off_vals.append(drive)
            if not org.alive:
                break
        base = float(np.mean(off_vals)) if off_vals else 0.0
        anticip_by_life.append((float(np.mean(pre_vals)) if pre_vals else 0.0) - base)

    profile = np.divide(profile_sum, profile_cnt, out=np.zeros(NBIN), where=profile_cnt > 0)
    profile -= profile[[0, 1, 18, 19]].mean()        # off-baseline this agent's profile
    return np.array(anticip_by_life), profile, org.cue_field.weights.copy(), org.cue_field.centers


def run_condition(paired: bool):
    """Run a fleet of N_AGENTS independent organisms; stack per-agent results for the error bars."""
    ant, prof, wts, centers = [], [], [], None
    for a in range(N_AGENTS):
        anticip, profile, weights, centers = run_agent(paired, a)
        ant.append(anticip)
        prof.append(profile)
        wts.append(weights)
        print(f"  {'paired' if paired else 'unpaired'} agent {a}: "
              f"late anticipation = {anticip[-20:].mean():+.3f}", flush=True)
    return np.array(ant), np.array(prof), np.array(wts), centers


def _mean_ci(arr2d, axis=0):
    """Mean and 95% CI half-width (1.96*SEM) across agents (arr2d = [agents, x])."""
    m = arr2d.mean(axis=axis)
    sem = arr2d.std(axis=axis, ddof=1) / np.sqrt(arr2d.shape[axis])
    return m, 1.96 * sem


def _rand_window(rng):
    # Centre spans the FULL light range (phase 0.12..0.88 maps L from ~0.13 up to 1.0 and back), so
    # across days food occurs at every light level and L stops predicting it -- a true unpaired
    # control. (Drawing only from the bright half would leave "bright = food" intact.)
    c = float(rng.uniform(0.12, 0.88))
    return (c - WINDOW_HALF, c + WINDOW_HALF)


def _smooth_rows(arr2d, k=10):
    return np.array([np.convolve(r, np.ones(k) / k, mode="valid") for r in arr2d])


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    print(f"running {N_AGENTS} agents x 2 conditions ...")
    ant_p, prof_p, w_p, centers = run_condition(paired=True)     # each [N_AGENTS, *]
    ant_u, prof_u, w_u, _ = run_condition(paired=False)

    # Headline: late anticipation index per agent -> mean +/- 95% CI across the fleet.
    late_p, late_u = ant_p[:, -20:].mean(1), ant_u[:, -20:].mean(1)
    cip = 1.96 * late_p.std(ddof=1) / np.sqrt(N_AGENTS)
    ciu = 1.96 * late_u.std(ddof=1) / np.sqrt(N_AGENTS)
    print(f"\nPAIRED   late anticipation index = {late_p.mean():+.3f} +/- {cip:.3f} (95% CI, "
          f"n={N_AGENTS})")
    print(f"UNPAIRED late anticipation index = {late_u.mean():+.3f} +/- {ciu:.3f} (95% CI, "
          f"n={N_AGENTS})")

    phase_centers = (np.arange(NBIN) + 0.5) / NBIN
    fig, ax = plt.subplots(1, 3, figsize=(16, 5.0))

    def band(axis, x, arr2d, color, label, ls="-", marker="o"):
        m, ci = _mean_ci(arr2d)
        axis.fill_between(x, m - ci, m + ci, color=color, alpha=0.18)
        axis.plot(x, m, marker=marker, ls=ls, color=color, lw=2, label=label)

    # A. within-day conditioned response (off-time baselined), mean +/- 95% CI over agents
    ax[0].axhline(0, color="0.7", lw=0.8)
    ax[0].axvspan(WINDOW[0], WINDOW[1], color="0.85", label="feeding window")
    ax[0].axvspan(PRE[0], PRE[1], color="tab:orange", alpha=0.15, label="pre-window (anticipation)")
    band(ax[0], phase_centers, prof_p, "tab:red", "paired (L predicts food)")
    band(ax[0], phase_centers, prof_u, "tab:gray", "unpaired (control)", ls="--", marker="s")
    ax[0].set_xlabel("phase of day (0=midnight, 0.5=noon)")
    ax[0].set_ylabel("conditioned drive vs off-time")
    ax[0].set_title("A. Approach ramps up THROUGH the pre-window\nto the feeding time -- only when "
                    "L predicts food")
    ax[0].legend(fontsize=7.5, loc="upper left")

    # B. anticipation across lives (learning curve), mean +/- 95% CI over agents
    sp, su = _smooth_rows(ant_p), _smooth_rows(ant_u)
    lives = np.arange(sp.shape[1])
    band(ax[1], lives, sp, "tab:red", "paired", marker="")
    band(ax[1], lives, su, "tab:gray", "unpaired", ls="--", marker="")
    ax[1].axhline(0, color="0.7", lw=0.8)
    ax[1].set_xlabel("life (10-life moving average)")
    ax[1].set_ylabel("anticipation index (pre-window − off-time)")
    ax[1].set_title("B. Anticipation EMERGES over lives\n(paired grows, unpaired stays flat)")
    ax[1].legend(fontsize=8)

    # C. learned cue-receptor weights vs light level, mean +/- 95% CI over agents
    ax[2].axhline(0, color="0.7", lw=0.8)
    band(ax[2], centers, w_p, "tab:red", "paired")
    band(ax[2], centers, w_u, "tab:gray", "unpaired", ls="--", marker="s")
    ax[2].set_xlabel("receptor centre = light level L (1.0 = noon)")
    ax[2].set_ylabel("learned association weight")
    ax[2].set_title("C. The learned weights peak at the\nhigh-light (midday) feeding time")
    ax[2].legend(fontsize=8)

    fig.suptitle(f"exp064 food anticipation -- mean ± 95% CI over {N_AGENTS} independent agents "
                 "per condition", fontsize=11, y=1.02)
    fig.tight_layout()
    out = FIG / "exp064_food_anticipation.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
