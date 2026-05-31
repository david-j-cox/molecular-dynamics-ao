"""Experiment 018 -- FR vs VR: post-reinforcement pause and within-ratio pattern.

Uses count-based timing (the timing model ticks per RESPONSE, so it times the
ratio). On FR the reinforcer always arrives at the same count, so the learned
count->value function is sharply peaked at n: value is low just after
reinforcement (count 0) -> a long post-reinforcement pause, then acceleration to a
run. On VR the reinforcer arrives at variable counts, so the function is spread
and value-at-low-count is higher -> little pause and a higher overall rate.

Reproduces: (1) a post-reinforcement pause that is much larger on FR than VR, and
(2) higher response rates on VR than FR.

Run:  python -m experiments.exp018_fr_vr_pause
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from behavioral_md.chamber import ChamberConfig, run_chamber
from behavioral_md.visualization import plot_fi_scallop

N_ORG, N_STEPS = 600, 20000
N_BINS = 8


def cfg():
    return ChamberConfig(
        timing_model="let", timing_clock="count", approach_gain=1.0, motiv_strength=0.0,
        learning_rate=0.1, value_extinction=0.04, restoring=1.0, temperature=0.5,
        emission_bias=0.7, act_tau=3.0, timing_gain=3.0, timing_states=60,
        timing_lr=0.15, let_flow=0.4)


def post_reinforcement_pause(r, warm):
    pres, reinf = r["presses"], r["reinforced"]
    pauses = []
    for o in range(pres.shape[1]):
        ri = np.where(reinf[warm:, o])[0]
        for k in ri[:-1]:
            nxt = np.where(pres[warm + k + 1:, o])[0]
            if len(nxt):
                pauses.append(nxt[0] + 1)
    return float(np.mean(pauses)) if pauses else float("nan")


def main() -> None:
    warm = N_STEPS // 2
    bins = np.linspace(0, 1, N_BINS + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    patterns = {}
    print(f"{'sched':>6} {'rate':>7} {'PRP (steps)':>12}")
    for sched, label in [("FR", "FR-20"), ("VR", "VR-20")]:
        r = run_chamber(sched, 20, cfg(), N_ORG, N_STEPS, seed=0)
        pr = r["presses"][warm:]
        ts = r["time_since_reinf"][warm:]
        scale = np.percentile(ts, 95)
        frac = np.clip(ts / scale, 0, 1)
        patterns[label] = np.array([pr[(frac >= bins[i]) & (frac < bins[i + 1])].mean()
                                    for i in range(N_BINS)])
        print(f"{sched:>6} {pr.mean():>7.3f} {post_reinforcement_pause(r, warm):>12.2f}")

    out = Path("outputs/figures/fr_vr_pattern.png")
    plot_fi_scallop(centers, patterns, out)   # response rate vs time-since-reinforcement
    print(f"Wrote {out}  (FR: pause then run; VR: flat high rate)")


if __name__ == "__main__":
    main()
