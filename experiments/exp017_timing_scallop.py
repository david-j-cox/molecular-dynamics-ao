"""Experiment 017 -- FI scallop across pluggable timing models.

Runs a fixed-interval schedule in the operant chamber under each toggleable
timing model (none/homeostatic, SET, BeT, LeT) and plots the within-interval
response-rate pattern. With no timer ("none") responding is flat (no scallop);
SET, BeT, and LeT each produce a scallop -- and with characteristically different
shapes (BeT more break-and-run, SET/LeT more graded acceleration). The timing
signal drives pressing (learned value off here to isolate timing) and an emission
threshold makes low drive a genuine pause.

Run:  python -m experiments.exp017_timing_scallop
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from behavioral_md.chamber import ChamberConfig, run_chamber
from behavioral_md.visualization import plot_fi_scallop

N_ORG, N_STEPS, FI = 400, 16000, 25
N_BINS = 8
MODELS = {"none": "None (homeostatic)", "set": "SET", "bet": "BeT", "let": "LeT"}


def main() -> None:
    base = dict(approach_gain=1.0, motiv_strength=0.0, learning_rate=0.0,
                restoring=1.0, temperature=0.5, emission_bias=1.2, act_tau=3.0,
                timing_gain=3.0, timing_states=30, timing_init=float(FI), timing_lr=0.15,
                set_threshold=0.6, set_width=0.12, bet_pace=0.7, let_flow=0.4)
    warm = N_STEPS // 2
    bins = np.linspace(0, 1, N_BINS + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    rates_by_model = {}
    print(f"FI-{FI} scallop, response rate by elapsed fraction ({N_ORG} organisms):")
    for key, label in MODELS.items():
        cfg = ChamberConfig(timing_model=key, **base)
        r = run_chamber("FI", FI, cfg, N_ORG, N_STEPS, seed=0)
        pr = r["presses"][warm:]
        frac = np.clip(r["time_since_reinf"][warm:] / FI, 0, 1)
        rates = np.array([pr[(frac >= bins[i]) & (frac < bins[i + 1])].mean()
                          for i in range(N_BINS)])
        rates_by_model[label] = rates
        print(f"  {label:>18}: scallop (last-first) = {rates[-1]-rates[0]:+.3f}")

    out = Path("outputs/figures/fi_scallop.png")
    plot_fi_scallop(centers, rates_by_model, out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
