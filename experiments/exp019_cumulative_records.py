"""Experiment 019 -- cumulative records (the classic operant visualization).

Plots cumulative responses vs time for a few artificial organisms, with reinforcer
deliveries marked as pips -- the traditional way to see schedule performance:
  FI (time clock, LeT): SCALLOPS -- flat after each reinforcer, accelerating into
    the next (concave rises with pips at the tops).
  FR (count clock, LeT): BREAK-AND-RUN -- a flat post-reinforcement pause then a
    steep run to the next reinforcer (a staircase).

Run:  python -m experiments.exp019_cumulative_records
"""

from __future__ import annotations

from pathlib import Path

from behavioral_md.chamber import ChamberConfig, run_chamber
from behavioral_md.visualization import plot_cumulative_record

N_ORG, N_STEPS = 50, 14000
N_SHOW, WINDOW = 3, 350
FI_PARAM, FR_PARAM = 50, 20


def fi_cfg():
    return ChamberConfig(timing_model="let", timing_clock="time", approach_gain=1.0,
                         motiv_strength=0.0, learning_rate=0.0, restoring=1.0,
                         temperature=0.3, emission_bias=1.3, act_tau=3.0, timing_gain=4.0,
                         timing_states=80, timing_lr=0.15, let_flow=0.25)


def fr_cfg():
    return ChamberConfig(timing_model="let", timing_clock="count", approach_gain=1.0,
                         motiv_strength=0.0, learning_rate=0.1, value_extinction=0.04,
                         restoring=1.0, temperature=0.3, emission_bias=0.7, act_tau=3.0,
                         timing_gain=4.0, timing_states=60, timing_lr=0.15, let_flow=0.4)


def main() -> None:
    warm = N_STEPS - WINDOW   # steady-state window at the end

    r = run_chamber("FI", FI_PARAM, fi_cfg(), N_ORG, N_STEPS, seed=0)
    plot_cumulative_record(r["presses"][warm:, :N_SHOW], r["reinforced"][warm:, :N_SHOW],
                           Path("outputs/figures/cumrec_fi.png"),
                           ylabel=f"Cumulative Responses\n(FI-{FI_PARAM}, 3 AOs offset)")
    print("Wrote outputs/figures/cumrec_fi.png  (FI scallops)")

    r = run_chamber("FR", FR_PARAM, fr_cfg(), N_ORG, N_STEPS, seed=0)
    plot_cumulative_record(r["presses"][warm:, :N_SHOW], r["reinforced"][warm:, :N_SHOW],
                           Path("outputs/figures/cumrec_fr.png"),
                           ylabel=f"Cumulative Responses\n(FR-{FR_PARAM}, 3 AOs offset)")
    print("Wrote outputs/figures/cumrec_fr.png  (FR break-and-run)")


if __name__ == "__main__":
    main()
