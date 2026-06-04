"""Closing the loop + a parametric sweep of the energy requirement.

Two follow-ups to first_principles.py:

1. CLOSING THE LOOP -- a behavioral population that actually lives and dies, choosing by a
   softmax over the DP-DERIVED survival values (not an imposed utility;
   survival.simulate_survival_choice). Its realized risk policy reproduces the energy-budget
   band, and its realized survival-by-starting-energy matches the planner's V(E) -- so the
   first-principles policy is executable behavior, not just a normative table.

2. PARAMETRIC requirement -- the requirement R = night_steps * metabolism is the one knob
   that sets where the band sits. Sweeping the night length, the dusk safe-suffices edge
   tracks R almost exactly: a heavier survival burden pushes the whole risk-prone band to
   higher reserves. The requirement is derived, and it is the right axis.

Run:   python studies/risk_sensitivity/parametric.py
Saves: studies/risk_sensitivity/figures/closing_the_loop.png
       studies/risk_sensitivity/figures/requirement_sweep.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from behavioral_md.survival import (  # noqa: E402
    risk_threshold,
    simulate_survival_choice,
    survival_dp,
)

FIG = Path(__file__).parent / "figures"
S, METAB, DAY = 0.05, 0.03, 24
SAFE = [(1.0, S)]
RISKY = [(0.5, 0.0), (0.5, 2 * S)]
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                     "axes.labelsize": 12, "xtick.labelsize": 10, "ytick.labelsize": 10,
                     "legend.fontsize": 10})


def closing_the_loop() -> None:
    res = survival_dp(SAFE, RISKY, DAY, 24, METAB)
    sim = simulate_survival_choice(res, SAFE, RISKY, n_org=8000, n_cycles=1, beta=40, seed=0,
                                   n_ebins=20)
    e = np.array(sim["energy_bins"])
    rby = np.array(sim["risky_by_energy"])
    sbs = np.array(sim["survival_by_start"])
    bc = np.array(sim["bin_count"])
    vbin = np.interp(e, res["energy"], res["value"])      # DP-predicted dawn survival V(E)
    rby = np.where(bc > 200, rby, np.nan)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 4.6))
    axL.plot(e, rby, color="black", marker="o", ms=5)
    axL.axhline(0.5, color="0.85", lw=0.6)
    axL.set_xlabel("Current energy")
    axL.set_ylabel("P(choose risky)")
    axL.set_title("Realized policy (behavioral)", fontsize=11)
    axL.set_ylim(0, 1)
    axR.plot(e, vbin, color="0.55", ls="--", lw=2, label="DP-predicted V(E)")
    axR.plot(e, sbs, color="black", marker="o", ms=5, label="Realized survival")
    axR.set_xlabel("Starting energy")
    axR.set_ylabel("P(survive the cycle)")
    axR.set_title("Realized survival matches the planner", fontsize=11)
    axR.set_ylim(0, 1.02)
    axR.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "closing_the_loop.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def requirement_sweep() -> None:
    nights = [8, 14, 20, 26, 32]
    Rs, edges = [], []
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 4.6))
    for i, night in enumerate(nights):
        res = survival_dp(SAFE, RISKY, DAY, night, METAB)
        thr = risk_threshold(res)
        R = res["night_requirement"]
        Rs.append(R)
        edges.append(float(np.nanmax(thr[-3:])))
        shade = str(0.15 + 0.6 * i / len(nights))
        axL.plot(thr, np.arange(DAY), lw=1.4, color=shade, label=f"R = {R:.2f}")
    axL.set_xlabel("Safe-suffices edge (energy)")
    axL.set_ylabel("Time of day (0 = dawn → dusk)")
    axL.set_title("The band's upper edge rises to R at dusk", fontsize=11)
    axL.legend(frameon=False, fontsize=9)

    axR.plot([0, 1], [0, 1], color="0.8", ls=":", lw=1)        # identity
    axR.plot(Rs, edges, color="black", marker="o", ms=6)
    axR.set_xlabel("Energy requirement  R = night steps × metabolism")
    axR.set_ylabel("Dusk safe-suffices edge")
    axR.set_title("The threshold tracks the requirement", fontsize=11)
    axR.set_xlim(0, 1)
    axR.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(FIG / "requirement_sweep.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("Dusk threshold vs requirement R:")
    for r, ed in zip(Rs, edges, strict=True):
        print(f"  R={r:.2f}  ->  dusk edge={ed:.2f}")


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    closing_the_loop()
    requirement_sweep()
    print(f"Saved {FIG/'closing_the_loop.png'} and {FIG/'requirement_sweep.png'}")


if __name__ == "__main__":
    main()
