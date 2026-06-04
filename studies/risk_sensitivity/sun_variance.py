"""The day/night sun as the source of risk: variance, not a fixed hazard.

Phase 5 added a global sun but found it could not produce risk-sensitivity, because its
"danger" was a stationary, deterministic hazard -- and risk-sensitivity is about VARIANCE.
This closes that loop. The sun now sets how UNPREDICTABLE foraging is: steady in full light
(midday), erratic in the dark (dawn/dusk). Same mean intake, variance set by darkness
(survival.sun_variance_risky + survival_dp_timevarying). We compare it to a control with the
SAME average variance spread evenly over the day -- so only the TIMING of the variance differs.

The result: high-variance foraging is a LIFELINE near the deadline and a LIABILITY far from
it. The "ruin" edge -- the lowest reserve from which an organism can still gamble its way to
surviving the night -- drops near dusk under the sun (a desperate forager is saved by the
dark's high variance) but rises at dawn (where the downside has all day to bite). The sun puts
the high variance exactly at dusk, when a behind-schedule organism most needs the gamble: the
Phase 5 "starving organism accepts night risk" intuition, finally emerging for the right
reason (variance).

Run:   python studies/risk_sensitivity/sun_variance.py
Saves: studies/risk_sensitivity/figures/sun_variance.png + sun_variance_summary.json
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from behavioral_md.survival import (  # noqa: E402
    sun_variance_risky,
    survival_dp,
    survival_dp_timevarying,
)

FIG = Path(__file__).parent / "figures"
DAY, NIGHT, METAB, S = 24, 24, 0.03, 0.05
W_MIN, W_MAX = 0.02, 0.12
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                     "axes.labelsize": 12, "xtick.labelsize": 10, "ytick.labelsize": 10,
                     "legend.fontsize": 10})


def _ruin_edge(res):
    e = res["energy"]
    pol = res["policy_risky"]
    out = np.full(pol.shape[0], np.nan)
    for t in range(pol.shape[0]):
        prone = np.where(pol[t] > 0.5)[0]
        if len(prone):
            out[t] = e[prone.min()]
    return out


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    risky_sun, light = sun_variance_risky(DAY, S, W_MIN, W_MAX)
    safe_by_step = [[(1.0, S)] for _ in range(DAY)]
    spread = np.array([abs(r[1][1] - S) for r in risky_sun])
    w_const = float(np.sqrt(np.mean(spread ** 2)))               # matched AVERAGE variance

    sun = survival_dp_timevarying(safe_by_step, risky_sun, NIGHT, METAB)
    con = survival_dp([(1.0, S)], [(0.5, S - w_const), (0.5, S + w_const)], DAY, NIGHT, METAB)
    ruin_sun, ruin_con = _ruin_edge(sun), _ruin_edge(con)
    t = np.arange(DAY)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 4.6))

    # Left: the sun sets the foraging variance (high in the dark).
    axL.fill_between(t, 0, 1, where=light < 0.5, color="0.92", lw=0)   # dark periods shaded
    axL.plot(t, light, color="black", lw=1.8, label="Daylight L(t)")
    axL.plot(t, spread / spread.max(), color="0.5", lw=1.8, ls="--",
             label="Foraging spread (norm.)")
    axL.set_xlabel("Time of day (0 = dawn → dusk)")
    axL.set_ylabel("Fraction")
    axL.set_title("The sun sets foraging variance (dark = erratic)", fontsize=11)
    axL.set_ylim(0, 1.05)
    axL.legend(loc="lower center", frameon=False)

    # Right: lowest survivable reserve (ruin edge) -- the dark dusk is a lifeline.
    axR.fill_between(t, 0, 1, where=light < 0.5, color="0.92", lw=0)
    axR.plot(ruin_sun, t, color="black", lw=1.8, marker="o", ms=4,
             label="Sun-modulated variance")
    axR.plot(ruin_con, t, color="0.55", lw=1.8, ls="--", marker="s", ms=4,
             label="Constant variance (same average)")
    axR.set_xlabel("Lowest survivable reserve (ruin edge)")
    axR.set_ylabel("Time of day (0 = dawn → dusk)")
    axR.set_title("Dark dusk is a lifeline; dark dawn a liability", fontsize=11)
    axR.set_xlim(0, 0.8)
    axR.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "sun_variance.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    (FIG / "sun_variance_summary.json").write_text(json.dumps(
        {"daylight": light.tolist(), "spread": spread.tolist(), "w_const": w_const,
         "ruin_sun": ruin_sun.tolist(), "ruin_const": ruin_con.tolist()}, indent=2))

    print(f"Matched average variance: sun spread {W_MIN}-{W_MAX}, constant = {w_const:.3f}")
    print("Lowest survivable reserve (ruin edge) -- lower = desperate organism still saveable:")
    print("  time   daylight   sun     constant")
    for tt in (0, 8, 16, 20, 23):
        print(f"   {tt:2d}     {light[tt]:.2f}     {ruin_sun[tt]:.2f}     {ruin_con[tt]:.2f}")
    print("\nNear dusk the sun's high variance LOWERS the ruin edge (a lifeline); at dawn it "
          "raises it (a liability). The sun puts variance where the deadline makes it pay.")
    print(f"Saved {FIG/'sun_variance.png'}")


if __name__ == "__main__":
    main()
