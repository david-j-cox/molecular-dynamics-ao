"""Nocturnal desperation foraging as realized behavior, not just an optimal table.

The companion DP study (`sun_variance.py`) showed the *policy*: under a sun that sets foraging
variance (steady at bright midday, erratic in the dark), the ruin edge -- the lowest reserve from
which gambling can still reach the night requirement -- drops near the dark dusk. Here we let a
population actually live it. Organisms are dropped into dusk behind on reserves, forage the few
remaining day-steps under the DP-optimal policy (drawing real intake from the time-varying
distributions), and then must outlast the night. We measure who survives.

The lifeline shows up as behavior: an organism behind at dusk survives the night from a LOWER
reserve under the sun's high-variance dark than under a constant-variance control with the same
average spread -- and its survival advantage peaks in the desperate band, exactly where a
behind-schedule forager has nothing to lose by gambling. The "starving organism accepts night
risk and is sometimes saved by it" intuition, realized in living and dying organisms.

Run:   python studies/risk_sensitivity/behavioral_sun.py
Saves: studies/risk_sensitivity/figures/behavioral_sun.png + behavioral_sun_summary.json
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from behavioral_md.survival import (  # noqa: E402
    simulate_dusk_survival,
    sun_variance_risky,
    survival_dp,
    survival_dp_timevarying,
)

FIG = Path(__file__).parent / "figures"
DAY, NIGHT, METAB, S = 24, 24, 0.03, 0.05
W_MIN, W_MAX = 0.02, 0.12
DUSK = 20                              # a dark, late day-step: few forage steps left
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                     "axes.labelsize": 12, "xtick.labelsize": 10, "ytick.labelsize": 10,
                     "legend.fontsize": 10})


def _e50(reserves, surv):
    """Lowest reserve from which at least half the population survives the night."""
    idx = np.where(surv >= 0.5)[0]
    return float(reserves[idx[0]]) if len(idx) else float("nan")


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    risky_sun, light = sun_variance_risky(DAY, S, W_MIN, W_MAX)
    safe_by_step = [[(1.0, S)] for _ in range(DAY)]
    spread = np.array([abs(r[1][1] - S) for r in risky_sun])
    w_const = float(np.sqrt(np.mean(spread ** 2)))               # matched AVERAGE variance

    sun = survival_dp_timevarying(safe_by_step, risky_sun, NIGHT, METAB)
    risky_con = [[(0.5, S - w_const), (0.5, S + w_const)] for _ in range(DAY)]
    con = survival_dp([(1.0, S)], [(0.5, S - w_const), (0.5, S + w_const)], DAY, NIGHT, METAB)

    reserves = np.linspace(0.30, 0.78, 25)
    sv_sun = simulate_dusk_survival(sun, safe_by_step, risky_sun, METAB, DUSK, reserves)
    sv_con = simulate_dusk_survival(con, safe_by_step, risky_con, METAB, DUSK, reserves)
    s_sun, s_con = sv_sun["survival"], sv_con["survival"]
    advantage = s_sun - s_con
    e50_sun, e50_con = _e50(reserves, s_sun), _e50(reserves, s_con)
    R = NIGHT * METAB

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 4.6))

    # Left: night survival vs dusk reserve -- the sun curve is shifted left (lives from less).
    axL.plot(reserves, s_sun, color="black", lw=2.0, marker="o", ms=4,
             label="Sun-modulated variance (dark dusk)")
    axL.plot(reserves, s_con, color="0.55", lw=2.0, ls="--", marker="s", ms=4,
             label="Constant variance (same average)")
    axL.axhline(0.5, color="0.8", lw=1.0, zorder=0)
    if not np.isnan(e50_sun):
        axL.plot([e50_sun], [0.5], "o", color="black", ms=8)
    if not np.isnan(e50_con):
        axL.plot([e50_con], [0.5], "s", color="0.55", ms=8)
    axL.set_xlabel("Reserve held at dusk")
    axL.set_ylabel("Fraction surviving the night")
    axL.set_ylim(-0.02, 1.02)
    axL.legend(loc="upper left", frameon=False)

    # Right: the survival advantage of the dark dusk, peaking in the desperate band.
    axR.fill_between(reserves, 0, advantage, where=advantage > 0, color="0.75", lw=0)
    axR.plot(reserves, advantage, color="black", lw=2.0, marker="o", ms=4)
    axR.axhline(0.0, color="0.6", lw=1.0)
    peak = reserves[int(np.argmax(advantage))]
    axR.annotate(f"+{advantage.max():.2f} at reserve {peak:.2f}",
                 xy=(peak, advantage.max()), xytext=(peak - 0.16, advantage.max() - 0.04),
                 fontsize=10, arrowprops=dict(arrowstyle="->", color="0.4"))
    axR.set_xlabel("Reserve held at dusk")
    axR.set_ylabel("Survival advantage of the dark dusk\n(sun − constant)")
    fig.tight_layout()
    fig.savefig(FIG / "behavioral_sun.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    (FIG / "behavioral_sun_summary.json").write_text(json.dumps(
        {"dusk_step": DUSK, "daylight_at_dusk": float(light[DUSK]), "w_const": w_const,
         "night_requirement": R, "reserves": reserves.tolist(),
         "survival_sun": s_sun.tolist(), "survival_const": s_con.tolist(),
         "e50_sun": e50_sun, "e50_const": e50_con,
         "peak_advantage": float(advantage.max())}, indent=2))

    print(f"Dropped into dusk (t={DUSK}, daylight {light[DUSK]:.2f}); night requirement = {R:.2f}.")
    print("Reserve needed at dusk for >=50% night survival (lower = saved from deeper trouble):")
    print(f"  sun (dark dusk) = {e50_sun:.3f}    constant = {e50_con:.3f}")
    print(f"Peak survival advantage of dark dusk: +{advantage.max():.2f} at reserve {peak:.2f}.")
    print(f"Saved {FIG/'behavioral_sun.png'}")


if __name__ == "__main__":
    main()
