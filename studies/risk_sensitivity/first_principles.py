"""Risk-sensitivity DERIVED from survival, not imposed by a utility.

exp030 reads option values through an assumed survival-shaped utility U(E); that builds in
the curvature that produces the energy-budget rule. Here the rule is derived from the bare
dynamics (energy reserve, metabolic drain, hard death at E<=0) with survival as the only
objective: a backward DP over one day (forage: safe vs risky) + night (forced fast).

The optimal policy is a risk-prone BAND in (energy, time-of-day), both edges emergent:
- upper edge = where the SAFE option already secures survival (risk-averse above); it rises
  through the day toward the night requirement R = night_steps * metabolism.
- lower edge = RUIN, where even the gamble cannot reach R (doomed either way -> indifferent);
  it rises late in the day as recovery time runs out.

So risk-proneness is bounded above (safe suffices) AND below (ruin) -- neither bound is a
parameter; both fall out of the survival problem. Contrast exp030's bump, centered on a
freely-chosen e_req.

Run:   python studies/risk_sensitivity/first_principles.py
Saves: studies/risk_sensitivity/figures/survival_policy_map.png + summary.json
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from behavioral_md.survival import risk_threshold, survival_dp  # noqa: E402

FIG = Path(__file__).parent / "figures"
DAY, NIGHT, METAB, S = 24, 24, 0.03, 0.05
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                     "axes.labelsize": 12, "xtick.labelsize": 10, "ytick.labelsize": 10,
                     "legend.fontsize": 10})


def band_edges(res: dict) -> tuple[np.ndarray, np.ndarray]:
    """Per day-step, the (lower = ruin, upper = safe-suffices) edges of the risk-prone
    zone -- the envelope of the energies at which the optimal policy gambles."""
    e = res["energy"]
    pol = res["policy_risky"]
    lo = np.full(pol.shape[0], np.nan)
    hi = np.full(pol.shape[0], np.nan)
    for t in range(pol.shape[0]):
        prone = np.where(pol[t] > 0.5)[0]
        if len(prone):
            lo[t] = e[prone.min()]
            hi[t] = e[prone.max()]
    return lo, hi


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    res = survival_dp([(1.0, S)], [(0.5, 0.0), (0.5, 2 * S)], DAY, NIGHT, METAB)
    e = res["energy"]
    thr = risk_threshold(res)
    R = res["night_requirement"]
    lo, hi = band_edges(res)
    t = np.arange(DAY)

    # The risk-prone band envelope vs time of day: ruin edge (lower) and safe-suffices
    # edge (upper). Below the band = doomed; above = safe already secures survival.
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.fill_betweenx(t, lo, hi, color="0.4", alpha=0.85, lw=0, label="risk-prone (gamble)")
    ax.plot(hi, t, color="black", lw=1.6, label="safe-suffices edge (risk-averse above)")
    ax.plot(lo, t, color="black", lw=1.2, ls="--", label="ruin edge (doomed below)")
    ax.axvline(R, color="black", ls=":", lw=1.2)
    ax.text(R, DAY - 1, " night requirement R", fontsize=10, va="top", ha="left")
    ax.set_xlabel("energy reserve")
    ax.set_ylabel("time of day (0 = dawn  ->  dusk)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, DAY - 1)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
    fig.tight_layout()
    path = FIG / "survival_policy_map.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)

    band = {}
    for t in (0, 8, 16, 23):
        prone = np.where(res["policy_risky"][t] > 0.5)[0]
        band[t] = [float(e[prone.min()]), float(e[prone.max()])] if len(prone) else None
    (FIG / "summary.json").write_text(json.dumps(
        {"night_requirement": R, "upper_edge_by_step": thr.tolist(), "band_samples": band},
        indent=2))

    print(f"Emergent night requirement R = night_steps * metabolism = {R:.2f}")
    print("Risk-prone BAND [ruin edge, safe-suffices edge] by time of day:")
    for t, b in band.items():
        msg = f"E in [{b[0]:.2f}, {b[1]:.2f}]" if b else "never risk-prone"
        print(f"  step {t:2d} ({'dusk' if t >= DAY - 3 else 'day'}): {msg}")
    print("\nUpper edge rises toward R as dusk nears; lower (ruin) edge rises as recovery "
          "time runs out. Both edges are derived, not imposed.")
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
