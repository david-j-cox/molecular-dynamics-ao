"""The energy-budget rule EVOLVES -- selection, with no utility, no DP, no learning rule.

The arc so far: exp030 IMPOSED a survival utility; first_principles.py DERIVED the policy by
dynamic programming; parametric.py executed it behaviorally. Here we remove even the planner.
A population carries a HERITABLE state-dependent risk trait -- a threshold linear in time of
day, theta(t) = a + b*(t/day) -- and gambles when its reserve is below theta(t). Organisms
forage through day/night cycles and die at E<=0; the survivors reproduce with mutation
(survival.evolve_risk_policy). Selection is the bare survival dynamics; nothing rewards
"gamble when hungry".

The rule emerges anyway. The evolved threshold rises through the day (b > 0) and converges on
the DP-optimal threshold at dusk, where the decision is most consequential and selection is
strongest; it is looser at dawn, where there is all day to recover and the choice barely
affects survival. So risk-sensitivity is not something we install -- it is what survival
selects for.

Run:   python studies/risk_sensitivity/evolution.py
Saves: studies/risk_sensitivity/figures/evolved_policy.png + evolution_summary.json
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from behavioral_md.survival import (  # noqa: E402
    evolve_risk_policy,
    risk_threshold,
    survival_dp,
)

FIG = Path(__file__).parent / "figures"
DAY, NIGHT, METAB, S = 24, 24, 0.03, 0.05
SAFE = [(1.0, S)]
RISKY = [(0.5, 0.0), (0.5, 2 * S)]
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                     "axes.labelsize": 12, "xtick.labelsize": 10, "ytick.labelsize": 10,
                     "legend.fontsize": 10})


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    ev = evolve_risk_policy(SAFE, RISKY, DAY, NIGHT, METAB, pop_size=3000,
                            n_generations=200, n_cycles=3, seed=0)
    dp = risk_threshold(survival_dp(SAFE, RISKY, DAY, NIGHT, METAB))
    gens = np.arange(len(ev["mean_a"]))
    evolved = np.array(ev["evolved_theta"])
    t = np.arange(DAY)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 4.6))

    # Left: the genome converging over generations (b = the time-of-day slope).
    axL.plot(gens, ev["mean_a"], color="black", lw=1.6, label="intercept a")
    axL.plot(gens, ev["mean_b"], color="0.5", lw=1.6, ls="--", label="time-of-day slope b")
    axL.axhline(0.0, color="0.85", lw=0.6)
    axL.set_xlabel("Generation")
    axL.set_ylabel("Mean trait value")
    axL.set_title("The risk trait evolves (slope b → positive)", fontsize=11)
    axL.legend(loc="center right", frameon=False)

    # Right: the evolved threshold vs the DP optimum over the day.
    axR.plot(evolved, t, color="black", lw=1.8, marker="o", ms=4, label="Evolved threshold")
    axR.plot(dp, t, color="0.55", lw=1.8, ls="--", label="DP-optimal threshold")
    axR.set_xlabel("Risk threshold θ (gamble below)")
    axR.set_ylabel("Time of day (0 = dawn → dusk)")
    axR.set_title("Evolved policy ≈ DP optimum (tightest at dusk)", fontsize=11)
    axR.set_xlim(0, 1)
    axR.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "evolved_policy.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    (FIG / "evolution_summary.json").write_text(json.dumps(
        {"final_survival": ev["survival"][-1],
         "evolved_a": float(np.nanmean(ev["mean_a"][-20:])),
         "evolved_b": float(np.nanmean(ev["mean_b"][-20:])),
         "evolved_theta": ev["evolved_theta"], "dp_threshold": dp.tolist()}, indent=2))

    print(f"Final survival rate: {ev['survival'][-1]:.2f}")
    print(f"Evolved trait: a={np.nanmean(ev['mean_a'][-20:]):.2f}  "
          f"b={np.nanmean(ev['mean_b'][-20:]):.2f}  (b>0 => threshold rises toward dusk)")
    print("Risk threshold theta(t), evolved vs DP-optimal:")
    for tt in (0, 8, 16, 23):
        print(f"  t={tt:2d}: evolved={evolved[tt]:.2f}  DP-optimal={dp[tt]:.2f}")
    print(f"Saved {FIG/'evolved_policy.png'}")


if __name__ == "__main__":
    main()
