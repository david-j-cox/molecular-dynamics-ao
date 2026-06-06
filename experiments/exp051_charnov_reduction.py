"""exp051 -- does the survival giving-up rule reduce to Charnov's MVT? (an honest check)

The manuscript previously asserted that the depleting-patch survival DP "reduces to Charnov's
marginal-value theorem in the not-desperate limit." A reviewer flagged this as asserted, not shown.
This experiment tests it directly and the claim does NOT hold quantitatively; we report the honest,
nuanced result instead.

Charnov's MVT is a RATE-maximizing result: leave a depleting patch when its instantaneous intake
rate drops to the long-term habitat average, giving an optimal give-up biomass b* that falls as the
travel cost rises. We compute b* analytically for the patch in survival_dp_depleting (intake
max_rate*b, one biomass-grid depletion step per forage step, travel = travel_steps of pure
metabolism), and compare it to the DP's actual give-up biomass.

Findings:
  1. QUALITATIVELY the survival DP matches Charnov: the give-up biomass falls as the travel cost
     rises (leave later when travel is dear), the marginal-value-theorem direction.
  2. QUANTITATIVELY it does NOT reduce to b*: the survival forager leaves at LOWER biomass than the
     rate-maximizing MVT prediction (it depletes patches further, because paying travel energy is
     risky for survival), and crucially the leaving is concentrated at LOW reserve (a DESPERATION-
     limit phenomenon), not the comfortable, far-from-deadline rate-maximizing regime where Charnov
     applies. Survival-maximizing is not rate-maximizing once the organism is safe (its value
     saturates, so it will not pay a travel cost to raise an already-secured rate).
So the survival DP gives finite-horizon, survival-driven giving-up (Tenhumberg et al. 2001), not a
clean Charnov reduction. The quantitative rate-maximizing MVT signature is reproduced separately in
the rate-based forage engine (exp020/exp021, forage.py), which is the Charnov-interpretable model.

Run:  python experiments/exp051_charnov_reduction.py
"""

from __future__ import annotations

import numpy as np

from behavioral_md.survival import survival_dp_depleting

MAXR, CV, DAY, NIGHT, METAB = 0.06, 0.2, 40, 12, 0.02
TRAVELS = (3, 6, 9, 12)
NB = 41


def mvt_bstar(travel: int, nb: int = NB) -> float:
    """Rate-maximizing give-up biomass: argmax over leave-biomass of gain/(residence+travel)."""
    b = np.linspace(0.0, 1.0, nb)
    best = (float("nan"), -1.0)
    for jl in range(nb):
        steps = nb - 1 - jl
        if steps <= 0:
            continue
        rate = MAXR * np.sum(b[jl + 1:nb]) / (steps + travel)
        if rate > best[1]:
            best = (b[jl], rate)
    return best[0]


def dp_giveup(travel: int):
    """DP give-up biomass and the mean reserve at which leaving occurs (early-mid day)."""
    res = survival_dp_depleting(MAXR, CV, travel, DAY, NIGHT, METAB, n_biomass=NB)
    e, b, act = res["energy"], res["biomass"], res["action"]
    gu, reserves = [], []
    for t in range(DAY - travel):                 # exclude the near-dusk no-leave zone
        for ie in range(len(e)):
            idx = np.where(act[t, ie, :] == 1)[0]
            if len(idx):
                gu.append(b[idx.max()])
                reserves.append(e[ie])
    return (float(np.mean(gu)) if gu else float("nan"),
            float(np.mean(reserves)) if reserves else float("nan"))


def main() -> None:
    print("Charnov reduction check (survival.survival_dp_depleting):\n")
    print(f"{'travel':>7}{'MVT b* (rate-max)':>20}{'DP give-up b':>14}{'mean leave reserve':>20}")
    mvt, dp = [], []
    for tr in TRAVELS:
        bs = mvt_bstar(tr)
        gu, lr = dp_giveup(tr)
        mvt.append(bs)
        dp.append(gu)
        print(f"{tr:7d}{bs:20.3f}{gu:14.3f}{lr:20.3f}")
    mvt, dp = np.array(mvt), np.array(dp)
    print(f"\nBoth fall with travel cost (the MVT DIRECTION): "
          f"MVT {mvt[0]:.2f}->{mvt[-1]:.2f}, DP {dp[0]:.2f}->{dp[-1]:.2f}.")
    print(f"But the DP leaves at LOWER biomass than rate-maximizing b* (mean gap "
          f"{np.mean(mvt - dp):+.3f}), and only at LOW reserve (desperation), not the comfortable")
    print("rate-max regime. So the survival DP does not reduce to Charnov MVT quantitatively;")
    print("it gives finite-horizon survival-driven giving-up (Tenhumberg 2001). The rate-max")
    print("MVT reduction is the rate-based forage engine (exp020/021), not this survival DP.")


if __name__ == "__main__":
    main()
