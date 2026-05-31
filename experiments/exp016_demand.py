"""Experiment 016 -- behavioral-economic demand curve (unit price).

Sweeps fixed-ratio size in the operant chamber. With fixed response effort and
reinforcer magnitude, FR size sets the UNIT PRICE = (responses per reinforcer x
effort) / magnitude. Consumption (reinforcers obtained per step) is measured and
plotted against unit price. The demand relation -- consumption falling as price
rises -- emerges from the effort-opposed press value (reinforced presses
strengthen it, unreinforced presses erode it) plus the metabolic cost of
responding; the organism never computes unit price.

Run:  python -m experiments.exp016_demand
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from behavioral_md.chamber import ChamberConfig, run_chamber
from behavioral_md.visualization import plot_demand, plot_demand_pair

N_ORG, N_STEPS = 400, 12000
FR_SIZES = [1, 2, 5, 10, 20, 50, 100, 200]


def main() -> None:
    cfg = ChamberConfig(approach_gain=1.0, motiv_strength=2.0, food_energy=0.15,
                        basal_metabolism=0.004, press_cost=0.006, energy_init=0.6,
                        restoring=1.0, temperature=0.5, learning_rate=0.1,
                        value_extinction=0.03, deficit_exponent=2.0)
    warm = N_STEPS // 2
    session = N_STEPS - warm   # steady-state window = one "session"
    prices, consumption, rates, n_consumed = [], [], [], []
    print(f"{'FR':>4} {'unit_price':>10} {'resp_rate':>10} {'cons/step':>10} {'n/session':>10}")
    for fr in FR_SIZES:
        r = run_chamber("FR", fr, cfg, N_ORG, N_STEPS, seed=0)
        rate = r["presses"][warm:].mean()
        cons = r["reinforced"][warm:].mean()               # reinforcers per step
        total = r["reinforced"][warm:].sum(axis=0).mean()  # per organism, per session
        price = fr * cfg.press_cost / cfg.food_energy
        prices.append(price)
        consumption.append(cons)
        rates.append(rate)
        n_consumed.append(total)
        print(f"{fr:>4} {price:>10.3f} {rate:>10.3f} {cons:>10.5f} {total:>10.1f}")

    # Elasticity = slope of log consumption vs log price.
    lp, lc = np.log(prices), np.log(consumption)
    slope = np.polyfit(lp, lc, 1)[0]
    print(f"\nDemand: consumption falls with unit price; overall log-log slope "
          f"(elasticity) = {slope:.2f}")
    out = Path("outputs/figures/demand_curve.png")
    plot_demand(np.array(prices), np.array(consumption), out)
    out2 = Path("outputs/figures/demand_curve_pair.png")
    plot_demand_pair(np.array(prices), np.array(consumption), np.array(n_consumed), out2)
    print(f"Wrote {out} and {out2} (session = {session} steps)")


if __name__ == "__main__":
    main()
