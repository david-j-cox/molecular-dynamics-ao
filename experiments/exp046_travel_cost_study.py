"""exp046 -- when does spatial travel invert the energy-budget rule? (reconciling exp037 and exp045)

exp037 (1D spatial) found that spatial travel INVERTS the energy-budget rule (risk-averse when low);
exp045 (2D) found travel only WEAKENS it. This isolates the responsible variable -- the per-trip
TRAVEL COST, the energy a failed risky trip wastes -- in a fast trip-based survival-credit forager.

Each day-encounter: choose safe or risky (softmax over survival values conditioned on energy x
time-of-day), pay a travel cost t (the round trip), then receive intake (safe S; risky 0 or 2S,
matched mean). The two options have the SAME net mean (S - t); the difference is the downside: a
risky 0-draw nets -t (the trip is wasted) while a safe trip nets S - t (the trip pays off). Survival
to dawn is credited back through an eligibility trace; the night drain sets the requirement R.

Finding (honest and nuanced):
- t = 0 (no travel): the energy-budget rule -- risk-prone below R (reversal > 0).
- small t: the rule INVERTS -- a wasted risky trip is a real loss, lethal when low, so the organism
  takes the reliable safe option when low (reversal < 0). This is exp037's inversion.
- large t: a starvation-desperation regime re-emerges (the harsh economy forces gambling), so the
  realized reversal is NON-MONOTONIC in travel cost.

The clean conclusion: introducing any travel cost flips the rule toward inversion (the asymmetric
risky-downside loss); the realized magnitude is economy-dependent, because travel cost cannot be
varied independently of the overall economy and the organism's energy distribution -- there is no
clean single-parameter phase boundary. exp037 (full inversion) and exp045 (weakening) are two points
in that economy-coupled space.

Run:  python experiments/exp046_travel_cost_study.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIG = Path("outputs/figures")
S = 0.10
NIGHT_R = 0.30


def run(t, *, n_org=4000, n_cycles=500, seed=0, cap=1.0, e_init=0.5, day=6,
        ebins=10, tbins=3, lr=0.15, decay=0.9, temp=0.05, measure_last=120):
    """Trip-based survival-credit forager with per-trip travel cost t. Returns reversal, curve."""
    rng = np.random.default_rng(seed)
    oi = np.arange(n_org)
    V = np.zeros((n_org, ebins, tbins, 2))
    el = np.zeros((n_org, ebins, tbins, 2))
    E = np.full(n_org, e_init)
    rf = np.zeros(ebins)
    fd = np.zeros(ebins)

    def credit(mask, target):
        if mask.any():
            V[mask] += lr * el[mask] * (target - V[mask])
            el[mask] = 0.0

    def dead_reset(dead):
        credit(dead, 0.0)
        E[dead] = e_init

    for cyc in range(n_cycles):
        measuring = cyc >= n_cycles - measure_last
        for ds in range(day):
            tb = min(int(ds / day * tbins), tbins - 1)
            eb = np.clip((E / cap * ebins).astype(int), 0, ebins - 1)
            q = V[oi, eb, tb, :] / temp
            q -= q.max(axis=1, keepdims=True)
            ez = np.exp(q)
            risky = rng.random(n_org) < ez[:, 1] / ez.sum(axis=1)
            up = rng.random(n_org) < 0.5
            intake = np.where(risky, np.where(up, 2 * S, 0.0), S)
            E = np.clip(E + intake - t, 0.0, cap)
            el *= decay
            el[oi, eb, tb, risky.astype(int)] += 1.0
            if measuring:
                np.add.at(fd, eb, 1.0)
                np.add.at(rf, eb, risky.astype(float))
            dead_reset(E <= 0.0)
        E = np.clip(E - NIGHT_R, 0.0, cap)
        el *= decay
        dead_reset(E <= 0.0)
        credit(np.ones(n_org, bool), 1.0)

    bins = (np.arange(ebins) + 0.5) / ebins
    curve = np.divide(rf, fd, out=np.full(ebins, np.nan), where=fd > 0)
    rev = np.nanmean(curve[bins < NIGHT_R]) - np.nanmean(curve[(bins >= NIGHT_R) & (bins < 0.9)])
    return rev, curve, bins


def main() -> None:
    costs = [0.0, 0.02, 0.04, 0.06, 0.08]
    revs, curves = [], {}
    for t in costs:
        rev, curve, bins = run(t)
        revs.append(rev)
        curves[t] = curve

    print(f"Travel-cost study (matched-mean safe vs risky; requirement R={NIGHT_R}):\n")
    print(f"{'travel cost t':14s} reversal (risk-prone below R if > 0)")
    for t, r in zip(costs, revs, strict=True):
        tag = "rule" if r > 0.05 else ("inverted" if r < -0.03 else "~flat")
        print(f"  {t:<12.2f} {r:+.3f}  ({tag})")
    print("\n  t=0 -> rule; small t -> inversion (wasted risky trip is lethal when low);")
    print("  large t -> starvation-desperation regime. Non-monotonic: travel cost is not separable")
    print("  from the economy. exp037 (inversion) and exp045 (weakening) are points in this space.")

    fig, (axr, axc) = plt.subplots(1, 2, figsize=(11, 4.3))
    axr.plot(costs, revs, "o-", color="tab:purple")
    axr.axhline(0, color="0.5", ls="--", lw=1)
    axr.set_xlabel("per-trip travel cost t")
    axr.set_ylabel("reversal (below - above R)")
    axr.set_title("Travel cost flips the rule toward inversion")
    for t, col in [(0.0, "tab:green"), (0.03, "tab:red"), (0.06, "0.5")]:
        rev, curve, bins = run(t)
        axc.plot(bins, curve, "o-", color=col, label=f"t={t:.2f}")
    axc.axvline(NIGHT_R, color="0.7", ls="--", lw=1, label=f"R={NIGHT_R}")
    axc.axhline(0.5, color="0.9", lw=1)
    axc.set_xlabel("energy reserve E")
    axc.set_ylabel("P(choose risky)")
    axc.set_ylim(0, 1)
    axc.legend(fontsize=8)
    axc.set_title("Rule (t=0) -> inversion (small t)")
    fig.suptitle("exp046: spatial travel cost and the energy-budget rule", fontsize=13)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    out = FIG / "exp046_travel_cost_study.png"
    fig.savefig(out, dpi=130)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
