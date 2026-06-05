"""exp037 -- approach B in a spatial loop, and what spatial travel does to the rule.

Follows exp036 (the survival-credit LEARNING ports to the atom weights, but a per-step
state-switching choice with the raw integrated activation cannot EXPRESS it). Two things are
resolved here:

1. EXPRESSION. The faithful readout of a movement atom is its drive (the steady-state of the damped
   accumulator), not the transient under-built activation. Orienting on the drive
   p(approach risky) = sigmoid((pull_R - pull_S)/T), with energy-state-conditioned history
   weights credited by the period-scale survival signal (B), the atom dynamics DO reproduce the
   energy-budget rule. The organism is a 1D forager; SAFE patch at x=-1 (constant), RISKY at x=+1
   (variable, matched mean); pull_p = (sens + W[state, p] + gain) * exp(-dist/range).

2. SPATIAL TRAVEL inverts the rule. The classic energy-budget rule assumes PER-STEP matched-mean
   options where a deadline is the only thing forcing risk. Real spatial foraging CONCENTRATES the
   cost of a failed gamble: a risky 0-draw means a whole trip's travel cost spent for nothing -- a
   large one-encounter energy drop that is lethal when already low. So with travel, the organism
   learns to take the reliable immediate intake (safe) when low: the reversal flips sign.

We run both conditions (intake every step vs intake only on patch contact) to isolate the effect,
against the imposed-utility reference (exp030).

Run:  python -m experiments.exp037_spatial_survival_choice
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from behavioral_md.chamber import ChamberConfig, run_risk_choice

FIG = Path("outputs/figures")
S = 0.10
SAFE = [(1.0, S)]
RISKY = [(0.5, 0.0), (0.5, 2 * S)]


def run_spatial(*, travel: bool, n_org=4000, n_cycles=700, seed=0, cap=1.0, e_init=0.5,
                day_steps=44, night_steps=12, step_cost=0.024, night_cost=0.025,
                ebins=10, tbins=3, lr=0.15, decay=0.92, T_orient=0.05,
                sens=0.3, mu=2.0, p=2.0, rng_range=1.2, move_speed=0.28, contact=0.2,
                predation_threshold=0.55, predation_prob=0.12, measure_last=150):
    """1D spatial forager; approach B with drive-readout emission. Returns P(feed risky), bins.

    travel=True: intake only on patch contact (a failed risky trip wastes the travel cost).
    travel=False: intake every step from the approached option (no travel-cost concentration).
    """
    rng = np.random.default_rng(seed)
    rp = np.array([o[0] for o in RISKY])
    rd = np.array([o[1] for o in RISKY])
    oi = np.arange(n_org)
    W = np.zeros((n_org, ebins, tbins, 2))
    elig = np.zeros((n_org, ebins, tbins, 2))
    x = np.zeros(n_org)
    E = np.full(n_org, e_init)
    risky_feeds = np.zeros(ebins)
    feeds = np.zeros(ebins)

    def ebin_of(e):
        return np.clip((e / cap * ebins).astype(int), 0, ebins - 1)

    def credit(mask, target):
        if mask.any():
            W[mask] += lr * elig[mask] * (target - W[mask])
            elig[mask] = 0.0

    def dead_reset(dead):
        credit(dead, 0.0)
        E[dead] = e_init
        x[dead] = 0.0

    def predation(e):
        return (e > predation_threshold) & (rng.random(n_org) < predation_prob)

    for cyc in range(n_cycles):
        measuring = cyc >= n_cycles - measure_last
        for ds in range(day_steps):
            tb = min(int(ds / day_steps * tbins), tbins - 1)
            eb = ebin_of(E)
            gain = mu * np.clip(1.0 - E / cap, 0.0, 1.0) ** p
            pull_s = (sens + W[oi, eb, tb, 0] + gain) * np.exp(-np.abs(x + 1.0) / rng_range)
            pull_r = (sens + W[oi, eb, tb, 1] + gain) * np.exp(-np.abs(x - 1.0) / rng_range)
            go_r = rng.random(n_org) < 1.0 / (1.0 + np.exp(-(pull_r - pull_s) / T_orient))
            x = np.clip(x + np.where(go_r, move_speed, -move_speed), -1.0, 1.0)
            elig *= decay
            elig[oi, eb, tb, go_r.astype(int)] += 1.0

            ridx = (rng.random(n_org)[:, None] < np.cumsum(rp)[None, :]).argmax(axis=1)
            intake = np.zeros(n_org)
            if travel:
                at_r = x >= 1.0 - contact
                at_s = x <= -1.0 + contact
                intake[at_r] = rd[ridx][at_r]
                intake[at_s] = S
                fed = at_r | at_s
            else:
                intake = np.where(go_r, rd[ridx], S)   # intake every step; no travel concentration
                at_r = go_r
                fed = np.ones(n_org, bool)
            if measuring and fed.any():
                fe = eb[fed]
                np.add.at(feeds, fe, 1.0)
                np.add.at(risky_feeds, fe, at_r[fed].astype(float))
            E = np.clip(E + intake - step_cost, 0.0, cap)
            if travel:
                x[fed] = 0.0                            # leave patch after feeding
            dead_reset((E <= 0.0) | predation(E))

        for _ in range(night_steps):
            E = np.clip(E - night_cost, 0.0, cap)
            elig *= decay
            dead_reset((E <= 0.0) | predation(E))
        credit(np.ones(n_org, bool), 1.0)

    bins = (np.arange(ebins) + 0.5) / ebins * cap
    risky_by_energy = np.divide(risky_feeds, feeds,
                                out=np.full(ebins, np.nan), where=feeds > 0)
    return risky_by_energy, bins


def main():
    R = 0.025 * 12  # night_cost * night_steps = 0.30
    cfg = ChamberConfig(temperature=0.02)
    imp = run_risk_choice(SAFE, RISKY, cfg, 4000, 1000, seed=0, cost=0.10, e_init=0.5, e_req=R)
    imp_curve = np.asarray(imp["risky_by_energy"], float)
    imp_bins = np.asarray(imp["energy_bins"], float)

    no_travel, bins = run_spatial(travel=False, seed=0)
    travel, _ = run_spatial(travel=True, seed=0)

    def adj(curve, b):
        lo = np.nanmean(curve[b < R])
        hi = np.nanmean(curve[(b >= R) & (b < 0.9)])
        return lo, hi
    print(f"P(feed risky) below vs above the emergent requirement (R={R:.2f}):")
    for name, cc, bb in [("imposed (exp030)", imp_curve, imp_bins),
                         ("atom dynamics, no travel", no_travel, bins),
                         ("atom dynamics, with travel", travel, bins)]:
        lo, hi = adj(cc, bb)
        print(f"  {name:28s} below {lo:.3f}  above {hi:.3f}  reversal {lo - hi:+.3f}")
    np.set_printoptions(precision=2, suppress=True)
    print("no-travel curve:", np.round(no_travel, 2))
    print("travel curve   :", np.round(travel, 2))

    plt.figure(figsize=(7, 4.5))
    plt.axvline(R, color="0.7", ls="--", lw=1, label=f"requirement R={R:.2f}")
    plt.axhline(0.5, color="0.9", lw=1)
    plt.plot(imp_bins, imp_curve, "o-", color="tab:blue", label="imposed utility (exp030)")
    plt.plot(bins, no_travel, "^-", color="tab:green",
             label="atom dynamics + survival credit (no travel): rule REPRODUCED")
    plt.plot(bins, travel, "s-", color="tab:red",
             label="+ spatial travel: rule INVERTS (failed trip wastes travel)")
    plt.xlabel("current energy reserve E")
    plt.ylabel("P(feed risky)")
    plt.title("exp037: the rule survives the atom dynamics, but spatial travel inverts it")
    plt.ylim(-0.02, 1.02)
    plt.legend(fontsize=7.5)
    plt.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    out = FIG / "exp037_spatial_survival_choice.png"
    plt.savefig(out, dpi=130)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
