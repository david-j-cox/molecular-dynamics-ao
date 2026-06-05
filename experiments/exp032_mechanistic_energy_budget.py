"""exp032 -- the mechanistic energy-budget test.

Question: does the distributed atom/force/Verlet organism reproduce Caraco's energy-budget rule
(risk-prone when hungry, risk-averse when fed) from its OWN dynamics, with no imposed survival
utility? exp030 (chamber.run_risk_choice) gets the rule by choosing via a softmax over an imposed
survival utility U(E): the curvature that drives the reversal is installed by hand. The choice here
instead emerges from the real engine primitives:

  - approach-SAFE and approach-RISKY drive atoms, each with sensitivity + a learned history weight;
  - the convex motivational gain  g = mu * deficit^p  amplifying food drive when the reserve is low;
  - damped Verlet integration of the two activations;
  - softmax emission over the two activations;
  - energy bookkeeping with death at E <= 0;
  - eligibility-gated Rescorla-Wagner learning of the history weights.

The safe and risky options are matched-mean (only the variance differs), so any energy-dependent
preference is risk sensitivity, not a mean effect.

Three curves, P(choose risky) binned by current energy:
  1. imposed     -- chamber.run_risk_choice (reference; the rule is built in via U(E)).
  2. mechanistic, teaching="energy"   -- history weight tracks the energy gain. PREDICTION: flat
       ~0.5, because the convex gain scales both options equally and matched means give equal
       learned values. The engine as-shipped has no route to variance preference.
  3. mechanistic, teaching="survival" -- the learning signal is the BARE survival outcome of the
       realized draw (did this outcome keep me alive, 1/0), exactly as the model-free survival
       learner uses (survival.simulate_model_free_choice). The convexity then comes from the death
       boundary, not an imposed utility. This is the candidate faithful mechanism.

Run:  python experiments/exp032_mechanistic_energy_budget.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from behavioral_md.chamber import ChamberConfig, run_risk_choice

FIG = Path("outputs/figures")

# Matched-mean options (mean S each); only the variance differs.
S = 0.05
SAFE = [(1.0, S)]
RISKY = [(0.5, 0.0), (0.5, 2 * S)]


def run_mechanistic(teaching: str, *, n_org=4000, n_steps=400, seed=0,
                    cap=1.0, e_init=0.5, cost=0.02, n_ebins=12,
                    T=0.3, mu=2.0, p=2.0, c=10.0, m=1.0, dt=0.1,
                    lr=0.05, elig_decay=0.95, sens=0.3):
    """Atom-dynamics risk choice. Returns (risky_by_energy[n_ebins], energy_bins, survival)."""
    rng = np.random.default_rng(seed)
    sp = np.array([o[0] for o in SAFE])
    sd = np.array([o[1] for o in SAFE])
    rp = np.array([o[0] for o in RISKY])
    rd = np.array([o[1] for o in RISKY])

    E = np.full(n_org, e_init)
    alive = np.ones(n_org, bool)
    # two approach atoms: activation x(t), x(t-dt); history weight; eligibility
    a_s = np.zeros(n_org)
    a_s_prev = np.zeros(n_org)
    a_r = np.zeros(n_org)
    a_r_prev = np.zeros(n_org)
    w_s = np.zeros(n_org)
    w_r = np.zeros(n_org)
    el_s = np.zeros(n_org)
    el_r = np.zeros(n_org)

    risky_count = np.zeros(n_ebins)
    bin_count = np.zeros(n_ebins)

    def draw(probs, deltas):
        idx = (rng.random(n_org)[:, None] < np.cumsum(probs)[None, :]).argmax(axis=1)
        return deltas[idx]

    for _ in range(n_steps):
        deficit = np.clip(1.0 - E / cap, 0.0, 1.0)
        gain = mu * deficit**p

        # Force on each approach atom (cue intensity = 1; both options always present):
        # F = readiness*(sensitivity + history + motivational gain) - damping*velocity.
        v_s = (a_s - a_s_prev) / dt
        v_r = (a_r - a_r_prev) / dt
        F_s = (sens + w_s + gain) - c * v_s
        F_r = (sens + w_r + gain) - c * v_r

        na_s = np.clip(2 * a_s - a_s_prev + (F_s / m) * dt**2, -10, 10)
        na_r = np.clip(2 * a_r - a_r_prev + (F_r / m) * dt**2, -10, 10)
        a_s_prev, a_s = a_s, na_s
        a_r_prev, a_r = a_r, na_r

        el_s = elig_decay * el_s + a_s
        el_r = elig_decay * el_r + a_r

        # softmax choice over the two activations
        z = np.stack([a_s, a_r], axis=1) / T
        z -= z.max(axis=1, keepdims=True)
        ez = np.exp(z)
        p_risky = ez[:, 1] / ez.sum(axis=1)
        choose_risky = (rng.random(n_org) < p_risky) & alive

        b = np.clip(E / cap * n_ebins, 0, n_ebins - 1).astype(int)
        np.add.at(bin_count, b[alive], 1.0)
        np.add.at(risky_count, b[alive], choose_risky[alive].astype(float))

        delta = np.where(choose_risky, draw(rp, rd), draw(sp, sd))
        post = np.clip(E + delta - cost, 0.0, cap)

        if teaching == "energy":
            mag = delta / (2 * S)           # normalized energy gain (matched mean -> equal)
        elif teaching == "survival":
            mag = (post > 0).astype(float)  # bare survival of the realized draw (death boundary)
        else:
            raise ValueError(teaching)

        # eligibility-gated RW on the CHOSEN option's history weight
        ur = choose_risky & alive
        us = (~choose_risky) & alive
        w_r = np.where(ur, np.clip(w_r + lr * el_r * (mag - w_r), -5, 5), w_r)
        w_s = np.where(us, np.clip(w_s + lr * el_s * (mag - w_s), -5, 5), w_s)

        E = np.where(alive, post, E)
        alive = alive & (E > 0)

    bins = (np.arange(n_ebins) + 0.5) / n_ebins * cap
    risky_by_energy = np.divide(risky_count, bin_count,
                                out=np.full(n_ebins, np.nan), where=bin_count > 0)
    return risky_by_energy, bins, alive.mean()


def main():
    cfg = ChamberConfig(temperature=0.02)  # sharp EU softmax (exp030 setting; default 0.5 is flat)
    # Harsh economy so the LOW-energy regime (where risk matters) is actually visited and
    # lethal: per-step cost slightly above the matched mean S, so safe alone slowly starves
    # and a low organism must gamble to recover. cost shared across all three curves.
    COST, EINIT, NSTEP, NORG = 0.05, 0.5, 1000, 5000
    # imposed-utility reference (the rule built in via U(E))
    imp = run_risk_choice(SAFE, RISKY, cfg, n_org=NORG, n_steps=NSTEP, seed=0,
                          cost=COST, e_init=EINIT)
    imp_curve = np.asarray(imp["risky_by_energy"], float)
    imp_bins = np.asarray(imp["energy_bins"], float)

    en_curve, bins, en_surv = run_mechanistic("energy", n_org=NORG, n_steps=NSTEP,
                                              cost=COST, e_init=EINIT)
    su_curve, _, su_surv = run_mechanistic("survival", n_org=NORG, n_steps=NSTEP,
                                           cost=COST, e_init=EINIT)

    R = 0.5  # e_req used by the imposed reference; the nominal requirement on this scale
    print("P(risky) JUST below vs JUST above the requirement (E = 0.5), and the reversal:")
    def adjacent(curve, b):
        lo = np.nanmean(curve[(b >= R - 0.12) & (b < R)])
        hi = np.nanmean(curve[(b >= R) & (b < R + 0.12)])
        return lo, hi
    for name, curve, bb in [("imposed", imp_curve, imp_bins),
                            ("mechanistic/energy", en_curve, bins),
                            ("mechanistic/survival", su_curve, bins)]:
        lo, hi = adjacent(curve, bb)
        print(f"  {name:22s}  below {lo:.3f}  above {hi:.3f}  reversal {lo - hi:+.3f}")
    print(f"survival fraction -- energy-teaching: {en_surv:.2f}   survival-teaching: {su_surv:.2f}")
    np.set_printoptions(precision=2, suppress=True)
    print("full curves (P_risky by energy bin):")
    print("  bins    ", np.round(bins, 2))
    print("  imposed ", np.round(imp_curve, 2))
    print("  mech/en ", np.round(en_curve, 2))
    print("  mech/su ", np.round(su_curve, 2))

    plt.figure(figsize=(7, 4.5))
    plt.axvline(R, color="0.7", ls="--", lw=1, label="requirement R")
    plt.plot(imp_bins, imp_curve, "o-", label="imposed utility (exp030)")
    plt.plot(bins, en_curve, "s-", label="mechanistic, energy teaching")
    plt.plot(bins, su_curve, "^-", label="mechanistic, survival teaching")
    plt.xlabel("current energy reserve E")
    plt.ylabel("P(choose risky)")
    plt.title("Mechanistic energy-budget test: does the rule emerge from atom dynamics?")
    plt.ylim(-0.02, 1.02)
    plt.legend(fontsize=8)
    plt.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    out = FIG / "exp032_mechanistic_energy_budget.png"
    plt.savefig(out, dpi=130)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
