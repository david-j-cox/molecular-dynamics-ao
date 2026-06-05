"""exp036 -- approach B on the atom substrate: the rule through the distributed dynamics.

exp033 proved approach B as a tabular value (V[state] + softmax). exp032 showed the atom dynamics
(force -> Verlet -> softmax) alone do not produce the rule. This experiment puts the two together:
the CHOICE is produced by the real engine primitives, but the learned values are the drive atoms'
energy-state-conditioned HISTORY WEIGHTS, credited by the period-scale survival signal (B).

Per step, each of two approach atoms (safe, risky) feels the engine force
    F = sensitivity + history_weight[state] + motivational_gain - damping * velocity
with the convex motivational gain g = mu * deficit^p (shared, so it cancels between options -- the
differentiation lives in the learned, state-conditioned history weights). The activation is advanced
by the engine's own ``verlet_update``; emission is a softmax over the activations. Learning is NOT
per-step: an eligibility trace over (energy bin, day-phase, option) is credited toward 1 if the
organism survives the day/night cycle and toward 0 if it dies (starvation below, optional predation
above) -- the bare survival fact scaled down onto the molecular choices (exp033 mechanism).

FINDING: the survival-credit LEARNING ports cleanly -- the state-conditioned history weights acquire
the rule (W(risky)-W(safe) > 0 below R, < 0 just above). But the per-step switching choice does
NOT express it: the damped-Verlet activation needs sustained drive to build magnitude, while the
energy state changes every step, so the activation never reflects the current state and the softmax
choice smears to ~0.5. Expressing a molar state-dependent policy through molecular dynamics needs
TIMESCALE SEPARATION -- molecular dynamics fast relative to the molar state, which the spatial
foraging loop supplies (commit to a patch over many steps; reserve changes slowly) but an abstract
one-shot choice does not. So the faithful atom-engine demonstration requires the spatial gridworld;
this experiment isolates why. Compared to the imposed reference (exp030) and the tabular B (exp033).

Run:  python -m experiments.exp036_atom_survival_choice
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from behavioral_md.atoms import verlet_update
from behavioral_md.chamber import ChamberConfig, run_risk_choice
from experiments.exp033_multilevel_reinforcement import run_survival_signal

FIG = Path("outputs/figures")
S = 0.05
SAFE = [(1.0, S)]
RISKY = [(0.5, 0.0), (0.5, 2 * S)]


def run_atom_survival(*, n_org=4000, n_cycles=500, seed=0, cap=1.0, e_init=0.5,
                      day_steps=12, night_steps=6, day_cost=0.05, night_cost=0.05,
                      ebins=10, tbins=3, lr=0.15, elig_decay=0.9, T=0.05,
                      sens=0.3, mu=2.0, p=2.0, c=10.0, m=1.0, dt=0.1,
                      predation_threshold=None, predation_prob=0.0, measure_last=80):
    """Approach B carried by the atom force/Verlet/softmax dynamics. Returns P(risky), bins."""
    rng = np.random.default_rng(seed)
    sp = np.array([o[0] for o in SAFE])
    sd = np.array([o[1] for o in SAFE])
    rp = np.array([o[0] for o in RISKY])
    rd = np.array([o[1] for o in RISKY])
    oi = np.arange(n_org)

    # state-conditioned history weights (the learned values) and the day eligibility over them
    W = np.zeros((n_org, ebins, tbins, 2))
    day_elig = np.zeros((n_org, ebins, tbins, 2))
    # atom activations for the two approach atoms: current and previous (Verlet buffer)
    a = np.zeros((n_org, 2))
    a_prev = np.zeros((n_org, 2))
    E = np.full(n_org, e_init)
    risky_hits = np.zeros(ebins)
    state_visits = np.zeros(ebins)

    def ebin_of(e):
        return np.clip((e / cap * ebins).astype(int), 0, ebins - 1)

    def draw(probs, deltas):
        idx = (rng.random(n_org)[:, None] < np.cumsum(probs)[None, :]).argmax(axis=1)
        return deltas[idx]

    def credit(mask, target):
        if mask.any():
            W[mask] += lr * day_elig[mask] * (target - W[mask])
            day_elig[mask] = 0.0

    def predation(e):
        if predation_threshold is None or predation_prob <= 0.0:
            return np.zeros(len(e), bool)
        return (e > predation_threshold) & (rng.random(len(e)) < predation_prob)

    def death_reset(dead):
        credit(dead, 0.0)
        E[dead] = e_init
        a[dead] = 0.0
        a_prev[dead] = 0.0

    for cyc in range(n_cycles):
        measuring = cyc >= n_cycles - measure_last
        for ds in range(day_steps):
            tb = min(int(ds / day_steps * tbins), tbins - 1)
            eb = ebin_of(E)
            gain = mu * np.clip(1.0 - E / cap, 0.0, 1.0) ** p          # convex motivational gain

            # engine force on each approach atom (cue intensity = 1); history weight is state-keyed
            w_now = W[oi, eb, tb, :]                                   # [O, 2]
            vel = (a - a_prev) / dt
            force = (sens + w_now + gain[:, None]) - c * vel           # [O, 2]
            a_new = np.clip(verlet_update(a, a_prev, force, m, dt), -10, 10)
            a_prev, a = a, a_new

            z = a / T
            z -= z.max(axis=1, keepdims=True)
            ez = np.exp(z)
            p_risky = ez[:, 1] / ez.sum(axis=1)
            choose_risky = rng.random(n_org) < p_risky
            if measuring:
                np.add.at(state_visits, eb, 1.0)
                np.add.at(risky_hits, eb, choose_risky.astype(float))

            day_elig *= elig_decay
            day_elig[oi, eb, tb, choose_risky.astype(int)] += 1.0

            delta = np.where(choose_risky, draw(rp, rd), draw(sp, sd))
            E = np.clip(E + delta - day_cost, 0.0, cap)
            death_reset((E <= 0.0) | predation(E))

        for _ in range(night_steps):
            E = np.clip(E - night_cost, 0.0, cap)
            death_reset((E <= 0.0) | predation(E))

        credit(np.ones(n_org, bool), 1.0)

    bins = (np.arange(ebins) + 0.5) / ebins * cap
    risky_by_energy = np.divide(risky_hits, state_visits,
                                out=np.full(ebins, np.nan), where=state_visits > 0)
    # diagnosis: did the state-conditioned weights LEARN the rule (independent of expression)?
    w_diff = (W[:, :, :, 1] - W[:, :, :, 0]).mean(axis=(0, 2))   # risky - safe, by energy bin
    return risky_by_energy, bins, w_diff


def main():
    R = 0.05 * 6
    cfg = ChamberConfig(temperature=0.02)
    imp = run_risk_choice(SAFE, RISKY, cfg, 4000, 1000, seed=0, cost=0.05, e_init=0.5, e_req=R)
    imp_curve = np.asarray(imp["risky_by_energy"], float)
    imp_bins = np.asarray(imp["energy_bins"], float)

    tab_curve, bins = run_survival_signal(n_cycles=500, seed=0)       # B, tabular value
    atom_curve, _, w_diff = run_atom_survival(n_cycles=500, seed=0)   # B, on the atom dynamics

    def adj(curve, b):
        lo = np.nanmean(curve[b < R])
        hi = np.nanmean(curve[(b >= R) & (b < 0.9)])
        return lo, hi
    print(f"P(risky) below vs above the emergent requirement (R={R:.2f}):")
    for name, cc, bb in [("imposed (exp030)", imp_curve, imp_bins),
                         ("B tabular (exp033)", tab_curve, bins),
                         ("B on atom dynamics (exp036)", atom_curve, bins)]:
        lo, hi = adj(cc, bb)
        print(f"  {name:30s} below {lo:.3f}  above {hi:.3f}  reversal {lo - hi:+.3f}")
    np.set_printoptions(precision=2, suppress=True)
    print("atom-port choice curve:", np.round(atom_curve, 2))
    print("atom-port LEARNED weights W(risky)-W(safe) by energy bin:", np.round(w_diff, 3))
    print(f"  (>0 below R={R:.2f}: weights LEARNED 'risky when low'; if the choice curve is")
    print("   nonetheless flat, the learning ported but the EXPRESSION did not -- see note below)")
    print("NOTE: the survival-credit LEARNING ports to the atom history weights (they acquire the")
    print("rule), but a per-step state-switching choice does not EXPRESS it: the damped-Verlet")
    print("activation needs sustained drive to build magnitude, while the state changes every")
    print("step, so the activation never reflects the state and the softmax smears to ~0.5.")
    print("Expressing a molar state-dependent policy through molecular dynamics needs timescale")
    print("separation -- the spatial foraging loop (commit to a patch; reserve changes slowly).")

    plt.figure(figsize=(7, 4.5))
    plt.axvline(R, color="0.7", ls="--", lw=1, label=f"emergent requirement R={R:.2f}")
    plt.plot(imp_bins, imp_curve, "o-", color="tab:blue", label="imposed utility (exp030)")
    plt.plot(bins, tab_curve, "s-", color="tab:green", label="B tabular value (exp033)")
    plt.plot(bins, atom_curve, "^-", color="tab:red",
             label="B on atom dynamics (force -> Verlet -> softmax)")
    plt.xlabel("current energy reserve E")
    plt.ylabel("P(choose risky)")
    plt.title("exp036: the energy-budget rule through the distributed atom mechanism")
    plt.ylim(-0.02, 1.02)
    plt.legend(fontsize=8)
    plt.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    out = FIG / "exp036_atom_survival_choice.png"
    plt.savefig(out, dpi=130)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
