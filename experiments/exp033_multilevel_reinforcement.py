"""exp033 -- multilevel reinforcement (approach A): the reinforcer is the only currency.

Follow-up to exp032 (the energy-budget rule does NOT emerge from per-step, mean-based learning).
Here we test the multilevel-signaling idea: survival is NEVER represented as a value. The only
signal is contact with food (the primary reinforcer); a choice is reinforced by the contacts that
FOLLOW it, and death simply truncates that stream. So a choice's learned value is "does taking this,
in this state, keep the reinforcement stream going?" -- coupling the molecular reinforcer to the
molar consequence (survival) without ever writing down a survival utility.

Two nested timescales of the SAME currency (contact):
  - within-day: each contact credits the recent choices that led to it (proximal operant).
  - across-night: the eligibility trace carries over the overnight fast, so the next morning's
    contacts credit the previous day's reserve-building choices -- "survival of the night signaled
    by contact with reinforcement." An organism that dies in the night has its trace zeroed
    (truncation): its choices earn no morning credit.

State (discriminative stimuli): interoceptive energy bin x circadian day-phase. The risky option
gives a contact only on its good draw (delta>0); the safe option contacts every step. So safe has
the higher contact RATE -- the rule can only emerge if the FUTURE stream opened by surviving
outweighs that immediate-rate advantage when the organism is low.

No survival scalar, no imposed utility: only contact (+1) and death (truncation). Compare P(risky)
vs energy to the imposed-utility reference (exp030).

Run:  python experiments/exp033_multilevel_reinforcement.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from behavioral_md.chamber import ChamberConfig, run_risk_choice

FIG = Path("outputs/figures")
S = 0.05
SAFE = [(1.0, S)]
RISKY = [(0.5, 0.0), (0.5, 2 * S)]


def run_multilevel(*, n_org=4000, n_cycles=400, seed=0, cap=1.0, e_init=0.5,
                   day_steps=12, night_steps=6, day_cost=0.05, night_cost=0.05,
                   ebins=10, tbins=3, lr=0.08, decay=0.9, T=0.05, measure_last=80):
    """Approach A. Returns P(risky) by energy bin (measured), bins, survival fraction."""
    rng = np.random.default_rng(seed)
    sp = np.array([o[0] for o in SAFE])
    sd = np.array([o[1] for o in SAFE])
    rp = np.array([o[0] for o in RISKY])
    rd = np.array([o[1] for o in RISKY])
    oi = np.arange(n_org)

    # V[o, ebin, tbin, option] = learned "keeps the contact stream going" value (0..1).
    V = np.zeros((n_org, ebins, tbins, 2))
    elig = np.zeros((n_org, ebins, tbins, 2))
    E = np.full(n_org, e_init)

    risky_hits = np.zeros(ebins)
    state_visits = np.zeros(ebins)

    def ebin_of(e):
        return np.clip((e / cap * ebins).astype(int), 0, ebins - 1)

    def draw(probs, deltas):
        idx = (rng.random(n_org)[:, None] < np.cumsum(probs)[None, :]).argmax(axis=1)
        return deltas[idx]

    for cyc in range(n_cycles):
        measuring = cyc >= n_cycles - measure_last
        for ds in range(day_steps):
            tb = min(int(ds / day_steps * tbins), tbins - 1)
            eb = ebin_of(E)

            # choice: softmax over the two option-values in the current state
            q = V[oi, eb, tb, :] / T
            q -= q.max(axis=1, keepdims=True)
            ez = np.exp(q)
            p_risky = ez[:, 1] / ez.sum(axis=1)
            choose_risky = rng.random(n_org) < p_risky

            if measuring:
                np.add.at(state_visits, eb, 1.0)
                np.add.at(risky_hits, eb, choose_risky.astype(float))

            # eligibility: decay everywhere, bump the (state, chosen option) just taken
            elig *= decay
            elig[oi, eb, tb, choose_risky.astype(int)] += 1.0

            # outcome + contact (risky contacts only on the good draw)
            delta = np.where(choose_risky, draw(rp, rd), draw(sp, sd))
            contact = delta > 0.0
            # credit every recently-eligible (state, option) for THIS contact: V -> 1
            V += lr * elig * (1.0 - V) * contact[:, None, None, None]

            E = np.clip(E + delta - day_cost, 0.0, cap)
            dead = E <= 0.0
            if dead.any():                       # truncation: dead choices earn no future credit
                elig[dead] = 0.0
                E[dead] = e_init                 # respawn body; V (learning) persists

        for _ in range(night_steps):             # forced overnight fast, no choice, no contact
            elig *= decay
            E = np.clip(E - night_cost, 0.0, cap)
            dead = E <= 0.0
            if dead.any():
                elig[dead] = 0.0
                E[dead] = e_init

    bins = (np.arange(ebins) + 0.5) / ebins * cap
    risky_by_energy = np.divide(risky_hits, state_visits,
                                out=np.full(ebins, np.nan), where=state_visits > 0)
    return risky_by_energy, bins


def run_survival_signal(*, n_org=4000, n_cycles=400, seed=0, cap=1.0, e_init=0.5,
                        day_steps=12, night_steps=6, day_cost=0.05, night_cost=0.05,
                        ebins=10, tbins=3, lr=0.15, decay=0.9, T=0.05, measure_last=80,
                        predation_threshold=None, predation_prob=0.0):
    """Approach B: survival is a DAILY-scale signal that scales down into the day's choices.

    Within the day, the molecular reinforcer (food contact) builds the energy reserve. At the
    DAILY scale, "lived to see another day" is the higher-level reinforcer: surviving the cycle
    credits the day's eligible (state, choice) pairs toward 1, dying credits them toward 0. The
    survival signal is a bare environmental fact (alive at dawn / dead), not a utility function;
    it is cashed out onto the within-day choices via the day's eligibility trace. V[state, option]
    then learns ~ P(survive the cycle | this option in this state).
    """
    rng = np.random.default_rng(seed)
    sp = np.array([o[0] for o in SAFE])
    sd = np.array([o[1] for o in SAFE])
    rp = np.array([o[0] for o in RISKY])
    rd = np.array([o[1] for o in RISKY])
    oi = np.arange(n_org)

    V = np.zeros((n_org, ebins, tbins, 2))
    day_elig = np.zeros((n_org, ebins, tbins, 2))     # choices made this cycle
    E = np.full(n_org, e_init)
    risky_hits = np.zeros(ebins)
    state_visits = np.zeros(ebins)

    def ebin_of(e):
        return np.clip((e / cap * ebins).astype(int), 0, ebins - 1)

    def draw(probs, deltas):
        idx = (rng.random(n_org)[:, None] < np.cumsum(probs)[None, :]).argmax(axis=1)
        return deltas[idx]

    def credit(mask, target):
        # V += lr * day_elig * (target - V) for organisms in mask; reset their trace.
        if mask.any():
            V[mask] += lr * day_elig[mask] * (target - V[mask])
            day_elig[mask] = 0.0

    def predation(e):
        # Optional second death source: a heavier reserve is slower/more visible, so predation
        # strikes above an upper threshold. Off by default (then dead = starvation only).
        if predation_threshold is None or predation_prob <= 0.0:
            return np.zeros(len(e), bool)
        return (e > predation_threshold) & (rng.random(len(e)) < predation_prob)

    for cyc in range(n_cycles):
        measuring = cyc >= n_cycles - measure_last
        for ds in range(day_steps):
            tb = min(int(ds / day_steps * tbins), tbins - 1)
            eb = ebin_of(E)
            q = V[oi, eb, tb, :] / T
            q -= q.max(axis=1, keepdims=True)
            ez = np.exp(q)
            p_risky = ez[:, 1] / ez.sum(axis=1)
            choose_risky = rng.random(n_org) < p_risky
            if measuring:
                np.add.at(state_visits, eb, 1.0)
                np.add.at(risky_hits, eb, choose_risky.astype(float))

            day_elig *= decay
            day_elig[oi, eb, tb, choose_risky.astype(int)] += 1.0

            delta = np.where(choose_risky, draw(rp, rd), draw(sp, sd))
            E = np.clip(E + delta - day_cost, 0.0, cap)
            dead = (E <= 0.0) | predation(E)  # starvation below, predation when too high
            credit(dead, 0.0)                 # died -> the cycle's choices credited toward 0
            E[dead] = e_init

        for _ in range(night_steps):
            E = np.clip(E - night_cost, 0.0, cap)
            dead = (E <= 0.0) | predation(E)
            credit(dead, 0.0)
            E[dead] = e_init

        credit(np.ones(n_org, bool), 1.0)     # survived the whole cycle -> choices toward 1

    bins = (np.arange(ebins) + 0.5) / ebins * cap
    risky_by_energy = np.divide(risky_hits, state_visits,
                                out=np.full(ebins, np.nan), where=state_visits > 0)
    return risky_by_energy, bins


def main():
    # The requirement is EMERGENT, not a free parameter: it is the reserve needed at dusk to
    # outlast the overnight fast, R = night_cost * night_steps. The imposed reference is matched
    # to it (e_req=R) for a fair comparison.
    R = 0.05 * 6  # night_cost * night_steps -> 0.30
    cfg = ChamberConfig(temperature=0.02)
    imp = run_risk_choice(SAFE, RISKY, cfg, 4000, 1000, seed=0, cost=0.05, e_init=0.5, e_req=R)
    imp_curve = np.asarray(imp["risky_by_energy"], float)
    imp_bins = np.asarray(imp["energy_bins"], float)

    a_curve, bins = run_multilevel()
    b_curve, _ = run_survival_signal()
    def adj(curve, b):
        lo = np.nanmean(curve[b < R])
        hi = np.nanmean(curve[(b >= R) & (b < 0.9)])  # exclude the saturated cap bin
        return lo, hi
    print(f"P(risky) below vs above the EMERGENT requirement (R = {R:.2f}):")
    for name, c, b in [("imposed (exp030)", imp_curve, imp_bins),
                       ("A: contact-only currency", a_curve, bins),
                       ("B: daily survival signal", b_curve, bins)]:
        lo, hi = adj(c, b)
        print(f"  {name:26s} below {lo:.3f}  above {hi:.3f}  reversal {lo - hi:+.3f}")
    np.set_printoptions(precision=2, suppress=True)
    print("A full curve:", np.round(a_curve, 2))
    print("B full curve:", np.round(b_curve, 2))
    print("bins        :", np.round(bins, 2))

    plt.figure(figsize=(7, 4.5))
    plt.axvline(R, color="0.7", ls="--", lw=1, label=f"emergent requirement R={R:.2f}")
    plt.plot(imp_bins, imp_curve, "o-", label="imposed utility (exp030)")
    plt.plot(bins, a_curve, "s-", color="tab:orange", label="A: contact is the only currency")
    plt.plot(bins, b_curve, "^-", color="tab:red", label="B: daily survival signal scales down")
    plt.xlabel("current energy reserve E")
    plt.ylabel("P(choose risky)")
    plt.title("exp033: multilevel reinforcement -- contact-only (A) vs daily survival (B)")
    plt.ylim(-0.02, 1.02)
    plt.legend(fontsize=8)
    plt.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    out = FIG / "exp033_multilevel_reinforcement.png"
    plt.savefig(out, dpi=130)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
