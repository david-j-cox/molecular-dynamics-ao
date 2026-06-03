"""Experiment 025 -- decoupling probability and delay sensitivity.

exp024 decoupled rate from amount with a magnitude-utility exponent (rho). A probe of
the remaining concatenated-matching dimensions showed the pattern generalizes: every
discriminability lever (beta, COD, ~temperature) scales ALL four sensitivities
together, while each graded dimension gets its own orthogonal curvature lever --

    amount  -> amount_exponent (rho):        value tracks amount**rho
    delay   -> delay_k:                      steepness of the hyperbolic delay discount
    prob    -> probability_exponent (sigma): reinforcement gated on prob**sigma  (NEW)

delay_k already existed; sigma is the new probability-weighting exponent (nonlinear
probability weighting, cf. prospect theory). Each leaves the other sensitivities
untouched because the other sweeps hold that dimension at its neutral value
(amount=1, prob=1, delay=0 -> 1**x = 1, discount(0) = 1). Verified flat in both the
surrogate and the stochastic engine.

This experiment fits two CROSSING (a_prob, a_delay) targets -- one wants high
probability / low delay sensitivity, the other the reverse -- using sigma and delay_k
(with beta as the rate anchor), and confirms the decoupling transfers to the
stochastic engine.

Run:  python -m experiments.exp025_prob_delay_decoupling
"""

from __future__ import annotations

import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from behavioral_md.experiment_utils import fit_matching_law
from behavioral_md.fit import fit_dims
from behavioral_md.matching import MatchConfig, make_matching_sim
from behavioral_md.matching_diff import (
    _ARM_PD,
    _PATCH_CUE,
    _PATCH_POS,
    _START,
    DELAY_PAIRS,
    PROB_PAIRS,
    default_params,
    soft_sensitivities_all,
)

N_ORG, N_STEPS = 400, 5000
# Search the probability/delay curvature levers + the rate anchor.
FREE = ("beta", "probability_exponent", "delay_k")
# Crossing targets (surrogate units): high prob / low delay vs low prob / high delay.
TARGET_A = {"prob": 0.50, "delay": 0.22}
TARGET_B = {"prob": 0.15, "delay": 0.34}


def measure_stochastic_pd(mcfg, keys):
    """Stochastic (a_prob, a_delay) via the exp013/exp014 procedures (identical VI)."""
    sim, initial_state = make_matching_sim(mcfg, _PATCH_POS, _PATCH_CUE, _START)
    arm = jnp.array(_ARM_PD)
    one = jnp.ones(2)

    log_B, log_P = [], []
    for i, (pL, pR) in enumerate(PROB_PAIRS):
        s0 = initial_state(N_ORG, jax.random.key(300 + i))
        t, _c, _a = sim(s0, keys, arm, one, jnp.array([pL, pR]))
        t = np.asarray(t)
        ok = (t[:, 0] > 0) & (t[:, 1] > 0)
        log_B.append(np.log(t[ok, 0] / t[ok, 1]))
        log_P.append(np.full(ok.sum(), np.log(pL / pR)))
    a_prob, _lb, _ = fit_matching_law(np.concatenate(log_P), np.concatenate(log_B))

    log_B, log_D = [], []
    for i, (dL, dR) in enumerate(DELAY_PAIRS):
        s0 = initial_state(N_ORG, jax.random.key(400 + i))
        t, _c, _a = sim(s0, keys, arm, one, one, jnp.array([dL, dR]))
        t = np.asarray(t)
        ok = (t[:, 0] > 0) & (t[:, 1] > 0)
        log_B.append(np.log(t[ok, 0] / t[ok, 1]))
        log_D.append(np.full(ok.sum(), np.log(dL / dR)))
    slope, _lb, _ = fit_matching_law(np.concatenate(log_D), np.concatenate(log_B))
    return a_prob, -slope          # a_delay = -slope (delay enters GML negatively)


def main() -> None:
    t0 = time.perf_counter()
    base = MatchConfig()
    keys = jax.random.split(jax.random.key(0), N_STEPS)

    s0 = soft_sensitivities_all(default_params(base, FREE), base)
    sp0, sd0 = float(s0["prob"]), float(s0["delay"])
    tp0, td0 = measure_stochastic_pd(base, keys)

    rows = []
    for name, target in [("A prob^/delay_v", TARGET_A), ("B prob_v/delay^", TARGET_B)]:
        fitted, hist = fit_dims(target, free=FREE)
        best = min(hist, key=lambda h: h["loss"])
        tp, td = measure_stochastic_pd(fitted, keys)
        rows.append((name, target, (best["prob"], best["delay"]), (tp, td), fitted))
    elapsed = time.perf_counter() - t0

    print(f"Decoupled prob/delay fit (sigma for prob, delay_k for delay) in {elapsed:.1f}s")
    print(f"  free params: {', '.join(FREE)}")
    print()
    print("                     a_prob                      a_delay")
    print("                target  soft  stoch    |    target  soft  stoch")
    print(f"  baseline         --   {sp0:.2f}  {tp0:.2f}    |      --   {sd0:.2f}  {td0:.2f}")
    for name, target, (sp, sd), (tp, td), _cfg in rows:
        print(f"  {name:14s}{target['prob']:.2f}  {sp:.2f}  {tp:.2f}    |    "
              f"{target['delay']:.2f}  {sd:.2f}  {td:.2f}")
    print()
    for name, _t, _s, _st, cfg in rows:
        ps = "  ".join(f"{k}={getattr(cfg, k):.3f}" for k in FREE)
        print(f"  {name:14s} fitted: {ps}")

    # Decoupling check: A higher a_prob AND lower a_delay than B (the two move
    # independently, off the diagonal the shared discriminability lever was confined to).
    (_a, _ta, _sa, (tpA, tdA), _cA) = rows[0]
    (_b, _tb, _sb, (tpB, tdB), _cB) = rows[1]
    prob_ok = tpA > tpB
    delay_ok = tdA < tdB
    print()
    print(f"  decoupling (stochastic): a_prob   A {tpA:.2f} > B {tpB:.2f}  "
          f"{'OK' if prob_ok else 'FAIL'}")
    print(f"  decoupling (stochastic): a_delay  A {tdA:.2f} < B {tdB:.2f}  "
          f"{'OK' if delay_ok else 'FAIL'}")

    out = Path("outputs/figures/decoupled_prob_delay.png")
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    pts = [("baseline", tp0, td0, "0.4"), ("A prob^/delay_v", tpA, tdA, "black"),
           ("B prob_v/delay^", tpB, tdB, "black")]
    for label, x, y, c in pts:
        ax.scatter([x], [y], s=60, color=c, zorder=3)
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(8, 4), fontsize=10)
    ax.set_xlabel("probability sensitivity a_prob")
    ax.set_ylabel("delay sensitivity a_delay")
    ax.set_title("Independent control of probability and delay sensitivity")
    ax.axhline(td0, color="0.85", lw=0.5)
    ax.axvline(tp0, color="0.85", lw=0.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
