"""Experiment 024 -- decoupling rate and amount sensitivity with a magnitude exponent.

exp023 fit the matching sensitivities, but found them positively COUPLED: the
organism's discriminability levers (temperature, approach_gain, beta) move a_rate and
a_amt together, so only aligned targets (both up / both down) were reachable. A probe
then showed the environmental changeover lever (patch separation, exp009) is no help --
it ALSO raises both sensitivities (a new result; exp009 only ever measured rate).

The fix is a lever that treats the two reinforcement dimensions asymmetrically: a
reinforcer-magnitude sensitivity exponent ``amount_exponent`` (rho), so the learned
value tracks ``amount**rho`` (utility curvature of magnitude). Because a_rate is
measured at equal amounts (amount=1, so amount**rho=1 for any rho), rho moves a_amt
while leaving a_rate untouched -- a clean orthogonal knob (verified: a_rate is flat to
3 decimals across rho in both the surrogate and the stochastic engine).

With beta setting a_rate and rho setting a_amt, the two sensitivities can be targeted
INDEPENDENTLY. This experiment fits two targets that CROSS -- one wants high rate /
low amount sensitivity, the other low rate / high amount -- a region the coupled levers
could never reach, and confirms the decoupling transfers to the stochastic engine.

Run:  python -m experiments.exp024_decoupled_fit
"""

from __future__ import annotations

import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from behavioral_md.experiment_utils import fit_matching_law
from behavioral_md.fit import fit
from behavioral_md.matching import MatchConfig, make_matching_sim
from behavioral_md.matching_diff import (
    _PATCH_CUE,
    _PATCH_POS,
    _START,
    AMOUNT_PAIRS,
    RATE_PAIRS,
    default_params,
    soft_sensitivities,
)

N_ORG, N_STEPS = 400, 4000
# Search all four levers: the three discriminability knobs + the magnitude exponent.
FREE = ("temperature", "approach_gain", "beta", "amount_exponent")
# Crossing targets (surrogate units): impossible on the old coupled manifold.
TARGET_A = (0.48, 0.35)   # high rate sensitivity, LOW amount sensitivity
TARGET_B = (0.40, 0.85)   # low rate sensitivity, HIGH amount sensitivity


def measure_stochastic(mcfg, keys):
    """Stochastic (a_rate, a_amt) via the exp008/exp011 procedures (see exp023)."""
    sim, initial_state = make_matching_sim(mcfg, _PATCH_POS, _PATCH_CUE, _START)

    log_B, log_R = [], []
    for i, (pL, pR) in enumerate(RATE_PAIRS):
        s0 = initial_state(N_ORG, jax.random.key(100 + i))
        t, r, _ = sim(s0, keys, jnp.array([pL, pR]))
        t, r = np.asarray(t), np.asarray(r)
        ok = (t[:, 0] > 0) & (t[:, 1] > 0) & (r[:, 0] > 0) & (r[:, 1] > 0)
        log_B.append(np.log(t[ok, 0] / t[ok, 1]))
        log_R.append(np.log(r[ok, 0] / r[ok, 1]))
    a_rate, _lb, _ = fit_matching_law(np.concatenate(log_R), np.concatenate(log_B))

    log_B, log_A = [], []
    for i, (aL, aR) in enumerate(AMOUNT_PAIRS):
        s0 = initial_state(N_ORG, jax.random.key(200 + i))
        t, _c, _a = sim(s0, keys, jnp.array([0.10, 0.10]), jnp.array([aL, aR]))
        t = np.asarray(t)
        ok = (t[:, 0] > 0) & (t[:, 1] > 0)
        log_B.append(np.log(t[ok, 0] / t[ok, 1]))
        log_A.append(np.full(ok.sum(), np.log(aL / aR)))
    a_amt, _lb, _ = fit_matching_law(np.concatenate(log_A), np.concatenate(log_B))
    return a_rate, a_amt


def main() -> None:
    t0 = time.perf_counter()
    base = MatchConfig()
    keys = jax.random.split(jax.random.key(0), N_STEPS)

    sr0, sa0 = soft_sensitivities(default_params(base, FREE), base)
    tr0, ta0 = measure_stochastic(base, keys)

    rows = []
    for name, target in [("A rate^/amt_v", TARGET_A), ("B rate_v/amt^", TARGET_B)]:
        fitted, hist = fit(target, free=FREE)
        best = min(hist, key=lambda h: h["loss"])
        tr, ta = measure_stochastic(fitted, keys)
        rows.append((name, target, (best["a_rate"], best["a_amt"]), (tr, ta), fitted))
    elapsed = time.perf_counter() - t0

    print(f"Decoupled fit (rate via beta, amount via amount_exponent) in {elapsed:.1f}s")
    print(f"  free params: {', '.join(FREE)}")
    print()
    print("                     a_rate                       a_amt")
    print("                target  soft  stoch    |    target  soft  stoch")
    print(f"  baseline         --   {sr0:.2f}  {tr0:.2f}    |      --   {sa0:.2f}  {ta0:.2f}")
    for name, target, (sr, sa), (tr, ta), _cfg in rows:
        print(f"  {name:13s} {target[0]:.2f}  {sr:.2f}  {tr:.2f}    |    "
              f"{target[1]:.2f}  {sa:.2f}  {ta:.2f}")
    print()
    for name, _t, _s, _st, cfg in rows:
        ps = "  ".join(f"{k}={getattr(cfg, k):.3f}" for k in FREE)
        print(f"  {name:13s} fitted: {ps}")

    # Decoupling check: the two fits must order OPPOSITELY on the two dimensions --
    # A has higher rate sensitivity AND lower amount sensitivity than B. On the old
    # coupled manifold the two sensitivities could only move together, so this
    # crossing is the signature of genuine independent control.
    (_a, _ta1, _sa1, (trA, taA), _cA) = rows[0]
    (_b, _tb1, _sb1, (trB, taB), _cB) = rows[1]
    rate_ok = trA > trB
    amt_ok = taA < taB
    print()
    print(f"  decoupling (stochastic): a_rate  A {trA:.2f} > B {trB:.2f}  "
          f"{'OK' if rate_ok else 'FAIL'}")
    print(f"  decoupling (stochastic): a_amt   A {taA:.2f} < B {taB:.2f}  "
          f"{'OK' if amt_ok else 'FAIL'}")

    # Figure: (a_rate, a_amt) for baseline + the two fits (stochastic). A and B land in
    # opposite corners -- off the diagonal the coupled levers were confined to.
    out = Path("outputs/figures/decoupled_fit.png")
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    pts = [("baseline", tr0, ta0, "0.4"), ("A rate^/amt_v", trA, taA, "black"),
           ("B rate_v/amt^", trB, taB, "black")]
    for label, x, y, c in pts:
        ax.scatter([x], [y], s=60, color=c, zorder=3)
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(8, 4), fontsize=10)
    ax.set_xlabel("rate sensitivity a_rate")
    ax.set_ylabel("amount sensitivity a_amt")
    ax.set_title("Independent control of the two matching sensitivities")
    ax.axhline(ta0, color="0.85", lw=0.5)
    ax.axvline(tr0, color="0.85", lw=0.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
