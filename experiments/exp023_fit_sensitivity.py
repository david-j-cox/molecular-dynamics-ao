"""Experiment 023 -- fitting organism parameters to target matching sensitivities.

The payoff of the differentiable engine: search organism parameters so the emergent
generalized-matching-law sensitivities (a_rate, a_amt) hit chosen targets, then confirm
the fit transfers to the true stochastic engine.

The sensitivities stay EMERGENT -- they fall out of the full stochastic dynamics, never
imposed by a chosen functional form (a molar closed-form would have to assume
``B_k ~ v_k**a``, writing the sensitivity in by hand). The search runs on the smooth,
deterministic Gumbel-softmax forward surrogate (``matching_diff``); reverse-mode
gradients through that ~1000-step recurrent rollout explode (~1000x too large,
sign-unstable), so we search it derivative-free (Nelder-Mead in ``fit``) rather than by
autodiff. The fitted parameters are then re-plugged into the real stochastic engine
(exp008/exp011 measurement) to verify transfer.

Free parameters: temperature, approach_gain, beta (lr_cue is excluded -- its effect does
not transfer; see matching_diff). On this two-patch preparation these levers move the two
sensitivities together (beta dominates), so the demonstration tunes them in aligned
directions (both up, both down); strong decoupling would need the environmental
changeover lever (patch separation, exp009), not organism parameters.

Run:  python -m experiments.exp023_fit_sensitivity
"""

from __future__ import annotations

import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from behavioral_md.experiment_utils import fit_matching_law
from behavioral_md.fit import FREE_PARAMS, fit
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
from behavioral_md.visualization import plot_matching

N_ORG, N_STEPS = 400, 4000
# Aligned targets (both sensitivities up / both down) in the surrogate's units, which
# read lower than the stochastic engine (a monotone compression). They sit near the
# edges of the surrogate's reachable range so the fit moves the parameters hard.
TARGET_UP = (0.50, 0.67)
TARGET_DOWN = (0.38, 0.56)


def measure_stochastic(mcfg, keys, want_points=False):
    """Stochastic (a_rate, a_amt) for a config, via the exp008/exp011 procedures.

    Mirrors those experiments exactly (rate: sweep VI pairs at equal amount; amount:
    sweep amount pairs at equal VI; pooled per-organism log ratios -> GML slope). When
    ``want_points`` is set, also returns the pooled (x, y, log_b) for the rate sweep so
    the matching plot can be drawn.
    """
    sim, initial_state = make_matching_sim(mcfg, _PATCH_POS, _PATCH_CUE, _START)

    log_B, log_R = [], []
    for i, (pL, pR) in enumerate(RATE_PAIRS):
        s0 = initial_state(N_ORG, jax.random.key(100 + i))
        t, r, _ = sim(s0, keys, jnp.array([pL, pR]))
        t, r = np.asarray(t), np.asarray(r)
        ok = (t[:, 0] > 0) & (t[:, 1] > 0) & (r[:, 0] > 0) & (r[:, 1] > 0)
        log_B.append(np.log(t[ok, 0] / t[ok, 1]))
        log_R.append(np.log(r[ok, 0] / r[ok, 1]))
    xr, yr = np.concatenate(log_R), np.concatenate(log_B)
    a_rate, log_b_r, _ = fit_matching_law(xr, yr)

    log_B, log_A = [], []
    for i, (aL, aR) in enumerate(AMOUNT_PAIRS):
        s0 = initial_state(N_ORG, jax.random.key(200 + i))
        t, _c, _a = sim(s0, keys, jnp.array([0.10, 0.10]), jnp.array([aL, aR]))
        t = np.asarray(t)
        ok = (t[:, 0] > 0) & (t[:, 1] > 0)
        log_B.append(np.log(t[ok, 0] / t[ok, 1]))
        log_A.append(np.full(ok.sum(), np.log(aL / aR)))
    a_amt, _lb, _ = fit_matching_law(np.concatenate(log_A), np.concatenate(log_B))

    if want_points:
        return a_rate, a_amt, (xr, yr, a_rate, log_b_r)
    return a_rate, a_amt


def main() -> None:
    t0 = time.perf_counter()
    base = MatchConfig()
    keys = jax.random.split(jax.random.key(0), N_STEPS)

    # Baselines.
    sr0, sa0 = soft_sensitivities(default_params(base), base)
    tr0, ta0 = measure_stochastic(base, keys)

    # Fit to aligned targets in each direction, then re-plug into the stochastic engine.
    rows = []
    for name, target in [("tune-up", TARGET_UP), ("tune-down", TARGET_DOWN)]:
        fitted, hist = fit(target)
        best = min(hist, key=lambda h: h["loss"])
        sr, sa = best["a_rate"], best["a_amt"]
        tr, ta = measure_stochastic(fitted, keys)
        rows.append((name, target, (sr, sa), (tr, ta), fitted))
    elapsed = time.perf_counter() - t0

    print(f"Fit organism params to target matching sensitivities in {elapsed:.1f}s")
    print(f"  free params: {', '.join(FREE_PARAMS)}  (search: Nelder-Mead on the "
          f"differentiable surrogate)")
    print()
    print("                  a_rate                       a_amt")
    print("             target  soft  stoch    |    target  soft  stoch")
    print(f"  baseline      --   {sr0:.2f}  {tr0:.2f}    |      --   {sa0:.2f}  {ta0:.2f}")
    for name, target, (sr, sa), (tr, ta), _cfg in rows:
        print(f"  {name:9s}  {target[0]:.2f}  {sr:.2f}  {tr:.2f}    |    "
              f"{target[1]:.2f}  {sa:.2f}  {ta:.2f}")
    print()
    for name, _t, _s, _st, cfg in rows:
        ps = "  ".join(f"{k}={getattr(cfg, k):.3f}" for k in FREE_PARAMS)
        print(f"  {name:9s} fitted: {ps}")

    # Transfer check: the stochastic sensitivities should be ordered down < base < up
    # for both dimensions (monotone bidirectional transfer of the fit).
    up = next(r for r in rows if r[0] == "tune-up")[3]
    down = next(r for r in rows if r[0] == "tune-down")[3]
    rate_ok = down[0] < tr0 < up[0]
    amt_ok = down[1] < ta0 < up[1]
    print()
    print(f"  transfer (stochastic): a_rate {down[0]:.2f} < {tr0:.2f} < {up[0]:.2f}  "
          f"{'OK' if rate_ok else 'FAIL'}")
    print(f"  transfer (stochastic): a_amt  {down[1]:.2f} < {ta0:.2f} < {up[1]:.2f}  "
          f"{'OK' if amt_ok else 'FAIL'}")

    # Figure: the emergent matching (rate) under the tune-up fitted organism.
    _tr, _ta, pts = measure_stochastic(rows[0][4], keys, want_points=True)
    xr, yr, a_rate, log_b = pts
    out = Path("outputs/figures/fit_sensitivity_rate.png")
    plot_matching(xr, yr, a_rate, log_b, out)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
