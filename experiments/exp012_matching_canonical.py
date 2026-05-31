"""Experiment 012 -- canonical concurrent VI-VI matching with counterbalancing.

Standard procedure: reinforcement-rate ratios 9:1, 3:1, 1:1, 1:3, 1:9 (total rate
held constant), and the cue<->side assignment COUNTERBALANCED so that only the
schedule -- not which cue or which side is rich -- controls responding.

Counterbalancing: half the organisms run with cue 0.2 on the left patch / cue 0.8
on the right; the other half with the cues swapped. The schedule ratio is defined
by SIDE (left:right), so cue identity is decorrelated from rich/lean.

Diagnostics:
- Pooled GML fit log(B_L/B_R) = a*log(R_L/R_R) + log b: with counterbalancing, log b
  should be ~0 (no residual side bias) if the organism discriminates cue->schedule.
- The 1:1 condition read directly per cue-assignment: a nonzero, sign-flipping
  allocation between the two assignments would reveal an intrinsic cue-value bias
  (which counterbalancing then cancels in the pooled fit).

Run:  python -m experiments.exp012_matching_canonical
"""

from __future__ import annotations

import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from behavioral_md.experiment_utils import fit_matching_law
from behavioral_md.matching import MatchConfig, make_matching_sim
from behavioral_md.visualization import plot_matching

# Odd grid (size 11, center 5) so the centered layout has equal wall margins on
# both sides -- the even-grid asymmetry trapped the organism near the closer wall
# and produced a spurious side bias (exp012 diagnosis). Separation 6.
N_ORG, N_STEPS = 600, 4000
GRID = 11
PATCH_POS = np.array([[2.0, 5.0], [8.0, 5.0]])      # left, right (separation 6)
START = [5.0, 5.0]
SENSOR_RANGE = 8.0
# Canonical ratios as (left, right) VI arming probs; total ~0.20 held constant.
RATIOS = {
    "9:1": (0.18, 0.02), "3:1": (0.15, 0.05), "1:1": (0.10, 0.10),
    "1:3": (0.05, 0.15), "1:9": (0.02, 0.18),
}
CUE_ASSIGNMENTS = {"cue0.2-left": (0.2, 0.8), "cue0.2-right": (0.8, 0.2)}


def run_cell(cue_pair, arm, key):
    mcfg = MatchConfig(grid_size=GRID, sensor_range=SENSOR_RANGE)
    sim, initial_state = make_matching_sim(mcfg, PATCH_POS, np.array(cue_pair), START)
    keys = jax.random.split(jax.random.key(0), N_STEPS)
    state0 = initial_state(N_ORG // 2, key)
    time_at, count, _amt = sim(state0, keys, jnp.array(arm))
    return np.asarray(time_at), np.asarray(count)


def main() -> None:
    t0 = time.perf_counter()
    log_B, log_R = [], []
    one_to_one = {}   # cue assignment -> median log(B_L/B_R) at 1:1
    for ai, (aname, cue_pair) in enumerate(CUE_ASSIGNMENTS.items()):
        for ri, (rname, arm) in enumerate(RATIOS.items()):
            B, _ = run_cell(cue_pair, arm, jax.random.key(10 * ai + ri))
            bL, bR = B[:, 0], B[:, 1]
            ok = (bL > 0) & (bR > 0)
            lb = np.log(bL[ok] / bR[ok])
            lr = np.full(ok.sum(), np.log(arm[0] / arm[1]))
            log_B.append(lb)
            log_R.append(lr)
            if rname == "1:1":
                one_to_one[aname] = float(np.median(lb))

    x = np.concatenate(log_R)
    y = np.concatenate(log_B)
    a, log_b, r2 = fit_matching_law(x, y)
    elapsed = time.perf_counter() - t0

    print(f"Canonical conc VI-VI (counterbalanced): {N_ORG} organisms x "
          f"{len(RATIOS)} ratios x 2 cue-assignments x {N_STEPS} steps in {elapsed:.1f}s\n")
    print(f"Pooled GML:  log(B_L/B_R) = {a:.2f} * log(R_L/R_R) + {log_b:+.2f}   R^2={r2:.3f}")
    print(f"  sensitivity a = {a:.2f}   bias log b = {log_b:+.2f} "
          f"({'~unbiased' if abs(log_b) < 0.1 else 'residual bias'})\n")
    print("1:1 condition, allocation log(B_L/B_R) by cue assignment:")
    for aname, v in one_to_one.items():
        print(f"  {aname:14s}: {v:+.2f}")
    cue_bias = (one_to_one["cue0.2-left"] - one_to_one["cue0.2-right"]) / 2
    print(f"  -> intrinsic cue bias (toward cue 0.8) = {cue_bias:+.2f} "
          f"(0 = pure schedule control)")

    out = Path("outputs/figures/matching_canonical.png")
    plot_matching(x, y, a, log_b, out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
