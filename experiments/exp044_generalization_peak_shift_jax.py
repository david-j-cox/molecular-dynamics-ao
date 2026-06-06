"""exp044 -- stimulus generalization and peak shift on the JAX fast path.

The NumPy demos (run_generalization_demo, run_peak_shift_demo) show these with the CueReceptorField.
This reproduces both on JAX, exercising the SAME mechanism the JAX engine uses for cues
(jax_engine.learn_with_cue): receptors tile the cue dimension with a Shepard kernel
exp(-beta*|value - center|), and learn by a summed-error (elemental) Rescorla-Wagner update. The
training run is one jitted lax.scan, vectorized over a population.

- Generalization: train at a single reinforced cue value -> the conditioned-response gradient peaks
  at the trained value and falls off smoothly with cue distance.
- Peak shift: train S+ (reinforced) against a nearby S- (non-reinforced) -> the peak moves AWAY from
  S-, past S+. The shared (summed) error term puts negative weight on the receptors S- excites, and
  because S- overlaps the S+ side of the gradient the net peak displaces to the far side.

Run:  python experiments/exp044_generalization_peak_shift_jax.py
"""

from __future__ import annotations

import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from behavioral_md.config import SimulationConfig
from behavioral_md.visualization import plot_generalization_gradient

FIG = Path("outputs/figures")
_CFG = SimulationConfig()
K = _CFG.n_cue_receptors                       # receptors tiling [0, 1]
BETA = _CFG.cue_generalization_beta            # Shepard kernel steepness (engine default 6.0)
LO, HI = _CFG.history_weight_min, _CFG.history_weight_max
CENTERS = jnp.linspace(0.0, 1.0, K)
RATE, LAM = 0.02, 1.0                           # controlled-procedure association rate / asymptote
N_AGENTS, N_BLOCKS, SENSE_NOISE = 200, 400, 0.02
V_PLUS, V_MINUS = 0.40, 0.55                     # S+ reinforced; S- (non-reinforced) just above it
PROBES = np.linspace(0.0, 1.0, 41)


@jax.jit
def _train(key, values, mags):
    """One jitted scan over presentations; returns learned receptor weights [O, K]."""
    noise = jax.random.normal(key, (values.shape[0], N_AGENTS)) * SENSE_NOISE

    def step(w, x):
        v, mag, ns = x
        sensed = jnp.clip(v + ns, 0.0, 1.0)                       # [O] noisy sensed cue
        act = jnp.exp(-BETA * jnp.abs(sensed[:, None] - CENTERS[None, :]))  # [O, K] Shepard tuning
        drive = jnp.sum(w * act, axis=1)                          # [O] conditioned response
        err = LAM * mag - drive                                   # summed (elemental) RW error
        w = jnp.clip(w + RATE * act * err[:, None], LO, HI)
        return w, None

    w0 = jnp.zeros((N_AGENTS, K))
    w, _ = jax.lax.scan(step, w0, (values, mags, noise))
    return w


def _gradient(w):
    """Probe each cue value with no learning: response [O, n_probe]."""
    act = jnp.exp(-BETA * jnp.abs(jnp.asarray(PROBES)[:, None] - CENTERS[None, :]))  # [P, K]
    return np.asarray(w @ act.T)


def main() -> None:
    key = jax.random.key(0)
    # Generalization: reinforced presentations at S+ only.
    gen_vals = jnp.full(N_BLOCKS, V_PLUS)
    gen_mags = jnp.ones(N_BLOCKS)
    # Peak shift: alternate S+ (reinforced) and S- (not), balanced.
    ps_vals = jnp.asarray(np.tile([V_PLUS, V_MINUS], N_BLOCKS))
    ps_mags = jnp.asarray(np.tile([1.0, 0.0], N_BLOCKS))

    t0 = time.perf_counter()
    gen_w = _train(key, gen_vals, gen_mags).block_until_ready()
    ps_w = _train(key, ps_vals, ps_mags).block_until_ready()
    dt = time.perf_counter() - t0

    gen_resp = _gradient(gen_w)
    ps_resp = _gradient(ps_w)
    gen_peak = PROBES[np.argmax(gen_resp.mean(0))]
    ps_peak = PROBES[np.argmax(ps_resp.mean(0))]

    print(f"JAX cue conditioning ({N_AGENTS} agents, {N_BLOCKS} blocks, jit+run {dt:.2f}s):\n")
    print(f"  Generalization: peak at {gen_peak:.2f}  (S+ = {V_PLUS:.2f}; should coincide)")
    print(f"  Peak shift:     peak at {ps_peak:.2f}  (S+ {V_PLUS:.2f}, S- {V_MINUS:.2f}; shifts")
    print(f"                  below S+, away from S-).  shift = {gen_peak - ps_peak:+.2f}")

    FIG.mkdir(parents=True, exist_ok=True)
    out_gen = plot_generalization_gradient(PROBES, gen_resp, V_PLUS,
                                           FIG / "exp044_generalization_jax.png")
    out_ps = plot_generalization_gradient(PROBES, ps_resp, V_PLUS,
                                          FIG / "exp044_peak_shift_jax.png", s_minus=V_MINUS)
    print(f"\nwrote {out_gen}\nwrote {out_ps}")


if __name__ == "__main__":
    main()
