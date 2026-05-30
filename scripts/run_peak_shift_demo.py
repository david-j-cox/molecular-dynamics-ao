"""Peak-shift demo: discrimination training shifts the gradient peak past S+.

After discriminating S+ (reinforced, value v_plus) from S- (non-reinforced,
value v_minus) on a single stimulus dimension, the peak of the generalization
gradient shifts AWAY from S- -- past S+ to a value never trained (Hanson, 1959).

Mechanism here: the cue receptor field uses a summed (elemental) prediction
error. S+ builds an excitatory gradient at v_plus that generalizes onto v_minus;
on S- presentations the resulting positive prediction at v_minus meets no
reinforcement, so the error is negative and those receptors go inhibitory. The
net (excitatory minus inhibitory) gradient peaks on the far side of S+ from S-.

This uses a CONTROLLED discrimination procedure (equal S+/S- presentations)
driving the same `CueReceptorField` the organism uses. We do this rather than
full foraging lives because survival foraging cannot balance exposure: a
reinforced organism camps at S+ (lots of exposure) while an unreinforced one
starves and leaves S- quickly (little exposure), so excitation would always swamp
inhibition. Per-agent sensory noise on the presented value gives the population
spread for the 95% CI.

Run:  python scripts/run_peak_shift_demo.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from behavioral_md.config import SimulationConfig
from behavioral_md.generalization import CueReceptorField
from behavioral_md.parallel import run_sweep
from behavioral_md.visualization import plot_generalization_gradient

FIG_DIR = Path("outputs/figures")
N_AGENTS = 200
N_BLOCKS = 400        # each block = one S+ and one S- presentation
V_PLUS = 0.40
V_MINUS = 0.55        # close enough that the inhibitory gradient overlaps the S+ side
RATE = 0.02           # association rate for the controlled procedure
LAM = 1.0
SENSE_NOISE = 0.02    # per-presentation noise on the sensed cue value
PROBE_VALUES = np.linspace(0.0, 1.0, 41)


def agent_worker(cell: dict[str, Any]) -> dict[str, Any]:
    seed = cell["seed"]
    rng = np.random.default_rng(seed)
    cfg = SimulationConfig()
    field = CueReceptorField(
        cfg.n_cue_receptors, cfg.cue_generalization_beta,
        cfg.history_weight_min, cfg.history_weight_max,
    )
    for _ in range(N_BLOCKS):
        for value, mag in ((V_PLUS, 1.0), (V_MINUS, 0.0)):
            sensed = float(np.clip(value + rng.normal(0.0, SENSE_NOISE), 0.0, 1.0))
            field.drive(sensed, 1.0)              # set receptor activations
            field.learn(1.0, mag, RATE, LAM)      # summed-error update
    return {"seed": seed, "gradient": [field.response(float(v)) for v in PROBE_VALUES]}


def main(n_agents: int = N_AGENTS) -> None:
    print(f"Peak-shift demo: {n_agents} agents, discrimination S+={V_PLUS} / S-={V_MINUS}, "
          f"{N_BLOCKS} balanced blocks...")
    results = run_sweep(agent_worker, [{"seed": s} for s in range(n_agents)], progress_every=50)
    responses = np.array([r["gradient"] for r in results])

    plot_generalization_gradient(
        PROBE_VALUES, responses, V_PLUS, FIG_DIR / "peak_shift.png", s_minus=V_MINUS
    )

    mean = responses.mean(axis=0)
    peak_v = float(PROBE_VALUES[int(np.argmax(mean))])
    direction = "shifted past S+, away from S-" if peak_v < V_PLUS else "NOT shifted"
    print(f"Empirical peak at cue value {peak_v:.3f}  (S+={V_PLUS}, S-={V_MINUS}) -> {direction}.")
    print(f"Wrote {FIG_DIR/'peak_shift.png'}")


if __name__ == "__main__":
    import argparse

    _ap = argparse.ArgumentParser()
    _ap.add_argument("--agents", type=int, default=N_AGENTS)
    main(_ap.parse_args().agents)
