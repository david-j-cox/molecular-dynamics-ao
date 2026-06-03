"""Experiment 026 -- behavioral momentum as mass-modulated decay (Nevin & Grace).

A multiple schedule presents a RICH (VI 5) and a LEAN (VI 40) component, each
signaled by its own stimulus and sharing the single press response. Each component
accrues a Pavlovian context->reinforcer value graded by its reinforcement RATE; that
value is the behavioral-momentum MASS. The mass divides the rate at which the learned
response value decays (a time-based, response-rate-independent decrement), so a
richly-reinforced response is more resistant to extinction.

This is the clean version of the extinction test that exp022 could not deliver: there,
resistance was read from the press RATE, where a saturating emission function plus a
shared energy-deficit floor confounded proportion-of-baseline with baseline height
(the rich component, higher on the logistic, fell more in proportion -> spurious
anti-momentum). Here motivation is small (the value carries responding) and resistance
is read from the response VALUE, which the mass protects directly. Because the decay is
purely multiplicative, the sessions-to-criterion measure on value is scale-free: with
momentum_mass_gain = 0 the rich and lean components reach 25% of their own baseline in
the SAME number of sessions (no momentum -- the control), so any separation at gain > 0
is attributable to mass alone.

Run:   python experiments/exp026_momentum_extinction.py
Saves: outputs/logs/exp026_momentum_extinction.json
       outputs/figures/exp026_momentum_extinction.png
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from behavioral_md.chamber import ChamberConfig, run_multiple_schedule
from behavioral_md.experiment_utils import save_results_json
from behavioral_md.visualization import plot_momentum

VI_RICH, VI_LEAN = 5.0, 40.0
N_ORG = 300
COMP_STEPS = 200
N_BASELINE = 15
N_DISRUPTION = 30
GAIN_MOMENTUM = 3.0     # mass gain for the momentum arm
SEED = 0


def _config(gain: float) -> ChamberConfig:
    # Small motivation so the press value (not a shared energy-deficit floor) carries
    # responding; time-based value decay at value_extinction/mass (already in
    # run_multiple_schedule). ctx (the mass) grows fast, decays slowly -> it persists
    # into the extinction phase rather than washing out within it.
    return ChamberConfig(
        motiv_strength=0.3, energy_init=0.6, emission_bias=0.6, temperature=0.5,
        ctx_drive_gain=0.6, momentum_mass_gain=gain, learning_rate=0.08,
        value_extinction=0.004, reinf_asymptote=1.0, ctx_learning_rate=0.05,
        ctx_omission_rate=0.0015, ctx_asymptote=2.0,
    )


def _sessions_to_criterion(value: np.ndarray, nb: int, frac: float = 0.25) -> list[int]:
    """Per component: extinction sessions until value falls to ``frac`` of its
    last-baseline level. Scale-free under multiplicative decay (baseline-independent)."""
    out = []
    for c in range(value.shape[1]):
        base = value[nb - 1, c]
        ext = value[nb:, c]
        below = np.where(ext <= frac * base)[0]
        out.append(int(below[0] + 1) if len(below) else int(len(ext)))
    return out


def _run(gain: float) -> dict:
    cfg = _config(gain)
    res = run_multiple_schedule([VI_RICH, VI_LEAN], cfg, N_ORG, COMP_STEPS,
                                N_BASELINE, N_DISRUPTION, disruptor="extinction", seed=SEED)
    nb = res["n_baseline"]
    s2c = _sessions_to_criterion(res["value"], nb)
    base = res["value"][nb - 1]
    # Value as proportion of its own baseline, full trajectory (for plotting).
    prop = res["value"] / np.where(base > 1e-9, base, 1.0)
    return {
        "ctx_mass": res["ctx"][nb - 1].tolist(),
        "baseline_value": base.tolist(),
        "sessions_to_25pct_value": s2c,
        "value_prop_by_session": prop.tolist(),
    }


def main() -> None:
    control = _run(0.0)
    momentum = _run(GAIN_MOMENTUM)
    results = {
        "components": {"rich_VI": VI_RICH, "lean_VI": VI_LEAN},
        "n_baseline": N_BASELINE, "n_disruption": N_DISRUPTION,
        "momentum_mass_gain": GAIN_MOMENTUM,
        "control_gain0": control, "momentum": momentum,
    }
    save_results_json("exp026_momentum_extinction.json", results)

    print(f"Multiple schedule: rich VI {VI_RICH}, lean VI {VI_LEAN}")
    print(f"Context value (mass): rich={momentum['ctx_mass'][0]:.2f}  "
          f"lean={momentum['ctx_mass'][1]:.2f}")
    c0, c1 = control["sessions_to_25pct_value"]
    m0, m1 = momentum["sessions_to_25pct_value"]
    print("\nSessions for response VALUE to fall to 25% of baseline under extinction:")
    print(f"  gain=0 (control):   rich={c0}  lean={c1}   "
          f"{'EQUAL (no momentum)' if c0 == c1 else 'separated'}")
    verdict = "MOMENTUM (rich resists more)" if m0 > m1 else "no momentum"
    print(f"  gain={GAIN_MOMENTUM} (momentum): rich={m0}  lean={m1}   {verdict}")

    prop = np.asarray(momentum["value_prop_by_session"])
    p = plot_momentum(prop[:, 0], prop[:, 1], N_BASELINE,
                      Path("outputs/figures/exp026_momentum_extinction.png"),
                      ylabel="Response value (prop. of baseline)")
    print("\nSaved outputs/logs/exp026_momentum_extinction.json")
    print(f"Saved {p}")


if __name__ == "__main__":
    main()
