"""Experiment 022 -- behavioral momentum (Nevin & Grace) in a multiple schedule.

Two components are presented successively, each signaled by its own stimulus: a
RICH component (VI 5) and a LEAN component (VI 40). The single press response is
shared, but each component stimulus carries a Pavlovian context->reinforcer value
that settles at a level graded by that component's reinforcement RATE (the
behavioral-momentum "mass"). After baseline, a disruptor is applied and resistance
to change is measured as each component's press rate relative to its own baseline.

Behavioral momentum theory predicts the RICH component is MORE resistant, because
resistance tracks the stimulus-reinforcer (Pavlovian) relation, not the response
rate. Here that emerges mechanistically: the rich component's higher context value
is a larger, non-motivational share of its press drive, so a SATIATION (prefeeding)
disruptor -- which removes the energy-deficit drive -- takes a smaller proportional
bite out of it.

Honest scope: momentum is robust under the satiation disruptor (shown here). Under
EXTINCTION it is NOT robustly reproduced in this energy-coupled world -- withholding
food is also starvation, which raises the deficit drive and masks/reverses response
extinction; and once value decays, both components fall to a shared emission floor,
so the proportion-of-baseline ordering is dominated by baseline height rather than
by mass. The extinction numbers are reported for transparency, not as a success.

Run:   python experiments/exp022_behavioral_momentum.py
Saves: outputs/logs/exp022_momentum.json
       outputs/figures/exp022_momentum_satiation.png
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from behavioral_md.chamber import ChamberConfig, run_multiple_schedule
from behavioral_md.experiment_utils import save_results_json
from behavioral_md.visualization import plot_momentum

VI_RICH, VI_LEAN = 5.0, 40.0
N_ORG = 300
COMP_STEPS = 300
N_BASELINE = 12
N_DISRUPTION = 10
SEED = 0


def _config() -> ChamberConfig:
    # Large deficit motivation (so satiation is a real disruptor) + a substantial
    # context drive (so the rich component's stimulus-reinforcer value meaningfully
    # protects it). mass_gain is left at 0: the satiation effect comes from the
    # context drive, not the inertia term.
    return ChamberConfig(
        motiv_strength=2.0, energy_init=0.5, emission_bias=1.2, temperature=0.5,
        ctx_drive_gain=0.8, momentum_mass_gain=0.0, reinf_asymptote=1.0,
    )


def _resistance(res: dict) -> dict:
    """Per-component baseline rate and mean resistance (proportion) over disruption."""
    pr = res["press_rate"]
    nb = res["n_baseline"]
    base = pr[nb - 3:nb].mean(0)
    disrupt = pr[nb:]
    resistance = (disrupt / base).mean(0)
    return {
        "baseline_rate": base.tolist(),
        "resistance": resistance.tolist(),
        "press_rate_by_session": pr.tolist(),
    }


def main() -> None:
    cfg = _config()
    sat = run_multiple_schedule([VI_RICH, VI_LEAN], cfg, N_ORG, COMP_STEPS,
                                N_BASELINE, N_DISRUPTION, disruptor="satiation", seed=SEED)
    ext = run_multiple_schedule([VI_RICH, VI_LEAN], cfg, N_ORG, COMP_STEPS,
                                N_BASELINE, N_DISRUPTION, disruptor="extinction", seed=SEED)

    s, e = _resistance(sat), _resistance(ext)
    reinf = sat["reinf_rate"][N_BASELINE - 3:N_BASELINE].mean(0)
    ctx = sat["ctx"][N_BASELINE - 1]
    results = {
        "components": {"rich_VI": VI_RICH, "lean_VI": VI_LEAN},
        "baseline_reinforcement_rate": reinf.tolist(),
        "context_value": ctx.tolist(),
        "satiation": s, "extinction": e,
    }
    save_results_json("exp022_momentum.json", results)

    print(f"Components: rich VI {VI_RICH}, lean VI {VI_LEAN}")
    print(f"Baseline reinforcement rate: rich={reinf[0]:.3f}  lean={reinf[1]:.3f}")
    print(f"Context value (mass):        rich={ctx[0]:.2f}   lean={ctx[1]:.2f}")
    print(f"Baseline press rate:         rich={s['baseline_rate'][0]:.3f}  "
          f"lean={s['baseline_rate'][1]:.3f}")
    verdict = ("MOMENTUM (rich more resistant)"
               if s["resistance"][0] > s["resistance"][1] else "no momentum")
    print("\nResistance (proportion of baseline retained under disruptor):")
    print(f"  satiation:  rich={s['resistance'][0]:.3f}  lean={s['resistance'][1]:.3f}  {verdict}")
    print(f"  extinction: rich={e['resistance'][0]:.3f}  lean={e['resistance'][1]:.3f}  "
          f"(not a clean test in this energy-coupled world -- see module docstring)")

    pr = np.asarray(sat["press_rate"])
    p = plot_momentum(pr[:, 0], pr[:, 1], N_BASELINE,
                      Path("outputs/figures/exp022_momentum_satiation.png"))
    print("\nSaved outputs/logs/exp022_momentum.json")
    print(f"Saved {p}")


if __name__ == "__main__":
    main()
