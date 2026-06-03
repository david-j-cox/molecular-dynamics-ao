"""Experiment 028 -- resurgence as an emergent property of choice.

Three phases in a concurrent chamber (the organism emits one of R1, R2, or a
background "other" behavior each step, by a softmax over their values -- behavior is
choice/matching):

  Phase 1 (train):  R1 reinforced (VI 5); R2 never.
  Phase 2 (alt):    R1 extinguished; R2 reinforced (VI 5).
  Phase 3 (test):   both extinguished.

Resurgence -- R1 responding recovering in phase 3 from its phase-2 suppressed level --
is NOT coded anywhere. It emerges because removing R2's reinforcement reallocates
choice back toward R1 (Shahan & Craig's "resurgence as choice", here a consequence of
the softmax allocation rather than an imposed model). The procedure is symmetric (R2 is
trained in phase 2 as R1 was in phase 1), so R1 and R2 converge to parity at test;
resurgence is the RISE of R1, not R1 exceeding R2.

The control isolates the cause: if R2 stays reinforced in phase 3, allocation does not
flow back and resurgence vanishes -- so it is the removal of alternative reinforcement,
not the passage of time or disinhibition, that drives recovery.

How much latent R1 strength survives phase 2 depends on the extinction-learning rule
(the same machinery built for items 1-2), and nothing resurgence-specific is added:
  - single value (RW):        R1 erodes toward the background floor -> bare choice.
  - + momentum mass:          training-history mass slows R1's phase-2 decay -> larger
                              resurgence (Nevin's resistance to change).
  - dual excitatory/inhib.:   omission grows a separate inhibition and PRESERVES R1's
                              excitation -> R1 stays much less suppressed in phase 2
                              (Konorski/Bouton) and resurges from a higher floor.

Run:   python experiments/exp028_resurgence.py
Saves: outputs/logs/exp028_resurgence.json
       outputs/figures/exp028_resurgence_single.png
       outputs/figures/exp028_resurgence_dual.png
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from behavioral_md.chamber import ChamberConfig, run_resurgence
from behavioral_md.experiment_utils import save_results_json
from behavioral_md.visualization import plot_resurgence

N_ORG = 500
PHASE_STEPS = 2500
BLOCK = 50
VI_R1 = VI_R2 = 5.0
R_OTHER = 0.2
SEED = 0


def _config(rule: str, gain: float) -> ChamberConfig:
    return ChamberConfig(
        value_rule=rule, momentum_mass_gain=gain, learning_rate=0.10,
        value_extinction=0.02, approach_gain=4.0, temperature=0.5, act_tau=3.0,
        inhib_rate=0.06, inhib_relax=0.12, inhib_passive_decay=0.005,
    )


def _resurgence(res: dict) -> dict:
    pb = res["phase_blocks"]
    r1, r2 = np.asarray(res["r1"]), np.asarray(res["r2"])
    end_p2 = float(r1[2 * pb - 3:2 * pb].mean())     # suppressed level entering test
    test_p3 = float(r1[2 * pb + 2:].mean())          # settled test level
    return {
        "r1_end_phase2": end_p2,
        "r1_test_phase3": test_p3,
        "resurgence": test_p3 - end_p2,
        "r1": r1.tolist(), "r2": r2.tolist(), "phase_blocks": pb,
    }


def _run(rule: str, gain: float, control: bool = False) -> dict:
    res = run_resurgence(_config(rule, gain), N_ORG, PHASE_STEPS, seed=SEED,
                         vi_r1=VI_R1, vi_r2=VI_R2, r_other=R_OTHER, block=BLOCK,
                         control_reinforce_r2=control)
    return _resurgence(res)


def main() -> None:
    arms = {
        "single_rw": _run("single", 0.0),
        "momentum": _run("single", 8.0),
        "dual": _run("dual", 0.0),
        "control_single_R2on": _run("single", 0.0, control=True),
    }
    results = {
        "phase_steps": PHASE_STEPS, "block": BLOCK, "vi": [VI_R1, VI_R2],
        "r_other": R_OTHER, "arms": arms,
    }
    save_results_json("exp028_resurgence.json", results)

    print("Resurgence = R1 response allocation rising in phase 3 (R2 removed)\n")
    print(f"  {'arm':22s}  R1 end-phase2  R1 test   resurgence")
    for name in ("single_rw", "momentum", "dual", "control_single_R2on"):
        a = arms[name]
        print(f"  {name:22s}  {a['r1_end_phase2']:.3f}        {a['r1_test_phase3']:.3f}    "
              f"{a['resurgence']:+.3f}")
    print("\n  The control (R2 still reinforced at test) shows ~0 resurgence: removing")
    print("  the alternative's reinforcement is what reallocates choice back to R1.")

    pb = arms["single_rw"]["phase_blocks"]
    ps = plot_resurgence(arms["single_rw"]["r1"], arms["single_rw"]["r2"], pb,
                         Path("outputs/figures/exp028_resurgence_single.png"))
    pm = plot_resurgence(arms["momentum"]["r1"], arms["momentum"]["r2"], pb,
                         Path("outputs/figures/exp028_resurgence_momentum.png"))
    pd_ = plot_resurgence(arms["dual"]["r1"], arms["dual"]["r2"], pb,
                          Path("outputs/figures/exp028_resurgence_dual.png"))
    print("\nSaved outputs/logs/exp028_resurgence.json")
    print(f"Saved {ps}")
    print(f"Saved {pm}")
    print(f"Saved {pd_}")


if __name__ == "__main__":
    main()
