"""Experiment 029 -- reinforcement/punishment asymmetry in concurrent choice.

Two responses are concurrently reinforced (VI) and, independently, punished (VI) in the
operant chamber (``chamber.run_punishment_choice``). Three accounts of how a punisher maps
to choice -- all of which suppress the punished response (mimicry) but make different
quantitative predictions:

  subtractive   (de Villiers, 1980): a punisher cancels c reinforcers of the SAME response.
  competitive   (Deluty, 1976):      a punisher strengthens the COMPETING responses.
  concatenated  (Critchfield/Klapes): power-law matching with a SEPARATE punishment
                                      sensitivity a_p.

Three results:
  1. Suppression -- all three reduce allocation to the punished response as its scheduled
     punishment rate rises (the common phenomenon).
  2. The de Villiers vs Deluty DISSOCIATION -- punishment suppression (log-odds shift)
     depends on the ALTERNATIVE's reinforcement rate with OPPOSITE slopes: subtractive
     rises with alternative richness, competitive falls (direct subtraction vs competitive
     reallocation -- their historic debate, the punishment analogue of the resurgence
     target-rate dissociation).
  3. Concatenated matching law with punishment -- with both responses punished, the
     emergent log(B1/B2) = -a_p*log(P1/P2) + bias is log-linear and recovers a punishment
     sensitivity that tracks the set ``pun_a_p``, separable from the reinforcement term.

Caveat (reported): fitting on OBTAINED punishment is confounded by response feedback --
heavy suppression means the punished response is rarely emitted, so it collects fewer
punishers and the obtained-rate axis can invert. Suppression curves use SCHEDULED rate.

Run:   python experiments/exp029_punishment_asymmetry.py
Saves: outputs/logs/exp029_punishment.json
       outputs/figures/exp029_punishment_suppression.png
       outputs/figures/exp029_punishment_dissociation.png
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from behavioral_md.chamber import ChamberConfig, run_punishment_choice
from behavioral_md.experiment_utils import save_results_json
from behavioral_md.visualization import (
    plot_punishment_dissociation,
    plot_punishment_suppression,
)

N_ORG = 600
N_STEPS = 5000
SEED = 0
INF = float("inf")


def _cfg(**kw) -> ChamberConfig:
    return ChamberConfig(pun_tau=800.0, pun_bump=0.04, pun_floor=0.1, **kw)


def _alloc(model, vi_reinf, vi_punish, **kw) -> float:
    res = run_punishment_choice(model, _cfg(**kw), N_ORG, N_STEPS,
                                vi_reinf=vi_reinf, vi_punish=vi_punish, seed=SEED)
    e = res["emit"].sum(0)
    return float(e[0] / e.sum())


def _logodds(model, alt_vi, punish, **kw) -> float:
    res = run_punishment_choice(model, _cfg(**kw), N_ORG, N_STEPS,
                                vi_reinf=[5.0, alt_vi], vi_punish=[punish, INF], seed=SEED)
    e = res["emit"].sum(0)
    return float(np.log(e[0] / e[1]))


def suppression_curves() -> dict:
    """Allocation to the punished target vs scheduled punishment rate, per model."""
    pun_vis = [INF, 40.0, 20.0, 12.0, 8.0, 5.0]
    rates = [0.0 if not np.isfinite(v) else 1.0 / v for v in pun_vis]
    models = {
        "subtractive": dict(pun_c=1.0, pun_sensitivity=1.0),
        "competitive": dict(pun_c=1.5, pun_sensitivity=1.0),
        "concatenated": dict(pun_a_r=1.0, pun_a_p=1.0),
    }
    curves = {
        name: [_alloc(name, [5.0, 5.0], [pv, INF], **kw) for pv in pun_vis]
        for name, kw in models.items()
    }
    return {"rates": rates, "curves": curves}


def dissociation() -> dict:
    """Log-odds punishment suppression vs alternative reinforcement (de Villiers vs Deluty)."""
    alt_vis = [20.0, 10.0, 6.0, 4.0]
    alt_rates = [1.0 / v for v in alt_vis]
    sub = [_logodds("subtractive", v, INF, pun_c=1.0, pun_sensitivity=1.0)
           - _logodds("subtractive", v, 10.0, pun_c=1.0, pun_sensitivity=1.0) for v in alt_vis]
    comp = [_logodds("competitive", v, INF, pun_c=1.5, pun_sensitivity=1.0)
            - _logodds("competitive", v, 10.0, pun_c=1.5, pun_sensitivity=1.0) for v in alt_vis]
    return {"alt_rates": alt_rates, "subtractive": sub, "competitive": comp}


def concatenated_recovery() -> dict:
    """Recover the punishment sensitivity a_p from a both-punished ratio sweep."""
    out = {}
    for set_ap in (0.5, 1.0, 1.5):
        logB, logP = [], []
        for pv0 in (15.0, 10.0, 7.0, 5.0, 4.0):
            res = run_punishment_choice("concatenated", _cfg(pun_a_r=1.0, pun_a_p=set_ap),
                                        N_ORG, N_STEPS, vi_reinf=[5.0, 5.0],
                                        vi_punish=[pv0, 15.0], seed=SEED)
            e, pc = res["emit"].sum(0), res["punished"].sum(0)
            logB.append(np.log(e[0] / e[1]))
            logP.append(np.log(pc[0] / pc[1]))
        slope, _ = np.polyfit(logP, logB, 1)
        r2 = float(np.corrcoef(logP, logB)[0, 1] ** 2)
        out[f"a_p={set_ap}"] = {"recovered_a_p": float(-slope), "r2": r2}
    return out


def main() -> None:
    supp = suppression_curves()
    diss = dissociation()
    conc = concatenated_recovery()
    save_results_json("exp029_punishment.json",
                      {"suppression": supp, "dissociation": diss, "concatenated": conc})

    print("1. Suppression of the punished response (allocation; all three models suppress):")
    for name, y in supp["curves"].items():
        print(f"   {name:13s} " + "  ".join(f"{v:.2f}" for v in y))
    print("\n2. de Villiers vs Deluty dissociation (log-odds suppression vs alt. reinf. rate):")
    print("   alt rate (1/VI): " + "  ".join(f"{r:.3f}" for r in diss["alt_rates"]))
    print("   subtractive:     " + "  ".join(f"{v:+.2f}" for v in diss["subtractive"])
          + "   (rises with alternative richness)")
    print("   competitive:     " + "  ".join(f"{v:+.2f}" for v in diss["competitive"])
          + "   (falls with alternative richness)")
    print("\n3. Concatenated matching law -- recovered punishment sensitivity a_p:")
    for k, v in conc.items():
        print(f"   set {k:9s} -> recovered a_p={v['recovered_a_p']:.2f}  (R^2={v['r2']:.3f})")

    p1 = plot_punishment_suppression(supp["rates"], supp["curves"],
                                     Path("outputs/figures/exp029_punishment_suppression.png"))
    p2 = plot_punishment_dissociation(diss["alt_rates"], diss["subtractive"],
                                      diss["competitive"],
                                      Path("outputs/figures/exp029_punishment_dissociation.png"))
    print("\nSaved outputs/logs/exp029_punishment.json")
    print(f"Saved {p1}")
    print(f"Saved {p2}")


if __name__ == "__main__":
    main()
