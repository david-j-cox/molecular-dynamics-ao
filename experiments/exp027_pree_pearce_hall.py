"""Experiment 027 -- partial-reinforcement extinction effect from Pearce-Hall.

Two groups acquire a single press response: CONTINUOUS reinforcement (CRF; every
press reinforced) vs PARTIAL reinforcement (PRF; each press reinforced with
probability p). Both are then extinguished. The partial-reinforcement extinction
effect (PREE) is that PRF responding persists LONGER in extinction.

The mechanism is Pearce-Hall associability: the effective learning/extinction rate is
scaled by a per-organism associability alpha that tracks the recent ABSOLUTE
prediction error |PE|. After PRF an omission is already partly expected (alpha low), so
the value decays slowly; after CRF the first omission is maximally surprising (alpha
spikes), so it decays fast.

The value's decay is TIME-BASED (per step), which makes the control clean: with a
constant associability ('fixed'), the value's extinction rate constant is identical for
CRF and PRF regardless of their different response rates and baselines, so both reach
25% of baseline in the SAME number of sessions -- no PREE. Any PREE under 'pearce_hall'
is therefore attributable to associability alone, not to a response-rate or baseline
confound.

Run:   python experiments/exp027_pree_pearce_hall.py
Saves: outputs/logs/exp027_pree.json
       outputs/figures/exp027_pree.png
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from behavioral_md.chamber import ChamberConfig, run_pree
from behavioral_md.experiment_utils import save_results_json
from behavioral_md.visualization import plot_pree

N_ORG = 400
N_TRAIN = 25
N_EXT = 30
SESS_STEPS = 40
REINF_PROB_PRF = 0.25
SEED = 0


def _config(rule: str) -> ChamberConfig:
    return ChamberConfig(
        associability_rule=rule, learning_rate=0.12, value_extinction=0.005,
        approach_gain=4.0, emission_bias=2.0, temperature=0.6,
        ph_eta=0.7, ph_init=0.4, ph_floor=0.05,
    )


def _sessions_to_criterion(value: np.ndarray, nt: int, frac: float = 0.25) -> int:
    base = value[nt - 1]
    ext = value[nt:]
    below = np.where(ext <= frac * base)[0]
    return int(below[0] + 1) if len(below) else int(len(ext))


def _run(rule: str) -> dict:
    cfg = _config(rule)
    crf = run_pree(1.0, cfg, N_ORG, N_TRAIN, N_EXT, SESS_STEPS, seed=SEED)
    prf = run_pree(REINF_PROB_PRF, cfg, N_ORG, N_TRAIN, N_EXT, SESS_STEPS, seed=SEED)
    out = {}
    for name, r in (("crf", crf), ("prf", prf)):
        base = r["rate"][N_TRAIN - 1]
        out[name] = {
            "sessions_to_25pct_value": _sessions_to_criterion(np.asarray(r["value"]), N_TRAIN),
            "rate_prop_by_session": (np.asarray(r["rate"]) / max(base, 1e-9)).tolist(),
            "assoc_at_ext_onset": float(r["assoc"][N_TRAIN]),
            "value_train_end": float(r["value"][N_TRAIN - 1]),
        }
    return out


def main() -> None:
    fixed = _run("fixed")
    ph = _run("pearce_hall")
    results = {
        "n_train": N_TRAIN, "n_ext": N_EXT, "reinf_prob_prf": REINF_PROB_PRF,
        "fixed": fixed, "pearce_hall": ph,
    }
    save_results_json("exp027_pree.json", results)

    def line(tag, d):
        sc, sp = d["crf"]["sessions_to_25pct_value"], d["prf"]["sessions_to_25pct_value"]
        verdict = "PREE (PRF persists longer)" if sp > sc else ("EQUAL" if sp == sc else "no PREE")
        ac, ap = d["crf"]["assoc_at_ext_onset"], d["prf"]["assoc_at_ext_onset"]
        print(f"  [{tag:11s}] sessions-to-25% value: CRF={sc}  PRF={sp}   {verdict}")
        print(f"                assoc at extinction onset: CRF={ac:.3f}  PRF={ap:.3f}")

    print(f"PREE: CRF (p=1.0) vs PRF (p={REINF_PROB_PRF}), then extinction\n")
    line("fixed", fixed)
    line("pearce_hall", ph)

    crf = np.asarray(ph["crf"]["rate_prop_by_session"])
    prf = np.asarray(ph["prf"]["rate_prop_by_session"])
    p = plot_pree(crf, prf, N_TRAIN, Path("outputs/figures/exp027_pree.png"))
    print("\nSaved outputs/logs/exp027_pree.json")
    print(f"Saved {p}")


if __name__ == "__main__":
    main()
