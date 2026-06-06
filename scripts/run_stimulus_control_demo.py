"""Operant stimulus control: a discriminative stimulus comes to control the emitted response.

The three-term contingency S: R -> C. A response is reinforced in the presence of S+ (a cue value)
and not in S-delta. Over training, responding comes under stimulus control: the organism emits the
response under S+ and withholds it under S-delta, and to a novel/neutral cue it responds at an
intermediate level (a generalization decrement).

This is the SAME associative mechanism as the generalization / peak-shift demos, not a separate
"operant" process. Stimulus-consequence (respondent) and response-consequence (operant) relations
are inseparable, as in real organisms: there is one cue->drive association (the CueReceptorField:
Shepard-tuned receptors, summed-error Rescorla-Wagner), updated by the three-term contingency -- it
changes only when a response meets a reinforcer in the presence of the cue. The demos differ only in
PROCEDURE (here reinforcement is response-contingent) and READOUT (here the emitted response, via a
logistic/Luce rule over the cue drive), not in kind.

On each presentation the response is reinforced only under S+; the credited (emitted) response then
strengthens the cue association under S+ and extinguishes it under S-delta. A controlled procedure
(balanced S+/S-delta exposure) is used, as in run_peak_shift_demo, because survival foraging cannot
balance exposure. Per-agent sensory noise gives the population spread for the 95% CI.

Run:  python scripts/run_stimulus_control_demo.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from behavioral_md.config import SimulationConfig
from behavioral_md.generalization import CueReceptorField
from behavioral_md.parallel import run_sweep

FIG_DIR = Path("outputs/figures")
N_AGENTS = 200
N_BLOCKS = 300        # each block = one S+ and one S-delta presentation
S_PLUS = 0.40         # discriminative stimulus for reinforcement
S_DELTA = 0.60        # signals non-reinforcement (extinction)
S_NEUTRAL = 0.85      # a novel cue, never trained (generalization-decrement probe)
RATE = 0.05
LAM = 1.0
BIAS, TEMP = 0.0, 0.5  # logistic (Luce) emission over the cue drive
SENSE_NOISE = 0.02
PROBE_VALUES = np.linspace(0.0, 1.0, 41)


def _p_respond(drive: float) -> float:
    return 1.0 / (1.0 + np.exp(-(drive - BIAS) / TEMP))


def agent_worker(cell: dict[str, Any]) -> dict[str, Any]:
    seed = cell["seed"]
    rng = np.random.default_rng(seed)
    cfg = SimulationConfig()
    field = CueReceptorField(cfg.n_cue_receptors, cfg.cue_generalization_beta,
                             cfg.history_weight_min, cfg.history_weight_max)
    acq_plus = np.zeros(N_BLOCKS)
    acq_delta = np.zeros(N_BLOCKS)
    for b in range(N_BLOCKS):
        for value, is_plus in ((S_PLUS, True), (S_DELTA, False)):
            sensed = float(np.clip(value + rng.normal(0.0, SENSE_NOISE), 0.0, 1.0))
            drive = field.drive(sensed, 1.0)          # cue->response drive (caches activations)
            p = _p_respond(drive)
            responded = rng.random() < p
            # operant three-term contingency: reinforce the response only under S+
            mag = 1.0 if (is_plus and responded) else 0.0
            elig = 1.0 if responded else 0.0          # credit the emitted response
            field.learn(elig, mag, RATE, LAM)
            (acq_plus if is_plus else acq_delta)[b] = p
    gradient = [_p_respond(field.response(float(v))) for v in PROBE_VALUES]
    return {"acq_plus": acq_plus, "acq_delta": acq_delta, "gradient": gradient}


def main(n_agents: int = N_AGENTS) -> None:
    print(f"Operant stimulus control: {n_agents} agents, S+={S_PLUS} (reinforced) vs "
          f"S-delta={S_DELTA} (extinction), {N_BLOCKS} blocks...")
    res = run_sweep(agent_worker, [{"seed": s} for s in range(n_agents)], progress_every=50)
    acq_plus = np.array([r["acq_plus"] for r in res]).mean(0)
    acq_delta = np.array([r["acq_delta"] for r in res]).mean(0)
    grad = np.array([r["gradient"] for r in res])
    gmean, gsem = grad.mean(0), grad.std(0, ddof=1) / np.sqrt(grad.shape[0])

    def at(v):
        return gmean[int(np.argmin(np.abs(PROBE_VALUES - v)))]
    print(f"  final P(respond): S+ {acq_plus[-1]:.2f}, S-delta {acq_delta[-1]:.2f}  "
          f"(discrimination index {acq_plus[-1] - acq_delta[-1]:+.2f})")
    print(f"  gradient at S+ {at(S_PLUS):.2f}, S-delta {at(S_DELTA):.2f}, "
          f"neutral {at(S_NEUTRAL):.2f} (generalization decrement)")

    fig, (axa, axg) = plt.subplots(1, 2, figsize=(11, 4.3))
    axa.plot(acq_plus, color="tab:green", label="S+ (reinforced)")
    axa.plot(acq_delta, color="tab:red", label="S-delta (extinction)")
    axa.set_xlabel("training block")
    axa.set_ylabel("P(respond)")
    axa.set_ylim(0, 1)
    axa.legend(fontsize=9)
    axa.set_title("Discrimination is acquired (responding diverges)")
    axg.plot(PROBE_VALUES, gmean, "-", color="black")
    axg.fill_between(PROBE_VALUES, gmean - 1.96 * gsem, gmean + 1.96 * gsem, color="0.8")
    marks = ((S_PLUS, ":", "S+"), (S_DELTA, "--", "S-delta"), (S_NEUTRAL, "-.", "neutral"))
    for v, ls, lab in marks:
        axg.axvline(v, color="black", ls=ls, lw=1.2)
        axg.text(v + 0.01, 0.95, lab, fontsize=9)
    axg.set_xlabel("test cue value")
    axg.set_ylabel("P(respond)")
    axg.set_ylim(0, 1)
    axg.set_title("Stimulus control: responding graded by cue")
    fig.suptitle("Operant stimulus control: S+ / S-delta / neutral", fontsize=13)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "stimulus_control.png"
    fig.savefig(out, dpi=130)
    print(f"Wrote {out}")


if __name__ == "__main__":
    import argparse

    _ap = argparse.ArgumentParser()
    _ap.add_argument("--agents", type=int, default=N_AGENTS)
    main(_ap.parse_args().agents)
