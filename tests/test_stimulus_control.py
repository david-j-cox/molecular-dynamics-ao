"""Operant stimulus control (scripts/run_stimulus_control_demo).

A response reinforced under S+ but not S-delta comes under stimulus control: after training the
emitted response is far more likely under S+ than S-delta (discrimination), and the post-training
gradient peaks on the S+ side. Self-contained: drives the engine's CueReceptorField with a logistic
(operant) emission, a few agents, single-process.
"""

from __future__ import annotations

import numpy as np

from behavioral_md.config import SimulationConfig
from behavioral_md.generalization import CueReceptorField

S_PLUS, S_DELTA = 0.40, 0.60
BIAS, TEMP, RATE, NOISE = 0.0, 0.5, 0.05, 0.02
PROBES = np.linspace(0.0, 1.0, 41)


def _p(drive):
    return 1.0 / (1.0 + np.exp(-(drive - BIAS) / TEMP))


def _train(seed, n_blocks=300):
    rng = np.random.default_rng(seed)
    cfg = SimulationConfig()
    f = CueReceptorField(cfg.n_cue_receptors, cfg.cue_generalization_beta,
                         cfg.history_weight_min, cfg.history_weight_max)
    p_plus = p_delta = 0.5
    for _ in range(n_blocks):
        for value, is_plus in ((S_PLUS, True), (S_DELTA, False)):
            sensed = float(np.clip(value + rng.normal(0.0, NOISE), 0.0, 1.0))
            p = _p(f.drive(sensed, 1.0))
            responded = rng.random() < p
            f.learn(1.0 if responded else 0.0, 1.0 if (is_plus and responded) else 0.0, RATE, 1.0)
            if is_plus:
                p_plus = p
            else:
                p_delta = p
    grad = np.array([_p(f.response(float(v))) for v in PROBES])
    return p_plus, p_delta, grad


def test_discrimination_acquired():
    runs = [_train(s) for s in range(6)]
    p_plus = np.mean([r[0] for r in runs])
    p_delta = np.mean([r[1] for r in runs])
    assert p_plus - p_delta > 0.2, f"S+ should exceed S-delta, got {p_plus:.2f}/{p_delta:.2f}"
    assert p_plus > 0.75, f"responding under S+ should be high, got {p_plus:.2f}"


def test_gradient_peaks_on_the_splus_side():
    grad = np.mean([_train(s)[2] for s in range(6)], axis=0)
    peak = PROBES[int(np.argmax(grad))]
    assert peak <= S_PLUS + 0.05, f"the gradient should peak at/below S+, got {peak:.2f}"
    # responding is lower at S-delta than at S+ (stimulus control)
    assert grad[np.argmin(np.abs(PROBES - S_DELTA))] < grad[np.argmin(np.abs(PROBES - S_PLUS))]
