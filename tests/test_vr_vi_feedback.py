"""VR >> VI from molar feedback sensitivity (experiments/exp047).

The per-press value rule alone gives little VR>VI difference; the response-reinforcer correlation
boost (chamber feedback_gain, Baum's correlation-based law) makes VR press rates clearly exceed VI's
at matched reinforcement rate. VR-30 and VI-60 deliver about the same reinforcement rate.
"""

from __future__ import annotations

from behavioral_md.chamber import ChamberConfig, run_chamber


def _run(sched, param, feedback_gain, n_steps=4000, warm=1800):
    cfg = ChamberConfig(motiv_strength=0.6, energy_init=0.6, emission_bias=1.0, temperature=0.5,
                        value_extinction=0.03, feedback_gain=feedback_gain)
    r = run_chamber(sched, param, cfg, 300, n_steps, seed=0)
    return r["presses"][warm:].mean(), r["reinforced"][warm:].mean()


def test_feedback_on_gives_vr_over_vi():
    p_vr, r_vr = _run("VR", 30, 50.0)
    p_vi, r_vi = _run("VI", 60, 50.0)
    assert abs(r_vr - r_vi) < 0.004, f"VR-30 / VI-60 should be reinforcement-matched, {r_vr}/{r_vi}"
    assert p_vr > 1.4 * p_vi, f"with feedback VR should clearly exceed VI, got {p_vr / p_vi:.2f}"


def test_feedback_amplifies_the_gap():
    off = _run("VR", 30, 0.0)[0] / _run("VI", 60, 0.0)[0]
    on = _run("VR", 30, 50.0)[0] / _run("VI", 60, 50.0)[0]
    assert on > off + 0.4, f"feedback should widen VR/VI: {off:.2f} -> {on:.2f}"
