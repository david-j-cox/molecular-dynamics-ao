"""Behavioral contrast emerges from the convex shared-hunger term (experiments/exp040).

Contrast is not installed as a relative-rate term: both components feed one energy reserve and the
drive carries a convex hunger term, so worsening one component makes the organism hungrier (positive
contrast in the other) and enriching it makes the organism sated (negative contrast). Setting
motiv_strength = 0 removes the hunger term and the per-component mechanism is purely local -> no
contrast. By convexity, positive contrast shows at a sated baseline and negative at a hungry one.
"""

from __future__ import annotations

from behavioral_md.chamber import ChamberConfig, run_contrast

ARGS = dict(n_org=300, comp_steps=300, n_baseline=15, n_phase2=12, changed=1, clamp_energy=False,
            seed=0)


def _cfg(motiv_strength, food_energy):
    return ChamberConfig(motiv_strength=motiv_strength, energy_init=0.5, emission_bias=1.2,
                         temperature=0.5, ctx_drive_gain=0.8, food_energy=food_energy,
                         deficit_exponent=2.0)


def _ratio(res):
    pr, nb, a = res["press_rate"], res["n_baseline"], res["other"]
    return pr[nb:, a][-3:].mean() / pr[nb - 3:nb, a].mean()


def test_positive_contrast_at_sated_baseline():
    r = _ratio(run_contrast(14.0, _cfg(1.5, 0.15), manipulation="extinction", **ARGS))
    assert r > 1.15, f"worsening B should raise A (positive contrast), got ratio {r:.2f}"


def test_negative_contrast_at_hungry_baseline():
    r = _ratio(run_contrast(20.0, _cfg(1.5, 0.06), manipulation="enrich", vi_phase2=4.0, **ARGS))
    assert r < 0.85, f"enriching B should lower A (negative contrast), got ratio {r:.2f}"


def test_contrast_knocked_out_without_hunger():
    pos = _ratio(run_contrast(14.0, _cfg(0.0, 0.15), manipulation="extinction", **ARGS))
    neg = _ratio(run_contrast(20.0, _cfg(0.0, 0.06), manipulation="enrich", vi_phase2=4.0, **ARGS))
    assert abs(pos - 1.0) < 0.05, f"no positive contrast without hunger, got {pos:.2f}"
    assert abs(neg - 1.0) < 0.05, f"no negative contrast without hunger, got {neg:.2f}"
