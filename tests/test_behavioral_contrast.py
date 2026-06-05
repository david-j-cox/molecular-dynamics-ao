"""Behavioral contrast: characterize what the engine does (experiments/exp040).

The chamber's per-component value and context are LOCAL (no cross-component term), so extinguishing
one component leaves the other unchanged when motivation is fixed (energy clamped): no associative
behavioral contrast. With energy free, a worsened component raises shared deprivation and
lifts the unchanged component's rate (a small positive contrast via motivation, not the local
mechanism). These lock in that characterization.
"""

from __future__ import annotations

from behavioral_md.chamber import ChamberConfig, run_contrast

CFG = ChamberConfig(motiv_strength=2.0, energy_init=0.5, emission_bias=1.2, temperature=0.5,
                    ctx_drive_gain=0.8, momentum_mass_gain=0.0, reinf_asymptote=1.0)
ARGS = dict(n_org=300, comp_steps=300, n_baseline=15, n_phase2=12, changed=1,
            manipulation="extinction", seed=0)


def _rates(res):
    pr, nb, a, b = res["press_rate"], res["n_baseline"], res["other"], res["changed"]
    return (pr[nb - 3:nb, a].mean(), pr[nb:, a][-3:].mean(),
            pr[nb - 3:nb, b].mean(), pr[nb:, b][-3:].mean())


def test_extinction_suppresses_the_changed_component():
    a_base, a_p2, b_base, b_p2 = _rates(run_contrast(10.0, CFG, clamp_energy=True, **ARGS))
    assert b_p2 < 0.6 * b_base, f"extinguished component should drop, got {b_base:.3f}->{b_p2:.3f}"


def test_no_associative_contrast_when_motivation_fixed():
    a_base, a_p2, _, _ = _rates(run_contrast(10.0, CFG, clamp_energy=True, **ARGS))
    assert abs(a_p2 - a_base) < 0.05 * a_base, (
        f"with energy clamped the unchanged component should not shift (no cross-component term), "
        f"got {a_base:.3f}->{a_p2:.3f}")


def test_shared_deprivation_gives_small_positive_contrast():
    a_base, a_p2, _, _ = _rates(run_contrast(10.0, CFG, clamp_energy=False, **ARGS))
    assert a_p2 > a_base, f"energy-free: component A should rise, got {a_base:.3f}->{a_p2:.3f}"
