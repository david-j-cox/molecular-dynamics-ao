"""Anticipatory contrast from a learned predicted-income term (experiments/exp041).

With energy clamped (current hunger fixed, so the exp040 current-reserve route is off), a learned
predicted-income discount produces anticipatory contrast with the correct sign: respond MORE before
a lean component, LESS before a rich one. Setting antic_discount = 0 removes it -> evidence it is
the learned predicted income, nothing installed.
"""

from __future__ import annotations

from behavioral_md.chamber import ChamberConfig, run_contrast

ARGS = dict(n_org=400, comp_steps=300, n_baseline=20, n_phase2=15, changed=1, clamp_energy=True,
            seed=0)


def _cfg(antic_discount):
    return ChamberConfig(motiv_strength=2.0, energy_init=0.5, emission_bias=1.2, temperature=0.5,
                         ctx_drive_gain=0.8, antic_lr=0.2, antic_discount=antic_discount)


def _ratio(res):
    pr, nb, a = res["press_rate"], res["n_baseline"], res["other"]
    return pr[nb:, a][-3:].mean() / pr[nb - 3:nb, a].mean()


def test_respond_more_before_a_lean_component():
    r = _ratio(run_contrast(14.0, _cfg(0.4), manipulation="extinction", **ARGS))
    assert r > 1.1, f"a lean component coming should raise current responding, got {r:.2f}"


def test_respond_less_before_a_rich_component():
    r = _ratio(run_contrast(14.0, _cfg(0.4), manipulation="enrich", vi_phase2=3.5, **ARGS))
    assert r < 0.95, f"a rich component coming should lower current responding, got {r:.2f}"


def test_anticipatory_contrast_knocked_out_without_discount():
    lean = _ratio(run_contrast(14.0, _cfg(0.0), manipulation="extinction", **ARGS))
    rich = _ratio(run_contrast(14.0, _cfg(0.0), manipulation="enrich", vi_phase2=3.5, **ARGS))
    assert abs(lean - 1.0) < 0.04 and abs(rich - 1.0) < 0.04, (
        f"no anticipatory contrast without the discount, got lean {lean:.2f}, rich {rich:.2f}")
