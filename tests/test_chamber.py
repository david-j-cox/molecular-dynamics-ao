"""Tests for the operant chamber, including the behavioral-momentum runner."""

from behavioral_md.chamber import ChamberConfig, run_multiple_schedule


def test_multiple_schedule_shapes():
    cfg = ChamberConfig()
    res = run_multiple_schedule([5.0, 40.0], cfg, n_org=20, comp_steps=50,
                                n_baseline=2, n_disruption=2, disruptor="satiation", seed=0)
    for key in ("press_rate", "reinf_rate", "ctx"):
        assert res[key].shape == (4, 2)


def test_momentum_under_satiation():
    """Rich component (higher reinforcement rate) is more resistant to satiation."""
    cfg = ChamberConfig(motiv_strength=2.0, energy_init=0.5, emission_bias=1.2,
                        ctx_drive_gain=0.8, reinf_asymptote=1.0)
    res = run_multiple_schedule([5.0, 40.0], cfg, n_org=120, comp_steps=300,
                                n_baseline=12, n_disruption=8, disruptor="satiation", seed=0)
    pr = res["press_rate"]
    base = pr[9:12].mean(0)
    resistance = (pr[12:] / base).mean(0)
    # Rich component (index 0) retains a larger proportion of baseline.
    assert resistance[0] > resistance[1]
    # And the rich context value (momentum "mass") exceeds the lean one.
    assert res["ctx"][11, 0] > res["ctx"][11, 1]
