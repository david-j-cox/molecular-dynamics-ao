"""Tests for the operant chamber: momentum, Pearce-Hall PREE, and resurgence."""

import numpy as np

from behavioral_md.chamber import (
    ChamberConfig,
    run_multiple_schedule,
    run_pree,
    run_punishment_choice,
    run_resurgence,
    run_risk_choice,
)

INF = float("inf")


def _sessions_to_crit(value, nt, frac=0.25):
    base = value[nt - 1]
    ext = np.asarray(value[nt:])
    below = np.where(ext <= frac * base)[0]
    return int(below[0] + 1) if len(below) else int(len(ext))


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


def _momentum_cfg(gain):
    return ChamberConfig(
        motiv_strength=0.3, energy_init=0.6, emission_bias=0.6, temperature=0.5,
        ctx_drive_gain=0.6, momentum_mass_gain=gain, learning_rate=0.08,
        value_extinction=0.004, reinf_asymptote=1.0, ctx_learning_rate=0.05,
        ctx_omission_rate=0.0015, ctx_asymptote=2.0,
    )


def test_momentum_mass_modulated_decay():
    """Mass divides the value decay: with gain>0 the rich response value resists
    extinction more than the lean; with gain=0 they decay identically (control)."""
    nb, nd = 15, 25
    ctrl = run_multiple_schedule([5.0, 40.0], _momentum_cfg(0.0), 200, 200, nb, nd,
                                 disruptor="extinction", seed=0)
    mom = run_multiple_schedule([5.0, 40.0], _momentum_cfg(3.0), 200, 200, nb, nd,
                                disruptor="extinction", seed=0)
    # Value-based, scale-free criterion: gain=0 -> equal (no momentum, the control).
    c_rich = _sessions_to_crit(ctrl["value"][:, 0], nb)
    c_lean = _sessions_to_crit(ctrl["value"][:, 1], nb)
    assert c_rich == c_lean
    # gain>0 -> rich (higher mass) resists more.
    m_rich = _sessions_to_crit(mom["value"][:, 0], nb)
    m_lean = _sessions_to_crit(mom["value"][:, 1], nb)
    assert m_rich > m_lean


def _pree_cfg(rule):
    return ChamberConfig(
        associability_rule=rule, learning_rate=0.12, value_extinction=0.005,
        approach_gain=4.0, emission_bias=2.0, temperature=0.6,
        ph_eta=0.7, ph_init=0.4, ph_floor=0.05,
    )


def test_pree_requires_pearce_hall():
    """Partial reinforcement persists longer in extinction (PREE) under Pearce-Hall;
    under a fixed associability CRF and PRF decay at the same rate (no PREE)."""
    nt, ne, ss = 25, 30, 40
    # Fixed control: time-based decay is scale-free, so CRF and PRF match exactly.
    fc = _pree_cfg("fixed")
    crf = run_pree(1.0, fc, 300, nt, ne, ss, seed=0)
    prf = run_pree(0.25, fc, 300, nt, ne, ss, seed=0)
    assert _sessions_to_crit(crf["value"], nt) == _sessions_to_crit(prf["value"], nt)
    # Pearce-Hall: PRF (omission less surprising) extinguishes more slowly -> PREE.
    pc = _pree_cfg("pearce_hall")
    crf2 = run_pree(1.0, pc, 300, nt, ne, ss, seed=0)
    prf2 = run_pree(0.25, pc, 300, nt, ne, ss, seed=0)
    assert _sessions_to_crit(prf2["value"], nt) > _sessions_to_crit(crf2["value"], nt)
    # Diagnostic: the first omission is more surprising after CRF than after PRF.
    assert crf2["assoc"][nt] > prf2["assoc"][nt]


def _resurgence(res):
    pb = res["phase_blocks"]
    r1 = np.asarray(res["r1"])
    end_p2 = r1[2 * pb - 3:2 * pb].mean()
    test_p3 = r1[2 * pb + 2:].mean()
    return end_p2, test_p3


def _resurgence_cfg(rule, gain):
    return ChamberConfig(
        value_rule=rule, momentum_mass_gain=gain, learning_rate=0.10,
        value_extinction=0.02, approach_gain=4.0, temperature=0.5, act_tau=3.0,
        inhib_rate=0.06, inhib_relax=0.12, inhib_passive_decay=0.005,
    )


def test_resurgence_emerges_and_control():
    """R1 recovers in phase 3 (resurgence) when R2 is extinguished; if R2 stays
    reinforced (control), allocation does not flow back -> no resurgence."""
    res = run_resurgence(_resurgence_cfg("single", 0.0), 200, 1500, seed=0,
                         r_other=0.2, block=50)
    end_p2, test_p3 = _resurgence(res)
    assert test_p3 > end_p2 + 0.03            # R1 rises at test
    ctrl = run_resurgence(_resurgence_cfg("single", 0.0), 200, 1500, seed=0,
                          r_other=0.2, block=50, control_reinforce_r2=True)
    c_end, c_test = _resurgence(ctrl)
    assert c_test < end_p2 + 0.03             # no rise when R2 keeps paying


def test_resurgence_dual_preserves_excitation():
    """The dual rule preserves R1's excitation, so it is far less suppressed at the
    end of phase 2 than under the single (RW) rule that erodes it."""
    single = run_resurgence(_resurgence_cfg("single", 0.0), 200, 1500, seed=0,
                            r_other=0.2, block=50)
    dual = run_resurgence(_resurgence_cfg("dual", 0.0), 200, 1500, seed=0,
                          r_other=0.2, block=50)
    assert _resurgence(dual)[0] > _resurgence(single)[0] + 0.05


def test_resurgence_as_choice_rac():
    """Resurgence as Choice (Shahan & Craig): a temporally-weighted reinforcement value
    with matching allocation. The target resurges (transient peak in phase 3) purely
    because the alternative's integrated value decays once its reinforcement stops;
    keeping the alternative reinforced (control) abolishes it. No preserved target
    strength is involved."""
    cfg = ChamberConfig(value_rule="rac", rac_tau=500.0, rac_bump=0.04,
                        rac_sensitivity=1.0, rac_floor=0.1)
    res = run_resurgence(cfg, 300, 1800, seed=0, r_other=0.2, block=50)
    pb = res["phase_blocks"]
    r1 = np.asarray(res["r1"])
    end_p2 = r1[2 * pb - 3:2 * pb].mean()
    peak_p3 = r1[2 * pb:].max()
    assert peak_p3 > end_p2 + 0.05                       # target resurges
    ctrl = run_resurgence(cfg, 300, 1800, seed=0, r_other=0.2, block=50,
                          control_reinforce_r2=True)
    cr1 = np.asarray(ctrl["r1"])
    assert cr1[2 * pb:].max() < end_p2 + 0.05            # control abolishes it


def _pun_alloc(model, alt_vi, punish, **kw):
    cfg = ChamberConfig(pun_tau=800.0, pun_bump=0.04, pun_floor=0.1, **kw)
    res = run_punishment_choice(model, cfg, 400, 3000, vi_reinf=[5.0, alt_vi],
                                vi_punish=[punish, INF], seed=0)
    e = res["emit"].sum(0)
    return e[0] / e.sum(), float(np.log(e[0] / e[1]))


def test_punishment_suppresses_target():
    """Adding punishment to one of two equally-reinforced responses suppresses it,
    in all three accounts."""
    for model, kw in (("subtractive", dict(pun_c=1.0)), ("competitive", dict(pun_c=1.5)),
                      ("concatenated", dict(pun_a_p=1.0))):
        base, _ = _pun_alloc(model, 5.0, INF, **kw)
        pun, _ = _pun_alloc(model, 5.0, 8.0, **kw)
        assert base > 0.45                       # ~equal allocation with no punishment
        assert pun < base - 0.05                 # punishment suppresses


def test_de_villiers_vs_deluty_dissociation():
    """Punishment suppression depends on the alternative's reinforcement rate with
    OPPOSITE slopes: subtractive rises with alternative richness, competitive falls."""
    def suppression(model, alt_vi, **kw):
        _, lo_base = _pun_alloc(model, alt_vi, INF, **kw)
        _, lo_pun = _pun_alloc(model, alt_vi, 10.0, **kw)
        return lo_base - lo_pun
    sub_lean = suppression("subtractive", 20.0, pun_c=1.0)
    sub_rich = suppression("subtractive", 4.0, pun_c=1.0)
    comp_lean = suppression("competitive", 20.0, pun_c=1.5)
    comp_rich = suppression("competitive", 4.0, pun_c=1.5)
    assert sub_rich > sub_lean                   # subtractive rises with alt richness
    assert comp_rich < comp_lean                 # competitive falls with alt richness


def test_concatenated_recovers_monotonic_ap():
    """A larger set punishment sensitivity yields a larger recovered a_p (separable,
    monotonic) from a both-punished ratio sweep."""
    def recover(set_ap):
        cfg = ChamberConfig(pun_tau=800.0, pun_bump=0.04, pun_floor=0.1,
                            pun_a_r=1.0, pun_a_p=set_ap)
        logB, logP = [], []
        for pv0 in (15.0, 8.0, 5.0):
            res = run_punishment_choice("concatenated", cfg, 400, 3000,
                                        vi_reinf=[5.0, 5.0], vi_punish=[pv0, 15.0], seed=0)
            e, pc = res["emit"].sum(0), res["punished"].sum(0)
            logB.append(np.log(e[0] / e[1]))
            logP.append(np.log(pc[0] / pc[1]))
        return -np.polyfit(logP, logB, 1)[0]
    assert recover(1.5) > recover(0.5)


def _risk_reversal(risky, util_shape):
    """(mean P(risky) just below requirement, just above) for one prep + utility."""
    cfg = ChamberConfig(temperature=0.02, energy_capacity=1.0)
    res = run_risk_choice([(1.0, 0.05)], risky, cfg, 4000, 800, seed=0, e_req=0.5,
                          util_width=0.08, cost=0.05, e_init=0.5, util_shape=util_shape)
    rb = np.asarray(res["risky_by_energy"])
    bc = np.asarray(res["bin_count"])
    b = np.asarray(res["energy_bins"])
    rb = np.where(bc >= 2000, rb, np.nan)
    below = np.nanmean(rb[(b < 0.5) & (b > 0.3)])
    above = np.nanmean(rb[(b > 0.5) & (b < 0.7)])
    return float(below), float(above)


def test_energy_budget_rule_reward_variance():
    """Caraco: risk-prone below the energy requirement, risk-averse above (reward
    variance) -- only under the survival utility; the linear control shows no reversal."""
    risky = [(0.5, 0.0), (0.5, 0.10)]                     # matched mean 0.05
    below, above = _risk_reversal(risky, "survival")
    assert below > above + 0.15                           # the reversal
    lb, la = _risk_reversal(risky, "linear")
    assert abs(lb - la) < 0.08                            # control: no reversal


def test_energy_budget_rule_predation_variance():
    """Same reversal with a predation downside (rich-but-risky vs lean-safe)."""
    risky = [(0.8, 0.0875), (0.2, -0.10)]                 # matched mean 0.05
    below, above = _risk_reversal(risky, "survival")
    assert below > above + 0.2
    lb, la = _risk_reversal(risky, "linear")
    assert abs(lb - la) < 0.08
