"""Tests for the first-principles survival DP (risk-sensitivity derived, not imposed)."""

import numpy as np

from behavioral_md.survival import evolve_risk_policy, risk_threshold, survival_dp


def _solve(day=24, night=24, metab=0.03, s=0.05):
    return survival_dp([(1.0, s)], [(0.5, 0.0), (0.5, 2 * s)], day, night, metab)


def test_night_requirement_is_emergent():
    """The requirement is night_steps * metabolism, not a parameter."""
    res = _solve(night=20, metab=0.03)
    assert np.isclose(res["night_requirement"], 0.6)


def test_risk_prone_band_exists():
    """Survival maximization makes the organism gamble for some reserves (the band)."""
    res = _solve()
    assert res["policy_risky"].sum() > 0


def test_risk_averse_when_well_fed():
    """With a full reserve the safe option already secures survival -- never gamble."""
    res = _solve()
    e = res["energy"]
    full = e >= 0.95
    assert res["policy_risky"][:, full].sum() == 0


def test_threshold_rises_toward_dusk():
    """The safe-suffices edge climbs through the day toward the night requirement."""
    res = _solve()
    thr = risk_threshold(res)            # upper edge per day-step (NaN where none)
    dawn = np.nanmax(thr[:3])
    dusk = np.nanmax(thr[-3:])
    assert dusk > dawn
    assert dusk <= res["night_requirement"] + 1e-6


def test_ruin_below_the_band_at_dusk():
    """At dusk, a near-empty reserve cannot be saved even by gambling -> not risk-prone
    (doomed either way), so risk-proneness is bounded below by ruin, not unbounded."""
    res = _solve()
    e = res["energy"]
    dusk = res["policy_risky"][-1]
    very_low = e < 0.2
    assert dusk[very_low].sum() == 0     # no gambling deep in the ruin region at dusk

def test_risk_policy_evolves_time_dependent_threshold():
    """Risk-sensitivity EVOLVES from selection alone: the time-of-day slope b goes
    positive (threshold rises toward dusk) and the evolved dusk threshold lands near the
    DP optimum, with no imposed utility, DP, or learning rule."""
    safe, risky = [(1.0, 0.05)], [(0.5, 0.0), (0.5, 0.10)]
    ev = evolve_risk_policy(safe, risky, 24, 24, 0.03, pop_size=2000, n_generations=120,
                            n_cycles=3, seed=0)
    b = np.nanmean(ev["mean_b"][-20:])
    assert b > 0.1                                   # threshold rises toward dusk
    assert 0.0 < ev["survival"][-1] < 1.0            # real selection pressure
    dp = risk_threshold(survival_dp(safe, risky, 24, 24, 0.03))
    evolved = np.array(ev["evolved_theta"])
    assert abs(evolved[-1] - dp[-1]) < 0.15          # evolved dusk threshold ~ DP optimum


def test_within_life_learning_discovers_the_rule():
    """Starting ignorant, the organism learns the risky option's variance from experience
    and its planned policy comes to gamble where the optimum prescribes (recall 0 -> 1);
    the learned threshold lands on the DP optimum."""
    from behavioral_md.survival import simulate_learning_choice
    safe, risky = [(1.0, 0.05)], [(0.5, 0.0), (0.5, 0.10)]
    r = simulate_learning_choice(safe, risky, 24, 24, 0.03, n_org=40, n_cycles=30, seed=0)
    rec = np.array(r["gamble_recall"])
    assert rec[0] < 0.2                                   # ignorant: does not gamble
    assert np.mean(rec[-10:]) > 0.8                       # learned: gambles in the band
    # Estimated variance converges to the truth (it discovered risky is risky).
    assert abs(r["risky_variance"][-1] - r["true_risky_variance"]) < 0.0005
    # Learned dusk threshold matches the DP optimum.
    assert abs(r["learned_theta"][-1] - r["dp_theta"][-1]) < 0.1


def test_model_free_learner_recovers_the_rule():
    """A model-free learner (tabular Q learned by Monte-Carlo from the survival signal, no
    model and no planning) recovers the energy-budget rule: gamble recall rises and the
    learned threshold lands near the DP optimum."""
    from behavioral_md.survival import simulate_model_free_choice
    safe, risky = [(1.0, 0.05)], [(0.5, 0.0), (0.5, 0.10)]
    r = simulate_model_free_choice(safe, risky, 24, 24, 0.03, n_org=200, n_cycles=200, seed=0)
    rec = np.array(r["gamble_recall"])
    assert np.mean(rec[-30:]) > np.mean(rec[:10]) + 0.1   # it learns (recall climbs)
    assert np.mean(rec[-30:]) > 0.8                       # recovers most of the band
    learned = np.array(r["learned_theta"], float)
    dp = np.array(r["dp_theta"], float)
    assert np.nanmean(np.abs(learned - dp)) < 0.1         # threshold ~ DP optimum
