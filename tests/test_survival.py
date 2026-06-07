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


def test_timevarying_dp_matches_constant():
    """With constant per-step outcomes, the time-varying DP reproduces the plain DP."""
    from behavioral_md.survival import survival_dp_timevarying
    safe, risky = [(1.0, 0.05)], [(0.5, 0.0), (0.5, 0.10)]
    a = survival_dp(safe, risky, 12, 12, 0.03)
    b = survival_dp_timevarying([safe] * 12, [risky] * 12, 12, 0.03)
    assert np.allclose(a["policy_risky"], b["policy_risky"])


def test_sun_variance_spread_peaks_in_dark():
    """The sun sets foraging spread: minimal at midday (bright), maximal at dawn/dusk
    (dark), with the mean matched at every step."""
    from behavioral_md.survival import sun_variance_risky
    risky, light = sun_variance_risky(24, 0.05, 0.02, 0.12)
    spread = np.array([abs(r[1][1] - 0.05) for r in risky])
    assert spread[light.argmax()] < spread[light.argmin()]      # bright steadier than dark
    for r in risky:                                             # matched mean each step
        assert abs(sum(p * d for p, d in r) - 0.05) < 1e-9


def test_sun_variance_dusk_lifeline():
    """High-variance dark dusk lowers the ruin edge (a desperate forager is saved) relative
    to a constant-variance control with the same average variance."""
    from behavioral_md.survival import (
        sun_variance_risky,
        survival_dp_timevarying,
    )
    day, night, metab, s = 24, 24, 0.03, 0.05
    risky_sun, _ = sun_variance_risky(day, s, 0.02, 0.12)
    spread = np.array([abs(r[1][1] - s) for r in risky_sun])
    w = float(np.sqrt(np.mean(spread ** 2)))
    sun = survival_dp_timevarying([[(1.0, s)]] * day, risky_sun, night, metab)
    con = survival_dp([(1.0, s)], [(0.5, s - w), (0.5, s + w)], day, night, metab)

    def ruin(res, t):
        prone = np.where(res["policy_risky"][t] > 0.5)[0]
        return res["energy"][prone.min()] if len(prone) else np.nan
    assert ruin(sun, 20) < ruin(con, 20) - 0.05         # dusk: the dark is a lifeline


def test_dusk_survival_lifeline_is_realized_behavior():
    """A population dropped into dusk behind on reserves SURVIVES the night more often under
    the sun's high-variance dark than under a matched constant-variance control -- the ruin-edge
    lifeline realized as who actually lives, and confined to the desperate (no edge once safe)."""
    from behavioral_md.survival import (
        simulate_dusk_survival,
        sun_variance_risky,
        survival_dp,
        survival_dp_timevarying,
    )
    day, night, metab, s = 24, 24, 0.03, 0.05
    risky_sun, _ = sun_variance_risky(day, s, 0.02, 0.12)
    safe_by_step = [[(1.0, s)] for _ in range(day)]
    w = float(np.sqrt(np.mean([abs(r[1][1] - s) ** 2 for r in risky_sun])))
    sun = survival_dp_timevarying(safe_by_step, risky_sun, night, metab)
    risky_con = [[(0.5, s - w), (0.5, s + w)] for _ in range(day)]
    con = survival_dp([(1.0, s)], [(0.5, s - w), (0.5, s + w)], day, night, metab)

    reserves = np.linspace(0.30, 0.95, 25)
    a = simulate_dusk_survival(sun, safe_by_step, risky_sun, metab, 20, reserves, seed=0)
    b = simulate_dusk_survival(con, safe_by_step, risky_con, metab, 20, reserves, seed=0)
    adv = a["survival"] - b["survival"]
    assert adv.max() > 0.1                               # a real lifeline in the desperate band
    assert adv.min() > -0.02                             # never a net liability at dusk
    assert adv[reserves > 0.85].max() < 0.02             # no edge once the reserve is already safe


def test_skewed_outcomes_fixes_mean_and_variance():
    """The continuous skewed distribution holds mean and variance fixed while the skew sign
    and magnitude track the parameter -- the setup that lets skew be probed where mean-variance
    theory predicts indifference."""
    from behavioral_md.survival import outcome_moments, skewed_outcomes
    for sk in (-1.2, -0.5, 0.0, 0.5, 1.2):
        m, var, skew = outcome_moments(skewed_outcomes(0.05, 0.06, sk))
        assert abs(m - 0.05) < 1e-9                       # mean fixed
        assert abs(var - 0.06 ** 2) < 1e-9               # variance fixed
        assert np.sign(round(skew, 6)) == np.sign(sk)    # skew sign follows the parameter
    s_lo = outcome_moments(skewed_outcomes(0.05, 0.06, 0.4))[2]
    s_hi = outcome_moments(skewed_outcomes(0.05, 0.06, 1.2))[2]
    assert s_hi > s_lo                                    # stronger parameter -> more skew


def test_continuous_outcomes_remove_the_comb():
    """A continuous gamble removes the two-point reachability comb on the safe-suffices edge,
    landing on the same dusk requirement -- the band is survival, not the discretization."""
    from behavioral_md.survival import skewed_outcomes
    day, night, metab, m, sd = 24, 24, 0.03, 0.05, 0.06
    tp = risk_threshold(survival_dp([(1.0, m)], [(0.5, m - sd), (0.5, m + sd)],
                                    day, night, metab, n_egrid=1601))
    co = risk_threshold(survival_dp([(1.0, m)], skewed_outcomes(m, sd, 0.0),
                                    day, night, metab, n_egrid=1601))
    assert np.nanstd(np.diff(co)) < 0.2 * np.nanstd(np.diff(tp))   # comb gone
    assert abs(np.nanmax(co[-3:]) - np.nanmax(tp[-3:])) < 1e-9     # same dusk requirement


def test_skew_preference_reverses_at_the_requirement():
    """The energy-budget rule extends to the third moment: at fixed mean and variance the
    survival policy is NOT skew-indifferent, and its preference reverses at the requirement --
    negative skew preferred below it (steady gains), positive skew above it (catastrophe
    avoidance)."""
    from behavioral_md.survival import skewed_outcomes
    day, night, metab, m, sd = 24, 24, 0.03, 0.05, 0.06
    R = night * metab

    def regime_adv(sk):
        res = survival_dp([(1.0, m)], skewed_outcomes(m, sd, sk), day, night, metab, n_egrid=1601)
        e = res["energy"]
        adv = res["q_risky"] - res["q_safe"]
        below = adv[:, (e > 0.05) & (e < R)].mean()
        above = adv[:, (e > R) & (e < 0.98)].mean()
        return below, above

    below_left, above_left = regime_adv(-1.2)
    below_right, above_right = regime_adv(1.2)
    assert below_left > below_right + 1e-4                # below R: prefers NEGATIVE skew
    assert above_right > above_left + 1e-4                # above R: prefers POSITIVE skew
    # The two regimes order oppositely in skew -> a genuine reversal, not a flat (mv) response.
    assert (below_left - below_right) * (above_left - above_right) < 0


def test_central_moments_match_outcome_moments():
    """central_moments returns the raw 2nd/3rd/4th central moments consistent with the
    normalized skewness reported by outcome_moments."""
    from behavioral_md.survival import central_moments, outcome_moments, skewed_outcomes
    g = skewed_outcomes(0.05, 0.06, 0.8)
    m, var, skew = outcome_moments(g)
    cm, mu2, mu3, mu4 = central_moments(g)
    assert abs(cm - m) < 1e-12
    assert abs(mu2 - var) < 1e-12
    assert abs(mu3 - skew * var ** 1.5) < 1e-9            # mu3 = skew * sigma^3
    assert mu4 > 0


def test_variance_preference_reversal_is_the_optimal_threshold():
    """The MECHANISM behind the energy-budget rule: the variance-preference field (a mean-
    preserving spread's survival advantage, ~ (1/2) V'' var) reverses sign EXACTLY at the optimal
    policy threshold -- i.e. the rule is the inflection (V'' = 0) of the emergent survival value,
    not an imposed utility. The reversal point tracks the DP threshold across the whole day."""
    from behavioral_md.survival import field_zero_crossing, moment_preference_fields
    f = moment_preference_fields(24, 24, 0.03, n_egrid=1201)
    e, thr = f["energy"], f["threshold"]
    zc = field_zero_crossing(f["variance"], e)
    ok = ~np.isnan(thr) & ~np.isnan(zc)
    assert ok.sum() >= 20
    assert np.corrcoef(zc[ok], thr[ok])[0, 1] > 0.95          # reversal sits on the threshold
    assert np.nanmean(np.abs(zc[ok] - thr[ok])) < 0.02       # to within ~a few grid cells
    # convex below (risk-prone), concave above (risk-averse): the sign of V''.
    assert f["variance"][:, (e > 0.05) & (e < 0.2)].mean() > 0
    assert f["variance"][:, e > 0.95].mean() <= 1e-6


def test_moment_fields_reverse_across_the_requirement():
    """All three measured preference fields are derivatives of one emergent value: the variance
    (V'') and skew (V''') preferences reverse sign across the requirement (the skew reversal
    reproduces richer_worlds: negative-skew below, positive-skew above)."""
    from behavioral_md.survival import moment_preference_fields
    f = moment_preference_fields(24, 24, 0.03, n_egrid=1201)
    e, thr = f["energy"], f["threshold"]
    R = f["night_requirement"]

    def regime(field):
        below, above = [], []
        for t in range(field.shape[0]):
            th = thr[t] if not np.isnan(thr[t]) else R
            b = (e > 0.05) & (e < th - 0.02)
            a = (e > th + 0.02) & (e < 0.95)
            below.append(field[t, b].mean() if b.sum() else np.nan)
            above.append(field[t, a].mean() if a.sum() else np.nan)
        return np.nanmean(below), np.nanmean(above)

    v_bl, v_ab = regime(f["variance"])
    s_bl, s_ab = regime(f["skew"])
    assert v_bl > 0 > v_ab                                    # variance: risk-prone -> averse
    assert s_bl < 0 < s_ab                                    # skew: neg-skew -> pos-skew


def test_kurtosis_outcomes_match_target_moments():
    """The 5-point generator hits the target mean, variance, and kurtosis with zero skew."""
    from behavioral_md.survival import central_moments, kurtosis_outcomes
    for K in (1.5, 3.0, 6.0):
        m, mu2, mu3, mu4 = central_moments(kurtosis_outcomes(0.05, 0.03, K))
        assert abs(m - 0.05) < 1e-9
        assert abs(mu2 - 0.03 ** 2) < 1e-9
        assert abs(mu3) < 1e-9                              # symmetric: zero skew
        assert abs(mu4 / mu2 ** 2 - K) < 1e-6              # kurtosis on target


def test_temperance_is_kurtosis_aversion_without_a_reversal():
    """Derived sign of temperance: at fixed mean/variance/skew the survival objective is
    kurtosis-AVERSE on both sides of R (V'''' < 0), unlike the variance and skew preferences which
    reverse at R. Measured on the smooth interior (the near-dusk step-function slices excluded)."""
    from behavioral_md.survival import kurtosis_outcomes, risk_threshold, survival_dp
    s, day, night, metab, sd = 0.05, 14, 16, 0.03, 0.03
    R = night * metab

    def adv(out):
        res = survival_dp([(1.0, s)], out, day, night, metab, n_egrid=1201)
        return res["q_risky"] - res["q_safe"], res

    kpref, res = adv(kurtosis_outcomes(s, sd, 7.0))
    klo, _ = adv(kurtosis_outcomes(s, sd, 1.5))
    kpref = kpref - klo
    e, thr = res["energy"], risk_threshold(res)
    below, above, deep = [], [], []
    for t in range(day - 2):                               # drop dusk step-function slices
        th = thr[t] if not np.isnan(thr[t]) else R
        b = (e > 0.05) & (e < th - 0.02)
        a = (e > th + 0.02) & (e < 0.85)
        d = e > 0.9
        if b.sum():
            below.append(kpref[t, b].mean())
        if a.sum():
            above.append(kpref[t, a].mean())
        if d.sum():
            deep.append(kpref[t, d].mean())
    assert np.nanmean(below) < 0 and np.nanmean(above) < 0   # averse on BOTH sides (no reversal)
    assert abs(np.nanmean(deep)) < 2e-4                      # vanishes deep-safe (the control)


def test_predation_off_is_byte_identical():
    """The predation arguments default to off and must not change the starvation-only DP."""
    from behavioral_md.survival import survival_dp
    safe, risky = [(1.0, 0.05)], [(0.5, 0.0), (0.5, 0.10)]
    a = survival_dp(safe, risky, 14, 16, 0.03, n_egrid=401)
    b = survival_dp(safe, risky, 14, 16, 0.03, n_egrid=401, predation_threshold=None)
    assert np.array_equal(a["value"], b["value"])
    assert np.array_equal(a["policy_risky"], b["policy_risky"])


def test_predation_makes_a_bounded_reserve_band():
    """A predation upper boundary turns the saturating survival value into a humped one: V stays
    high in the band between the night requirement and the boundary and falls above it (the
    starvation-predation reserve target; McNamara & Houston 1990)."""
    from behavioral_md.survival import survival_dp
    safe, risky = [(1.0, 0.05)], [(0.5, 0.0), (0.5, 0.10)]
    res = survival_dp(safe, risky, 14, 16, 0.03, n_egrid=801,
                      predation_threshold=0.8, predation_prob=0.2)
    e, v = res["energy"], res["value"]
    in_band = np.interp(0.6, e, v)
    above = np.interp(0.95, e, v)
    assert in_band > above + 0.3                  # high in the band, low above the boundary
    assert above < 0.2                            # being fat is nearly lethal


def test_reversals_survive_predation_and_split_into_a_band():
    """Both the variance (energy-budget) and skew reversals survive a second death source: still
    risk-prone / negative-skew below R, and the safe band R..x_r becomes risk-averse / positive-
    skew, with predation sharpening in-band aversion (the rule splits into a band, not inverts)."""
    from behavioral_md.survival import moment_preference_fields
    day, night, metab, xr = 14, 16, 0.03, 0.8
    R = night * metab
    f = moment_preference_fields(day, night, metab, n_egrid=1201,
                                 predation_threshold=xr, predation_prob=0.2)
    e = f["energy"]
    below = (e > 0.05) & (e < R - 0.02)
    band = (e > R + 0.02) & (e < xr - 0.04)
    assert f["variance"][:, below].mean() > 0          # risk-prone below R survives
    assert f["variance"][:, band].mean() < 0           # band is risk-averse
    assert f["skew"][:, below].mean() < 0 < f["skew"][:, band].mean()   # skew reversal survives


def test_patch_choice_is_a_three_way_energy_budget_rule():
    """With a menu of safe (low-variance), rich (high-mean), and wild (high-variance) patches,
    the survival-optimal choice is the safe patch when comfortable, the rich (rate-maximizing)
    patch below the requirement with time, and the wild (variance) patch near the deadline."""
    from behavioral_md.survival import skewed_outcomes, survival_dp_patches
    menu = [skewed_outcomes(0.045, 0.02, 0.0),       # 0 safe
            skewed_outcomes(0.060, 0.04, 0.0),       # 1 rich
            skewed_outcomes(0.050, 0.11, 0.0)]       # 2 wild
    res = survival_dp_patches(menu, 30, 24, 0.03)
    e, choice = res["energy"], res["choice"]
    hi = np.argmin(np.abs(e - 0.90))
    mid = np.argmin(np.abs(e - 0.50))
    assert np.all(choice[:, hi] == 0)                # comfortable -> safe (low variance) all day
    assert choice[2, mid] == 1                       # below R, early -> rich (rate-maximizing)
    assert choice[-1, mid] == 2                       # below R, at dusk -> wild (high variance)


def test_giving_up_is_finite_horizon():
    """A depleting patch with a travel cost: the forager leaves readily mid-day (MVT relocation)
    but stops leaving near dusk, and the leaving deadline moves earlier as travel gets costlier --
    a finite-horizon effect infinite-horizon MVT cannot express."""
    from behavioral_md.survival import survival_dp_depleting
    day, night, metab = 22, 16, 0.03
    R = night * metab

    def last_leave(travel):
        d = survival_dp_depleting(0.12, 0.6, travel, day, night, metab,
                                  n_biomass=21, n_egrid=401)
        eb, bb, act = d["energy"], d["biomass"], d["action"]
        sel = np.ix_((eb > 0.2) & (eb < R), (bb >= 0.3) & (bb <= 0.6))
        pl = np.array([act[t][sel].mean() for t in range(day)])
        return pl, (int(np.where(pl > 0.3)[0].max()) if (pl > 0.3).any() else -1)

    pl2, ll2 = last_leave(2)
    _, ll6 = last_leave(6)
    assert pl2[: day // 2].max() > 0.8               # leaves readily through the day (MVT)
    assert pl2[-2:].max() < 0.2                       # stops leaving at the deadline
    assert ll2 > ll6                                  # costlier travel -> stops leaving earlier
