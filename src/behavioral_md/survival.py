"""First-principles survival dynamics: risk-sensitivity DERIVED, not imposed.

The risk-sensitive choice in :mod:`behavioral_md.chamber` (``run_risk_choice``) reads the
expected value of each option through an *assumed* survival-shaped utility ``U(E)``. That
imposes the curvature that produces the energy-budget rule -- circular. Here the same rule
is **derived** from the bare dynamics already in the engine: an energy reserve, a metabolic
drain, and a hard death boundary (E <= 0). The only objective is survival.

A single day/night cycle: by DAY the organism forages, choosing a SAFE or a RISKY option to
maximize the probability of surviving the cycle; by NIGHT it cannot forage and simply burns
``metabolism`` each step, dying if its reserve hits zero. The "requirement" -- the reserve
needed at dusk -- is therefore ``night_steps * metabolism``, an *emergent* quantity, not a
parameter. Backward dynamic programming gives the exact survival value of each option at
every (energy, time-of-day); the optimal risk policy is read straight off it.

This answers what an imposed utility cannot: the OPTIMAL policy is a state- and time-
dependent THRESHOLD (risk-prone for every energy below it, risk-averse above), with the
threshold rising through the day toward the night requirement -- not a bump that fades at
the extremes. The fade in the softmax/utility version is a property of bounded-rational
choice over a saturating value, not of survival itself.
"""

from __future__ import annotations

import numpy as np


def _survive_next(energy_grid: np.ndarray, e_next: np.ndarray, value: np.ndarray,
                  cap: float) -> np.ndarray:
    """Value of landing at ``e_next``: 0 if dead (<=0), else interpolate ``value``."""
    return np.where(e_next > 0.0,
                    np.interp(np.clip(e_next, 0.0, cap), energy_grid, value), 0.0)


def survival_dp(safe_outcomes, risky_outcomes, day_steps: int, night_steps: int,
                metabolism: float, cap: float = 1.0, n_egrid: int = 401,
                predation_threshold: float | None = None, predation_prob: float = 0.0) -> dict:
    """Exact survival DP over one day/night cycle; derive the optimal risk policy.

    ``safe_outcomes`` / ``risky_outcomes`` are lists of ``(probability, intake)`` for the
    two foraging options (usually matched-mean so they differ only in variance). Returns a
    dict with the energy grid and, per forward day-step (0 = dawn, day_steps-1 = dusk), the
    survival value of each option ``q_safe``/``q_risky`` [day_steps, n_egrid], the optimal
    ``policy_risky`` (1 where the risky option strictly maximizes survival), the survival
    probability ``value`` = max of the two, and the emergent ``night_requirement`` =
    ``night_steps * metabolism``.

    Optional PREDATION as a second death source above an upper reserve boundary (the starvation-
    predation trade-off; McNamara & Houston 1990): if ``predation_threshold`` is set, each DAY step
    spent with a reserve above it incurs a per-step death probability ``predation_prob`` (a heavier,
    slower, more visible animal is killed more often). Default ``None`` is starvation-only and
    byte-identical to the single-boundary DP. With predation on, carrying a high reserve is costly,
    so the value function turns concave just below the upper boundary: the optimal policy becomes a
    twin-threshold rule, risk-prone below the night requirement and risk-averse as it approaches the
    predation boundary, targeting a reserve band between the two.
    """
    e = np.linspace(0.0, cap, n_egrid)
    value = (e > 0.0).astype(float)                 # end of cycle: survived iff alive
    # Per-step predation survival vs reserve: 1.0 below the boundary, 1 - predation_prob above it.
    if predation_threshold is None:
        pred_survive = np.ones_like(e)
    else:
        pred_survive = np.where(e > predation_threshold, 1.0 - predation_prob, 1.0)

    # Night, worked backward: forced drain, no choice, die if it takes you to <= 0.
    for _ in range(night_steps):
        value = _survive_next(e, e - metabolism, value, cap)

    q_safe = np.zeros((day_steps, n_egrid))
    q_risky = np.zeros((day_steps, n_egrid))
    cont = np.zeros((day_steps, n_egrid))           # continuation value V_{t+1} per day-step
    for t in range(day_steps - 1, -1, -1):          # dusk -> dawn
        cont[t] = value                             # value currently holds V_{t+1}
        qs = sum(p * _survive_next(e, e + d - metabolism, value, cap) for p, d in safe_outcomes)
        qr = sum(p * _survive_next(e, e + d - metabolism, value, cap) for p, d in risky_outcomes)
        qs = pred_survive * qs                      # pay mass-dependent predation while foraging
        qr = pred_survive * qr
        q_safe[t] = qs
        q_risky[t] = qr
        value = np.maximum(qs, qr)                  # optimal: take the better option

    policy_risky = (q_risky > q_safe + 1e-12).astype(float)
    return {"energy": e, "q_safe": q_safe, "q_risky": q_risky, "cont": cont,
            "policy_risky": policy_risky, "value": value,
            "night_requirement": float(night_steps * metabolism),
            "day_steps": day_steps, "night_steps": night_steps, "metabolism": metabolism}


def risk_threshold(result: dict) -> np.ndarray:
    """Per day-step, the highest energy at which the optimal policy is still risk-prone.

    This is the energy-budget threshold: below it the organism gambles, above it plays
    safe. It rises through the day toward the night requirement. NaN where the policy is
    risk-prone nowhere (already safe at dawn for all reserves)."""
    e = result["energy"]
    pol = result["policy_risky"]
    out = np.full(pol.shape[0], np.nan)
    for t in range(pol.shape[0]):
        prone = np.where(pol[t] > 0.5)[0]
        if len(prone):
            out[t] = e[prone.max()]
    return out


def softmax_policy(result: dict, beta: float) -> np.ndarray:
    """A bounded-rational policy: P(choose risky) = sigma(beta*(q_risky - q_safe)).

    With large ``beta`` this approaches the optimal threshold; with small ``beta`` it is
    graded and FADES toward 0.5 where both options are near-certain death (ruin) or near-
    certain survival -- the bump shape an imposed survival utility also produces. The
    threshold itself is unchanged; only the sharpness differs.
    """
    d = result["q_risky"] - result["q_safe"]
    return 1.0 / (1.0 + np.exp(-beta * d))


def simulate_survival_choice(result: dict, safe_outcomes, risky_outcomes, n_org: int,
                             n_cycles: int, beta: float = 40.0, seed: int = 0,
                             cap: float = 1.0, n_ebins: int = 20):
    """Closing the loop: a BEHAVIORAL population that actually lives and dies, choosing by
    softmax over the DP-DERIVED survival values (no imposed utility).

    Each organism cycles through day (forage; choose by ``sigma(beta*(q_risky - q_safe))``
    at its current energy and time-of-day, looked up from ``result``) and night (forced
    fast), with a hard death at E <= 0; energy carries across cycles. Organisms start spread
    across the reserve so the whole policy is sampled. Returns the realized risky-choice
    fraction binned by current energy (``risky_by_energy`` [n_ebins], ``energy_bins``), the
    bin counts, and the survival fraction -- the energy-budget rule as executed behavior,
    derived from survival alone.
    """
    rng = np.random.default_rng(seed)
    eg = result["energy"]
    qs_t, qr_t = result["q_safe"], result["q_risky"]
    day, night, metab = result["day_steps"], result["night_steps"], result["metabolism"]
    sp = np.array([o[0] for o in safe_outcomes])
    sd = np.array([o[1] for o in safe_outcomes])
    rp = np.array([o[0] for o in risky_outcomes])
    rd = np.array([o[1] for o in risky_outcomes])

    energy = np.linspace(0.02, cap, n_org)            # spread so all bins are visited
    start_energy = energy.copy()
    alive = np.ones(n_org, bool)
    risky_count = np.zeros(n_ebins)
    bin_count = np.zeros(n_ebins)

    def sample(p, d):
        return d[(rng.random(n_org)[:, None] < np.cumsum(p)[None, :]).argmax(1)]

    for _ in range(n_cycles):
        for t in range(day):
            qs = np.interp(np.clip(energy, 0, cap), eg, qs_t[t])
            qr = np.interp(np.clip(energy, 0, cap), eg, qr_t[t])
            p_risky = 1.0 / (1.0 + np.exp(-beta * (qr - qs)))
            choose_risky = (rng.random(n_org) < p_risky) & alive
            b = np.clip(energy / cap * n_ebins, 0, n_ebins - 1).astype(int)
            np.add.at(bin_count, b[alive], 1.0)
            np.add.at(risky_count, b[alive], choose_risky[alive].astype(float))
            intake = np.where(choose_risky, sample(rp, rd), sample(sp, sd))
            energy = np.where(alive, np.clip(energy + intake - metab, 0.0, cap), energy)
            alive = alive & (energy > 0.0)
        for _ in range(night):
            energy = np.where(alive, np.clip(energy - metab, 0.0, cap), energy)
            alive = alive & (energy > 0.0)

    bins = [(i + 0.5) / n_ebins * cap for i in range(n_ebins)]
    # Realized survival binned by STARTING energy (compare to the DP's predicted V(E)).
    sb = np.clip(start_energy / cap * n_ebins, 0, n_ebins - 1).astype(int)
    surv_n = np.zeros(n_ebins)
    surv_alive = np.zeros(n_ebins)
    np.add.at(surv_n, sb, 1.0)
    np.add.at(surv_alive, sb, alive.astype(float))
    return {"energy_bins": bins,
            "risky_by_energy": (risky_count / np.maximum(bin_count, 1)).tolist(),
            "bin_count": bin_count.tolist(),
            "survival_by_start": (surv_alive / np.maximum(surv_n, 1)).tolist(),
            "survival": float(alive.mean())}


def evolve_risk_policy(safe_outcomes, risky_outcomes, day_steps: int, night_steps: int,
                       metabolism: float, pop_size: int = 3000, n_generations: int = 250,
                       n_cycles: int = 3, e_init: float = 0.5, mutation: float = 0.04,
                       cap: float = 1.0, seed: int = 0) -> dict:
    """Evolve the risk policy from selection alone -- no utility, no DP, no learning rule.

    Each organism carries a HERITABLE state-dependent risk trait: a threshold that is linear
    in time of day, ``theta(t) = a + b * (t / day_steps)``; it gambles (risky) when its
    reserve is below ``theta(t)``, otherwise plays safe. Organisms forage through
    ``n_cycles`` day/night cycles (day = choose; night = forced fast) and die at E <= 0; the
    survivors reproduce, offspring inheriting ``(a, b)`` with Gaussian mutation. Selection is
    the bare survival dynamics -- nothing rewards "gamble when hungry".

    Over generations the population converges on the adaptive policy; ``theta(t)`` should
    track the DP-optimal threshold (the energy-budget rule), now EVOLVED rather than derived
    or imposed. Returns the per-generation mean ``(a, b)`` and survival, and the final
    evolved threshold over the day.
    """
    rng = np.random.default_rng(seed)
    a = rng.uniform(0.0, cap, pop_size)
    b = rng.uniform(-cap, cap, pop_size)
    sp = np.array([o[0] for o in safe_outcomes])
    sd = np.array([o[1] for o in safe_outcomes])
    rp = np.array([o[0] for o in risky_outcomes])
    rd = np.array([o[1] for o in risky_outcomes])

    def sample(p, d):
        return d[(rng.random(pop_size)[:, None] < np.cumsum(p)[None, :]).argmax(1)]

    hist_a, hist_b, hist_surv = [], [], []
    for _ in range(n_generations):
        energy = np.full(pop_size, e_init)
        alive = np.ones(pop_size, bool)
        for _c in range(n_cycles):
            for t in range(day_steps):
                theta = a + b * (t / day_steps)
                gamble = (energy < theta) & alive
                intake = np.where(gamble, sample(rp, rd), sample(sp, sd))
                energy = np.where(alive, np.clip(energy + intake - metabolism, 0.0, cap), energy)
                alive = alive & (energy > 0.0)
            for _ in range(night_steps):
                energy = np.where(alive, np.clip(energy - metabolism, 0.0, cap), energy)
                alive = alive & (energy > 0.0)
        surv = np.where(alive)[0]
        hist_a.append(float(a[surv].mean()) if len(surv) else np.nan)
        hist_b.append(float(b[surv].mean()) if len(surv) else np.nan)
        hist_surv.append(len(surv) / pop_size)
        if len(surv) == 0:                                  # extinction: reseed (rare if tuned)
            a = rng.uniform(0.0, cap, pop_size)
            b = rng.uniform(-cap, cap, pop_size)
            continue
        parents = surv[rng.integers(0, len(surv), pop_size)]
        a = np.clip(a[parents] + rng.normal(0, mutation, pop_size), 0.0, cap)
        b = np.clip(b[parents] + rng.normal(0, mutation, pop_size), -cap, cap)

    phase = np.arange(day_steps) / day_steps
    evolved_theta = float(np.nanmean(hist_a[-20:])) + float(np.nanmean(hist_b[-20:])) * phase
    return {"mean_a": hist_a, "mean_b": hist_b, "survival": hist_surv,
            "evolved_theta": np.clip(evolved_theta, 0.0, cap).tolist(),
            "day_steps": day_steps}


def simulate_learning_choice(safe_outcomes, risky_outcomes, day_steps: int, night_steps: int,
                             metabolism: float, n_org: int = 60, n_cycles: int = 40,
                             e_init: float = 0.5, cap: float = 1.0, n_egrid: int = 201,
                             seed: int = 0) -> dict:
    """Within-life learning: organisms LEARN the option distributions from experience and
    plan survival on their own estimate -- the distributions are no longer handed to them.

    Each organism starts ignorant (a single pseudo-observation of each option at the overall
    mean, so it is initially indifferent and therefore explores). Each cycle it re-solves the
    survival DP on its CURRENT empirical estimate of the two options' outcome distributions,
    forages the day/night cycle under that policy, observes the true outcomes (updating its
    estimate), and respawns at ``e_init`` on death while keeping what it has learned. Nothing
    tells it which option is risky; it finds out.

    As experience accumulates the estimate converges on the true distributions, so the planned
    policy converges on the DP optimum and the energy-budget rule appears. Returns per-cycle
    population means: ``gamble_recall`` (of the states where the true optimum gambles, the
    fraction the learned plan also gambles), ``risky_variance`` (estimated), ``survival``, and
    the final learned threshold ``learned_theta`` over the day vs the true ``dp_theta``.
    """
    rng = np.random.default_rng(seed)
    s_vals = np.array([o[1] for o in safe_outcomes])
    s_p = np.array([o[0] for o in safe_outcomes])
    r_vals = np.array([o[1] for o in risky_outcomes])
    r_p = np.array([o[0] for o in risky_outcomes])
    grand_mean = 0.5 * ((s_vals * s_p).sum() + (r_vals * r_p).sum())

    # True reference policy + threshold (the optimum the learners should approach).
    true = survival_dp(safe_outcomes, risky_outcomes, day_steps, night_steps, metabolism,
                       cap=cap, n_egrid=n_egrid)
    true_policy = true["policy_risky"]
    gamble_mask = true_policy > 0.5
    dp_theta = risk_threshold(true).tolist()

    # Per organism: counts of observed outcomes for each option's true support, seeded with
    # one pseudo-observation at the grand mean so the organism begins indifferent (explores).
    s_support = np.concatenate([s_vals, [grand_mean]])
    r_support = np.concatenate([r_vals, [grand_mean]])
    s_count = np.zeros((n_org, len(s_support)))
    r_count = np.zeros((n_org, len(r_support)))
    s_count[:, -1] = 1.0
    r_count[:, -1] = 1.0

    energy = np.full(n_org, e_init)
    accuracy, variance, survival = [], [], []

    def emp(vals, counts_row):
        tot = counts_row.sum()
        return list(zip((counts_row / tot).tolist(), vals.tolist(), strict=True))

    def sample_true(p, v):
        return v[(rng.random() < np.cumsum(p)).argmax()]

    for cyc in range(n_cycles):
        # Decaying epsilon-greedy exploration: the planned policy never tries the risky
        # option while both look like point masses at the mean, so it must be nudged to
        # sample (and thereby discover) the variance. Exploration anneals away as it learns.
        epsilon = max(0.05, 0.45 * (1.0 - cyc / n_cycles))
        cyc_acc = np.zeros(n_org)
        cyc_var = np.zeros(n_org)
        alive = np.ones(n_org, bool)
        for i in range(n_org):
            res = survival_dp(emp(s_support, s_count[i]), emp(r_support, r_count[i]),
                              day_steps, night_steps, metabolism, cap=cap, n_egrid=n_egrid)
            pol = res["policy_risky"]
            # Recall on the gamble states: of the (energy, time) cells where the TRUE optimum
            # gambles, what fraction does the learned plan also gamble? (Insensitive overall
            # accuracy is dominated by the risk-averse majority; this isolates the learning.)
            cyc_acc[i] = float(pol[gamble_mask].mean()) if gamble_mask.any() else 1.0
            rc = r_count[i] / r_count[i].sum()
            mu = (rc * r_support).sum()
            cyc_var[i] = float((rc * (r_support - mu) ** 2).sum())
            e = energy[i]
            for t in range(day_steps):               # day: forage under the learned plan
                b = min(int(e / cap * n_egrid), n_egrid - 1)
                gamble = pol[t, b] > 0.5             # the planned choice
                if rng.random() < epsilon:
                    gamble = not gamble             # explore the other option
                if gamble:
                    out = sample_true(r_p, r_vals)
                    r_count[i, np.argmin(np.abs(r_support - out))] += 1
                else:
                    out = sample_true(s_p, s_vals)
                    s_count[i, np.argmin(np.abs(s_support - out))] += 1
                e = min(max(e + out - metabolism, 0.0), cap)
                if e <= 0.0:
                    break
            for _n in range(night_steps):            # night: forced fast
                e = max(e - metabolism, 0.0)
                if e <= 0.0:
                    break
            alive[i] = e > 0.0
            energy[i] = e if e > 0.0 else e_init     # respawn, keep what was learned
        accuracy.append(float(cyc_acc.mean()))
        variance.append(float(cyc_var.mean()))
        survival.append(float(alive.mean()))

    true_var = float((r_p * (r_vals - (r_p * r_vals).sum()) ** 2).sum())
    # Final learned threshold: re-plan on the population-pooled estimate.
    pooled = survival_dp(emp(s_support, s_count.sum(0)), emp(r_support, r_count.sum(0)),
                         day_steps, night_steps, metabolism, cap=cap, n_egrid=n_egrid)
    return {"gamble_recall": accuracy, "risky_variance": variance, "survival": survival,
            "true_risky_variance": true_var, "learned_theta": risk_threshold(pooled).tolist(),
            "dp_theta": dp_theta}


def simulate_model_free_choice(safe_outcomes, risky_outcomes, day_steps: int, night_steps: int,
                               metabolism: float, n_org: int = 200, n_cycles: int = 300,
                               n_ebins: int = 25, alpha: float = 0.1, cap: float = 1.0,
                               seed: int = 0) -> dict:
    """Model-FREE survival learning: no planning, no model of the distributions.

    Each organism holds a tabular value ``Q[energy_bin, time_of_day, action]`` and learns it
    by Monte-Carlo from the bare SURVIVAL signal: it forages a day/night cycle (epsilon-greedy
    over Q), and at the end every (state, action) it visited is updated toward 1 if it survived
    the cycle and 0 if it died. Nothing models the option distributions and nothing plans;
    survival values are learned directly from living and dying. Each cycle starts from a random
    reserve (for state coverage) -- the organism experiences many days from many states.

    As Q converges to the true survival probabilities, the greedy policy becomes the energy-
    budget rule. Returns per-cycle population-mean ``gamble_recall`` (of the states where the
    true optimum gambles, the fraction the greedy Q-policy also gambles) and ``survival``, plus
    the final learned threshold ``learned_theta`` vs the true ``dp_theta``.
    """
    rng = np.random.default_rng(seed)
    s_vals = np.array([o[1] for o in safe_outcomes])
    s_p = np.array([o[0] for o in safe_outcomes])
    r_vals = np.array([o[1] for o in risky_outcomes])
    r_p = np.array([o[0] for o in risky_outcomes])

    # True reference policy, downsampled to the coarse energy bins the learner uses.
    true = survival_dp(safe_outcomes, risky_outcomes, day_steps, night_steps, metabolism, cap=cap)
    centers = (np.arange(n_ebins) + 0.5) / n_ebins * cap
    true_gamble = np.zeros((n_ebins, day_steps), bool)
    for t in range(day_steps):
        true_gamble[:, t] = np.interp(centers, true["energy"], true["policy_risky"][t]) > 0.5
    dp_theta = risk_threshold(true).tolist()

    idx = np.arange(n_org)
    q = np.full((n_org, n_ebins, day_steps, 2), 0.5)     # neutral init
    recall, survival = [], []

    def draw(p, v):
        return v[(rng.random(n_org)[:, None] < np.cumsum(p)[None, :]).argmax(1)]

    for cyc in range(n_cycles):
        eps = max(0.05, 0.3 * (1.0 - cyc / n_cycles))
        energy = rng.uniform(0.0, cap, n_org)            # random start (state coverage)
        alive = np.ones(n_org, bool)
        b_rec = np.zeros((n_org, day_steps), int)
        a_rec = np.zeros((n_org, day_steps), int)
        on = np.zeros((n_org, day_steps), bool)
        for t in range(day_steps):
            b = np.clip((energy / cap * n_ebins).astype(int), 0, n_ebins - 1)
            gamble = q[idx, b, t, 1] > q[idx, b, t, 0]
            flip = rng.random(n_org) < eps               # epsilon-greedy exploration
            gamble = np.where(flip, ~gamble, gamble)
            b_rec[:, t], a_rec[:, t], on[:, t] = b, gamble.astype(int), alive
            out = np.where(gamble, draw(r_p, r_vals), draw(s_p, s_vals))
            energy = np.where(alive, np.clip(energy + out - metabolism, 0.0, cap), energy)
            alive = alive & (energy > 0.0)
        for _ in range(night_steps):
            energy = np.where(alive, np.clip(energy - metabolism, 0.0, cap), energy)
            alive = alive & (energy > 0.0)

        survived = alive.astype(float)                   # the only learning signal
        for t in range(day_steps):                       # Monte-Carlo backup of the outcome
            bi, ai, m = b_rec[:, t], a_rec[:, t], on[:, t]
            cur = q[idx, bi, t, ai]
            q[idx, bi, t, ai] = np.where(m, cur + alpha * (survived - cur), cur)

        # Recall of the AGGREGATE learned value function (pooled experience). Individual
        # Monte-Carlo Q-tables are high-variance; the population mean is the clean readout.
        mg = q.mean(0)[:, :, 1] > q.mean(0)[:, :, 0]     # [n_ebins, day_steps]
        recall.append(float(mg[true_gamble].mean()))
        survival.append(float(survived.mean()))

    # Final learned threshold from the population-mean greedy policy.
    mean_greedy = (q.mean(0)[:, :, 1] > q.mean(0)[:, :, 0])     # [n_ebins, day_steps]
    learned_theta = []
    for t in range(day_steps):
        prone = np.where(mean_greedy[:, t])[0]
        learned_theta.append(float(centers[prone.max()]) if len(prone) else float("nan"))
    return {"gamble_recall": recall, "survival": survival,
            "learned_theta": learned_theta, "dp_theta": dp_theta}


def survival_dp_timevarying(safe_by_step, risky_by_step, night_steps: int, metabolism: float,
                            cap: float = 1.0, n_egrid: int = 401) -> dict:
    """Survival DP with TIME-VARYING option distributions over the day.

    ``safe_by_step`` / ``risky_by_step`` are length-``day_steps`` lists, each a list of
    ``(probability, intake)`` for that day-step -- so the foraging variance can be set by a
    day/night light cycle (darker -> more variable). Otherwise identical to :func:`survival_dp`
    (backward DP over day + night, death at E <= 0); returns the same keys.
    """
    day_steps = len(risky_by_step)
    e = np.linspace(0.0, cap, n_egrid)
    value = (e > 0.0).astype(float)
    for _ in range(night_steps):
        value = _survive_next(e, e - metabolism, value, cap)

    q_safe = np.zeros((day_steps, n_egrid))
    q_risky = np.zeros((day_steps, n_egrid))
    for t in range(day_steps - 1, -1, -1):
        qs = sum(p * _survive_next(e, e + d - metabolism, value, cap) for p, d in safe_by_step[t])
        qr = sum(p * _survive_next(e, e + d - metabolism, value, cap) for p, d in risky_by_step[t])
        q_safe[t] = qs
        q_risky[t] = qr
        value = np.maximum(qs, qr)

    policy_risky = (q_risky > q_safe + 1e-12).astype(float)
    return {"energy": e, "q_safe": q_safe, "q_risky": q_risky, "policy_risky": policy_risky,
            "value": value, "night_requirement": float(night_steps * metabolism),
            "day_steps": day_steps, "night_steps": night_steps, "metabolism": metabolism}


def simulate_dusk_survival(result: dict, safe_by_step, risky_by_step, metabolism: float,
                           dusk_step: int, reserves, n_org: int = 4000, cap: float = 1.0,
                           seed: int = 0) -> dict:
    """Realized night survival of organisms dropped into dusk behind on reserves.

    Each organism starts the day-step ``dusk_step`` at a given ``reserve``, forages the
    remaining day-steps under the DP's optimal risk policy (``result["policy_risky"]``, gambling
    where it prescribes), drawing its actual intake from the TIME-VARYING ``risky_by_step`` /
    ``safe_by_step`` for that step, then fasts the night (burning ``metabolism`` each step, dying
    at E <= 0). Returns the fraction surviving the night for each starting ``reserve`` -- the
    ruin edge as realized behavior, not a table. ``result`` should come from
    :func:`survival_dp_timevarying` (sun) or :func:`survival_dp` (constant control).
    """
    rng = np.random.default_rng(seed)
    eg = result["energy"]
    pol = result["policy_risky"]
    day_steps = result["day_steps"]
    night_steps = result["night_steps"]
    survival = np.zeros(len(reserves))
    for i, e0 in enumerate(reserves):
        E = np.full(n_org, float(e0))
        alive = np.ones(n_org, bool)
        for t in range(dusk_step, day_steps):
            b = np.clip((E / cap * len(eg)).astype(int), 0, len(eg) - 1)
            gamble = (pol[t][b] > 0.5) & alive
            ro = risky_by_step[t]
            rv = np.array([d for _, d in ro])
            rp = np.cumsum([p for p, _ in ro])
            out_r = rv[(rng.random(n_org)[:, None] < rp).argmax(1)]
            out_s = safe_by_step[t][0][1]
            out = np.where(gamble, out_r, out_s)
            E = np.where(alive, np.clip(E + out - metabolism, 0.0, cap), E)
            alive &= E > 0.0
        for _ in range(night_steps):
            E = np.where(alive, E - metabolism, E)
            alive &= E > 0.0
        survival[i] = alive.mean()
    return {"reserves": np.asarray(reserves, float), "survival": survival}


def sun_variance_risky(day_steps: int, mean: float, w_min: float, w_max: float):
    """Per-day-step risky outcomes whose SPREAD tracks darkness (a day/night sun).

    Daylight ``L(t) = sin(pi*(t+0.5)/day_steps)`` is 0 at dawn/dusk and 1 at midday. The risky
    option is a matched-mean two-point gamble ``{mean - w(t), mean + w(t)}`` with spread
    ``w(t) = w_min`` in full light (steady foraging) rising to ``w_max`` in the dark (erratic
    foraging). Returns ``(risky_by_step, daylight)`` for :func:`survival_dp_timevarying`.
    """
    light = np.sin(np.pi * (np.arange(day_steps) + 0.5) / day_steps)
    w = w_min + (w_max - w_min) * (1.0 - light)
    risky_by_step = [[(0.5, mean - wt), (0.5, mean + wt)] for wt in w]
    return risky_by_step, light


def skewed_outcomes(mean: float, std: float, skew: float, n_points: int = 121,
                    n_sigma: float = 4.0):
    """A CONTINUOUS outcome distribution with a chosen mean, std, and skew sign/strength.

    A discretized standard normal ``z`` is warped by ``(exp(skew*z) - 1) / skew`` (which tends
    to ``z`` as ``skew -> 0``), then standardized so the result has exactly ``mean`` and ``std``
    regardless of ``skew``. ``skew > 0`` is right-skewed (a rare large gain over frequent small
    losses -- a "lottery"); ``skew < 0`` is left-skewed (frequent small gains over a rare large
    loss -- a "disaster"); ``skew == 0`` is a symmetric (Gaussian) gamble. Returned as a list of
    ``(probability, intake)`` for :func:`survival_dp`, so a *richer* world than the two-point
    gamble -- it removes the two-point reachability comb and lets skew be probed at fixed
    mean/variance (where mean-variance theory predicts indifference).
    """
    z = np.linspace(-n_sigma, n_sigma, n_points)
    w = np.exp(-0.5 * z ** 2)
    w /= w.sum()
    s = abs(skew)
    base = z if s < 1e-9 else (np.exp(s * z) - 1.0) / s
    if skew < 0:
        base = -base
    base = base - (w * base).sum()                       # standardize to mean 0, var 1
    base = base / np.sqrt((w * base ** 2).sum())
    return list(zip(w, mean + std * base, strict=True))


def outcome_moments(outcomes):
    """Mean, variance, and skewness of a discrete ``[(probability, value), ...]`` distribution."""
    p = np.array([pi for pi, _ in outcomes])
    x = np.array([xi for _, xi in outcomes])
    m = float((p * x).sum())
    var = float((p * (x - m) ** 2).sum())
    sd = np.sqrt(var)
    skew = float((p * ((x - m) / sd) ** 3).sum()) if sd > 0 else 0.0
    return m, var, skew


def central_moments(outcomes):
    """Mean and the 2nd/3rd/4th CENTRAL moments ``(mean, mu2, mu3, mu4)`` of a discrete
    ``[(probability, value), ...]`` distribution. ``mu2`` is the variance, ``mu3 = skew*sigma^3``
    the (signed) third central moment, ``mu4 = kurtosis*sigma^4`` the fourth -- the quantities
    that enter the moment expansion of the survival value (:func:`moment_dominance`)."""
    p = np.array([pi for pi, _ in outcomes])
    x = np.array([xi for _, xi in outcomes])
    m = float((p * x).sum())
    d = x - m
    return m, float((p * d ** 2).sum()), float((p * d ** 3).sum()), float((p * d ** 4).sum())


def _symmetric_three_point(mean: float, var: float, tail_prob: float):
    """Symmetric 3-point gamble ``{mean +/- a, mean}`` with given variance and tail probability.

    Outcomes ``mean +/- a`` each occur with probability ``tail_prob`` and ``mean`` with the rest;
    ``a = sqrt(var / (2*tail_prob))`` fixes the variance, so its (excess) kurtosis ``= 1/(2 q) - 3``
    is set by ``tail_prob`` ALONE at fixed mean and variance (skew stays 0). Small ``tail_prob`` ->
    leptokurtic (rare fat tails); large ``tail_prob`` -> platykurtic (a near-two-point spread)."""
    a = np.sqrt(var / (2.0 * tail_prob))
    return [(tail_prob, mean - a), (1.0 - 2.0 * tail_prob, mean), (tail_prob, mean + a)]


def mean_preserving_spread_advantage(mean: float, gamble, day_steps: int, night_steps: int,
                                     metabolism: float, **dp_kw):
    """Survival advantage ``adv(t, e) = q_risky - q_safe`` of ``gamble`` over the SAFE (degenerate-
    at-``mean``) option: the value change a mean-preserving spread buys at each (day-step, reserve).

    Because the safe option lands deterministically at ``e* = e + mean - metabolism``, the advantage
    is exactly the moment expansion of the emergent continuation value ``V`` about ``e*``::

        adv ~ (1/2) V''(e*) mu2 + (1/6) V'''(e*) mu3 + (1/24) V''''(e*) mu4 + ...

    Returns ``(advantage [day_steps, n_egrid], dp_result)``."""
    res = survival_dp([(1.0, mean)], gamble, day_steps, night_steps, metabolism, **dp_kw)
    return res["q_risky"] - res["q_safe"], res


def moment_preference_fields(day_steps: int, night_steps: int, metabolism: float,
                             mean: float = 0.05, sigma: float = 0.04, skew: float = 0.9,
                             kurt_tail=(0.08, 0.40), **dp_kw) -> dict:
    """Variance, skew, and kurtosis preference fields of the EMERGENT survival value, each isolated
    by a mean-preserving spread of the DP itself (no finite differencing of a gridded value).

    For a common foraging mean and variance ``sigma**2``:
      - ``variance`` field = advantage of a symmetric gamble        (sign = sign of V'', curvature)
      - ``skew`` field = half-difference of ``+skew`` and ``-skew`` gambles  (sign = sign of V''',
        prudence), matched in mean and variance
      - ``kurtosis`` field = high-kurtosis minus low-kurtosis 3-point gamble (sign = sign of
        V'''', temperance), matched in mean, variance, and skew (``kurt_tail`` = the tail probs)

    Each field is ``[day_steps, n_egrid]`` over the reserve grid. The variance field reverses sign
    (risk-prone below -> risk-averse above) exactly at the optimal ``threshold`` (the value
    sigmoid's inflection, where V'' = 0); the skew field reverses there too. This is the energy-
    budget rule generalized: a survival/ruin objective is a whole-distribution functional, so the
    optimal policy is sensitive to ALL moments, each governed by a successive derivative of one
    emergent value function -- and the reversals are those derivatives' sign changes across the
    requirement. Returns the three fields, ``energy``, the per-day-step ``threshold``, and ``R``.
    """
    var = sigma ** 2
    sym = skewed_outcomes(mean, sigma, 0.0)
    pos = skewed_outcomes(mean, sigma, skew)
    neg = skewed_outcomes(mean, sigma, -skew)
    hi = _symmetric_three_point(mean, var, kurt_tail[0])
    lo = _symmetric_three_point(mean, var, kurt_tail[1])
    a_var, res = mean_preserving_spread_advantage(mean, sym, day_steps, night_steps, metabolism,
                                                  **dp_kw)
    a_pos = mean_preserving_spread_advantage(mean, pos, day_steps, night_steps, metabolism,
                                             **dp_kw)[0]
    a_neg = mean_preserving_spread_advantage(mean, neg, day_steps, night_steps, metabolism,
                                             **dp_kw)[0]
    a_hi = mean_preserving_spread_advantage(mean, hi, day_steps, night_steps, metabolism,
                                            **dp_kw)[0]
    a_lo = mean_preserving_spread_advantage(mean, lo, day_steps, night_steps, metabolism,
                                            **dp_kw)[0]
    return {"energy": res["energy"], "variance": a_var, "skew": 0.5 * (a_pos - a_neg),
            "kurtosis": a_hi - a_lo, "threshold": risk_threshold(res),
            "night_requirement": res["night_requirement"], "day_steps": day_steps}


def field_zero_crossing(field: np.ndarray, energy: np.ndarray, lo: float = 0.04,
                        hi: float = 0.95, smooth: int = 15) -> np.ndarray:
    """Per day-step, the reserve of a field's primary positive->negative zero-crossing in
    ``(lo, hi)`` (lightly box-smoothed first). For the variance preference field this is the risk
    threshold (the value's inflection, V'' = 0); NaN where the field never crosses down."""
    out = np.full(field.shape[0], np.nan)
    k = np.ones(smooth) / smooth
    for t in range(field.shape[0]):
        s = np.convolve(field[t], k, mode="same")
        idx = np.where((s[:-1] > 0) & (s[1:] <= 0) & (energy[:-1] > lo) & (energy[:-1] < hi))[0]
        if len(idx):
            out[t] = energy[idx[0]]
    return out


def survival_dp_patches(patches, day_steps: int, night_steps: int, metabolism: float,
                        cap: float = 1.0, n_egrid: int = 801) -> dict:
    """Survival DP over a MENU of foraging patches (multi-patch risk-sensitive choice).

    ``patches`` is a list of outcome distributions ``[[(p, intake), ...], ...]`` -- e.g. a
    low-variance "safe" patch, a high-mean "rich" patch, and a high-variance "wild" patch. Each
    day-step the organism forages whichever patch maximizes its probability of surviving the cycle
    (backward DP over day + night, death at E <= 0; the binary :func:`survival_dp` is the two-patch
    case). Returns ``choice[t, e]`` = index of the optimal patch at each (time, energy), plus the
    per-patch survival values ``q[t, patch, e]``. The energy-budget rule selects among the menu:
    the safe patch when comfortable (above the requirement), the rich (rate-maximizing) patch when
    below it with time to climb, and the wild (variance) patch when the deadline leaves no time to
    climb steadily.
    """
    e = np.linspace(0.0, cap, n_egrid)
    value = (e > 0.0).astype(float)
    for _ in range(night_steps):
        value = _survive_next(e, e - metabolism, value, cap)
    n_patch = len(patches)
    q = np.zeros((day_steps, n_patch, n_egrid))
    for t in range(day_steps - 1, -1, -1):
        for j, outs in enumerate(patches):
            q[t, j] = sum(p * _survive_next(e, e + d - metabolism, value, cap) for p, d in outs)
        value = q[t].max(0)
    return {"energy": e, "q": q, "choice": q.argmax(1), "value": value,
            "night_requirement": float(night_steps * metabolism), "day_steps": day_steps,
            "night_steps": night_steps, "metabolism": metabolism}


def survival_dp_depleting(max_rate: float, cv: float, travel_steps: int, day_steps: int,
                          night_steps: int, metabolism: float, n_biomass: int = 41,
                          cap: float = 1.0, n_egrid: int = 601) -> dict:
    """Survival DP for a DEPLETING patch with a travel cost (risk-sensitive MVT).

    The organism is in a patch of biomass ``b in [0, 1]``; foraging draws a noisy intake of mean
    ``max_rate * b`` (coefficient of variation ``cv``, via :func:`skewed_outcomes`) and depletes
    the patch one biomass-grid step. Or it can TRAVEL to a fresh (full) patch, paying
    ``travel_steps`` steps of pure metabolism with no intake. It maximizes survival over the
    day/night cycle (death at E <= 0). Returns ``action[t, e, b]`` (0 = forage, 1 = leave) and the
    value function. Classic marginal-value-theorem leaving emerges, but with a finite-horizon
    twist: leaving stops near dusk, once too few steps remain to recoup the travel cost before the
    night fast -- a deadline that infinite-horizon MVT lacks.
    """
    e = np.linspace(0.0, cap, n_egrid)
    biomass = np.linspace(0.0, 1.0, n_biomass)
    night = (e > 0.0).astype(float)
    for _ in range(night_steps):
        night = _survive_next(e, e - metabolism, night, cap)
    dists = [skewed_outcomes(max_rate * b, max(cv * max_rate * b, 1e-9), 0.0, n_points=41)
             for b in biomass]
    value = np.zeros((day_steps + 1, n_egrid, n_biomass))
    value[day_steps, :, :] = night[:, None]
    action = np.zeros((day_steps, n_egrid, n_biomass), int)
    for t in range(day_steps - 1, -1, -1):
        arrive = min(t + travel_steps, day_steps)
        travel_q = _survive_next(e, e - travel_steps * metabolism, value[arrive, :, -1], cap)
        for j in range(n_biomass):
            depleted = max(j - 1, 0)
            forage_q = sum(p * _survive_next(e, e + d - metabolism, value[t + 1, :, depleted], cap)
                           for p, d in dists[j])
            value[t, :, j] = np.maximum(forage_q, travel_q)
            action[t, :, j] = (travel_q > forage_q + 1e-12).astype(int)
    return {"energy": e, "biomass": biomass, "action": action, "value": value,
            "night_requirement": float(night_steps * metabolism), "day_steps": day_steps,
            "night_steps": night_steps, "travel_steps": travel_steps, "metabolism": metabolism}
