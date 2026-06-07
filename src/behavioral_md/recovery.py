"""Sequence-based parameter recovery and identifiability of the requirement R.

Roadmap 2.1, the credibility item. Houston & Rosenstrom (2024, Biol. Rev.) argue that the field's
evidence on state-dependent (energy-budget) risk sensitivity is mixed because the diagnostic signal
lives in the *sequence* of choices, not in the aggregate preference: "the devil is in the sequence".
This module makes that claim exact and testable inside the survival-DP framework.

The behaving organism chooses by a softmax over the DP-derived survival values at its CURRENT state:

    P(risky | E, t; theta) = sigma( beta * ( q_risky(E,t;theta) - q_safe(E,t;theta) ) )

where ``theta`` are the latent parameters -- the overnight requirement ``R`` (which fixes WHERE the
risk preference flips: q_risky - q_safe crosses zero near E = R at dusk), the daytime metabolism
``m_day`` (which sets the reserve drift away from the boundary, shaping the curve far from R), and
the softmax temperature ``beta`` (which sets HOW SHARP the flip is). q depends only on (m_day, R)
through the DP; beta enters only the choice rule.

Two readouts of the same behaving population:

* SEQUENCE: every (reserve, time, choice) triple is kept, so the analyst sees the whole conditional
  choice curve P(risky | E). Its crossing point identifies R (a LOCATION) and its slope identifies
  beta (a SCALE) -- two different features of one curve, so the two parameters separate.
* AGGREGATE: only the overall risky-choice proportion and the reserve-occupancy histogram survive
  (the choice-to-reserve pairing is discarded). One scalar prediction, so any (R, beta)
  on its level set fits equally -- a ridge. R is NOT identifiable from aggregate preference.

So R is recoverable from sequences and lost in aggregate -- a mechanistic account of why the field's
aggregate-preference evidence is mixed (exp055). The conditional-choice likelihood is exact and
cheap (no intractable DP-likelihood; ~0.3 ms per candidate (m_day, R)), pure numpy/scipy.

References: Houston & Rosenstrom 2024 (Biol. Rev., the named open problem); Wilson & Collins 2019
(eLife; recovery protocol -- simulate from known theta, refit, confusion matrix and parameter
scatter); Beaumont 2010 (ABC, the fallback when a likelihood is intractable -- not needed here, the
conditional likelihood is closed form). The companion learning-rate vs softmax-temperature
of a value-learner (Comput. Brain Behav. 2022) is exp056.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from behavioral_md.survival import survival_dp

# Default stimuli: a safe point intake and a mean-preserving risky spread (matched mean = 0.05).
SAFE = [(1.0, 0.05)]
RISKY = [(0.5, 0.0), (0.5, 0.10)]
DAY, NIGHT = 14, 16
CAP = 1.0


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Logistic, argument clipped so a large (unbounded during a fit) beta cannot overflow."""
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))


def solve(m_day: float, R: float, *, safe=SAFE, risky=RISKY, day: int = DAY, night: int = NIGHT,
          n_egrid: int = 301, cap: float = CAP) -> dict:
    """Survival DP with the daytime burn and the overnight requirement decoupled.

    ``R`` is the overnight demand (the boundary location); the night burn is ``R / night``.
    ``m_day`` is the daytime per-step burn (reserve drift). Returns the usual ``survival_dp`` dict.
    """
    return survival_dp(safe, risky, day, night, metabolism=m_day, cap=cap, n_egrid=n_egrid,
                       metabolism_night=R / night)


def _advantage_at(result: dict, reserve: np.ndarray, tstep: np.ndarray) -> np.ndarray:
    """Interpolate q_risky - q_safe at each decision's (reserve, time-of-day)."""
    e = result["energy"]
    d = result["q_risky"] - result["q_safe"]          # [day_steps, n_egrid]
    res = np.asarray(reserve, float)
    out = np.empty(res.shape, float)
    r_clip = np.clip(res, e[0], e[-1])
    for t in np.unique(tstep):
        m = tstep == t
        out[m] = np.interp(r_clip[m], e, d[t])
    return out


def simulate_choice_sequences(m_day: float, R: float, beta: float, *, n_org: int = 80,
                              n_cycles: int = 12, seed: int = 0, safe=SAFE, risky=RISKY,
                              day: int = DAY, night: int = NIGHT, n_egrid: int = 301,
                              cap: float = CAP) -> dict:
    """Generate trial-by-trial choice sequences from the behaving softmax planner.

    Organisms start spread across the reserve, forage day/night cycles choosing risky with
    probability ``sigma(beta * (q_risky - q_safe))`` at their current (reserve, time), and respawn
    (spread again) on starvation so the whole policy keeps being sampled. EVERY daytime decision of
    a live organism is recorded. Returns flat arrays ``reserve``, ``tstep``, ``chose_risky`` (one
    entry per decision) plus the true ``params``. This is the data both readouts are computed from.
    """
    rng = np.random.default_rng(seed)
    res = solve(m_day, R, safe=safe, risky=risky, day=day, night=night, n_egrid=n_egrid, cap=cap)
    e_grid = res["energy"]
    qs_t, qr_t = res["q_safe"], res["q_risky"]
    m_night = R / night
    sp = np.array([o[0] for o in safe])
    sd = np.array([o[1] for o in safe])
    rp = np.array([o[0] for o in risky])
    rd = np.array([o[1] for o in risky])

    energy = np.linspace(0.04, cap, n_org)
    alive = np.ones(n_org, bool)
    rec_E, rec_t, rec_c = [], [], []

    def sample(p, d):
        return d[(rng.random(n_org)[:, None] < np.cumsum(p)[None, :]).argmax(1)]

    for _ in range(n_cycles):
        for t in range(day):
            qs = np.interp(np.clip(energy, 0, cap), e_grid, qs_t[t])
            qr = np.interp(np.clip(energy, 0, cap), e_grid, qr_t[t])
            p_risky = _sigmoid(beta * (qr - qs))
            choose_risky = rng.random(n_org) < p_risky
            # Record the decisions of currently-live organisms.
            rec_E.append(energy[alive].copy())
            rec_t.append(np.full(int(alive.sum()), t))
            rec_c.append(choose_risky[alive].astype(np.int8))
            intake = np.where(choose_risky, sample(rp, rd), sample(sp, sd))
            energy = np.where(alive, np.clip(energy + intake - m_day, 0.0, cap), energy)
            alive = alive & (energy > 0.0)
        for _ in range(night):
            energy = np.where(alive, np.clip(energy - m_night, 0.0, cap), energy)
            alive = alive & (energy > 0.0)
        # Respawn the dead, spread across the reserve, so sampling continues.
        dead = ~alive
        if dead.any():
            energy[dead] = rng.uniform(0.04, cap, int(dead.sum()))
            alive[dead] = True

    return {"reserve": np.concatenate(rec_E), "tstep": np.concatenate(rec_t),
            "chose_risky": np.concatenate(rec_c).astype(float),
            "params": {"m_day": m_day, "R": R, "beta": beta}}


# --- likelihoods -------------------------------------------------------------------------------

def sequence_negloglik(m_day: float, R: float, beta: float, data: dict, **soln) -> float:
    """Negative log-likelihood of the observed (state, choice) sequence under (m_day, R, beta).

    The exact conditional-choice likelihood: each decision contributes log P(choice | its reserve
    and time; theta). This is the SEQUENCE readout -- it uses the choice-to-reserve pairing.
    """
    res = solve(m_day, R, **soln)
    adv = _advantage_at(res, data["reserve"], data["tstep"])
    p = _sigmoid(beta * adv)
    p = np.clip(p, 1e-12, 1 - 1e-12)
    c = data["chose_risky"]
    return -float(np.sum(c * np.log(p) + (1 - c) * np.log(1 - p)))


def _occupancy(data: dict, day: int, n_bins: int = 40, cap: float = CAP):
    """Reserve x time occupancy weights (the histogram the aggregate analyst keeps)."""
    edges = np.linspace(0, cap, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    w = np.zeros((day, n_bins))
    b = np.clip(np.digitize(data["reserve"], edges) - 1, 0, n_bins - 1)
    for t in range(day):
        m = data["tstep"] == t
        if m.any():
            np.add.at(w[t], b[m], 1.0)
    return centers, w


def aggregate_negloglik(m_day: float, R: float, beta: float, data: dict, occ, day: int,
                        **soln) -> float:
    """Negative log-likelihood of only the OVERALL risky-choice proportion (binomial).

    The aggregate analyst keeps the reserve-occupancy histogram ``occ`` (centers, weights) and the
    total risky count, but NOT which choice happened at which reserve. The predicted marginal
    risky probability is the occupancy-weighted average of sigma(beta*(q_risky - q_safe)); the data
    is one binomial count. Any (R, beta) reproducing that scalar fits equally -- the source of
    R's non-identifiability.
    """
    centers, w = occ
    res = solve(m_day, R, **soln)
    e = res["energy"]
    d = res["q_risky"] - res["q_safe"]
    wsum = w.sum()
    p_mean = 0.0
    for t in range(day):
        if w[t].sum() == 0:
            continue
        adv = np.interp(np.clip(centers, e[0], e[-1]), e, d[t])
        p_mean += np.sum(w[t] * _sigmoid(beta * adv))
    p_mean = np.clip(p_mean / wsum, 1e-12, 1 - 1e-12)
    n = data["chose_risky"].size
    k = float(data["chose_risky"].sum())
    return -float(k * np.log(p_mean) + (n - k) * np.log(1 - p_mean))


# --- fitting -----------------------------------------------------------------------------------

# Search in a transformed space so the optimizer is unconstrained: positive params via log.
def _pack(m_day, R, beta):
    return np.log([m_day, R, beta])


def _unpack(x):
    m_day, R, beta = np.exp(x)
    return float(m_day), float(R), float(beta)


# Multi-start grid. The neg-loglik has a degenerate plateau at beta -> 0 (every choice 50/50, a
# flat region a single far start can fall into), so we restart from a coarse net of (R, beta) and
# keep the best optimum. m_day starts at a fixed plausible value; it is well-behaved.
_START_R = (0.34, 0.42, 0.50, 0.58)
_START_BETA = (5.0, 14.0)
_STARTS = tuple(_pack(0.035, R, b) for b in _START_BETA for R in _START_R)


def _multistart(obj, x0=None) -> tuple[np.ndarray, float]:
    starts = list(_STARTS) if x0 is None else [np.asarray(x0, float)]
    best_x, best_f = None, np.inf
    for s in starts:
        r = minimize(obj, s, method="Nelder-Mead",
                     options={"xatol": 1e-4, "fatol": 1e-4, "maxiter": 1200})
        if r.fun < best_f:
            best_x, best_f = r.x, float(r.fun)
    return best_x, best_f


def fit_sequence(data: dict, x0=None, **soln) -> dict:
    """Maximum-likelihood recovery of (m_day, R, beta) from the full sequence (multi-start)."""
    day = soln.get("day", DAY)
    soln = {k: v for k, v in soln.items() if k != "day"} | {"day": day}

    def obj(x):
        m_day, R, beta = _unpack(x)
        return sequence_negloglik(m_day, R, beta, data, **soln)

    x, f = _multistart(obj, x0)
    m_day, R, beta = _unpack(x)
    return {"m_day": m_day, "R": R, "beta": beta, "negloglik": f}


def fit_aggregate(data: dict, x0=None, n_bins: int = 40, **soln) -> dict:
    """Maximum-likelihood recovery from the aggregate proportion + occupancy only (multi-start)."""
    day = soln.get("day", DAY)
    cap = soln.get("cap", CAP)
    occ = _occupancy(data, day, n_bins=n_bins, cap=cap)

    def obj(x):
        m_day, R, beta = _unpack(x)
        return aggregate_negloglik(m_day, R, beta, data, occ, day, **soln)

    x, f = _multistart(obj, x0)
    m_day, R, beta = _unpack(x)
    return {"m_day": m_day, "R": R, "beta": beta, "negloglik": f}


def profile_loglik_R(data: dict, R_grid: np.ndarray, true: dict, *, aggregate: bool = False,
                     n_bins: int = 40, **soln) -> np.ndarray:
    """Profile log-likelihood over R: at each R, minimize the neg-loglik over (m_day, beta).

    Returns the profile relative to its own maximum (so 0 = best). A sharp peak at R_true means R is
    identifiable; a flat profile means it is not. ``aggregate`` switches between the two readouts.
    """
    day = soln.get("day", DAY)
    cap = soln.get("cap", CAP)
    occ = _occupancy(data, day, n_bins=n_bins, cap=cap) if aggregate else None
    out = np.empty(R_grid.size)
    for i, R in enumerate(R_grid):
        def obj(x, R=R):
            m_day, beta = np.exp(x)
            if aggregate:
                return aggregate_negloglik(m_day, R, beta, data, occ, day, **soln)
            return sequence_negloglik(m_day, R, beta, data, **soln)
        x0 = np.log([true["m_day"], true["beta"]])
        r = minimize(obj, x0, method="Nelder-Mead",
                     options={"xatol": 1e-4, "fatol": 1e-4, "maxiter": 800})
        out[i] = -r.fun
    return out - out.max()


def fisher_info_map(m_day: float, R: float, beta: float, *, param: str = "R", n_bins: int = 40,
                    eps: float = 1e-3, **soln) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-cell (reserve, time) Fisher information for one parameter, from one Bernoulli choice.

    A choice at state (E,t) is Bernoulli(p) with p = sigma(beta*(q_risky-q_safe)); its Fisher
    information about a parameter is (dp/dtheta)^2 / (p(1-p)). This map shows WHERE in state space
    each parameter is identifiable: cells near the requirement carry the information about R; cells
    away from it (deep in the day, far from the boundary) carry the information about the daytime
    metabolism. Returns (reserve_centers, time_steps, info[time, reserve]). dp/dtheta is a central
    finite difference; the choice probability is evaluated at the true theta.
    """
    day = soln.get("day", DAY)
    cap = soln.get("cap", CAP)
    centers = np.linspace(0, cap, n_bins)

    def p_grid(md, r, b):
        res = solve(md, r, **soln)
        e = res["energy"]
        d = res["q_risky"] - res["q_safe"]
        out = np.empty((day, n_bins))
        for t in range(day):
            adv = np.interp(np.clip(centers, e[0], e[-1]), e, d[t])
            out[t] = _sigmoid(b * adv)
        return out

    p0 = p_grid(m_day, R, beta)
    if param == "R":
        h = R * eps
        dp = (p_grid(m_day, R + h, beta) - p_grid(m_day, R - h, beta)) / (2 * h)
    elif param == "m_day":
        h = m_day * eps
        dp = (p_grid(m_day + h, R, beta) - p_grid(m_day - h, R, beta)) / (2 * h)
    elif param == "beta":
        h = beta * eps
        dp = (p_grid(m_day, R, beta + h) - p_grid(m_day, R, beta - h)) / (2 * h)
    else:
        raise ValueError(f"unknown param {param!r}")
    denom = np.clip(p0 * (1 - p0), 1e-9, None)
    info = dp ** 2 / denom
    return centers, np.arange(day), info


# === Companion: a generic value-learner and its learning-rate/softmax-temperature degeneracy =====
# exp056. The survival PLANNER above has a structurally identifiable parameter (R) because the
# choice is state-dependent: which option is favored depends on the reserve relative to R, so the
# sequence of (reserve, choice) pairs pins R. A generic reinforcement-learning VALUE-LEARNER on the
# SAME matched-mean risk choice has no such structure: it tracks a scalar value per option by a
# delta rule and chooses by softmax. Its two parameters -- learning rate ``alpha`` and softmax
# temperature ``beta`` -- are partially CONFOUNDED in choice sequences (a faster learner with a
# flatter policy mimics a slower learner with a sharper one), the well-known bandit alpha-beta
# degeneracy (Comput. Brain Behav. 2022; Daw 2011). So "the devil is in the sequence" cuts both
# ways: sequences fix R for the structured survival model, but not (alpha, beta) for the
# unstructured value-learner -- identifiability comes from the state-dependent structure, not from
# having sequence data per se.

# Matched-mean risk choice for the value-learner: a sure safe payoff vs a 0/1 risky payoff of equal
# mean (so any preference is a sampling/learning artifact, not a value difference -- the same
# matched stimuli the planner faces).
RL_SAFE_PAYOFF = 0.5
RL_RISKY_HIGH, RL_RISKY_LOW, RL_RISKY_P = 1.0, 0.0, 0.5


def rl_simulate(alpha: float, beta: float, n_trials: int = 400, seed: int = 0,
                q_init: float = 0.5) -> dict:
    """A tabular delta-rule + softmax value-learner on the matched-mean risk choice.

    Two scalar values (safe, risky), updated ``Q[a] += alpha * (reward - Q[a])`` on the chosen
    option; choice by ``P(risky) = sigma(beta * (Q_risky - Q_safe))``. Returns choice/reward arrays.
    """
    rng = np.random.default_rng(seed)
    q = np.array([q_init, q_init], float)
    choices = np.empty(n_trials, np.int8)
    rewards = np.empty(n_trials, float)
    for t in range(n_trials):
        p = 1.0 / (1.0 + np.exp(-beta * (q[1] - q[0])))
        a = int(rng.random() < p)
        if a == 0:
            r = RL_SAFE_PAYOFF
        else:
            r = RL_RISKY_HIGH if rng.random() < RL_RISKY_P else RL_RISKY_LOW
        choices[t] = a
        rewards[t] = r
        q[a] += alpha * (r - q[a])
    return {"choices": choices, "rewards": rewards,
            "params": {"alpha": alpha, "beta": beta}}


def rl_negloglik(alpha: float, beta: float, choices, rewards, q_init: float = 0.5) -> float:
    """Negative trial-by-trial log-likelihood: replay the delta-rule Q updates on the OBSERVED
    rewards (value trajectory fixed by alpha and the data), scoring each choice by softmax.
    """
    q = np.array([q_init, q_init], float)
    ll = 0.0
    for a, r in zip(choices, rewards, strict=True):
        p = 1.0 / (1.0 + np.exp(-beta * (q[1] - q[0])))
        p = min(max(p, 1e-12), 1 - 1e-12)
        ll += np.log(p if a == 1 else 1 - p)
        q[a] += alpha * (r - q[a])
    return -float(ll)


_RL_STARTS = ((0.1, 4.0), (0.3, 8.0), (0.05, 12.0), (0.5, 3.0))
RL_BETA_MAX = 40.0             # cap the softmax temperature: it runs to infinity when a short
#                                sequence looks deterministic (a real identifiability failure that
#                                otherwise sends fits to absurd values). Standard recovery practice.


def rl_fit(choices, rewards, q_init: float = 0.5) -> dict:
    """Maximum-likelihood recovery of (alpha, beta) for the value-learner (multi-start).

    alpha is logit-bounded to (0, 1); beta is positive via log and capped at ``RL_BETA_MAX``.
    """
    def unpack(x):
        return (float(1.0 / (1.0 + np.exp(-x[0]))),
                float(min(np.exp(x[1]), RL_BETA_MAX)))

    def obj(x):
        a, b = unpack(x)
        return rl_negloglik(a, b, choices, rewards, q_init=q_init)

    best_x, best_f = None, np.inf
    for a0, b0 in _RL_STARTS:
        x0 = [np.log(a0 / (1 - a0)), np.log(b0)]
        r = minimize(obj, x0, method="Nelder-Mead",
                     options={"xatol": 1e-4, "fatol": 1e-4, "maxiter": 800})
        if r.fun < best_f:
            best_x, best_f = r.x, float(r.fun)
    alpha, beta = unpack(best_x)
    return {"alpha": alpha, "beta": beta, "negloglik": best_f}


def rl_loglik_surface(choices, rewards, alphas: np.ndarray, betas: np.ndarray,
                      q_init: float = 0.5) -> np.ndarray:
    """Negative-log-likelihood surface over an (alpha, beta) grid, relative to its own minimum.

    Returns ``S[i, j]`` for ``alphas[i]``, ``betas[j]``. A long diagonal valley (low NLL along an
    alpha-beta anticorrelated ridge) is the degeneracy.
    """
    s = np.array([[rl_negloglik(a, b, choices, rewards, q_init=q_init) for b in betas]
                  for a in alphas])
    return s - s.min()
