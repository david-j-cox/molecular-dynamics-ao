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
                metabolism: float, cap: float = 1.0, n_egrid: int = 401) -> dict:
    """Exact survival DP over one day/night cycle; derive the optimal risk policy.

    ``safe_outcomes`` / ``risky_outcomes`` are lists of ``(probability, intake)`` for the
    two foraging options (usually matched-mean so they differ only in variance). Returns a
    dict with the energy grid and, per forward day-step (0 = dawn, day_steps-1 = dusk), the
    survival value of each option ``q_safe``/``q_risky`` [day_steps, n_egrid], the optimal
    ``policy_risky`` (1 where the risky option strictly maximizes survival), the survival
    probability ``value`` = max of the two, and the emergent ``night_requirement`` =
    ``night_steps * metabolism``.
    """
    e = np.linspace(0.0, cap, n_egrid)
    value = (e > 0.0).astype(float)                 # end of cycle: survived iff alive

    # Night, worked backward: forced drain, no choice, die if it takes you to <= 0.
    for _ in range(night_steps):
        value = _survive_next(e, e - metabolism, value, cap)

    q_safe = np.zeros((day_steps, n_egrid))
    q_risky = np.zeros((day_steps, n_egrid))
    for t in range(day_steps - 1, -1, -1):          # dusk -> dawn
        qs = sum(p * _survive_next(e, e + d - metabolism, value, cap) for p, d in safe_outcomes)
        qr = sum(p * _survive_next(e, e + d - metabolism, value, cap) for p, d in risky_outcomes)
        q_safe[t] = qs
        q_risky[t] = qr
        value = np.maximum(qs, qr)                  # optimal: take the better option

    policy_risky = (q_risky > q_safe + 1e-12).astype(float)
    return {"energy": e, "q_safe": q_safe, "q_risky": q_risky,
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
