"""Differentiable stochastic surrogate of the concurrent-matching rollout.

The stochastic matching engine (``matching.py``) measures generalized-matching-law
sensitivities, but it is not differentiable: actions are sampled with
``jax.random.categorical``, VI patches arm with a Bernoulli draw, and reinforcement
passes through a probability gate. Gradients of a sensitivity ``a`` with respect to
the organism parameters cannot flow through those discrete operations.

Why a deterministic (expected-value) surrogate does NOT work here: in this engine
*under*matching (a < 1) is produced by the **action sampling noise** -- the organism
keeps wandering to the poorer patch instead of always choosing the richer one. Any
expected-value relaxation removes that noise, sharpens the softmax choice, and
*over*matches (measured a_rate ~ 2.7-3.7, and its slope no longer tracks the
stochastic one -- gradients would not transfer). Sensitivity is a noise phenomenon,
so the surrogate must stay stochastic.

This module keeps the EXACT ``matching.py`` particle rollout (2-D position, damped
Verlet, value-weighted force, cue-receptor Rescorla-Wagner learning) and changes only
the non-differentiable steps:

- **Gumbel-softmax actions** (replaces ``categorical``): the categorical sample
  ``argmax(logits + g)`` (g ~ Gumbel) is relaxed to ``y = softmax((logits + g)/TAU)``,
  a soft one-hot; the expected displacement is ``y @ move_dirs``. As ``TAU -> 0`` this
  recovers the true sample. The matching ``temperature`` stays inside the logits
  (``logits = activation / temperature``), so its role is unchanged; ``TAU`` is a
  separate relaxation knob. This is the reparameterization trick: the Gumbel noise is
  drawn from FIXED keys (common random numbers), so the loss is a deterministic smooth
  function of the parameters and gradients are exact for the relaxed objective.
- **Relaxed Bernoulli VI arming** (replaces the Bernoulli draws): a patch arms with
  a relaxed-Bernoulli sample ``sigmoid((logit(arm) + logistic_noise)/TAU_B)`` (again
  reparameterized from fixed keys), accumulated into a continuous armed occupancy
  ``armed' = armed + (1-armed)*newly``, with soft contact
  ``soft_in = sigmoid((consume_radius - dist)/WIDTH)`` and soft disarm on
  reinforcement. Keeping the arming STOCHASTIC matters: an expected/deterministic
  arming smooths away the VI timing noise, which flips the sign of the ``lr_cue``
  sensitivity (a higher learning rate should chase that noise and *increase*
  undermatching, as in the stochastic engine).

A population of organisms (different Gumbel keys) is rolled out so the summed
allocation is smooth, and -- because each organism still follows a noisy particle
trajectory -- contacts at the two patches are temporally separated, which is what lets
the shared-error elemental RW rule learn a separate value (hence a separate amount
sensitivity) per cue.

The free parameters gradients flow through are ``temperature``, ``approach_gain``, and
``beta`` (the cue-tuning width); they are passed in a ``params`` dict and everything
else is taken from the fixed ``MatchConfig``. ``beta`` is the strong, sign-consistent
lever on the rate sensitivity (and, with ``temperature``, spans the 2-D
(a_rate, a_amt) target space), while ``temperature``/``approach_gain`` set the amount
sensitivity. ``lr_cue`` is deliberately NOT a free parameter: in the stochastic engine
a higher learning rate sharply lowers the rate sensitivity (a noise-accumulation
effect), but the relaxed surrogate does not reproduce that dependence, so its ``lr_cue``
gradient does not transfer -- including it would steer the fit in a direction that does
not hold in the real (stochastic) sim. See the per-parameter transfer probe in the
lab notebook.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from behavioral_md._kernels import exp_falloff, safe_unit
from behavioral_md.matching import _MOVES, MatchConfig, _tuning

# Gumbel-softmax relaxation temperature (separate from the matching temperature).
# Smaller -> closer to a true categorical sample but higher-variance gradients.
TAU = 0.5
# Relaxed-Bernoulli temperature for the VI arming draw.
TAU_B = 0.5
# Soft-contact width: length scale (grid cells) over which presence turns on at
# ``consume_radius``. Larger -> softer/flatter presence.
WIDTH = 0.5

# Parameters the search can vary; the rollout reads each from the params dict if
# present, else from MatchConfig. lr_cue is NOT here on purpose -- its rate-sensitivity
# effect does not transfer from the relaxed surrogate to the stochastic engine (see
# docstring). The other three are the per-dimension curvature levers, each orthogonal
# to the rate anchor (the rate/amount/prob/delay sweeps hold the other dimensions at
# their neutral value, so a lever only moves its own sensitivity):
#   amount_exponent (rho)      -> a_amt  (value tracks amount**rho)
#   probability_exponent (sigma) -> a_prob (reinforcement gated on prob**sigma)
#   delay_k                    -> a_delay (steepness of the hyperbolic delay discount)
TUNABLE = ("temperature", "approach_gain", "beta",
           "amount_exponent", "probability_exponent", "delay_k")

# Default free set: the discriminability levers, which move both sensitivities together.
# Add "amount_exponent" to the free set (see fit.fit(free=...)) to decouple a_amt.
FREE_PARAMS = ("temperature", "approach_gain", "beta")


def default_params(mcfg: MatchConfig, free=FREE_PARAMS) -> dict:
    """The free-parameter dict at a config's values (the search starting point)."""
    return {k: float(getattr(mcfg, k)) for k in free}


def make_matching_sim_soft(mcfg: MatchConfig, patch_pos, patch_cue, start):
    """Return ``rollout(params, arm, amount, n_steps, n_org, key) -> (B, count)``.

    ``patch_pos`` [P,2], ``patch_cue`` [P], ``start`` [2] are the fixed geometry/cues
    (as in ``make_matching_sim``). ``params`` is a dict over ``FREE_PARAMS``; ``arm``
    and ``amount`` are per-patch schedules [P]. ``prob`` is held at 1 and ``delay`` at
    0, matching the exp008/exp011 procedures. Outputs are per-patch [P], summed over
    the population and the ``n_steps`` steps: behavior allocation ``B`` (soft presence)
    and expected reinforcer ``count``. Differentiable w.r.t. ``params``.
    """
    centers = jnp.linspace(0.0, 1.0, mcfg.n_receptors)       # [K]
    patch_pos = jnp.asarray(patch_pos, float)                # [P, 2]
    patch_cue = jnp.asarray(patch_cue, float)                # [P]
    start = jnp.asarray(start, float)                        # [2]
    move_dirs = _MOVES                                       # [M=5, 2]
    n_patch = patch_pos.shape[0]

    def rollout(params, arm, amount, n_steps, n_org=128, key=None, prob=None, delay=None):
        # Each tunable comes from params if the search exposes it, else from mcfg.
        temperature = params.get("temperature", mcfg.temperature)
        approach_gain = params.get("approach_gain", mcfg.approach_gain)
        beta = params.get("beta", mcfg.beta)
        rho = params.get("amount_exponent", mcfg.amount_exponent)
        sigma = params.get("probability_exponent", mcfg.probability_exponent)
        delay_k = params.get("delay_k", mcfg.delay_k)
        lr_cue = mcfg.lr_cue                             # fixed (does not transfer)
        arm = jnp.asarray(arm, float)                        # [P]
        amount = jnp.asarray(amount, float)                  # [P]
        prob = jnp.ones(n_patch) if prob is None else jnp.asarray(prob, float)
        delay = jnp.zeros(n_patch) if delay is None else jnp.asarray(delay, float)
        # Magnitude utility curvature (a_amt lever); rho==1.0 keeps the linear path.
        mag = amount if (isinstance(rho, float) and rho == 1.0) else amount ** rho
        # Delay discount (a_delay lever via delay_k); hyperbolic unless config says exp.
        if mcfg.delay_discount == "exponential":
            disc = jnp.exp(-delay / mcfg.delay_tau)
        else:
            disc = 1.0 / (1.0 + delay_k * delay)
        eff_amount = mag * disc                              # [P]
        # Probability weighting (a_prob lever); sigma==1.0 keeps the linear gate.
        prob_eff = prob if (isinstance(sigma, float) and sigma == 1.0) else prob ** sigma
        cue_act = _tuning(centers, beta, patch_cue)          # [P, K] (beta is free)

        # Fixed common-random-number noise (reparameterization): Gumbel for the action
        # [T, O, M] and logistic for the relaxed-Bernoulli VI arming [T, O, P].
        if key is None:
            key = jax.random.key(0)
        k_act, k_arm = jax.random.split(key)
        gumbel = jax.random.gumbel(k_act, (n_steps, n_org, move_dirs.shape[0]))
        u = jax.random.uniform(k_arm, (n_steps, n_org, n_patch),
                               minval=_EPS, maxval=1.0 - _EPS)
        logistic = jnp.log(u) - jnp.log1p(-u)            # logistic(0,1) noise
        logit_arm = jnp.log(arm) - jnp.log1p(-arm)       # [P] arming log-odds

        def step(state, noise):
            g, lg = noise                                # gumbel [O,M], logistic [O,P]
            pos = state["pos"]                               # [O, 2]
            w = state["w"]                                   # [O, K]
            diff = patch_pos[None, :, :] - pos[:, None, :]   # [O, P, 2]
            dist = jnp.linalg.norm(diff, axis=2)             # [O, P]
            unit = safe_unit(diff, dist)
            intensity = exp_falloff(dist, mcfg.sensor_range)            # [O, P]
            value = w @ cue_act.T                            # [O, P] learned cue value
            pull = (approach_gain * value * intensity)[:, :, None]      # [O, P, 1]
            dots = unit @ move_dirs.T                        # [O, P, M]
            force = jnp.sum(pull * dots, axis=1)             # [O, M]

            vel = (state["act"] - state["prev"]) / mcfg.dt
            net = force - mcfg.damping * vel
            new_act = jnp.clip(2 * state["act"] - state["prev"] + net * mcfg.dt**2,
                               -10.0, 10.0)

            # Gumbel-softmax action (reparameterized sample), then expected move.
            logits = new_act / temperature                   # [O, M]
            y = jax.nn.softmax((logits + g) / TAU, axis=1)   # [O, M] soft one-hot
            new_pos = jnp.clip(pos + y @ move_dirs, 0.0, mcfg.grid_size - 1)

            d_new = jnp.linalg.norm(patch_pos[None, :, :] - new_pos[:, None, :], axis=2)
            soft_in = jax.nn.sigmoid((mcfg.consume_radius - d_new) / WIDTH)   # [O, P]

            # Relaxed-Bernoulli arming draw, then soft-OR into the persistent armed
            # occupancy and soft disarm on reinforcement.
            newly = jax.nn.sigmoid((logit_arm[None, :] + lg) / TAU_B)   # [O, P]
            p_arm = state["armed_p"] + (1.0 - state["armed_p"]) * newly
            r = soft_in * p_arm * prob_eff[None, :]          # [O, P] expected reinforcement
            armed_p = p_arm * (1.0 - soft_in * prob_eff[None, :])

            # Cue-receptor RW update on the expected armed contact / reinforcement.
            opp = soft_in * p_arm                            # [O, P] expected armed contact
            contact_cue_act = jnp.einsum("op,pk->ok", opp, cue_act)     # [O, K]
            target = jnp.sum(r * (mcfg.reinf_asymptote * eff_amount)[None, :], axis=1)
            v_pred = jnp.sum(w * contact_cue_act, axis=1)
            err = target - v_pred
            w_new = jnp.clip(w + lr_cue * contact_cue_act * err[:, None], -5.0, 5.0)

            new_state = {"pos": new_pos, "act": new_act, "prev": state["act"],
                         "w": w_new, "armed_p": armed_p}
            return new_state, (soft_in, r)

        state0 = {
            "pos": jnp.broadcast_to(start, (n_org, 2)),
            "act": jnp.zeros((n_org, move_dirs.shape[0])),
            "prev": jnp.zeros((n_org, move_dirs.shape[0])),
            "w": jnp.zeros((n_org, mcfg.n_receptors)),
            "armed_p": jnp.zeros((n_org, n_patch)),
        }
        _, (presence, count) = jax.lax.scan(step, state0, (gumbel, logistic))
        # Sum over time then population -> per-patch totals [P].
        return presence.sum(axis=(0, 1)), count.sum(axis=(0, 1))

    return rollout


# --- Tier-1 validation: soft surrogate vs the stochastic engine -------------------

# Same two-patch geometry / cue layout / sweeps as exp008 and exp011.
_PATCH_POS = [[2.0, 5.0], [8.0, 5.0]]
_PATCH_CUE = [0.2, 0.8]
_START = [5.0, 5.0]
RATE_PAIRS = [(0.02, 0.18), (0.04, 0.16), (0.06, 0.14), (0.10, 0.10),
              (0.14, 0.06), (0.16, 0.04), (0.18, 0.02)]
AMOUNT_PAIRS = [(0.3, 1.7), (0.5, 1.5), (0.7, 1.3), (1.0, 1.0),
                (1.3, 0.7), (1.5, 0.5), (1.7, 0.3)]
# Probability (exp013) and delay (exp014) sweeps, run at identical VI (ARM_PD).
PROB_PAIRS = [(0.9, 0.1), (0.75, 0.25), (0.5, 0.5), (0.25, 0.75), (0.1, 0.9)]
DELAY_PAIRS = [(1.0, 9.0), (2.0, 6.0), (4.0, 4.0), (6.0, 2.0), (9.0, 1.0)]
_ARM_AMT = [0.10, 0.10]    # equal VI for the amount sweep (exp011)
_ARM_PD = [0.12, 0.12]     # equal VI for the probability/delay sweeps (exp013/14)
_EPS = 1e-6


def _slope(x, y):
    """Least-squares slope of y on x (the differentiable GML sensitivity)."""
    xc = x - x.mean()
    yc = y - y.mean()
    return jnp.sum(xc * yc) / (jnp.sum(xc * xc) + 1e-12)


def soft_sensitivities(params, mcfg: MatchConfig | None = None,
                       n_steps: int = 1500, n_org: int = 128, key=None):
    """Soft-surrogate (a_rate, a_amt) at ``params``, mirroring exp008/exp011.

    rate: sweep arming-probability pairs at equal amount; amount: sweep amount pairs
    at equal VI. x is the programmed log schedule ratio, y the log behavior-allocation
    ratio (population-summed presence); the sensitivity is the slope.
    """
    mcfg = MatchConfig() if mcfg is None else mcfg
    if key is None:
        key = jax.random.key(0)
    rollout = make_matching_sim_soft(mcfg, _PATCH_POS, _PATCH_CUE, _START)

    def behavior_logratio(arm, amount):
        b, _c = rollout(params, arm, amount, n_steps, n_org, key)
        return jnp.log((b[0] + _EPS) / (b[1] + _EPS))

    arms = jnp.array(RATE_PAIRS)
    x_rate = jnp.log(arms[:, 0] / arms[:, 1])
    y_rate = jax.vmap(lambda p: behavior_logratio(p, jnp.ones(2)))(arms)
    a_rate = _slope(x_rate, y_rate)

    amts = jnp.array(AMOUNT_PAIRS)
    x_amt = jnp.log(amts[:, 0] / amts[:, 1])
    y_amt = jax.vmap(lambda a: behavior_logratio(jnp.array([0.10, 0.10]), a))(amts)
    a_amt = _slope(x_amt, y_amt)
    return a_rate, a_amt


def soft_sensitivities_all(params, mcfg: MatchConfig | None = None,
                           n_steps: int = 1500, n_org: int = 128, key=None) -> dict:
    """All four soft sensitivities as a dict {rate, amt, prob, delay}.

    Mirrors exp008/011/013/014: each dimension is swept while the others are held at
    their neutral value (amount=1, prob=1, delay=0), so each curvature lever moves only
    its own sensitivity. The rate sweep regresses on the PROGRAMMED arm ratio (the
    surrogate's clean x); a_delay is the negated slope (delay enters the GML with a
    negative sign). Used for the prob/delay fitting (exp025); ``soft_sensitivities``
    keeps the cheaper rate+amt pair for exp023/exp024.
    """
    mcfg = MatchConfig() if mcfg is None else mcfg
    if key is None:
        key = jax.random.key(0)
    rollout = make_matching_sim_soft(mcfg, _PATCH_POS, _PATCH_CUE, _START)
    one, zero = jnp.ones(2), jnp.zeros(2)

    def blr(arm, amount, prob, delay):
        b, _c = rollout(params, arm, amount, n_steps, n_org, key, prob=prob, delay=delay)
        return jnp.log((b[0] + _EPS) / (b[1] + _EPS))

    arms = jnp.array(RATE_PAIRS)
    a_rate = _slope(jnp.log(arms[:, 0] / arms[:, 1]),
                    jax.vmap(lambda p: blr(p, one, one, zero))(arms))
    amts = jnp.array(AMOUNT_PAIRS)
    a_amt = _slope(jnp.log(amts[:, 0] / amts[:, 1]),
                   jax.vmap(lambda a: blr(jnp.array(_ARM_AMT), a, one, zero))(amts))
    probs = jnp.array(PROB_PAIRS)
    a_prob = _slope(jnp.log(probs[:, 0] / probs[:, 1]),
                    jax.vmap(lambda p: blr(jnp.array(_ARM_PD), one, p, zero))(probs))
    dels = jnp.array(DELAY_PAIRS)
    a_delay = -_slope(jnp.log(dels[:, 0] / dels[:, 1]),
                      jax.vmap(lambda d: blr(jnp.array(_ARM_PD), one, one, d))(dels))
    return {"rate": a_rate, "amt": a_amt, "prob": a_prob, "delay": a_delay}


def validate_soft_vs_stochastic(n_steps: int = 1500, n_org: int = 128):
    """Tier-1 check: soft sensitivities sit in the undermatching band and move with
    the parameters the same way the stochastic engine does.

    Returns a dict of named numbers; the ``__main__`` block prints them with pass
    flags. Equality is not required -- only that soft a_rate is in (0, 1) and that
    sweeping a parameter yields a soft a_rate positively correlated with the
    stochastic one (so gradients transfer). The sweep uses ``beta`` because it is the
    strong, sign-consistent lever on a_rate (a flat lever like temperature gives an
    uninformative correlation of two near-constant curves).
    """
    import numpy as np

    from behavioral_md.matching import make_matching_sim

    mcfg = MatchConfig()
    a_rate0, a_amt0 = soft_sensitivities(default_params(mcfg), mcfg, n_steps, n_org)

    sweep = [4.0, 5.0, 6.0, 7.5, 9.0]
    soft_curve, stoch_curve = [], []
    keys = jax.random.split(jax.random.key(0), 4000)
    for bval in sweep:
        sp = {**default_params(mcfg), "beta": bval}
        ar, _ = soft_sensitivities(sp, mcfg, n_steps, n_org)
        soft_curve.append(float(ar))

        m = mcfg._replace(beta=bval)
        sim, initial_state = make_matching_sim(m, _PATCH_POS, _PATCH_CUE, _START)
        logB, logR = [], []
        for i, (pL, pR) in enumerate(RATE_PAIRS):
            st0 = initial_state(400, jax.random.key(100 + i))
            time_at, reinforced, _ = sim(st0, keys, jnp.array([pL, pR]))
            t = np.asarray(time_at)
            rr = np.asarray(reinforced)
            ok = (t[:, 0] > 0) & (t[:, 1] > 0) & (rr[:, 0] > 0) & (rr[:, 1] > 0)
            logB.append(np.log(t[ok, 0] / t[ok, 1]))
            logR.append(np.log(rr[ok, 0] / rr[ok, 1]))
        x = np.concatenate(logR)
        y = np.concatenate(logB)
        stoch_curve.append(float(np.polyfit(x, y, 1)[0]))

    corr = float(np.corrcoef(np.array(soft_curve), np.array(stoch_curve))[0, 1])
    return {
        "soft_a_rate_default": float(a_rate0),
        "soft_a_amt_default": float(a_amt0),
        "beta_sweep": sweep,
        "soft_a_rate_curve": soft_curve,
        "stoch_a_rate_curve": stoch_curve,
        "sweep_correlation": corr,
    }


if __name__ == "__main__":
    res = validate_soft_vs_stochastic()
    ar = res["soft_a_rate_default"]
    aa = res["soft_a_amt_default"]
    corr = res["sweep_correlation"]
    print(f"soft a_rate (default params) = {ar:.3f}   "
          f"{'OK' if 0.0 < ar < 1.0 else 'OUT OF BAND'} (expect 0<a<1)")
    print(f"soft a_amt  (default params) = {aa:.3f}")
    print("   beta    soft_a_rate  stoch_a_rate")
    for bv, s, t in zip(res["beta_sweep"], res["soft_a_rate_curve"],
                        res["stoch_a_rate_curve"], strict=False):
        print(f"   {bv:6.2f}     {s:7.3f}     {t:7.3f}")
    print(f"soft-vs-stochastic correlation = {corr:.3f}   "
          f"{'OK' if corr > 0.7 else 'WEAK (consider molar fallback)'}")
