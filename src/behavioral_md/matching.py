"""Concurrent VI-VI matching via discriminative cues (vectorized, JAX).

Two food patches sit at fixed locations, each marked by a distinct value on the
single cue dimension (e.g. cue 0.2 = "green", cue 0.8 = "red") and running its own
variable-interval (VI) schedule. Patches are NOT different food channels -- they
are the same reinforcer under different discriminative stimuli, so the organism
learns the cue->reinforcement-rate relation (via a value-tuned receptor
population, as in `generalization.py`) and that learned value drives how strongly
it approaches each patch. One cue dimension scales to any number of patches.

The organism moves in 2D under a damped-Verlet movement model and emits actions by
softmax (the matching law). Approach force toward patch k is
``learned_value(cue_k) * intensity_k`` projected onto each movement direction.
A VI timer arms each patch (Bernoulli, mean interval 1/arm_prob); a reinforcer is
collected when the organism is within ``consume_radius`` of an armed patch, which
trains the cue receptors and disarms the patch. Travel between patches is the
changeover-delay (COD) analog (set by patch separation).

Behavior allocation (time/contact at each patch) is measured against obtained
reinforcement and fit to the generalized matching law
``log(B1/B2) = a*log(R1/R2) + log b``. Energy/death are omitted: this is a
steady-state allocation preparation (the organism is always motivated to forage).
The reinforcer model carries amount/delay/probability hooks for the concatenated
matching law later; only rate (VI) is exercised here.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

# Action set: 4 movement directions + stay.
_MOVES = jnp.array([[0.0, 1.0], [0.0, -1.0], [-1.0, 0.0], [1.0, 0.0], [0.0, 0.0]])


class MatchConfig(NamedTuple):
    # Odd grid so a centered layout has equal wall margins on both sides; an even
    # grid puts the center at a half-cell and makes patch-to-wall margins
    # asymmetric, which traps the organism near the closer wall (a spurious side
    # bias). See exp012 diagnosis.
    grid_size: int = 11
    sensor_range: float = 8.0
    consume_radius: float = 1.0
    n_receptors: int = 21
    beta: float = 6.0           # cue tuning width
    lr_cue: float = 0.02        # cue-receptor learning rate (summed error)
    reinf_asymptote: float = 1.0
    damping: float = 10.0
    dt: float = 0.1
    temperature: float = 0.3    # softmax (matching) temperature
    approach_gain: float = 3.0  # scales learned value -> movement force
    # Delay of reinforcement reduces the reinforcer's EFFICACY (its strengthening
    # effect), the empirical delay-discounting finding -- NOT a credit-assignment
    # / eligibility-decay effect. In the standard procedure the response occurs,
    # the experiment blacks out, the delay elapses, and the reinforcer is then
    # delivered, so attribution is unambiguous; delay simply makes the delivered
    # reinforcer a weaker strengthener. Modeled as a discount on its teaching signal.
    delay_discount: str = "hyperbolic"   # "hyperbolic" (Mazur) | "exponential"
    delay_k: float = 0.5        # hyperbolic: efficacy *= 1/(1 + k*D)
    delay_tau: float = 5.0      # exponential: efficacy *= exp(-D/tau)


def _tuning(centers, beta, value):
    """Shepard tuning of each receptor to a cue value. value: [...]; -> [..., K]."""
    return jnp.exp(-beta * jnp.abs(value[..., None] - centers))


def make_matching_sim(mcfg: MatchConfig, patch_pos, patch_cue, start):
    """Return a jitted ``sim(keys, arm_prob) -> (time_at, reinforced)`` over a population.

    ``patch_pos`` [P,2], ``patch_cue`` [P], ``start`` [2] are fixed geometry/cues.
    ``arm_prob`` [P] is the per-step VI arming probability per patch (runtime arg,
    so the schedule can be swept without recompiling). Outputs are per-organism
    [O, P]: steps spent at each patch (allocation) and reinforcers collected.
    """
    centers = jnp.linspace(0.0, 1.0, mcfg.n_receptors)
    patch_pos = jnp.asarray(patch_pos, float)
    patch_cue = jnp.asarray(patch_cue, float)
    cue_act = _tuning(centers, mcfg.beta, patch_cue)        # [P, K] (fixed per patch)
    start = jnp.asarray(start, float)
    move_dirs = _MOVES                                       # [5, 2]

    def discount(delay):
        """Efficacy of a reinforcer delivered after ``delay``, in (0, 1]; 1 at 0.

        Represents reduced strengthening by delayed reinforcement (empirical delay
        discounting), not a credit-assignment effect.
        """
        if mcfg.delay_discount == "exponential":
            return jnp.exp(-delay / mcfg.delay_tau)
        return 1.0 / (1.0 + mcfg.delay_k * delay)            # hyperbolic (Mazur)

    def step(state, sched):
        # Per-patch reinforcer dimensions (concatenated matching law):
        # arm_prob [P] = rate (VI), amount [P] = magnitude, prob [P] = probability
        # of reinforcement per armed contact, delay [P] = reinforcer delay.
        arm_prob, amount, prob, delay = sched
        pos = state["pos"]
        diff = patch_pos[None, :, :] - pos[:, None, :]       # [O, P, 2]
        dist = jnp.linalg.norm(diff, axis=2)                 # [O, P]
        unit = jnp.where(dist[..., None] > 1e-9, diff / jnp.clip(dist[..., None], 1e-9, None), 0.0)
        intensity = jnp.exp(-dist / mcfg.sensor_range)       # [O, P]
        value = state["w"] @ cue_act.T                       # [O, P] learned value of each cue
        # Movement force per direction: sum over patches of value*intensity*(dir.move).
        pull = (mcfg.approach_gain * value * intensity)[:, :, None]   # [O, P, 1]
        dots = unit @ move_dirs.T                            # [O, P, 5]
        force = jnp.sum(pull * dots, axis=1)                 # [O, 5]  (stay col -> 0)

        vel = (state["act"] - state["prev"]) / mcfg.dt
        net = force - mcfg.damping * vel
        # damped Verlet (mass = 1): x' = 2x - x_prev + net*dt^2
        new_act = 2 * state["act"] - state["prev"] + net * mcfg.dt**2
        new_act = jnp.clip(new_act, -10.0, 10.0)

        key, sub = jax.random.split(state["key"])
        z = new_act / mcfg.temperature
        z = z - z.max(axis=1, keepdims=True)
        probs = jnp.exp(z) / jnp.exp(z).sum(axis=1, keepdims=True)
        action = jax.random.categorical(sub, jnp.log(probs))             # [O]
        new_pos = jnp.clip(pos + move_dirs[action], 0.0, mcfg.grid_size - 1)

        # VI: arm each patch (Bernoulli). A contact with an armed patch is a
        # collection OPPORTUNITY; it is reinforced with probability prob_k.
        key, sub2 = jax.random.split(key)
        newly = jax.random.uniform(sub2, state["armed"].shape) < arm_prob[None, :]
        armed = state["armed"] | newly
        d_new = jnp.linalg.norm(patch_pos[None, :, :] - new_pos[:, None, :], axis=2)
        in_range = d_new <= mcfg.consume_radius              # [O, P]
        opportunity = in_range & armed                       # [O, P] armed contact
        key, sub3 = jax.random.split(key)
        roll = jax.random.uniform(sub3, opportunity.shape)
        reinforced = opportunity & (roll < prob[None, :])    # PROBABILITY gate
        armed = armed & (~reinforced)                        # disarm only on reinforcement

        # Train cue receptors on every armed contact (summed/elemental RW error):
        # reinforced contacts -> target lambda * amount * delay-discount(D)
        # (AMOUNT and DELAY terms); non-reinforced contacts -> target 0 (extinction
        # trial -> partial reinforcement makes the value track PROBABILITY ~ p).
        eff_amount = amount * discount(delay)                # [P] amount x delay-discount
        opp_f = opportunity.astype(float)                    # [O, P]
        contact_cue_act = jnp.einsum("op,pk->ok", opp_f, cue_act)            # [O, K]
        target = jnp.sum(reinforced.astype(float) * (mcfg.reinf_asymptote * eff_amount)[None, :],
                         axis=1)                             # [O]
        v_pred = jnp.sum(state["w"] * contact_cue_act, axis=1)
        err = target - v_pred
        w = jnp.clip(state["w"] + mcfg.lr_cue * contact_cue_act * err[:, None], -5.0, 5.0)

        new_state = {"pos": new_pos, "act": new_act, "prev": state["act"],
                     "w": w, "armed": armed, "key": key}
        rein_f = reinforced.astype(jnp.float32)
        # Outputs: presence at each patch, reinforcer counts, amount obtained.
        return new_state, (in_range.astype(jnp.float32), rein_f, rein_f * amount[None, :])

    def initial_state(n_org, key):
        P = patch_pos.shape[0]
        return {
            "pos": jnp.broadcast_to(start, (n_org, 2)),
            "act": jnp.zeros((n_org, 5)),
            "prev": jnp.zeros((n_org, 5)),
            "w": jnp.zeros((n_org, mcfg.n_receptors)),
            "armed": jnp.zeros((n_org, P), bool),
            "key": key,
        }

    @jax.jit
    def sim(state0, keys, arm_prob, amount=None, prob=None, delay=None):
        """state0 from initial_state; keys [T,2] per-step rng; arm_prob [P] (rate).

        Concatenated-law reinforcer dimensions per patch (each [P], runtime args):
        ``amount`` magnitude (default 1), ``prob`` probability of reinforcement per
        armed contact (default 1), ``delay`` reinforcer delay (default 0). Returns
        per-organism, summed over T steps: (time_at [O,P], count [O,P],
        amount_obtained [O,P]).
        """
        p = patch_pos.shape[0]
        amount = jnp.ones(p) if amount is None else amount
        prob = jnp.ones(p) if prob is None else prob
        delay = jnp.zeros(p) if delay is None else delay

        def scan_step(st, k):
            st = {**st, "key": k}
            return step(st, (arm_prob, amount, prob, delay))

        _, (presence, count, amt) = jax.lax.scan(scan_step, state0, keys)
        return presence.sum(axis=0), count.sum(axis=0), amt.sum(axis=0)

    return sim, initial_state
