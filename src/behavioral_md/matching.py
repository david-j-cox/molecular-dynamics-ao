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
    grid_size: int = 10
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

    def step(state, sched):
        arm_prob, amount = sched          # arm_prob [P] (rate), amount [P] (magnitude)
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

        # VI: arm each patch (Bernoulli), then collect at an armed patch in range.
        key, sub2 = jax.random.split(key)
        newly = jax.random.uniform(sub2, state["armed"].shape) < arm_prob[None, :]
        armed = state["armed"] | newly
        d_new = jnp.linalg.norm(patch_pos[None, :, :] - new_pos[:, None, :], axis=2)
        in_range = d_new <= mcfg.consume_radius              # [O, P]
        collect = in_range & armed                           # [O, P]
        armed = armed & (~collect)

        # Train cue receptors on collection (summed/elemental RW error). The
        # teaching magnitude is the collected reinforcer's AMOUNT, so the cue's
        # learned value tracks reinforcement magnitude (the concatenated-law
        # amount term), not just its occurrence.
        collect_f = collect.astype(float)                    # [O, P]
        mag = jnp.sum(collect_f * amount[None, :], axis=1)    # [O] amount collected
        coll_cue_act = jnp.einsum("op,pk->ok", collect_f, cue_act)   # [O, K]
        v_pred = jnp.sum(state["w"] * coll_cue_act, axis=1)  # [O]
        err = mcfg.reinf_asymptote * mag - v_pred
        w = jnp.clip(state["w"] + mcfg.lr_cue * coll_cue_act * err[:, None], -5.0, 5.0)

        new_state = {"pos": new_pos, "act": new_act, "prev": state["act"],
                     "w": w, "armed": armed, "key": key}
        # Outputs: presence at each patch, reinforcer counts, and amount obtained.
        return new_state, (in_range.astype(jnp.float32), collect_f, collect_f * amount[None, :])

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
    def sim(state0, keys, arm_prob, amount=None):
        """state0 from initial_state; keys [T,2] per-step rng; arm_prob [P] (rate).

        ``amount`` [P] is the reinforcer magnitude per patch (defaults to all 1 =
        rate-only). Returns per-organism, summed over T steps:
        (time_at [O,P], count [O,P], amount_obtained [O,P]).
        """
        if amount is None:
            amount = jnp.ones(patch_pos.shape[0])

        def scan_step(st, k):
            st = {**st, "key": k}
            return step(st, (arm_prob, amount))

        _, (presence, count, amt) = jax.lax.scan(scan_step, state0, keys)
        return presence.sum(axis=0), count.sum(axis=0), amt.sum(axis=0)

    return sim, initial_state
