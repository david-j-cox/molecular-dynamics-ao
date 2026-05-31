"""Operant chamber: a single press response under schedules of reinforcement.

No navigation -- the organism either presses or not each step. Pressing is driven
by the learned value of pressing plus an energy-deficit motivation, integrated
with the same damped-Verlet dynamics and emitted stochastically (softmax over
press / no-press). A schedule decides when a press produces a reinforcer:

  FR n : every nth press is reinforced (fixed ratio)
  VR n : each press reinforced with probability 1/n (variable ratio)
  FI t : the first press after t steps since the last reinforcer (fixed interval)
  VI t : the first press after a variable interval (Bernoulli arming, mean t)

The energy budget is included on purpose: a reinforcer raises energy, lowering the
deficit motivation (a post-reinforcement pause); energy then drains (basal
metabolism + press effort), raising motivation again and accelerating responding.
This is the ground-up bet that within-schedule patterns (FR break-and-run, FI
scallop) and the VR>VI rate difference emerge from energy-driven motivation +
value learning, rather than being imposed. Vectorized over organisms (NumPy);
the per-step loop is sequential because schedules carry state.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from behavioral_md.timing import make_timing_model


@dataclass
class ChamberConfig:
    dt: float = 0.1
    damping: float = 10.0
    # Restoring force toward baseline (a spring): without it a constant drive
    # ramps activation to the clip and pressing saturates. With it, equilibrium
    # activation ~ drive/restoring, so response rate is GRADED by drive.
    restoring: float = 1.0
    # Press activation is a leaky integrator of the drive (the overdamped limit of
    # damped Verlet): act <- (1-1/tau)*act + (1/tau)*drive. Fast enough to track a
    # within-interval timing ramp without the overshoot a 2nd-order oscillator
    # produces (which spikes responding right after reinforcement instead of pausing).
    act_tau: float = 3.0           # leaky-integrator time constant (steps)
    temperature: float = 0.5      # softmax over press/no-press
    emission_bias: float = 0.0    # response threshold: p_press = logistic((act - bias)/temp)
    approach_gain: float = 1.0
    learning_rate: float = 0.05
    value_extinction: float = 0.02   # unreinforced press erodes press value toward 0
    reinf_asymptote: float = 1.0
    # Energy budget (drives post-reinforcement pausing + acceleration).
    energy_init: float = 0.6
    energy_capacity: float = 1.0
    basal_metabolism: float = 0.004
    press_cost: float = 0.002     # response effort
    food_energy: float = 0.25     # energy per reinforcer
    deficit_exponent: float = 2.0
    motiv_strength: float = 1.0
    # Pluggable interval-timing model (see timing.py): none|homeostatic|set|bet|let.
    timing_model: str = "none"
    timing_clock: str = "time"     # "time" (tick/step -> FI) | "count" (tick/press -> FR)
    timing_gain: float = 2.0       # scales the timing contribution into the press drive
    timing_lr: float = 0.1         # timing-model learning rate
    timing_states: int = 25        # number of behavioral states (BeT/LeT)
    timing_init: float = 20.0      # SET initial expected time (steps)
    timing_seed: int = 0
    set_threshold: float = 0.7     # SET comparator: respond when accum/t_ref > thr
    set_width: float = 0.1
    bet_pace: float = 0.6          # BeT P(state advance) per step
    let_flow: float = 0.5          # LeT fraction of activation flowing forward/step
    # Pavlovian component->reinforcer association (behavioral-momentum "mass").
    # In a multiple schedule each component stimulus accrues a context value via
    # an omission-RW rule: it grows toward ctx_asymptote on reinforced steps and
    # decays toward 0 on non-reinforced steps WHILE THE COMPONENT IS PRESENT, so
    # its steady state is graded by that component's reinforcement RATE (~ p*lambda)
    # -- NOT by responding. It feeds the press drive additively, so a rich
    # component sustains responding under disruption (resistance to change). The
    # omission rate is kept well below the acquisition rate so the association is
    # slow to erode -- the source of momentum / persistence.
    ctx_learning_rate: float = 0.05    # context value growth on reinforced steps
    ctx_omission_rate: float = 0.002   # context value decay per non-reinforced step
    ctx_asymptote: float = 2.0         # lambda for the context->reinforcer value
    ctx_drive_gain: float = 0.5        # how much context value adds to the press drive
    # Behavioral momentum = MASS = resistance to CHANGE (the project's core Verlet
    # metaphor). The context->reinforcer association confers inertia: it divides
    # the rate at which the response value extinguishes, so mass_c = 1 +
    # momentum_mass_gain * ctx_c. With gain 0 the context value only adds to the
    # drive (no inertia) -- which gives momentum under satiation but REVERSES under
    # extinction. With gain > 0 reinforcement rate slows value erosion, so the rich
    # component resists extinction too (Nevin & Grace). Opt-in (default 0) so the
    # mechanism can be shown to be necessary rather than assumed.
    momentum_mass_gain: float = 0.0


def run_chamber(schedule: str, param: float, cfg: ChamberConfig,
                n_org: int, n_steps: int, seed: int = 0):
    """Run an operant chamber; return per-step records for the population.

    ``schedule`` in {FR, VR, FI, VI}; ``param`` is n (ratio) or t (interval steps).
    Returns a dict with arrays: presses[T,O], reinforced[T,O],
    time_since_reinf[T,O] (steps since last reinforcer at each step).
    """
    rng = np.random.default_rng(seed)
    timing = make_timing_model(cfg.timing_model, n_org, cfg)
    w = np.zeros(n_org)                       # learned value of pressing
    act = np.zeros(n_org)
    energy = np.full(n_org, cfg.energy_init)
    count = np.zeros(n_org)                   # presses since last reinf (FR/VR)
    timer = np.zeros(n_org)                   # steps since last reinf (FI)
    armed = np.zeros(n_org, bool)             # VI
    tsr = np.zeros(n_org)                     # time since reinforcement

    presses = np.zeros((n_steps, n_org))
    reinforced_rec = np.zeros((n_steps, n_org))
    tsr_rec = np.zeros((n_steps, n_org))

    ones = np.ones(n_org)
    prev_press = np.zeros(n_org)              # for the count clock (tick per response)
    for t in range(n_steps):
        # Advance the timing state: per step (time clock) or per response (count clock,
        # using the previous step's press so this step's contribution reflects the count).
        timing.tick(ones if cfg.timing_clock == "time" else prev_press)
        deficit = np.clip(1.0 - energy / cfg.energy_capacity, 0.0, None) ** cfg.deficit_exponent
        # Press drive = learned value + energy-deficit motivation + timing signal.
        drive = cfg.approach_gain * (
            w + cfg.motiv_strength * deficit + cfg.timing_gain * timing.contribution()
        )
        # Leaky integrator (overdamped limit): tracks the drive with time constant
        # act_tau, fast enough to follow a within-interval timing ramp.
        alpha = 1.0 / cfg.act_tau
        act = np.clip((1.0 - alpha) * act + alpha * drive, -10.0, 10.0)

        # Logistic emission with a response threshold (bias): low drive -> a pause.
        p_press = 1.0 / (1.0 + np.exp(-(act - cfg.emission_bias) / cfg.temperature))
        press = rng.random(n_org) < p_press
        prev_press = press.astype(float)

        # Schedule: which presses are reinforced. count = presses since last
        # reinforcer (tracked for all schedules, for within-ratio analysis).
        count += press
        if schedule == "FR":
            reinforced = press & (count >= param)
        elif schedule == "VR":
            reinforced = press & (rng.random(n_org) < 1.0 / param)
        elif schedule == "FI":
            timer += 1.0
            reinforced = press & (timer >= param)
            timer = np.where(reinforced, 0.0, timer)
        elif schedule == "VI":
            armed |= rng.random(n_org) < 1.0 / param
            reinforced = press & armed
            armed = armed & (~reinforced)
        else:
            raise ValueError(f"unknown schedule {schedule}")

        # Energy bookkeeping: pressing costs effort (response cost), food restores.
        energy = energy - cfg.basal_metabolism - press * cfg.press_cost
        energy = np.clip(energy + reinforced * cfg.food_energy, 0.0, cfg.energy_capacity)
        # Press value: reinforced press strengthens it; UNREINFORCED press erodes it
        # (extinction). On ratio schedules every press has the same reinforcement
        # probability so the value stays high; on interval schedules, faster
        # pressing makes more presses unreinforced and erodes it -> lower rate.
        # Unit price (effort/food) and the demand curve emerge; not computed.
        unreinf_press = press & (~reinforced)
        w = np.where(reinforced, w + cfg.learning_rate * (cfg.reinf_asymptote - w), w)
        w = np.where(unreinf_press, w - cfg.value_extinction * w, w)

        # Timing model learns when reinforcement occurred and resets its marker.
        timing.update(reinforced)
        # Post-reinforcement: the organism pauses to consume -> press activation
        # resets, so a new interval starts from a low rate (clean scallop / PRP).
        act = np.where(reinforced, 0.0, act)
        count = np.where(reinforced, 0.0, count)   # reset count-since-reinforcer

        tsr_rec[t] = tsr
        presses[t] = press
        reinforced_rec[t] = reinforced
        tsr = np.where(reinforced, 0.0, tsr + 1.0)

    return {"presses": presses, "reinforced": reinforced_rec, "time_since_reinf": tsr_rec}


def run_concurrent_chamber(vi_params, cfg: ChamberConfig, n_org: int, n_steps: int,
                           seed: int = 0, efforts=None):
    """Concurrent chamber: M mutually exclusive responses, each on its own VI.

    At each step the organism emits exactly ONE response (softmax over response
    activations -- behavior is choice). Each response's reinforcer arms on its own
    VI timer (mean interval = vi_params[r]) and is collected only by emitting that
    response while armed. This makes a single response's rate sensitive to the
    reinforcement available for OTHER behavior (Herrnstein's R_e), instead of
    saturating. ``vi_params`` is a length-M list of mean intervals (one per
    response; treat response 0 as the "lever", the rest as background behavior).

    ``efforts`` is a length-M per-response ENERGY cost (defaults to cfg.press_cost
    for every response). Only the EMITTED response's effort is charged, so
    responding on a higher-effort response costs more energy -- a real cost-benefit
    that, with a lower-effort alternative, makes "jamming the lever" non-free.

    Returns emission counts [O, M] and reinforcer counts [O, M] (summed over the
    second half of the run, i.e. steady state).
    """
    rng = np.random.default_rng(seed)
    m = len(vi_params)
    vi = np.asarray(vi_params, float)
    effort = (np.full(m, cfg.press_cost) if efforts is None else np.asarray(efforts, float))
    w = np.zeros((n_org, m))
    act = np.zeros((n_org, m))
    prev = np.zeros((n_org, m))
    energy = np.full(n_org, cfg.energy_init)
    armed = np.zeros((n_org, m), bool)
    warm = n_steps // 2
    emit_count = np.zeros((n_org, m))
    reinf_count = np.zeros((n_org, m))

    for t in range(n_steps):
        deficit = np.clip(1.0 - energy / cfg.energy_capacity, 0.0, None) ** cfg.deficit_exponent
        drive = cfg.approach_gain * (w + cfg.motiv_strength * deficit[:, None])
        vel = (act - prev) / cfg.dt
        net = drive - cfg.damping * vel - cfg.restoring * act
        new = np.clip(2 * act - prev + net * cfg.dt**2, -10.0, 10.0)
        prev, act = act, new

        # Emit exactly one response (softmax / matching over response activations).
        z = act / cfg.temperature
        z = z - z.max(axis=1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(axis=1, keepdims=True)
        cum = np.cumsum(p, axis=1)
        draw = rng.random((n_org, 1))
        emitted = (draw < cum).argmax(axis=1)               # [O] chosen response
        onehot = np.zeros((n_org, m), bool)
        onehot[np.arange(n_org), emitted] = True

        # VI arming per response (time-based); collect only the emitted+armed one.
        armed |= rng.random((n_org, m)) < (1.0 / vi)[None, :]
        reinforced = onehot & armed
        armed = armed & (~reinforced)

        # Charge the EMITTED response's effort (response-contingent), plus basal.
        step_effort = effort[emitted]
        energy = energy - cfg.basal_metabolism - step_effort
        energy = np.clip(energy + reinforced.any(axis=1) * cfg.food_energy,
                         0.0, cfg.energy_capacity)
        w = np.where(reinforced, w + cfg.learning_rate * (cfg.reinf_asymptote - w), w)

        if t >= warm:
            emit_count += onehot
            reinf_count += reinforced

    return {"emit": emit_count, "reinforced": reinf_count, "steps": n_steps - warm}


def run_multiple_schedule(vi_params, cfg: ChamberConfig, n_org: int,
                          comp_steps: int, n_baseline: int, n_disruption: int,
                          disruptor: str = "extinction", seed: int = 0):
    """Multiple schedule for behavioral momentum (Nevin & Grace).

    K components are presented successively, each ``comp_steps`` steps, one after
    another within a session; ``vi_params[k]`` is component k's mean VI interval
    (small = rich, large = lean). The single press response is shared, but each
    component stimulus carries its own Pavlovian context value (cfg.ctx_*), which
    settles at a level graded by that component's reinforcement RATE and feeds the
    press drive. After ``n_baseline`` sessions a disruptor is applied for
    ``n_disruption`` sessions:

      'extinction'  -- reinforcement withheld in every component.
      'satiation'   -- energy clamped to capacity (deficit ~ 0), reinforcement
                       still delivered (a prefeeding / motivational disruptor).

    Resistance to change is the per-component press rate during disruption relative
    to its own baseline. Momentum predicts the RICH component is more resistant
    (its rate falls more slowly in proportion). Returns per-session, per-component
    press rate, reinforcement rate, and end context value. Arrays [n_sessions, K].
    """
    rng = np.random.default_rng(seed)
    vi = np.asarray(vi_params, float)
    k = len(vi)
    n_sessions = n_baseline + n_disruption

    v = np.zeros((n_org, k))                 # per-component press value (response->reinf)
    ctx = np.zeros((n_org, k))               # per-component context value (Pavlovian mass)
    energy = np.full(n_org, cfg.energy_init)
    alpha = 1.0 / cfg.act_tau

    press_rate = np.zeros((n_sessions, k))   # mean over organisms + component steps
    reinf_rate = np.zeros((n_sessions, k))
    ctx_rec = np.zeros((n_sessions, k))

    for s in range(n_sessions):
        disrupt = s >= n_baseline
        for c in range(k):
            act = np.zeros(n_org)            # fresh activation entering a component
            armed = np.zeros(n_org, bool)
            presses = np.zeros(n_org)
            reinfs = np.zeros(n_org)
            for _ in range(comp_steps):
                if disrupt and disruptor == "satiation":
                    energy = np.full(n_org, cfg.energy_capacity)  # prefed / sated
                elif disrupt and disruptor == "extinction":
                    # Maintain deprivation: extinction withholds the RESPONSE->food
                    # contingency, not food-as-survival. Holding energy constant
                    # keeps motivation fixed so we measure response extinction, not
                    # a starvation-driven motivation surge (which would mask it).
                    energy = np.full(n_org, cfg.energy_init)
                deficit = np.clip(1.0 - energy / cfg.energy_capacity, 0.0, None)
                deficit = deficit**cfg.deficit_exponent
                drive = cfg.approach_gain * (
                    v[:, c] + cfg.ctx_drive_gain * ctx[:, c] + cfg.motiv_strength * deficit
                )
                act = np.clip((1.0 - alpha) * act + alpha * drive, -10.0, 10.0)
                p_press = 1.0 / (1.0 + np.exp(-(act - cfg.emission_bias) / cfg.temperature))
                press = rng.random(n_org) < p_press

                armed |= rng.random(n_org) < (1.0 / vi[c])
                withhold = disrupt and disruptor == "extinction"
                reinforced = press & armed & (not withhold)
                armed = armed & (~reinforced)

                energy = energy - cfg.basal_metabolism - press * cfg.press_cost
                energy = np.clip(energy + reinforced * cfg.food_energy, 0.0, cfg.energy_capacity)

                # MASS: the context->reinforcer association confers inertia, dividing
                # the rate at which the response value changes (resistance to change).
                # Higher in the rich component -> slower decay -> resistance.
                mass = 1.0 + cfg.momentum_mass_gain * ctx[:, c]
                # Per-component response value. Reinforced -> grow toward asymptote.
                # Otherwise decay toward 0 at value_extinction/mass PER STEP (time-
                # based, NOT per press): Nevin's mass resists change per unit time,
                # so a vigorous response is not penalised by self-extinction. The
                # baseline equilibrium is graded by reinforcement RATE (rich higher).
                v[:, c] = np.where(
                    reinforced, v[:, c] + cfg.learning_rate * (cfg.reinf_asymptote - v[:, c]),
                    v[:, c] - (cfg.value_extinction / mass) * v[:, c],
                )
                # Pavlovian context value: omission-RW while the component is present
                # (grows on reinforcement, slow decay otherwise) -> rate-graded mass.
                ctx[:, c] = np.where(
                    reinforced,
                    ctx[:, c] + cfg.ctx_learning_rate * (cfg.ctx_asymptote - ctx[:, c]),
                    ctx[:, c] - cfg.ctx_omission_rate * ctx[:, c],
                )
                act = np.where(reinforced, 0.0, act)  # consume pause

                presses += press
                reinfs += reinforced
            press_rate[s, c] = (presses / comp_steps).mean()
            reinf_rate[s, c] = (reinfs / comp_steps).mean()
            ctx_rec[s, c] = ctx[:, c].mean()

    return {"press_rate": press_rate, "reinf_rate": reinf_rate, "ctx": ctx_rec,
            "n_baseline": n_baseline, "n_disruption": n_disruption, "vi": vi}

