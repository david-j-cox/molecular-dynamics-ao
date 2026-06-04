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
    # --- Pearce-Hall associability (PREE) ---------------------------------------
    # 'fixed' = constant acquisition/extinction rates (the Rescorla-Wagner default).
    # 'pearce_hall' = the effective rate is scaled by a per-response associability
    # alpha that EMAs toward the recent ABSOLUTE prediction error |PE|, so a
    # surprising outcome learns fast and a well-predicted one learns slowly. After
    # PARTIAL reinforcement an omission is already partly expected (|PE| small), so
    # alpha is low and extinction is slow; after CONTINUOUS reinforcement the first
    # omission is maximally surprising (|PE| large), so alpha spikes and extinction
    # is fast. The partial-reinforcement extinction effect (PREE) thus emerges.
    associability_rule: str = "fixed"     # 'fixed' | 'pearce_hall'
    ph_eta: float = 0.3                   # EMA rate of alpha toward |PE|
    ph_init: float = 0.5                  # initial associability
    ph_floor: float = 0.05                # associability floor (never fully stops learning)
    # --- Dual excitatory/inhibitory value rule (resurgence) ---------------------
    # 'single' = one response value updated RW-style (extinction erodes it).
    # 'dual' = a preserved excitation w+ and a separate inhibition w- (Konorski/
    # Bouton); the value the drive reads is net = w+ - w-. Omission grows w- and
    # leaves w+ intact, so an extinguished response keeps a LATENT excitatory
    # strength -- which re-emerges (resurges) when a competing response is removed.
    # Mirrors learning.DualExcitatoryInhibitory, vectorized for the chamber.
    value_rule: str = "single"            # 'single' | 'dual' | 'rac'
    inhib_rate: float = 0.05              # w- growth toward asymptote on omission
    inhib_relax: float = 0.1              # w- relax toward 0 on reinforcement
    inhib_passive_decay: float = 0.0      # per-step multiplicative w- decay (recovery)
    # --- Resurgence as Choice (Shahan & Craig, 2017): value_rule='rac' ----------
    # A molar choice account, NOT a local response-strength account. Each response's
    # value is a TEMPORALLY-WEIGHTED (leaky-integrated) tally of the reinforcers it has
    # produced: v <- (1 - 1/rac_tau)*v + rac_bump*reinforced. Allocation MATCHES relative
    # value (power law with exponent rac_sensitivity), over R1, R2, and a fixed
    # extraneous source (r_other). Resurgence needs no preserved/protected target
    # strength: when the alternative's reinforcement stops, its integrated value DECAYS,
    # so the target's RELATIVE value (v_T / sum v) recovers -- a pure choice effect whose
    # size depends on the time scale rac_tau relative to the phase durations.
    rac_tau: float = 2000.0              # temporal-weighting time constant (steps)
    rac_bump: float = 0.01               # value increment per reinforcer
    rac_floor: float = 0.1               # baseline value of an explicit response (exploration)
    rac_sensitivity: float = 1.0         # matching-law exponent on relative value
    # --- Punishment of concurrent choice (run_punishment_choice) ----------------
    # Each response is on its own reinforcement VI and (independently) punishment VI.
    # Reinforcement and punishment each train a leaky-integrated value (vr, vp) with
    # the same temporal weighting. The three accounts of how punishment maps to choice:
    #   'subtractive'  (de Villiers, 1980): score = vr - pun_c*vp; punishment directly
    #                  cancels a response's own reinforcement value (pun_c = reinforcers
    #                  cancelled per punisher). Matched with exponent pun_sensitivity.
    #   'competitive'  (Deluty, 1976): punishment STRENGTHENS competitors --
    #                  score_i = vr_i + pun_c*sum_{j!=i} vp_j; the punished response is
    #                  suppressed only relatively (its alternatives gain value).
    #   'concatenated' (Critchfield/Klapes): power-law matching with SEPARATE
    #                  sensitivities -- B_i proportional to vr_i^a_r * vp_i^(-a_p) --
    #                  so log(B1/B2) = a_r*log(vr1/vr2) - a_p*log(vp1/vp2) + bias.
    pun_tau: float = 800.0               # temporal weighting of vr and vp (steps)
    pun_bump: float = 0.04               # value increment per reinforcer / punisher
    pun_floor: float = 0.1               # baseline reinforcement value (exploration)
    pun_c: float = 1.0                   # de Villiers c / Deluty competition weight
    pun_sensitivity: float = 1.0         # matching exponent (subtractive/competitive)
    pun_a_r: float = 1.0                 # concatenated: reinforcement sensitivity
    pun_a_p: float = 1.0                 # concatenated: punishment sensitivity


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
    value_rec = np.zeros((n_sessions, k))    # end-of-session response value (mass acts here)

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
            value_rec[s, c] = v[:, c].mean()

    return {"press_rate": press_rate, "reinf_rate": reinf_rate, "ctx": ctx_rec,
            "value": value_rec, "n_baseline": n_baseline,
            "n_disruption": n_disruption, "vi": vi}


def run_pree(reinf_prob: float, cfg: ChamberConfig, n_org: int, n_train: int,
             n_ext: int, sess_steps: int, seed: int = 0):
    """Single response: acquire under reinforcement probability ``reinf_prob`` then
    extinguish. ``reinf_prob == 1.0`` is continuous reinforcement (CRF); a smaller
    value is partial reinforcement (PRF). Used to demonstrate the partial-
    reinforcement extinction effect (PREE) with Pearce-Hall associability.

    Energy is decoupled (no metabolic budget): responding is driven purely by the
    learned press value, so we measure RESPONSE extinction rather than a starvation-
    driven motivation surge. The value update is acquisition-on-reinforcement (press-
    contingent, toward lambda) plus a TIME-BASED decay every step (toward 0) scaled
    by associability. Time-based (not press-contingent) decay is what makes the
    'fixed' control clean: with a constant associability the value's extinction rate
    constant is identical for CRF and PRF (no PREE) regardless of their different
    response rates -- so any PREE under 'pearce_hall' is attributable to
    associability alone. Under Pearce-Hall, after PRF an omission is less surprising
    (|PE| small -> low alpha), so the decay constant is smaller -> slower extinction
    -> PREE; after CRF the first omission is maximally surprising (alpha spikes).

    Returns per-session arrays (length n_train+n_ext): mean press ``rate``, mean
    press ``value``, mean ``assoc`` (associability), and ``n_train``. The PREE
    readout is the extinction-phase decay constant of ``value`` (computed by the
    experiment), which is baseline-free.
    """
    rng = np.random.default_rng(seed)
    ph = cfg.associability_rule == "pearce_hall"
    w = np.zeros(n_org)                        # learned value of pressing
    alpha = np.full(n_org, cfg.ph_init)        # Pearce-Hall associability
    act = np.zeros(n_org)
    a_int = 1.0 / cfg.act_tau
    n_sess = n_train + n_ext

    rate = np.zeros(n_sess)
    value = np.zeros(n_sess)
    assoc = np.zeros(n_sess)
    for s in range(n_sess):
        train = s < n_train
        presses = np.zeros(n_org)
        for _ in range(sess_steps):
            drive = cfg.approach_gain * w
            act = np.clip((1.0 - a_int) * act + a_int * drive, -10.0, 10.0)
            p_press = 1.0 / (1.0 + np.exp(-(act - cfg.emission_bias) / cfg.temperature))
            press = rng.random(n_org) < p_press
            reinforced = press & train & (rng.random(n_org) < reinf_prob)

            a_old = alpha if ph else 1.0
            # Acquisition: reinforced press moves value toward lambda (press-
            # contingent). Decay: value forgets toward 0 every step (time-based),
            # so under 'fixed' the extinction rate constant is response-rate- and
            # baseline-independent -- a clean control for PREE.
            w = np.where(reinforced, w + cfg.learning_rate * a_old * (cfg.reinf_asymptote - w), w)
            w = w - cfg.value_extinction * a_old * w
            # Associability tracks |prediction error| sampled when the organism
            # presses (it experiences the outcome): reinforced -> PE = lambda - w;
            # unreinforced press -> PE = 0 - w.
            if ph:
                target = np.where(reinforced, cfg.reinf_asymptote, 0.0)
                pe = np.abs(target - w)
                alpha = np.where(
                    press,
                    np.clip((1.0 - cfg.ph_eta) * alpha + cfg.ph_eta * pe, cfg.ph_floor, 1.0),
                    alpha,
                )
            presses += press
        rate[s] = (presses / sess_steps).mean()
        value[s] = w.mean()
        assoc[s] = alpha.mean()

    return {"rate": rate, "value": value, "assoc": assoc, "n_train": n_train}


def run_resurgence(cfg: ChamberConfig, n_org: int, phase_steps: int, seed: int = 0,
                   vi_r1: float = 5.0, vi_r2: float = 5.0, r_other: float = 0.15,
                   block: int = 50, control_reinforce_r2: bool = False):
    """Three-phase concurrent resurgence (R1 = target, R2 = alternative).

    Phase 1 (train):  R1 reinforced on VI ``vi_r1``; R2 never.
    Phase 2 (alt):    R1 on extinction; R2 reinforced on VI ``vi_r2``.
    Phase 3 (test):   both on extinction (or, with ``control_reinforce_r2=True``,
                      R2 stays reinforced -- the control that should ABOLISH
                      resurgence, isolating removal-of-alternative-reinforcement
                      as the cause).

    Each step the organism emits exactly ONE option -- R1, R2, or a background
    "other" behavior with a fixed extraneous value ``r_other`` (Herrnstein's R_e) --
    by a softmax (matching/choice) over the three option activations. Energy is
    decoupled, so "extinction" withholds only the response->reinforcer contingency.

    Resurgence (R1 responding RECOVERING in phase 3 from its phase-2 suppressed
    level) is NOT computed anywhere; it emerges from choice reallocation -- when R2's
    reinforcement is removed, allocation flows back toward R1. The procedure is
    symmetric (R2 is trained in phase 2 as R1 was in phase 1), so R1 and R2 converge
    to parity at test; resurgence is the RISE of R1, not R1 exceeding R2. The value
    rule sets how much latent R1 strength survives phase 2: ``value_rule='single'``
    with ``momentum_mass_gain=0`` erodes R1 toward the background floor (bare choice);
    ``momentum_mass_gain>0`` slows R1's phase-2 decay (mass persists from training);
    ``value_rule='dual'`` preserves R1's excitation (w+) so it stays less suppressed.

    Returns per-block R1/R2 response rate (fraction of organisms), phase boundaries
    (in blocks), and the block size.
    """
    rng = np.random.default_rng(seed)
    dual = cfg.value_rule == "dual"
    rac = cfg.value_rule == "rac"
    ph = cfg.associability_rule == "pearce_hall"
    a_int = 1.0 / cfg.act_tau
    lam = cfg.reinf_asymptote
    lo, hi = -10.0, 10.0

    w = np.zeros((n_org, 2))                   # single-value rule
    wp = np.zeros((n_org, 2))                  # dual: excitation
    wm = np.zeros((n_org, 2))                  # dual: inhibition
    vr = np.zeros((n_org, 2))                  # RaC: temporally-weighted reinforcement value
    alpha = np.full((n_org, 2), cfg.ph_init)   # Pearce-Hall associability
    mtr = np.zeros((n_org, 2))                 # reinforcement-history trace -> mass
    act = np.zeros((n_org, 2))
    armed = np.zeros((n_org, 2), bool)
    # The mass trace grows with delivered reinforcement and decays SLOWLY, so the
    # training-history "mass" persists across the (long) extinction phase rather than
    # washing out within it -- the durable resistance-to-change Nevin's momentum
    # requires (cf. the slow-decaying Pavlovian context value in run_multiple_schedule).
    mass_grow = 0.05
    mass_decay = 0.0003

    # Phase schedule: which response is reinforced (-1 = none) and on what VI.
    phase3 = (1, vi_r2) if control_reinforce_r2 else (-1, 0.0)
    phases = [(0, vi_r1), (1, vi_r2), phase3]
    n_steps = phase_steps * len(phases)
    r1 = np.zeros(n_steps)
    r2 = np.zeros(n_steps)

    other_z = cfg.approach_gain * r_other / cfg.temperature
    t = 0
    for reinf_target, vi in phases:
        for _ in range(phase_steps):
            if rac:
                # Molar choice: allocation MATCHES relative temporally-weighted value
                # (power law) over [R1, R2, extraneous]. No leaky-integrator activation.
                base = np.concatenate([vr + cfg.rac_floor,
                                       np.full((n_org, 1), r_other)], axis=1)
                pw = base ** cfg.rac_sensitivity
                p = pw / pw.sum(axis=1, keepdims=True)
            else:
                net = (wp - wm) if dual else w
                net = np.clip(net, lo, hi)
                act = np.clip((1.0 - a_int) * act + a_int * cfg.approach_gain * net, lo, hi)
                # Softmax (local matching) over [R1, R2, other]; emit one option.
                z = np.concatenate([act / cfg.temperature,
                                    np.full((n_org, 1), other_z)], axis=1)
                z = z - z.max(axis=1, keepdims=True)
                p = np.exp(z)
                p /= p.sum(axis=1, keepdims=True)
            emitted = (rng.random((n_org, 1)) < np.cumsum(p, axis=1)).argmax(axis=1)
            emit01 = np.zeros((n_org, 2), bool)
            hit = emitted < 2
            emit01[np.arange(n_org)[hit], emitted[hit]] = True

            # Reinforcement: only the phase's target response, when emitted + armed.
            reinf01 = np.zeros((n_org, 2), bool)
            if reinf_target >= 0:
                armed[:, reinf_target] |= rng.random(n_org) < (1.0 / vi)
                got = emit01[:, reinf_target] & armed[:, reinf_target]
                reinf01[:, reinf_target] = got
                armed[:, reinf_target] &= ~got
            omit01 = emit01 & ~reinf01

            mtr = mtr + mass_grow * reinf01 * (1.0 - mtr) - mass_decay * mtr
            mass = 1.0 + cfg.momentum_mass_gain * mtr
            a_old = alpha if ph else np.ones((n_org, 2))

            if rac:
                # Temporally-weighted reinforcement value: decay every step, increment
                # on reinforcement. Resurgence falls out of the matching above as the
                # alternative's value decays in phase 3 (no preserved target strength).
                vr = (1.0 - 1.0 / cfg.rac_tau) * vr + cfg.rac_bump * reinf01
                pre = vr
            elif dual:
                if cfg.inhib_passive_decay > 0.0:
                    wm *= (1.0 - cfg.inhib_passive_decay)
                wp += reinf01 * (cfg.learning_rate * a_old * (lam - wp))
                wm += reinf01 * (cfg.inhib_relax * (0.0 - wm))
                wm += omit01 * (cfg.inhib_rate * a_old * (lam - wm))
                wm = np.clip(wm, 0.0, hi)
                pre = wp - wm
            else:
                pre = w.copy()
                w += reinf01 * (cfg.learning_rate * a_old * (lam - w))
                w += omit01 * (-(cfg.value_extinction * a_old / mass) * w)

            if ph:
                target = np.where(reinf01, lam, 0.0)
                pe = np.abs(target - pre)
                alpha = np.where(emit01,
                                 np.clip((1.0 - cfg.ph_eta) * alpha + cfg.ph_eta * pe,
                                         cfg.ph_floor, 1.0),
                                 alpha)

            r1[t] = emit01[:, 0].mean()
            r2[t] = emit01[:, 1].mean()
            t += 1

    nb = n_steps // block
    r1b = r1[:nb * block].reshape(nb, block).mean(1)
    r2b = r2[:nb * block].reshape(nb, block).mean(1)
    phase_blocks = phase_steps // block
    return {"r1": r1b, "r2": r2b, "phase_blocks": phase_blocks, "block": block,
            "n_phases": len(phases)}


def run_punishment_choice(model: str, cfg: ChamberConfig, n_org: int, n_steps: int,
                          vi_reinf, vi_punish, seed: int = 0, warmup_frac: float = 0.5):
    """Concurrent M-alternative choice with reinforcement AND punishment schedules.

    Each response is on its own reinforcement VI (``vi_reinf[i]``) and, independently,
    its own punishment VI (``vi_punish[i]``; use ``inf`` or ``0`` for no punishment).
    A VI arms Bernoulli (mean interval = VI); the armed reinforcer/punisher is collected
    when that response is emitted. Reinforcement and punishment each train a leaky-
    integrated value (vr, vp, temporal weighting ``pun_tau``). One response is emitted
    per step by matching over a model-specific score:

      'subtractive'  (de Villiers, 1980):  score_i = (vr_i + floor) - pun_c*vp_i
                     -- punishment cancels a response's OWN reinforcement value.
      'competitive'  (Deluty, 1976):       score_i = (vr_i + floor) + pun_c*sum_{j!=i} vp_j
                     -- punishment strengthens COMPETITORS (relative suppression).
      'concatenated' (Critchfield/Klapes): B_i ~ (vr_i+floor)^a_r * (vp_i+floor)^(-a_p)
                     -- separate sensitivities; log(B1/B2)=a_r log(vr1/vr2)-a_p log(vp1/vp2).

    Returns steady-state (second ``1-warmup_frac`` of the run) per-organism counts:
    emit [O,M] (allocation), reinforced [O,M], punished [O,M], and the step count.
    """
    rng = np.random.default_rng(seed)
    m = len(vi_reinf)
    vir = np.asarray(vi_reinf, float)
    vip = np.asarray(vi_punish, float)
    arm_r = np.where(np.isfinite(vir) & (vir > 0), 1.0 / np.where(vir > 0, vir, 1.0), 0.0)
    arm_p = np.where(np.isfinite(vip) & (vip > 0), 1.0 / np.where(vip > 0, vip, 1.0), 0.0)

    vr = np.zeros((n_org, m))
    vp = np.zeros((n_org, m))
    armed_r = np.zeros((n_org, m), bool)
    armed_p = np.zeros((n_org, m), bool)
    decay = 1.0 - 1.0 / cfg.pun_tau
    eps = 1e-9
    warm = int(n_steps * warmup_frac)
    emit_count = np.zeros((n_org, m))
    reinf_count = np.zeros((n_org, m))
    punish_count = np.zeros((n_org, m))

    for t in range(n_steps):
        base_r = vr + cfg.pun_floor
        if model == "subtractive":
            score = np.clip(base_r - cfg.pun_c * vp, eps, None)
            pw = score ** cfg.pun_sensitivity
        elif model == "competitive":
            others_p = vp.sum(axis=1, keepdims=True) - vp
            score = np.clip(base_r + cfg.pun_c * others_p, eps, None)
            pw = score ** cfg.pun_sensitivity
        elif model == "concatenated":
            pw = base_r ** cfg.pun_a_r * (vp + cfg.pun_floor) ** (-cfg.pun_a_p)
        else:
            raise ValueError(f"unknown punishment model {model}")
        p = pw / pw.sum(axis=1, keepdims=True)
        emitted = (rng.random((n_org, 1)) < np.cumsum(p, axis=1)).argmax(axis=1)
        onehot = np.zeros((n_org, m), bool)
        onehot[np.arange(n_org), emitted] = True

        armed_r |= rng.random((n_org, m)) < arm_r[None, :]
        armed_p |= rng.random((n_org, m)) < arm_p[None, :]
        got_r = onehot & armed_r
        got_p = onehot & armed_p
        armed_r &= ~got_r
        armed_p &= ~got_p

        vr = decay * vr + cfg.pun_bump * got_r
        vp = decay * vp + cfg.pun_bump * got_p

        if t >= warm:
            emit_count += onehot
            reinf_count += got_r
            punish_count += got_p

    return {"emit": emit_count, "reinforced": reinf_count, "punished": punish_count,
            "steps": n_steps - warm}


def run_risk_choice(safe_outcomes, risky_outcomes, cfg: ChamberConfig, n_org: int,
                    n_steps: int, seed: int = 0, e_req: float = 0.5,
                    util_width: float = 0.08, cost: float = 0.02, e_init: float = 0.5,
                    n_ebins: int = 12, util_shape: str = "survival"):
    """Energy-budget-rule risk-sensitive choice between a SAFE and a RISKY option.

    Each option is a list of ``(probability, energy_delta)`` outcomes; the two are usually
    matched-mean so the only difference is VARIANCE. Each step the organism chooses by a
    softmax over the EXPECTED SURVIVAL UTILITY of each option AT ITS CURRENT ENERGY E:

        U(E) = logistic((E - e_req) / util_width)   ~  P(survive | energy E)

    U is CONVEX below the requirement ``e_req`` and CONCAVE above it. By Jensen's
    inequality the variable (risky) option's spread is then favored when starving
    (convex -> risk-prone) and disfavored when well-fed (concave -> risk-averse): the
    preference reverses at ``e_req`` -- the energy-budget rule (Caraco, 1980). Nothing
    encodes "be risk-prone when hungry"; it falls out of maximizing a survival-shaped
    utility. Energy drains by ``cost`` each step and is replenished by the chosen outcome;
    an organism is frozen (dead) once E <= 0.

    Returns the risky-choice fraction binned by current energy (``risky_by_energy``
    [n_ebins], ``energy_bins`` centers), the overall risky fraction, and survival.
    """
    rng = np.random.default_rng(seed)
    sp = np.array([o[0] for o in safe_outcomes], float)
    sd = np.array([o[1] for o in safe_outcomes], float)
    rp = np.array([o[0] for o in risky_outcomes], float)
    rd = np.array([o[1] for o in risky_outcomes], float)
    cap = cfg.energy_capacity

    def util(e):
        # 'survival' = a sigmoid in energy ~ P(survive): convex below e_req (risk-prone),
        # concave above (risk-averse). 'linear' = risk-neutral utility (the control: with
        # matched-mean options it yields no energy-budget reversal).
        if util_shape == "linear":
            return e / cap
        return 1.0 / (1.0 + np.exp(-(e - e_req) / util_width))

    def expected_utility(e, probs, deltas):
        # sum_o p_o * U(clip(E + delta_o - cost)); broadcast over organisms.
        post = np.clip(e[:, None] + (deltas - cost)[None, :], 0.0, cap)
        return (probs[None, :] * util(post)).sum(axis=1)

    energy = np.full(n_org, e_init)
    alive = np.ones(n_org, bool)
    risky_count = np.zeros(n_ebins)
    bin_count = np.zeros(n_ebins)
    total_risky = total = 0

    for _ in range(n_steps):
        eu = np.stack([expected_utility(energy, sp, sd),
                       expected_utility(energy, rp, rd)], axis=1)   # [O, 2]
        z = eu / cfg.temperature
        z = z - z.max(axis=1, keepdims=True)
        p_risky = np.exp(z[:, 1]) / np.exp(z).sum(axis=1)
        choose_risky = (rng.random(n_org) < p_risky) & alive

        # Record choice binned by the DECISION-time energy (living organisms only).
        b = np.clip(energy / cap * n_ebins, 0, n_ebins - 1).astype(int)
        np.add.at(bin_count, b[alive], 1.0)
        np.add.at(risky_count, b[alive], choose_risky[alive].astype(float))
        total += int(alive.sum())
        total_risky += int(choose_risky.sum())

        # Sample the chosen option's outcome and update energy (dead organisms frozen).
        out_safe = sd[(rng.random(n_org)[:, None] < np.cumsum(sp)[None, :]).argmax(1)]
        out_risky = rd[(rng.random(n_org)[:, None] < np.cumsum(rp)[None, :]).argmax(1)]
        delta = np.where(choose_risky, out_risky, out_safe) - cost
        energy = np.where(alive, np.clip(energy + delta, 0.0, cap), energy)
        alive = alive & (energy > 0.0)

    bins = [(i + 0.5) / n_ebins * cap for i in range(n_ebins)]
    risky_by_energy = (risky_count / np.maximum(bin_count, 1)).tolist()
    return {"energy_bins": bins, "risky_by_energy": risky_by_energy,
            "bin_count": bin_count.tolist(), "overall_risky": total_risky / max(total, 1),
            "survival": float(alive.mean())}

