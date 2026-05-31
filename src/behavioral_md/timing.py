"""Pluggable interval-timing models for the operant chamber.

Each model turns "time since the last reinforcer" into a contribution to the
press drive, and learns (from reinforcement) when reinforcement tends to occur, so
that responding becomes timed (the FI scallop, FR/VR pause). They are selected by
``ChamberConfig.timing_model`` and all share one interface, vectorized over
organisms (NumPy, length-O arrays):

    tick()                 advance the internal state one step
    contribution() -> [O]  the timing signal added to the press drive (~[0, 1])
    update(reinforced)     learn from this step's reinforcement, then reset the
                           marker (timer/state) for reinforced organisms

The four models span the theoretical landscape (see the lab notebook):

  "none"  Homeostatic -- NO timer. Contribution is 0; timing, if any, comes from
          the chamber's energy-deficit dynamics. Maximally first-principles, but
          cannot distinguish fixed vs variable requirements.
  "set"   Scalar Expectancy Theory (Gibbon/Church) -- pacemaker-accumulator clock
          + reference memory (expected time) + comparator. Fits data well but is
          explicitly cognitivist (an internal represented-and-read time). Included
          for comparison; NOT first-principles.
  "bet"   Behavioral Theory of Timing (Killeen & Fetterman) -- a stochastic
          pacemaker drives transitions through behavioral STATES; the operant
          comes under control of the state usually present at reinforcement.
          Timing grounded in behavior; mild internal-pacemaker flavor.
  "let"   Learning to Time (Machado) -- a serial chain of behavioral states whose
          activation flows after the marker, with LEARNED state->response links
          (RW). No comparator/stored duration. Most behavioral / least mentalist;
          maps onto the two-tier atom architecture.
"""

from __future__ import annotations

import numpy as np


class TimingModel:
    # ``advance`` is a per-organism [O] amount to advance the internal state this
    # tick: all-ones for a TIME clock (one unit per step), or the press mask for a
    # COUNT clock (one unit per response). This is what lets the SAME models time
    # intervals (FI) or counts (FR) -- the difference is only what drives the tick.
    def tick(self, advance: np.ndarray) -> None: ...
    def contribution(self) -> np.ndarray: ...
    def update(self, reinforced: np.ndarray) -> None: ...


class HomeostaticTiming(TimingModel):
    """No explicit timer (timing, if any, is carried by the energy budget)."""

    def __init__(self, n_org, cfg):
        self._zero = np.zeros(n_org)

    def tick(self, advance):
        pass

    def contribution(self):
        return self._zero

    def update(self, reinforced):
        pass


class SETTiming(TimingModel):
    """Scalar Expectancy Theory: accumulator clock + reference memory + comparator."""

    def __init__(self, n_org, cfg):
        self.accum = np.zeros(n_org)                       # pulses since marker
        self.t_ref = np.full(n_org, cfg.timing_init)       # learned expected time
        self.lr = cfg.timing_lr
        self.thr = cfg.set_threshold                       # respond when accum/t_ref > thr
        self.width = cfg.set_width

    def tick(self, advance):
        self.accum += advance                              # pulses (per step) or counts (per press)

    def contribution(self):
        # Comparator: responding ramps on as the clock nears the expected time.
        return 1.0 / (1.0 + np.exp(-(self.accum / self.t_ref - self.thr) / self.width))

    def update(self, reinforced):
        # Reference memory moves toward the time the reinforcer actually occurred.
        self.t_ref = np.where(reinforced,
                              self.t_ref + self.lr * (self.accum - self.t_ref), self.t_ref)
        self.accum = np.where(reinforced, 0.0, self.accum)


class BeTTiming(TimingModel):
    """Behavioral Theory of Timing: stochastic transitions through behavioral states."""

    def __init__(self, n_org, cfg):
        self.k = cfg.timing_states
        self.state = np.zeros(n_org, dtype=int)            # current behavioral state
        self.pace = cfg.bet_pace                            # P(advance) per step (pacemaker)
        self.v = np.zeros((n_org, self.k))                 # learned value of each state
        self.lr = cfg.timing_lr
        self.rng = np.random.default_rng(cfg.timing_seed)
        self._idx = np.arange(n_org)

    def tick(self, advance):
        step = (self.rng.random(self.state.shape) < self.pace) & (advance > 0)
        self.state = np.minimum(self.state + step, self.k - 1)

    def contribution(self):
        return self.v[self._idx, self.state]

    def update(self, reinforced):
        cur = self.v[self._idx, self.state]
        self.v[self._idx, self.state] = np.where(reinforced, cur + self.lr * (1.0 - cur), cur)
        self.state = np.where(reinforced, 0, self.state)


class LeTTiming(TimingModel):
    """Learning to Time: serial state activation flows after the marker;
    learned state->response links (RW). Most behavioral / least mentalist."""

    def __init__(self, n_org, cfg):
        self.k = cfg.timing_states
        self.a = np.zeros((n_org, self.k))                 # state activations (a bump)
        self.a[:, 0] = 1.0
        self.flow = cfg.let_flow                            # fraction flowing forward per step
        self.w = np.zeros((n_org, self.k))                 # learned state->response links
        self.lr = cfg.timing_lr

    def tick(self, advance):
        # Activation flows forward along the chain (a traveling bump), advanced for
        # organisms whose clock ticked this step (time: all; count: those that pressed).
        move = self.flow * self.a * advance[:, None]
        self.a = self.a - move
        self.a[:, 1:] += move[:, :-1]                      # last state's flow is absorbed

    def contribution(self):
        return np.sum(self.w * self.a, axis=1)

    def update(self, reinforced):
        # RW on the active states (states active at reinforcement gain link strength).
        # Per-state error so links climb to ~1 at the reinforced state (not damped
        # by a shared summed prediction, which kept the signal tiny).
        dw = self.lr * self.a * (1.0 - self.w)
        self.w = np.where(reinforced[:, None], self.w + dw, self.w)
        reset = reinforced[:, None]
        new_a = np.zeros_like(self.a)
        new_a[:, 0] = 1.0
        self.a = np.where(reset, new_a, self.a)


def make_timing_model(name: str, n_org: int, cfg) -> TimingModel:
    return {
        "none": HomeostaticTiming, "homeostatic": HomeostaticTiming,
        "set": SETTiming, "bet": BeTTiming, "let": LeTTiming,
    }[name](n_org, cfg)
