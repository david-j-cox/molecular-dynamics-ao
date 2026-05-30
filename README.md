# Behavioral Molecular Dynamics Simulation Engine

A Python framework that models an organism interacting with an environment using
an analogy to **molecular dynamics**, built to stay **objective and
non-mentalistic**: every term reduces to something measurable (energy expended,
distance moved, stimulus intensity, response strength), never a posited internal
state.

The organism is **not** a reinforcement-learning policy. It is a *distributed
dynamical system* of many coupled, response-capable units ("behavioral atoms").
Environmental stimuli act as **force fields**; learning history changes the
forces those stimuli produce; behavior emerges from integrating the coupled
dynamics over time. Survival is governed by an **energy budget** drawn from
behavioral ecology.

---

## Conceptual overview

In molecular dynamics, particles move through space under forces. Here,
behavioral atoms move through *activation space* under history-dependent
stimulus forces, integrated with a damped Verlet update. Behavior (the emitted
action) is read out from the atom activations.

| Engine term | Behavioral meaning (objective) |
|---|---|
| observation | sensory stimulation available to the organism |
| atom activation | response strength / response tendency |
| mass | behavioral inertia, resistance to change |
| force | current effective behavioral pull of the stimulus context given history |
| history weight | learning history (accumulated effect of consequences) |
| action | emitted behavior |
| reward/consequence | **change in energy** (food adds, danger removes); trains learning |
| energy reserve | physical state that governs survival and motivation |

### Two-tier architecture (functional response classes)

Learning lives on **drive atoms** — non-directional, sign-stable *functional
response classes* (`approach_food`, `avoid_danger`, `approach_light`,
`orient_to_cue`), each tagged with the `stimulus` it tracks and a `valence`
(+approach / −avoid). Their learned dispositions are **expressed** through
**movement atoms** (`move_up/down/left/right`) — topographies that carry no
learning of their own. This mirrors the behavior-analytic definition of an
**operant as a functional class defined by its outcome, not its topography**:
"approach food" is the operant; it is expressed through whatever movements the
environment makes available. (`consume` is a consummatory response; `pause` and
`explore` modulate movement.)

---

## Architecture

This engine grew from a hand sketch tying together the governing behavioral
equations, a "multiple-context" hierarchy of response units, and a dynamic
energy budget:

![Original design sketch](docs/architecture/original_sketch.jpg)

The diagrams below capture the **current implementation** in that spirit.

### Per-timestep loop

```mermaid
flowchart LR
  ENV["Environment<br/>stimulus fields, renewable food patch, danger"]
  SENSE["Sense<br/>intensity = exp(-d / range), direction, food_contact"]
  FORCE["Force calculator<br/>F = sensory + history + motivational + coupling - fatigue"]
  INT["Damped Verlet integration<br/>activation update with c * velocity damping"]
  EMIT["Emission (softmax / matching)<br/>P(action) proportional to exp(activation / T)"]
  ENERGY["Energy budget<br/>E += intake - metabolism - movement cost; death if E <= 0"]
  LEARN["Learning rule (Rescorla-Wagner)<br/>dw = lr * elig * intensity * (lambda*mag - w)"]
  ENV -->|observation| SENSE
  SENSE --> FORCE
  FORCE --> INT
  INT --> EMIT
  EMIT -->|action| ENV
  ENV -->|consequence: energy| ENERGY
  ENV -->|appetitive / aversive| LEARN
  ENERGY --> FORCE
  LEARN --> FORCE
```

### Two-tier atom architecture (the "multiple-context" hierarchy)

Stimuli drive sign-stable **drive atoms** that carry the learning history; those
are *expressed* through topographic **movement atoms** via the live stimulus
geometry. Learning lives on the drive atoms, not the movements.

```mermaid
flowchart TB
  subgraph STIM["Stimuli (sensed)"]
    F["food"]
    DG["danger"]
    LT["light"]
    CU["cue"]
  end
  subgraph DRIVE["Drive atoms (functional classes; learned weight w lives here)"]
    AF["approach_food (+)"]
    AD["avoid_danger (-)"]
    AL["approach_light (+)"]
    OC["orient_to_cue (+)"]
  end
  subgraph MOTOR["Movement atoms (topographies; no learning)"]
    MU["move_up"]
    MD["move_down"]
    ML["move_left"]
    MR["move_right"]
  end
  CON["consume<br/>contact-gated consummatory"]
  MOD["pause / explore<br/>modulate via coupling"]
  ACT(["Emitted action (softmax)"])
  STIM --> DRIVE
  DRIVE -->|"express: valence * activation * (stim_dir . move_dir)"| MOTOR
  F -->|contact| CON
  MOD --> MOTOR
  MOTOR --> ACT
  CON --> ACT
```

### Dynamic energy budget

```mermaid
flowchart LR
  INTAKE["food intake (+)"] --> E(["Energy reserve E in 0..1"])
  BAS["basal metabolism (-)"] --> E
  MOV["movement / rest cost (-)"] --> E
  E -->|"deficit = (1 - E/E_cap)^p (convex)"| MOT["motivational force<br/>on approach_food and consume"]
  E -->|"E <= 0"| DEATH(["death: starvation or danger"])
```

### Quantitative laws from the sketch — implementation status

| Law (sketch) | In the engine | Status |
|---|---|---|
| Generalized matching law | softmax emission (Luce rule); concurrent matching needs multiple patches | partial |
| Rescorla-Wagner | `learning.RescorlaWagner` (with omission decay / extinction) | implemented |
| Temporal weighting | eligibility trace (`EligibilityTrace`) | implemented (related) |
| Behavioral momentum | atom `mass`; momentum-modulated extinction | planned |
| Delay/probability discounting | — | planned |
| Demand (exponential) | — | planned |
| Unit price | — | planned |
| Dynamic energy budget | `config` energy terms + `organism` bookkeeping | implemented |

---

## Governing equations

Notation: atom `i` has scalar activation `x_i` (state), mass `m_i`, timestep
`dt`. Stimulus channels `s ∈ {food, danger, light, cue}` each have intensity
`I_s` and unit direction `u_s` toward the source.

**Sensing (Shepard's universal law of generalization).** Intensity falls off
exponentially with distance `d_s`; a sharp binary `food_contact` signals being
on food:
```
I_s = exp(-d_s / sensor_range)
food_contact = 1 if d_food <= consume_radius else 0
```

**Behavioral force** (spec decomposition):
```
F_i = sensory_i + history_i + motivational_i + coupling_i - fatigue_i
```
- *Drive atoms* (non-directional):
  `sensory_i  = readiness_i * Σ_s sensitivity_i[s] * I_s^kᵢ`
  `history_i  = readiness_i * Σ_s w_i[s] * I_s^kᵢ`         (`w` = learned weights)
  `motivational_i = readiness_i * g(E) * I_food`           (food drive only)
  where `kᵢ` is the atom's `contact_exponent` (>1 sharpens to contact).
- *Movement atoms* (direction `d_i`) carry no sensitivity/history; they
  **express** the drive atoms through the live geometry:
  `F_express_i = Σ_k valence_k * x_k * (u_{stim_k} · d_i)`   (sum over drive atoms k)
- *Consummatory* (`consume`): driven by the sharp `food_contact` signal, so it
  fires only at food and (via coupling) holds the organism there.
- *Coupling*: `coupling_i = Σ_j C[i,j] * x_j` (e.g. `consume → movement = −1.5`
  so feeding suppresses locomotion; `pause → movement < 0`; `explore → movement > 0`).

**Convex marginal value of energy** (state-dependent foraging): a depleted
reserve produces stronger food-seeking, because energy is worth more nearer the
death boundary:
```
g(E) = motivational_strength * (1 - E/E_cap) ** deficit_exponent      (deficit_exponent ≥ 1, convex)
```

**Damped Verlet integration** (driven, dissipative dynamics; the overdamped
regime makes activation track current drive like a leaky accumulator, and gives
a Brunt–Väisälä-type restoring/relaxation reading):
```
v_i        = (x_i(t) - x_i(t-dt)) / dt
F_net      = F_i - c * v_i                              (c = damping_coef; c=0 -> literal pure Verlet)
x_i(t+dt)  = 2*x_i(t) - x_i(t-dt) + (F_net / m_i) * dt^2
x_i(t+dt)  = clip(x_i(t+dt), activation_min, activation_max)
```

**Action emission (Luce choice rule / matching law).** Over the action atoms:
```
P(action a) = exp(x_a / T) / Σ_b exp(x_b / T)           (T = softmax_temperature; T->0 -> argmax)
```

**Energy budget (objective bookkeeping).** Each step:
```
intake       = food_intake_rate           if in contact with food, else 0
energy_delta = intake - danger_loss * danger_contact
expenditure  = basal_metabolism + (move_cost if moved else rest_cost)
E(t+1)       = clip(E(t) + energy_delta - expenditure, 0, E_cap)
death (episode ends) when E <= 0     (cause: starvation, or danger if in contact)
```

**Eligibility trace (temporal weighting of credit).** Recency-weighted:
```
e_i(t) = eligibility_decay * e_i(t-1) + x_i(t)          (sweet spot ~0.9-0.95; 0.99 too long)
```

**Learning (two-tier, valence-split Rescorla–Wagner).** The consequence model
splits the consequence into `(appetitive, aversive)` teaching signals.
Approach-valence drives learn from `appetitive` (reinforcement), avoid-valence
drives from `aversive` (punishment); both *strengthen* the disposition (valence
handles direction in the expression). For drive atom `i` over present channels
`s`:
```
mag = appetitive if valence_i > 0 else aversive
RW:      Δw_i[s] = lr * e_i * I_s * (λ * mag - V_pred)
linear:  Δw_i[s] = lr * mag * e_i * I_s
w_i[s]  = clip(w_i[s] + Δw_i[s], history_weight_min, history_weight_max)
```
`credit_assignment` selects how `V_pred`/channels are handled:
`rw_independent` (per-channel error; enables cue conditioning/generalization),
`rw_competitive` (shared error → blocking/overshadowing), `source_only`.

---

## Mapping to established biobehavioral principles

| Mechanism | Grounding |
|---|---|
| `exp(-distance)` sensing & cue similarity | Shepard's universal law of generalization |
| softmax emission | Luce choice rule / generalized matching law (T ≈ sensitivity) |
| mass + velocity damping | behavioral momentum (resistance to change); MD Langevin friction; leaky competing accumulator |
| energy budget (vs. a motivating operation) | behavioral ecology / state-dependent foraging; objective energy bookkeeping |
| convex deficit drive | marginal value of energy / risk-sensitive foraging |
| valence-split RW; reinforcement ≠ punishment | Rescorla–Wagner; de Villiers (subtractive), Deluty (competitive), Klapes & Riley 2018, Klapes & McDowell 2025; Rasmussen & Newland 2008 |
| operant = functional class, not topography | two-tier drive→movement design |
| eligibility trace | temporal/recency weighting of credit |

---

## Key assumptions (current version)

- **Continuous world, no goal episodes.** Eating does not end the episode; the
  episode ends only at **death** (energy depletion) or truncation. Lives are long
  (target ~10,000 steps; 1 step = 15 min, 96 steps = 1 day).
- **Objectivism.** No mentalistic constructs. Motivation = energy reserve;
  consequence = change in energy; "value" is never posited.
- **Grid world** (default 10×10), discrete actions, for inspectability first.
- **Sensing is local** (`exp(-d/sensor_range)`), not omniscient.
- **Learning signals are normalized per event** (~1), while the energy *reserve*
  uses raw energy units (per-step energy amounts are too small to be useful RW
  targets). The reinforcement/punishment **asymmetry** is deferred to the
  pluggable `ConsequenceModel` variants (currently only `DeltaEnergy`).
- **Reproducible**: all randomness is seeded via `SimulationConfig.seed`.
- See `ToDO.txt` for what is not yet implemented (day/night, operant schedules,
  multiple food patches, visualization, demos, tests, asymmetry models).

---

## Repository layout

```
src/behavioral_md/
  config.py              # SimulationConfig (all parameters, pydantic)
  atoms.py               # BehavioralAtom, verlet_update, default_atom_set (two-tier)
  forces.py              # ForceCalculator (drives + movement expression), coupling
  consequence.py         # ConsequenceModel (DeltaEnergy default; asymmetry stubs)
  learning.py            # EligibilityTrace + valence-split RW history update
  organism.py            # Organism: sense -> force -> damped Verlet -> emit -> learn; energy/death
  simulation.py          # run_episode / run_simulation + long-format DataLogger
  environments/
    gridworld.py         # BehavioralFieldEnv (Gymnasium); stimulus fields, energy, death
experiments/             # reproducible parameter sweeps (exp001-003) + parallel helper
docs/lab_notebook.md     # running record of every experiment and decision
outputs/                 # logs/ and figures/ (generated)
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
python -m experiments.exp003_learning_curve   # learning curve across lives
```
(pygame is optional — `pip install pygame` — for the live render; it has no
prebuilt wheel on some new Python versions.)

## Status

- [x] **Phase 1** — package scaffold, `SimulationConfig`, `BehavioralFieldEnv`
- [x] **Phase 2** — `BehavioralAtom`, force model, Verlet integrator
- [x] **Phase 3** — `Organism`, learning, simulation loop, logging
- [x] **Phase 4** — damped Verlet + softmax; objective energy budget; persistent
  lives; two-tier valence-split learning; consummatory competition. First
  positive acquisition curve (foraging & survival improve across lives).
- [ ] **Phase 5+** — day/night light cycle, operant schedules + matching,
  multiple food patches, visualization, demos, tests (see `ToDO.txt`).
