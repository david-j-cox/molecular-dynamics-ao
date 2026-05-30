# Lab Notebook — Behavioral Molecular Dynamics Engine

A running record of what we tried, what the data showed, and why we made each
decision. Newest entries at the bottom. Each experiment that produces numbers
has a reproducible script under `experiments/` and raw output under
`outputs/logs/` (gitignored; regenerate by running the script). Key numbers are
copied into the relevant entry so the notebook stands alone.

Conventions:
- **Q** = question/hypothesis, **M** = method, **R** = results, **I** =
  interpretation, **D** = decision/next step.
- Dates are absolute.

---

## 2026-05-30 — Build log (Phases 1–3)

- **Phase 1**: package scaffold, `SimulationConfig`, `BehavioralFieldEnv`
  (Gymnasium-compliant; passes `check_env`). pygame made optional (no Py3.14
  wheel).
- **Phase 2**: `BehavioralAtom`, `verlet_update`, `ForceCalculator`. Decisions
  recorded with the user:
  - Learning rule: **both** RW and linear, selectable (`learning_rule`).
  - Dynamics: **literal pure Verlet** (no restoring/damping term) — user choice,
    over my flagged concern about stability.
  - Emission: **argmax now**, softmax later — user choice.
  - Mass: **static** for v1.
  - Behavior-environment relation: **directional projection** — a stimulus drives
    a movement atom ∝ `dot(stimulus_unit_vector, atom_direction)`; sign of the
    history weight encodes approach (+) vs. avoid (−).
- **Phase 3**: `Organism`, `learning.py` (eligibility trace + history update),
  `simulation.py` (loop + long-format logging). Credit assignment made
  **selectable**: `rw_competitive | rw_independent | source_only`.

---

## 2026-05-30 — Exp 001: Why doesn't the organism reach food?

**Script:** `experiments/exp001_dynamics_and_emission.py`
**Raw output:** `outputs/logs/exp001_results.json`
**Fixed layout:** start (1,1), food (6,6), danger (9,0), light (0,9), cue (3,3);
10×10 grid, Manhattan start→food = 10.

### Q
The first acquisition run consumed food in **0/30 episodes**; latency censored at
the 120-step cap every episode; all history weights stayed 0 (no consumption →
no consequence → no learning). Why?

### M
Instrument a single episode (print per-step action, position, atom activations),
then sweep candidate dynamics/emission rules over 12 seeds and record reach rate
+ latency. Candidate variations (velocity damping, softmax emission) are
prototyped *outside* core code in the experiment script; core `Organism` is
unchanged.

### R — Bug 1: `consume` dominated from a distance
With `consume.sensitivity[food]=1.0` and a broad sensory field
(`sensor_range=10`), at the start cell (distance 7.07) food intensity ≈ 0.49, so
`consume`'s drive ≈ 0.49 — larger than any movement atom's projected drive
(≈ 0.07). The organism emitted `eat` on every step and never moved:

```
t= 0 act=eat pos=[1,1] consume_act=0.005  (all move atoms ≈ 0)
t=10 act=eat pos=[1,1] consume_act=0.313
t=30 act=eat pos=[1,1] consume_act=1.714   <- still sitting on start cell
```

**Fix applied to core:** consummatory responses are *contact*-gated, not distal.
Added a per-atom `contact_exponent`; `consume` uses `intensity**6`, so its drive
is ≈ 0 at distance and ≈ 1 only on the food cell (where movement atoms get zero
food drive because the food direction vector is zero). Movement atoms keep the
linear distal field (`exponent=1`).

### R — Bug 2: pure Verlet + argmax pins against walls
After fixing Bug 1 the organism moves but **never reaches food**. Sample
trajectory under the literal spec (pure Verlet, argmax), food at (6,6):

```
(1,2)(1,3)(1,4)(1,5)(1,6)(1,7)(1,8)(1,9)  -> runs up the left wall
(2,9)(3,9)(4,9)(5,9)(6,9)(7,9)(8,9)(9,9)  -> across the top wall
(9,9)(9,9)(9,9)... pinned at the corner for the rest of the episode
```

Mechanism: Verlet makes activation the **double integral of force**
(force→acceleration). `move_up` banks a large activation on the way up; when the
organism passes food's row the force on `move_up` reverses, but its *activation*
stays the argmax winner long after, so it keeps emitting "up" into the wall.
Deterministic argmax has no way to shed the banked activation → limit cycle /
wall-pinning.

### R — Reach rate over 12 seeds (the key table)

| Configuration | reaches food | median latency | min dist if failed |
|---|---|---|---|
| `pure_verlet_argmax` (literal spec) | **0 / 12** | — | 3.0 |
| `pure_verlet_softmax` (T=1.0) | 1 / 12 | 29 | 3.16 |
| `damping_c2_argmax` | 0 / 12 | — | 3.0 |
| `overdamped_c20_softmax` (T=0.3, food_sens=1.5, sensor_range=5) | **10 / 12** | 114 | 0.5 |

(Numbers are from `outputs/logs/exp001_results.json`, n=12 seeds, 200-step cap.)

Earlier finer sweeps (not all in the script): softmax alone is seed-fragile
(1–4/12 across T=1–3); damping alone with argmax still pins (0/12 at c=0.5…2);
overdamped + softmax is the only combination that reliably reaches, robust to
food/no-food danger placement (8–10/12).

### I — interpretation (physics ↔ behavior)
- **Velocity damping** `−c·(x−x_prev)/dt` makes each atom a *damped* harmonic
  oscillator. As damping grows the second-order Verlet dynamics become
  effectively first-order ("overdamped"), so activation stops being an inertial
  integrator and becomes a **leaky accumulator of current drive**
  (Usher–McClelland leaky competing accumulator). This is also standard in real
  MD: Langevin dynamics adds exactly a `−γv` friction term — damping is not a
  departure from the MD analogy.
- **Softmax emission** = the **Luce choice rule / matching law** (temperature ↔
  matching sensitivity). It breaks the argmax limit cycle and makes response
  allocation lawful/graded rather than winner-take-all.

### D — OPEN DECISION (with user)
Proposed: change *defaults* to velocity-damping (overdamped) + softmax emission,
keeping pure-Verlet and argmax reachable via config so nothing is lost. Awaiting
user input on the dynamics framing before editing core. Bug-1 fix
(`contact_exponent`) already applied to core and committed.

**RESOLVED 2026-05-30:** user approved overdamped Verlet + softmax as defaults,
literal forms kept as config switches. See the Phase-4 design entry below.

---

## 2026-05-30 — Phase 4 design: long-life energy-budget foraging (PLAN)

Major redirection agreed with the user. This entry is the design of record; all
numeric values marked *(proposed)* are starting points to calibrate, not commitments.

### Settled design decisions (with rationale)

1. **Eating does not end the episode.** Model life in the long run. Lives run
   ~10,000 steps. Episodes end only on **death** (energy reserve depleted).
2. **Timescale:** 1 step = 15 min; **96 steps = 1 day** *(expandable)*. A
   10,000-step life ≈ 104 days.
3. **Motivation -> objective energy budget** (behavioral ecology), replacing the
   hunger/MO construct. Energy is a conserved reserve with literal in/out flows.
   Nothing is "valued" or "established."
4. **Energy -> behavior coupling: convex marginal value of energy.** The
   food-directed force rises steeply as the reserve approaches starvation
   (state-dependent / risk-sensitive foraging). Objective: energy is worth more
   nearer the death boundary, so foraging pressure is a function of the
   measurable reserve, not a posited motivational state.
5. **Light = global ambient sun** `L(t) in [0,1]`, cycling over the 96-step day.
   It is a discriminative *context / setting event*, sensed as a scalar (no
   direction to move toward). Distinct from the localized discriminative stimuli
   (old "light"/"cue" point sources = S+/S-delta/neutral signals) which remain a
   separate construct to develop later.
6. **Darkness raises risk, graded with a low floor.** `danger_sensed =
   danger_true * (floor + (1-floor)*L)`, floor ~0.2 *(proposed)*. Night vision is
   poor but nonzero; a near-starving organism rationally accepts elevated night
   risk to feed (couples to #4).
7. **Food = multiple renewable patches** (lean 2-3) with **logistic regrowth**,
   growth rate proportional to `L` (grows in daylight), drawn down on
   consumption. Food is also sensed more strongly in daylight (same graded form
   as danger). Multiple independently-regrowing patches make patch-foraging
   (marginal-value theorem) and **concurrent-schedule matching** emerge.
8. **Schedules: both, on one engine.** (a) The grid yields interval-like
   schedules and matching *naturally* from concurrent patch regrowth (faithful to
   the molar/foraging reading of operant behavior -- Baum, Rachlin). (b) A
   separate minimal **operant chamber** (feeder + `press` response, no
   navigation) reuses the same Organism/atoms/forces/learning for textbook-clean
   FR/VR/FI/VI curves and conc VI-VI matching.
9. **Dynamics/emission defaults:** overdamped Verlet (velocity damping `-c*v`) +
   softmax emission (Luce/matching). Literal pure-Verlet and argmax stay as
   config switches. (Justified by Exp 001.)
10. **Death is a dependent variable.** Log time-to-death, cause, and the energy
    trajectory leading up to it, so survival/mortality patterns can defend the
    modeling choices in write-up.

### Objective energy model (proposed equations + numbers)

Reserve `E` normalized to [0, 1]; `E_init = 0.5`. Per step:

```
E <- E - basal - move_cost(action) + intake - danger_cost
basal        = 0.01            (proposed; matches user's metabolic anchor)
move_cost    = 0.005 for a move, ~0.001 for pause/no-op   (proposed)
intake       = energy drawn from the patch when consuming  (see patches)
danger_cost  = 0.1 * danger_sensed   (injury = energy loss; objective)  (proposed)
death when E <= 0  -> episode ends
```

- **Consequence = change in energy from contingent events** (food gain +,
  danger loss -). This is what feeds the learning update -- the reinforcer is
  literally energy, not an abstract +/-1. (Baseline metabolism/movement are
  non-contingent costs, not learning consequences.)
- **Convex coupling:** food-directed drive multiplied by `g(E) = (1 - E)**2`
  *(proposed; 1/E variant available)*.

### Food patch dynamics (proposed)

Per patch, biomass `B in [0, K]`, `K = 1`:
```
B <- B + r * L * B * (1 - B/K)        # logistic growth, daylight-gated
r = 0.05 (proposed)
on consume at patch: bite = min(B, 0.2);  B -= bite;  intake = yield * bite
yield = 1.0 (proposed)
food_sensed = B * (floor_f + (1-floor_f)*L),  floor_f ~0.3 (proposed)
```

### Day/night (proposed)

```
L(t) = 0.5 * (1 - cos(2*pi * (t mod 96) / 96))   # 0 at midnight, 1 at midday
```

### Compute / parallelism (proposed)

Start with **multiprocessing across independent lives/conditions** (e.g. 64
lives/condition); near-linear speedup, no rewrite. JAX vectorization deferred.

### Logging / metrics additions

Add to the per-step log: `energy`, `light`, per-patch `biomass`, `cause_of_death`.
New metrics: survival curves, time-to-death distributions, death-cause breakdown,
day/night activity and mortality, patch allocation vs. relative intake (matching),
schedule response patterns (chamber).

### Proposed build order (phases)

- **4a** Dynamics/emission defaults: damping + softmax in core (controller must
  work before a 10k-step life is viable).
- **4b** Energy budget: reserve, costs, food=energy intake, danger=energy loss,
  death; convex coupling replacing `hunger`. Death-pattern logging.
- **4c** Persistent episodes (eating no longer terminates); long lives.
- **4d** Multi-patch logistic food + daylight-gated growth/visibility.
- **5**  Global ambient sun + graded light-gating of danger and food.
- **6**  Operant chamber env + schedule layer; FR/VR/FI/VI + conc VI-VI matching.
- **7**  Parallel runner; metrics + figures incl. death-pattern analysis.

### Open / proposed (awaiting confirmation)
- All *(proposed)* numbers above (energy magnitudes, growth rates, floors).
- `press` atom for the chamber (vs. reusing `consume`).
- Parallelism approach (multiprocessing first vs. JAX now).
- Discriminative-stimulus (S+/S-delta) semantics -- deferred by user.
