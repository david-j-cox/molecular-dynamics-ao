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
