# Behavioral Molecular Dynamics Simulation Engine

A Python framework that models an organism interacting with an environment using
an analogy to **molecular dynamics**.

The organism is **not** a reinforcement-learning policy. It is a *distributed
dynamical system* composed of many response-capable units ("behavioral atoms").
Each atom has a current state, a prior state, sensitivity to stimuli, response
inertia (mass), fatigue, and a learning history. Environmental stimuli act as
**force fields**; learning history changes the direction and magnitude of the
forces those stimuli produce. Behavior emerges from the coupled dynamics of
these units over time.

## Core idea

In molecular dynamics, particles move through space according to forces. Here,
behavioral atoms move through *activation/state space* according to
history-dependent stimulus forces, integrated with a Verlet update:

```
x_i(t + dt) = 2·x_i(t) − x_i(t − dt) + (F_i(t) / m_i)·dt²
```

where the behavioral force is

```
F_i(t) = sensory_drive + history_drive + motivational_drive + coupling_drive − fatigue
```

- **observation** = sensory stimulation
- **action** = emitted behavior (from relative atom activations)
- **reward** = consequence event that updates learning history
- **history weights** = learning history
- **mass** = behavioral inertia / resistance to change
- **force** = current effective behavioral pull of a stimulus context given history

Gymnasium is used **only** as the environment interface — not for RL training.

## Status

Built incrementally. Current progress:

- [x] **Phase 1** — package scaffold, `SimulationConfig`, `BehavioralFieldEnv`
- [x] **Phase 2** — `BehavioralAtom`, force calculation (directional projection), Verlet integrator
- [ ] Phase 3 — `Organism`, learning rule, simulation loop, logging
- [ ] Phase 4 — visualization, metrics, tests
- [ ] Phase 5 — acquisition / extinction / generalization demos

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
pytest
python scripts/run_demo.py
```

## Repository layout

```
src/behavioral_md/
  config.py              # SimulationConfig (pydantic)
  atoms.py               # BehavioralAtom
  forces.py              # force calculation
  learning.py            # eligibility traces + history-weight updates
  organism.py            # Organism: atoms + forces + integrator + emitter
  simulation.py          # environment <-> organism loop + logging
  visualization.py       # matplotlib + pygame outputs
  metrics.py             # latency, gradients, etc.
  environments/
    gridworld.py         # BehavioralFieldEnv (Gymnasium)
scripts/                 # run_demo, run_extinction_demo, run_generalization_demo
tests/                   # pytest suite
outputs/                 # logs/ and figures/ (generated)
```
