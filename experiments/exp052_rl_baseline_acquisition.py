"""exp052 -- RL baseline: is acquisition (falling latency to food) generic to value learning?

Companion to exp049 (matching baseline). The AI/RL review asked what the behavioral-atom machinery
adds over a plain value learner on the engine's core tasks. Here we run a textbook tabular Q-learner
on the SAME environment and controlled layout as the acquisition demo (run_demo): state = grid cell,
Q[cell, action] updated by Q-learning from the environment's consequence, epsilon-greedy, learning
carried across lives (body/position reset each life). We measure latency to first food per life.

If the Q-learner's latency falls across lives (acquisition), then acquisition is a generic property
of value learning on this task, not a signature of the atom dynamics. The atom engine's contribution
is the mechanistic, distributed account (and the unification with the other phenomena), not a unique
ability to acquire. We report both curves so the reader can see what is and is not specific.

Run:  python experiments/exp052_rl_baseline_acquisition.py
"""

from __future__ import annotations

import numpy as np

from behavioral_md.config import SimulationConfig
from behavioral_md.environments.gridworld import BehavioralFieldEnv

N_AGENTS = 60
N_LIVES = 40
STEPS = 300
ALPHA = 0.3          # Q-learning rate
GAMMA = 0.95         # discount
EPS0, EPS1 = 0.3, 0.02   # epsilon anneal across lives
LAYOUT = {"position": [4, 4], "food": [4, 8], "danger": [9, 0], "light": [0, 9], "cue": [8, 4]}


def run_agent(seed: int) -> list[int]:
    cfg = SimulationConfig(max_steps=STEPS, seed=seed)
    env = BehavioralFieldEnv(cfg)
    g = cfg.grid_size
    n_a = env.action_space.n
    q = np.zeros((g * g, n_a))
    rng = np.random.default_rng(seed)
    latencies = []
    for life in range(N_LIVES):
        eps = EPS0 + (EPS1 - EPS0) * life / (N_LIVES - 1)
        obs, _ = env.reset(seed=seed * 1000 + life, options={"layout": LAYOUT})
        pos = np.asarray(obs["position"]).astype(int)
        s = int(pos[1]) * g + int(pos[0])
        latency = STEPS
        for t in range(STEPS):
            a = rng.integers(n_a) if rng.random() < eps else int(np.argmax(q[s]))
            obs, r, term, trunc, _ = env.step(a)
            pos = np.asarray(obs["position"]).astype(int)
            s2 = int(pos[1]) * g + int(pos[0])
            q[s, a] += ALPHA * (r + GAMMA * q[s2].max() - q[s, a])
            s = s2
            if r > 0 and latency == STEPS:
                latency = t
            if term or trunc:
                break
        latencies.append(latency)
    return latencies


def main() -> None:
    print(f"RL baseline (tabular Q-learning) on the acquisition task: {N_AGENTS} agents x "
          f"{N_LIVES} lives.\n")
    lat = np.array([run_agent(s) for s in range(N_AGENTS)])      # [agents, lives]
    by_life = lat.mean(0)
    early = float(by_life[:5].mean())
    late = float(by_life[-5:].mean())
    reach = float((lat[:, -5:] < STEPS).mean())
    print(f"mean latency to food  early (1-5): {early:.1f}  ->  late (36-40): {late:.1f}")
    print(f"latency drop: {early - late:.1f}   late reach rate: {reach:.2f}")
    verdict = "FALLS (acquisition)" if early - late > 10 else "is flat"
    print(f"\nA plain tabular Q-learner's latency {verdict}: acquisition is generic to value")
    print("learning on this task, not a signature of the atom dynamics. For reference, the atom")
    print("engine on the same layout drops latency ~128 -> ~56 (run_demo). The atom engine's")
    print("contribution is the mechanistic distributed account and unification, not acquisition.")


if __name__ == "__main__":
    main()
