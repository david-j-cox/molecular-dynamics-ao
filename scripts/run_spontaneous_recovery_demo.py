"""Spontaneous recovery demo (dual excitatory/inhibitory rule).

A trained approach is extinguished, then the organism spends a REST interval during
which food is simply ABSENT (the SAME arena, food withheld -- no spatial/context change,
just the passage of time with nothing to respond to), then is re-tested with food back.
Under the Konorski/Bouton dual rule the extinction inhibition (w-) decays passively over
the rest interval while the excitation (w+) is preserved, so the net association -- and
food-directed behavior -- RETURNS at re-test: spontaneous recovery driven purely by
elapsed time (context held constant, gating off), distinct from renewal.

The rest interval keeps the identical layout but sets food_present=False, so there is no
food signal/contact (no further extinction) -- only the time-based passive w- decay acts.
Measures across lives (population, 95% CI): net association, w+, w- for approach_food.

Run:  python scripts/run_spontaneous_recovery_demo.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from behavioral_md.config import SimulationConfig
from behavioral_md.environments import BehavioralFieldEnv
from behavioral_md.organism import Organism
from behavioral_md.parallel import run_sweep
from behavioral_md.visualization import plot_dual_components

FIG_DIR = Path("outputs/figures")
N_AGENTS = 100
N_TRAIN, N_EXT, N_REST, N_TEST = 15, 15, 10, 10
STEPS = 300
REST_STEPS = 50  # length of each food-free rest life (the retention interval)
PASSIVE_DECAY = 0.02
LAYOUT = {"position": [4, 4], "food": [4, 8], "danger": [9, 0], "light": [0, 9], "cue": [8, 4]}


def _life(env, org, cfg, reinforce, food_present, seed, n_steps):
    """Run one life; return (net, w_plus, w_minus, steps_at_food) for approach_food."""
    obs, info = env.reset(seed=seed, options={
        "layout": LAYOUT, "food_reinforces": reinforce, "food_present": food_present})
    org.reset(obs)
    af = org.atoms[org.index["approach_food"]]
    steps_at_food = 0
    for _ in range(n_steps):
        org.step(obs)
        action = org.emit_action()
        obs, _r, term, trunc, info = env.step(action)
        org.update_history(obs, action, info)
        if float(obs["food_contact"][0]) > 0.0:
            steps_at_food += 1
        if (not org.alive) or term or trunc:
            break
    return (af.history_weights["food"], af.w_plus["food"], af.w_minus["food"], steps_at_food)


def agent_worker(cell: dict[str, Any]) -> dict[str, Any]:
    seed = cell["seed"]
    cfg = SimulationConfig(max_steps=STEPS, seed=seed, sensor_range=8.0,
                           learning_model="dual_exc_inhib",
                           inhibition_passive_decay=PASSIVE_DECAY)
    env, org = BehavioralFieldEnv(cfg), Organism(cfg)
    rows = []
    for ep in range(N_TRAIN + N_EXT + N_REST + N_TEST):
        if ep < N_TRAIN:                              # acquire (food present, reinforced)
            phase, reinforce, present, n_steps = "train", True, True, STEPS
        elif ep < N_TRAIN + N_EXT:                    # extinguish (food present, unrewarded)
            phase, reinforce, present, n_steps = "ext", False, True, STEPS
        elif ep < N_TRAIN + N_EXT + N_REST:           # rest: SAME arena, food absent
            phase, reinforce, present, n_steps = "rest", False, False, REST_STEPS
        else:                                         # re-test (food present, unrewarded)
            phase, reinforce, present, n_steps = "test", False, True, STEPS
        net, wp, wm, saf = _life(env, org, cfg, reinforce, present, seed * 1000 + ep, n_steps)
        rows.append({"seed": seed, "episode": ep, "phase": phase, "net": net,
                     "w_plus": wp, "w_minus": wm, "steps_at_food": saf})
    return {"rows": rows}


def main(n_agents: int = N_AGENTS) -> None:
    print(f"Spontaneous recovery demo: {n_agents} agents x "
          f"({N_TRAIN} train + {N_EXT} ext + {N_REST} rest + {N_TEST} test) lives...")
    results = run_sweep(agent_worker, [{"seed": s} for s in range(n_agents)], progress_every=25)
    df = pd.DataFrame([r for res in results for r in res["rows"]])

    t1, t2, t3 = N_TRAIN, N_TRAIN + N_EXT, N_TRAIN + N_EXT + N_REST
    plot_dual_components(df, transitions=[t1, t2, t3],
                         path=FIG_DIR / "spontaneous_recovery.png",
                         title="Spontaneous recovery (dual exc/inhib)")

    g = df.groupby("episode")
    net = g["net"].mean()
    net_end_ext = net.iloc[t2 - 1]
    net_recovered = net.iloc[t3 - 1]            # end of the rest interval (recovery peak)
    net_first_test = net.iloc[t2 + N_REST]      # first re-test life (re-extinction begins)
    rest_contacts = df[df["phase"] == "rest"]["steps_at_food"].mean()
    print(f"  net  end-of-extinction: {net_end_ext:.3f}  ->  end-of-rest (recovered): "
          f"{net_recovered:.3f}  ->  first re-test life: {net_first_test:.3f} (re-extinguishing)")
    print(f"  w+ end-ext {g['w_plus'].mean().iloc[t2-1]:.3f} (preserved); "
          f"w- end-ext {g['w_minus'].mean().iloc[t2-1]:.3f} -> "
          f"end-rest {g['w_minus'].mean().iloc[t3-1]:.3f} (passively decayed)")
    print(f"  rest-phase food contacts (should be ~0): {rest_contacts:.2f}")
    print(f"  spontaneous recovery (net rebounds over the rest interval): "
          f"{'OK' if net_recovered > net_end_ext + 0.1 else 'FAIL'}")
    print(f"Wrote {FIG_DIR/'spontaneous_recovery.png'}")


if __name__ == "__main__":
    import argparse

    _ap = argparse.ArgumentParser()
    _ap.add_argument("--agents", type=int, default=N_AGENTS)
    main(_ap.parse_args().agents)
