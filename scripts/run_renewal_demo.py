"""Renewal demo (ABA vs ABB) for the dual excitatory/inhibitory rule.

Extinction inhibition (w-) is context-specific while excitation (w+) is context-general.
Two groups acquire in context A and extinguish in context B, then are tested:

  ABA -- test back in the ORIGINAL context A: the B-context inhibition does not apply
         (the gate -> 0), so the preserved excitation drives behavior again -- renewal.
  ABB -- test in the EXTINCTION context B (control): inhibition still applies, so
         responding stays extinguished.

Context is a scalar surfaced in the observation (obs["context"]); the dual rule tags w-
with the context it was learned in and gates it at readout by Shepard similarity. Both
test phases are unreinforced; the ABA vs ABB difference isolates the context effect.

Run:  python scripts/run_renewal_demo.py
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
N_TRAIN, N_EXT, N_TEST = 15, 15, 10
STEPS = 300
CTX_A, CTX_B = 0.0, 1.0
LAYOUT = {"position": [4, 4], "food": [4, 8], "danger": [9, 0], "light": [0, 9], "cue": [8, 4]}


def _life(env, org, cfg, reinforce, context, seed):
    """Run one life; return (net, w_plus, w_minus, steps_at_food) for approach_food."""
    obs, info = env.reset(seed=seed, options={
        "layout": LAYOUT, "food_reinforces": reinforce, "context": context})
    org.reset(obs)
    af = org.atoms[org.index["approach_food"]]
    steps_at_food = 0
    for _ in range(cfg.max_steps):
        org.step(obs)
        action = org.emit_action()
        obs, _r, term, trunc, info = env.step(action)
        org.update_history(obs, action, info)
        if float(obs["food_contact"][0]) > 0.0:
            steps_at_food += 1
        if (not org.alive) or term or trunc:
            break
    return (af.history_weights["food"], af.w_plus["food"], af.w_minus["food"], steps_at_food)


def _run_arm(test_ctx: float, seed: int) -> list[dict[str, Any]]:
    cfg = SimulationConfig(max_steps=STEPS, seed=seed, sensor_range=8.0,
                           learning_model="dual_exc_inhib", context_gating=True)
    env, org = BehavioralFieldEnv(cfg), Organism(cfg)
    rows = []
    for ep in range(N_TRAIN + N_EXT + N_TEST):
        if ep < N_TRAIN:
            reinforce, ctx = True, CTX_A           # acquire in A
        elif ep < N_TRAIN + N_EXT:
            reinforce, ctx = False, CTX_B          # extinguish in B
        else:
            reinforce, ctx = False, test_ctx       # test in A (ABA) or B (ABB)
        net, wp, wm, saf = _life(env, org, cfg, reinforce, ctx, seed * 1000 + ep)
        rows.append({"seed": seed, "episode": ep, "net": net, "w_plus": wp,
                     "w_minus": wm, "steps_at_food": saf})
    return rows


def agent_worker(cell: dict[str, Any]) -> dict[str, Any]:
    seed = cell["seed"]
    return {"aba": _run_arm(CTX_A, seed), "abb": _run_arm(CTX_B, seed)}


def main(n_agents: int = N_AGENTS) -> None:
    print(f"Renewal demo (ABA vs ABB): {n_agents} agents x "
          f"({N_TRAIN} acquire A + {N_EXT} extinguish B + {N_TEST} test) lives...")
    results = run_sweep(agent_worker, [{"seed": s} for s in range(n_agents)], progress_every=25)
    aba = pd.DataFrame([r for res in results for r in res["aba"]])
    abb = pd.DataFrame([r for res in results for r in res["abb"]])

    plot_dual_components(aba, transitions=[N_TRAIN, N_TRAIN + N_EXT],
                         path=FIG_DIR / "renewal.png",
                         title="Renewal (ABA: acquire A, extinguish B, test A)")

    test0 = N_TRAIN + N_EXT
    aba_net = aba.groupby("episode")["net"].mean().iloc[test0]
    abb_net = abb.groupby("episode")["net"].mean().iloc[test0]
    ext_net = aba.groupby("episode")["net"].mean().iloc[test0 - 1]
    # The net association the force reads is the clean readout of renewal. (Food sits
    # near the start, so raw food-contact counts are floored by incidental proximity
    # and do not discriminate motivation here -- the net does.)
    print(f"  net  end-of-extinction (ctx B): {ext_net:.3f}")
    print(f"  first test life net:   ABA (ctx A) {aba_net:.3f}  vs  ABB (ctx B) {abb_net:.3f}")
    print(f"  renewal (net returns in the original context A, not in B):  "
          f"{'OK' if aba_net > abb_net + 0.1 else 'FAIL'}")
    print(f"Wrote {FIG_DIR/'renewal.png'}")


if __name__ == "__main__":
    import argparse

    _ap = argparse.ArgumentParser()
    _ap.add_argument("--agents", type=int, default=N_AGENTS)
    main(_ap.parse_args().agents)
