"""exp060 -- is the second-order (inertial) Verlet term load-bearing, or decorative?

Reviewers (and the manuscript's own analysis) suggested the overdamped Verlet limit "is" a leaky
competing accumulator (Usher & McClelland 2001), so the molecular-dynamics framing might be
decorative. This audit shows that claim is IMPRECISE and the term is in fact load-bearing.

The key distinction (panel A): DAMPING is not LEAK. The engine's atom is damped (a velocity
friction -c*v) but has NO restoring/spring term, so its update is a damped INTEGRATOR: under a
constant drive the activation RAMPS, and once driven up PERSISTS after the drive is removed
(memory/hysteresis; at low damping it also overshoots/carries momentum). A leaky accumulator
x <- x + dt*(force/m - leak*x) has a restoring term (leak), so it TRACKS the current drive and
DECAYS to baseline when the drive stops. The overdamped limit of a damped *harmonic oscillator*
(m x'' + c x' + k x = F) is the leaky accumulator only because of the spring k; with k = 0 (the
engine's atom) the limit is a pure integrator, not a leaky accumulator -- the equivalence fails.

Two load-bearing consequences the leaky accumulator cannot reproduce:
  * ACQUISITION CURVE (panel B). The ramp makes a naive organism SLOW (activation must build before
    it drives decisive approach), so latency starts high and falls as learning steepens the ramp --
    the realistic acquisition curve. A leaky organism tracks the drive immediately, so it is fast
    from life 1 (no naive-slow phase) at ANY softmax temperature -- lowering temperature does not
    recover the curve, it just makes it faster from the start. The integrator shapes the curve.
  * PERSISTENCE after stimulus removal (panel C). Activation 20 steps after a stimulus offset is
    ~100% under Verlet (it holds / perseverates) and ~10% under leaky (it decays) -- behavior
    outlasting its eliciting cue (perseveration), which the leaky accumulator cannot hold.

VERDICT: the second-order/integrator term is LOAD-BEARING, and "damped Verlet = leaky accumulator"
conflates damping (velocity friction) with leak (a restoring spring the atom lacks). The atom is a
damped integrator: it ramps (shaping acquisition) and persists (perseveration), both absent from the
leaky accumulator. The missing restoring force is supplied behaviorally -- by fatigue (exp059) or
coupling inhibition -- not by the integrator. The MD framing earns its keep.

Run:  python experiments/exp060_integrator_audit.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from behavioral_md.atoms import verlet_update  # noqa: E402
from behavioral_md.config import SimulationConfig  # noqa: E402
from behavioral_md.environments.gridworld import BehavioralFieldEnv  # noqa: E402
from behavioral_md.experiment_utils import weak_innate_atoms  # noqa: E402
from behavioral_md.organism import Organism  # noqa: E402
from behavioral_md.parallel import run_sweep  # noqa: E402
from behavioral_md.simulation import run_episode  # noqa: E402

FIG = Path("outputs/figures")

# Acquisition audit (exp048 layout, smaller fleet).
N_AGENTS, N_LIVES, STEPS = 24, 28, 300
LAYOUT = {"position": [4, 4], "food": [4, 8], "danger": [9, 0], "light": [0, 9], "cue": [8, 4]}


# --- A. step-response of one atom: persistence (Verlet) vs decay (leaky) -----------------------
def step_response(integrator: str, c: float = 10.0, leak: float = 1.0, F0: float = 1.0,
                  on: int = 30, total: int = 75, mass: float = 1.0, dt: float = 0.1) -> np.ndarray:
    x = np.array([0.0])
    xp = np.array([0.0])
    out = []
    for t in range(total):
        force = F0 if t < on else 0.0
        if integrator == "leaky":
            x = np.clip(x + dt * (force / mass - leak * x), -10, 10)
        else:
            vel = (x - xp) / dt
            xn = np.clip(verlet_update(x, xp, force - c * vel, mass, dt), -10, 10)
            xp, x = x, xn
        out.append(float(x[0]))
    return np.array(out)


# --- C. organism-level persistence after stimulus removal --------------------------------------
def _food_obs(on: bool) -> dict:
    z = np.zeros(2)
    o = {f"{s}_vector": z.copy() for s in ("food", "danger", "light", "cue")}
    o.update({f"{s}_intensity": np.array([0.0]) for s in ("food", "danger", "light", "cue")})
    o["food_intensity"] = np.array([1.0 if on else 0.0])
    o["food_vector"] = np.array([1.0, 0.0])
    o["food_contact"] = np.array([0.0])
    o["cue_value"] = np.array([0.0])
    o["context"] = np.array([0.0])
    return o


def persistence(integrator: str, c: float = 10.0, leak: float = 1.0, pulse: int = 25,
                after: int = 20) -> float:
    """Fraction of the at-offset approach activation still present ``after`` steps later."""
    kw: dict[str, Any] = dict(seed=0, learning_rate=0.0, integrator=integrator, damping_coef=c)
    if integrator == "leaky":
        kw["leak_coef"] = leak
    org = Organism(SimulationConfig(**kw), rng=np.random.default_rng(0))
    org.reset(_food_obs(True))
    acts = []
    for t in range(pulse + after + 1):
        org.step(_food_obs(t < pulse))
        acts.append(org.activation("approach_food"))
    off = acts[pulse - 1]
    return acts[pulse + after] / off if abs(off) > 1e-6 else 0.0


# --- B. acquisition learning curve under each integrator ---------------------------------------
def _worker(cell: dict[str, Any]) -> dict[str, Any]:
    kw = dict(n_episodes=N_LIVES, max_steps=STEPS, seed=cell["seed"], sensor_range=8.0,
              reinforcement_asymptote=2.0, integrator=cell["integrator"])
    if cell["integrator"] == "leaky":
        kw["leak_coef"] = 1.0
    cfg = SimulationConfig(**kw)
    env = BehavioralFieldEnv(cfg)
    org = Organism(cfg, atoms=weak_innate_atoms(0.2))
    rows = []
    for ep in range(N_LIVES):
        r = run_episode(env, org, cfg, ep, None, {"layout": LAYOUT}, seed=cell["seed"] * 1000 + ep)
        rows.append({"episode": ep, "latency": r["latency"]})
    return {"summaries": rows}


def acquisition_curve(integrator: str) -> np.ndarray:
    cells = [{"seed": s, "integrator": integrator} for s in range(N_AGENTS)]
    res = run_sweep(_worker, cells, progress_every=200)
    df = pd.DataFrame([r for x in res for r in x["summaries"]])
    return df.groupby("episode")["latency"].mean().to_numpy()


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)

    sr = {f"verlet c={c}": step_response("verlet", c=c) for c in (2.0, 10.0)}
    sr["leaky leak=1"] = step_response("leaky")
    pers = {"verlet c=10\n(default)": persistence("verlet", c=10.0),
            "verlet c=2\n(low damping)": persistence("verlet", c=2.0),
            "leaky\nleak=1": persistence("leaky")}
    acq_v = acquisition_curve("verlet")
    acq_l = acquisition_curve("leaky")

    print("Integrator audit -- damped Verlet integrator vs first-order leaky accumulator:\n")
    print("C. persistence (activation 20 steps after a drive pulse ends, fraction of at-offset):")
    for name, v in pers.items():
        print(f"   {name.replace(chr(10), ' '):26s} persistence = {v:.2f}")
    print("   -> Verlet HOLDS (perseverates); leaky DECAYS. Load-bearing for response persistence.")
    print(f"\nB. acquisition latency verlet {acq_v[:4].mean():.0f}->{acq_v[-4:].mean():.0f}"
          f"  vs  leaky {acq_l[:4].mean():.0f}->{acq_l[-4:].mean():.0f}")
    print("   -> the Verlet ramp gives a realistic slow-naive acquisition curve; leaky tracks the "
          "drive and is fast from life 1 (no naive-slow phase). The integrator shapes the curve.")
    print("\nVERDICT: load-bearing. 'damped Verlet = leaky' conflates damping (velocity friction) "
          "with leak (a restoring spring the atom lacks).")

    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.6))

    for name, tr in sr.items():
        ls = "--" if "leaky" in name else "-"
        ax[0].plot(tr, ls, lw=2, label=name)
    ax[0].axvspan(0, 30, color="0.92", label="drive ON")
    ax[0].set_xlabel("step (drive on 0–30, off after)")
    ax[0].set_ylabel("atom activation")
    ax[0].set_title("A. Damping ≠ leak: the Verlet integrator ramps then\nHOLDS; leaky tracks then "
                    "decays (restoring term)")
    ax[0].legend(fontsize=8, loc="center right")

    lives = np.arange(len(acq_v))
    ax[1].plot(lives, acq_v, "o-", color="tab:blue", ms=3, label="damped Verlet")
    ax[1].plot(lives, acq_l, "s--", color="tab:orange", ms=3, label="first-order leaky")
    ax[1].set_xlabel("life")
    ax[1].set_ylabel("latency to food (steps)")
    ax[1].set_title("B. Acquisition: the ramp gives a slow-naive curve;\nleaky is fast from life 1 "
                    "(integrator shapes it)")
    ax[1].legend(fontsize=8)

    names = list(pers)
    colors = ["tab:blue", "tab:cyan", "tab:orange"]
    ax[2].bar(names, [pers[n] for n in names], color=colors)
    ax[2].axhline(1.0, color="0.7", lw=0.8, ls=":")
    ax[2].set_ylabel("activation retained 20 steps after stimulus offset")
    ax[2].set_title("C. Persistence after stimulus removal:\nVerlet holds, leaky cannot → "
                    "LOAD-BEARING")
    ax[2].set_ylim(0, 1.35)

    fig.tight_layout()
    out = FIG / "exp060_integrator_audit.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
