"""Reinforcement/punishment asymmetry in two preparations.

The asymmetry between how reinforcement and punishment control behavior is studied here
in BOTH worlds of the engine:

  1. Concurrent choice (operant chamber, ``chamber.run_punishment_choice``): three
     accounts of how a punisher maps to allocation -- subtractive (de Villiers, 1980),
     competitive (Deluty, 1976), concatenated (Critchfield/Klapes). They all suppress the
     punished response but DISSOCIATE on how that suppression depends on the alternative's
     reinforcement rate (subtractive rises, competitive falls). Quantified in exp029; the
     dissociation panel is reproduced here for the side-by-side.

  2. Open foraging (survival world, the pluggable ``ConsequenceModel``): danger is the
     punisher. Under the subtractive consequence model a punisher trains avoidance ``c``
     times more strongly than a reinforcer trains approach. Sweeping ``c`` traces an
     approach-avoidance gradient -- learned avoidance rises, food intake and danger
     contacts fall, and a high enough punishment sensitivity becomes maladaptive
     (the organism over-avoids the danger that blocks the path to food and starves).

Run:   python studies/punishment_asymmetry/compare_models.py
Saves: studies/punishment_asymmetry/figures/*.png + summary.json
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from behavioral_md.chamber import ChamberConfig, run_punishment_choice  # noqa: E402
from behavioral_md.config import SimulationConfig  # noqa: E402
from behavioral_md.environments import BehavioralFieldEnv  # noqa: E402
from behavioral_md.experiment_utils import weak_innate_atoms  # noqa: E402
from behavioral_md.organism import Organism  # noqa: E402

FIG = Path(__file__).parent / "figures"
INF = float("inf")
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                     "axes.labelsize": 12, "xtick.labelsize": 10,
                     "ytick.labelsize": 10, "legend.fontsize": 10})


# --- Preparation 1: concurrent choice (chamber) ----------------------------------- #
def chamber_dissociation() -> dict:
    """Log-odds punishment suppression vs alternative reinforcement (de Villiers/Deluty)."""
    def logodds(model, alt_vi, punish, **kw):
        cfg = ChamberConfig(pun_tau=800.0, pun_bump=0.04, pun_floor=0.1, **kw)
        res = run_punishment_choice(model, cfg, 600, 5000, vi_reinf=[5.0, alt_vi],
                                    vi_punish=[punish, INF], seed=0)
        e = res["emit"].sum(0)
        return float(np.log(e[0] / e[1]))

    alt_vis = [20.0, 10.0, 6.0, 4.0]
    rates = [1.0 / v for v in alt_vis]
    sub = [logodds("subtractive", v, INF, pun_c=1.0) - logodds("subtractive", v, 10.0, pun_c=1.0)
           for v in alt_vis]
    comp = [logodds("competitive", v, INF, pun_c=1.5) - logodds("competitive", v, 10.0, pun_c=1.5)
            for v in alt_vis]
    return {"alt_rates": rates, "subtractive": sub, "competitive": comp}


# --- Preparation 2: open foraging (survival world) -------------------------------- #
# Food straight up column 5 from the start; danger off to the side (column 2, midway)
# with a local sensor range, so it is an AVOIDABLE obstacle (a detour cost), not a hard
# block -- giving a graded approach-avoidance tradeoff as punishment sensitivity rises.
_LAYOUT = {"position": [5, 1], "food": [5, 9], "danger": [2, 5],
           "light": [0, 0], "cue": [9, 9]}
_SENSOR_RANGE = 5.0
_N_AGENTS, _N_LIVES, _STEPS = 36, 25, 250


def _forage_agent(c: float, seed: int) -> tuple[float, float, float]:
    """One agent under the subtractive consequence model with punishment weight c.

    Returns (food consumed, danger contacts, learned avoidance weight), averaged over
    the asymptotic lives. c == 1 is the symmetric (delta-energy) baseline.
    """
    kw = ({} if c == 1.0 else
          {"consequence_model": "subtractive", "punishment_weight": c})
    cfg = SimulationConfig(n_episodes=_N_LIVES, max_steps=_STEPS, seed=seed,
                           sensor_range=_SENSOR_RANGE, **kw)
    env = BehavioralFieldEnv(cfg)
    org = Organism(cfg, atoms=weak_innate_atoms(0.3))
    food = danger = 0.0
    for ep in range(_N_LIVES):
        obs, info = env.reset(seed=seed * 1000 + ep, options={"layout": _LAYOUT})
        org.reset(obs)
        for _ in range(_STEPS):
            org.step(obs)
            action = org.emit_action()
            obs, _r, term, trunc, info = env.step(action)
            org.update_history(obs, action, info)
            if ep >= _N_LIVES - 8:                       # asymptotic lives
                food += float(info.get("food_consumed", 0.0))
                danger += float(info.get("danger_contact", 0.0) > 0.0)
            if (not org.alive) or term or trunc:
                break
    w_avoid = org.atoms[org.index["avoid_danger"]].history_weights["danger"]
    return food / 8.0, danger / 8.0, float(w_avoid)


def foraging_gradient() -> dict:
    cs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    food, danger, avoid = [], [], []
    for c in cs:
        vals = np.array([_forage_agent(c, s) for s in range(_N_AGENTS)])
        f, d, w = vals.mean(0)
        food.append(f)
        danger.append(d)
        avoid.append(w)
    return {"c": cs, "food": food, "danger_contacts": danger, "avoid_weight": avoid}


# --- Figure: the asymmetry in both preparations ----------------------------------- #
def figure(diss: dict, forage: dict) -> Path:
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 4.6))

    a = np.asarray(diss["alt_rates"])
    axL.plot(a, diss["subtractive"], color="black", ls="-", marker="o", ms=5,
             label="subtractive (de Villiers)")
    axL.plot(a, diss["competitive"], color="0.45", ls="--", marker="s", ms=5,
             label="competitive (Deluty)")
    axL.set_title("Concurrent choice (chamber):\npunishment suppression vs. alternative")
    axL.set_xlabel("alternative reinforcement rate (1/VI)")
    axL.set_ylabel("punishment suppression (log-odds)")
    axL.set_ylim(bottom=0)
    axL.legend(frameon=False)

    c = np.asarray(forage["c"])
    axR.plot(c, forage["food"], color="black", ls="-", marker="o", ms=5, label="food consumed")
    axR.plot(c, forage["danger_contacts"], color="0.45", ls="--", marker="s", ms=5,
             label="danger contacts")
    ax2 = axR.twinx()
    ax2.plot(c, forage["avoid_weight"], color="0.0", ls=":", marker="^", ms=5,
             label="learned avoidance")
    ax2.set_ylabel("learned avoidance weight")
    axR.set_title("Open foraging (survival):\napproach-avoidance vs. punishment sensitivity")
    axR.set_xlabel("punishment sensitivity c (subtractive)")
    axR.set_ylabel("per-life count")
    lines = axR.get_lines() + ax2.get_lines()
    axR.legend(lines, [ln.get_label() for ln in lines], frameon=False, loc="upper center")

    fig.suptitle("Reinforcement/punishment asymmetry in two preparations", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = FIG / "asymmetry_both_preparations.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    diss = chamber_dissociation()
    forage = foraging_gradient()

    print("Chamber -- de Villiers vs Deluty (log-odds suppression vs alt. reinf. rate):")
    print("  alt rate: " + "  ".join(f"{r:.3f}" for r in diss["alt_rates"]))
    print("  subtractive: " + "  ".join(f"{v:+.2f}" for v in diss["subtractive"]) + "  (rises)")
    print("  competitive: " + "  ".join(f"{v:+.2f}" for v in diss["competitive"]) + "  (falls)")
    print("\nForaging -- approach-avoidance vs punishment sensitivity c:")
    print("  c:               " + "  ".join(f"{x:.0f}" for x in forage["c"]))
    print("  food/life:       " + "  ".join(f"{x:.1f}" for x in forage["food"]))
    print("  danger/life:     " + "  ".join(f"{x:.1f}" for x in forage["danger_contacts"]))
    print("  avoidance weight:" + "  ".join(f"{x:.2f}" for x in forage["avoid_weight"]))

    (FIG / "summary.json").write_text(json.dumps({"chamber": diss, "foraging": forage}, indent=2))
    p = figure(diss, forage)
    print(f"\nSaved {p} + summary.json")


if __name__ == "__main__":
    main()
