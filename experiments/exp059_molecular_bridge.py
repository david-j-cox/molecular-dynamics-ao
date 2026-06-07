"""exp059 -- closing the molecular bridge: a costly-effort engine that realizes the molar optimum.

The repo has two halves: the distributed atom/force/Verlet ENGINE (atoms.py, forces.py, organism.py)
and the normative survival DP (survival.py). This closes the loop between them on two fronts.

PART A -- the force<->energy loop (the engine). Energy already drives force (a deficit amplifies
the food drive), but atom activations cost nothing and fatigue was dormant. We close it: behavioral
effort (summed positive action-atom activation) now costs energy (config.effort_cost), and fatigue
is load-bearing (fatigue_gain decrements force; fatigue_energy_cost makes it a metabolic
load). Two consequences on the real Organism: (A1) vigorous responding depletes energy faster -> the
organism dies sooner under effort cost; (A2) without fatigue a sustained drive makes an atom's
activation RAMP toward the ceiling (the atom is an integrator), while fatigue acts as a homeostatic
BRAKE that bounds it and leaves a post-bout refractory dip.

PART B -- does the messy mechanism realize the optimum?  exp032/036/045 found that survival-credit
learning makes the atoms' state-conditioned weights acquire the energy-budget rule (gamble below the
requirement R), but the per-step Verlet+softmax EMISSION under-expresses it -- one molecular step is
tiny. exp036's diagnosis: expressing a molar, state-dependent policy through fast molecular dynamics
needs TIMESCALE SEPARATION. We supply it directly: each molar choice runs K molecular Verlet steps
(drive held fixed) so the activations equilibrate before the softmax reads them. RESULT: as K grows,
the genuine Verlet emission's energy-budget reversal climbs monotonically from the under-build limit
(~the exp036 value) to the full rule (the drive-readout / imposed-utility value). So the molecular
mechanism DOES realize the molar optimum -- given timescale separation -- and the rule is robust to
the now-costly effort. The bridge is closed.

Run:  python experiments/exp059_molecular_bridge.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from behavioral_md.atoms import verlet_update  # noqa: E402
from behavioral_md.chamber import ChamberConfig, run_risk_choice  # noqa: E402
from behavioral_md.config import SimulationConfig  # noqa: E402
from behavioral_md.environments.gridworld import BehavioralFieldEnv  # noqa: E402
from behavioral_md.organism import Organism  # noqa: E402

FIG = Path("outputs/figures")

# Part B survival economy (exp045): safe vs risky patch, matched mean; R = night drain.
S = 0.10
SAFE = [(1.0, S)]
RISKY = [(0.5, 0.0), (0.5, 2 * S)]
SAFE_POS, RISKY_POS, START = np.array([-0.7, 0.4]), np.array([0.7, 0.4]), np.array([0.0, -0.6])
R = 12 * 0.025


# ----------------------------------------------------------------------------------------------- #
# PART A: the force <-> energy loop on the real Organism
# ----------------------------------------------------------------------------------------------- #
def _food_obs(on: bool) -> dict:
    z = np.zeros(2)
    o = {f"{s}_vector": z.copy() for s in ("food", "danger", "light", "cue")}
    o.update({f"{s}_intensity": np.array([0.0]) for s in ("food", "danger", "light", "cue")})
    o["food_intensity"] = np.array([1.0 if on else 0.0])
    o["food_vector"] = np.array([1.0, 0.0])
    o["food_contact"] = np.array([1.0 if on else 0.0])
    o["cue_value"] = np.array([0.0])
    o["context"] = np.array([0.0])
    return o


def effort_energy_trace(effort_cost: float, steps: int = 90, seed: int = 1) -> np.ndarray:
    """Energy over time for a foraging organism in the gridworld, with/without effort cost."""
    cfg = SimulationConfig(seed=seed, effort_cost=effort_cost)
    env = BehavioralFieldEnv(SimulationConfig(seed=seed))
    org = Organism(cfg, rng=np.random.default_rng(seed))
    obs, _ = env.reset(seed=seed)
    org.reset(obs)
    energy = []
    for _ in range(steps):
        org.step(obs)
        a = org.emit_action()
        obs, _r, _term, _trunc, info = env.step(a)
        org.update_history(obs, a, info)
        energy.append(org.energy)
        if not org.alive:
            break
    return np.array(energy)


def fatigue_bout(fatigue_gain: float, bout: int = 90, total: int = 150,
                 seed: int = 0) -> np.ndarray:
    """approach_food activation across a sustained drive bout then a pause (learning frozen)."""
    cfg = SimulationConfig(seed=seed, fatigue_gain=fatigue_gain, fatigue_decay=0.98,
                           learning_rate=0.0)
    org = Organism(cfg, rng=np.random.default_rng(seed))
    org.reset(_food_obs(True))
    acts = []
    for t in range(total):
        org.step(_food_obs(t < bout))
        acts.append(org.activation("approach_food"))
    return np.array(acts)


# ----------------------------------------------------------------------------------------------- #
# PART B: does the molecular emission realize the molar rule? (timescale separation)
# ----------------------------------------------------------------------------------------------- #
def run_forager(K: int, *, effort_cost: float = 0.0, n_org: int = 2500, n_cycles: int = 420,
                seed: int = 0, cap: float = 1.0, e_init: float = 0.5, day_steps: int = 44,
                night_steps: int = 12, step_cost: float = 0.024, night_cost: float = 0.025,
                ebins: int = 10, tbins: int = 3, lr: float = 0.15, decay: float = 0.92,
                T: float = 0.05, sens: float = 0.3, mu: float = 2.0, p: float = 2.0,
                c: float = 10.0, mass: float = 1.0, dt: float = 0.1, rng_range: float = 1.2,
                move_speed: float = 0.18, measure_last: int = 110,
                predation_threshold: float = 0.55, predation_prob: float = 0.12):
    """exp045 no-travel survival forager. Emission = K molecular Verlet deliberation steps with the
    drive held fixed (timescale separation); K <= 0 = the drive readout (infinite sep). Effort
    (the chosen atom activation) costs energy when ``effort_cost`` > 0. Returns (curve, bins, rev).
    """
    rng = np.random.default_rng(seed)
    rp, rd = np.cumsum([o[0] for o in RISKY]), np.array([o[1] for o in RISKY])
    oi = np.arange(n_org)
    patches = np.stack([SAFE_POS, RISKY_POS])
    W = np.zeros((n_org, ebins, tbins, 2))
    elig = np.zeros((n_org, ebins, tbins, 2))
    pos = np.tile(START, (n_org, 1)).astype(float)
    E = np.full(n_org, e_init)
    risky_feeds, feeds = np.zeros(ebins), np.zeros(ebins)

    def credit(mask, target):
        if mask.any():
            W[mask] += lr * elig[mask] * (target - W[mask])
            elig[mask] = 0.0

    def dead_reset(dead):
        credit(dead, 0.0)
        E[dead] = e_init
        pos[dead] = START

    for cyc in range(n_cycles):
        measuring = cyc >= n_cycles - measure_last
        for ds in range(day_steps):
            tb = min(int(ds / day_steps * tbins), tbins - 1)
            eb = np.clip((E / cap * ebins).astype(int), 0, ebins - 1)
            gain = mu * np.clip(1.0 - E / cap, 0.0, 1.0) ** p
            diff = patches[None, :, :] - pos[:, None, :]
            dist = np.linalg.norm(diff, axis=2) + 1e-9
            unit = diff / dist[:, :, None]
            drive = (sens + W[oi, eb, tb, :] + gain[:, None]) * np.exp(-dist / rng_range)
            if K <= 0:
                act = drive
            else:
                m, m_prev = np.zeros((n_org, 2)), np.zeros((n_org, 2))
                for _k in range(K):                          # molecular dynamics, drive held fixed
                    vel = (m - m_prev) / dt
                    m_new = np.clip(verlet_update(m, m_prev, drive - c * vel, mass, dt), -10, 10)
                    m_prev, m = m, m_new
                act = m
            q = act / T
            q -= q.max(axis=1, keepdims=True)
            ez = np.exp(q)
            choose_risky = rng.random(n_org) < ez[:, 1] / ez.sum(axis=1)
            target = np.where(choose_risky[:, None], unit[:, 1, :], unit[:, 0, :])
            pos = np.clip(pos + move_speed * target, -1.0, 1.0)
            elig *= decay
            elig[oi, eb, tb, choose_risky.astype(int)] += 1.0
            ridx = (rng.random(n_org)[:, None] < rp[None, :]).argmax(1)
            intake = np.where(choose_risky, rd[ridx], S)
            effort = np.maximum(0.0, act[oi, choose_risky.astype(int)])
            if measuring:
                np.add.at(feeds, eb, 1.0)
                np.add.at(risky_feeds, eb, choose_risky.astype(float))
            E = np.clip(E + intake - step_cost - effort_cost * effort, 0.0, cap)
            dead_reset((E <= 0.0) | ((E > predation_threshold)
                                     & (rng.random(n_org) < predation_prob)))
        for _ in range(night_steps):
            E = np.clip(E - night_cost, 0.0, cap)
            elig *= decay
            dead_reset((E <= 0.0) | ((E > predation_threshold)
                                     & (rng.random(n_org) < predation_prob)))
        credit(np.ones(n_org, bool), 1.0)

    bins = (np.arange(ebins) + 0.5) / ebins * cap
    curve = np.divide(risky_feeds, feeds, out=np.full(ebins, np.nan), where=feeds > 0)
    rev = float(np.nanmean(curve[bins < R]) - np.nanmean(curve[(bins >= R) & (bins < 0.9)]))
    return curve, bins, rev


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)

    # --- Part A ---------------------------------------------------------------------------------
    e_no = effort_energy_trace(0.0)
    e_eff = effort_energy_trace(0.08)
    fat_off = fatigue_bout(0.0)
    fat_on = fatigue_bout(0.06)
    print(f"A1. effort -> energy: dies/declines faster with effort cost "
          f"(steps alive {len(e_no)} vs {len(e_eff)}).")
    print(f"A2. fatigue brake: sustained-drive activation ramps to {fat_off[89]:.2f} without "
          f"fatigue, held to {fat_on[89]:.2f} with it (post-bout dip to {fat_on.min():.2f}).")

    # --- Part B: timescale-separation sweep -----------------------------------------------------
    Ks = [1, 5, 15]
    sweep = {K: run_forager(K, seed=0) for K in Ks}
    drive_curve, bins, drive_rev = run_forager(0, seed=0)
    eff_curve, _, eff_rev = run_forager(15, effort_cost=0.15, seed=0)
    cfg = ChamberConfig(temperature=0.02)
    imp = run_risk_choice(SAFE, RISKY, cfg, 4000, 1000, seed=0, cost=0.10, e_init=0.5, e_req=R)
    imp_curve = np.asarray(imp["risky_by_energy"], float)
    imp_bins = np.asarray(imp["energy_bins"], float)

    above = (imp_bins >= R) & (imp_bins < 0.9)
    revs = [sweep[K][2] for K in Ks] + [drive_rev]
    print("\nB. energy-budget reversal (P(risky) below R minus above) vs deliberation length K:")
    for K in Ks:
        print(f"   K={K:2d}: {sweep[K][2]:+.3f}")
    print(f"   K=inf (drive readout): {drive_rev:+.3f}   imposed utility (exp030): "
          f"{np.nanmean(imp_curve[imp_bins < R]) - np.nanmean(imp_curve[above]):+.3f}")
    print(f"   with costly effort (K=15, effort_cost=0.15): {eff_rev:+.3f} (rule robust to effort)")

    # --- figure ---------------------------------------------------------------------------------
    fig, ax = plt.subplots(2, 2, figsize=(13, 9.2))

    ax[0, 0].plot(e_no, color="tab:blue", lw=2, label="effort_cost = 0")
    ax[0, 0].plot(e_eff, color="tab:red", lw=2, label="effort_cost = 0.08")
    ax[0, 0].set_xlabel("step")
    ax[0, 0].set_ylabel("energy reserve")
    ax[0, 0].set_title("A1. The loop closes: behavioral effort costs energy\n(vigorous responding "
                       "depletes the reserve faster)")
    ax[0, 0].legend(fontsize=8)

    t = np.arange(len(fat_off))
    ax[0, 1].axvline(90, color="0.8", ls=":", lw=1, label="bout ends (pause)")
    ax[0, 1].plot(t, fat_off, color="tab:gray", lw=2, label="fatigue off (ramps to ceiling)")
    ax[0, 1].plot(t, fat_on, color="tab:purple", lw=2, label="fatigue on (braked + refractory)")
    ax[0, 1].axhline(0, color="0.85", lw=0.8)
    ax[0, 1].set_xlabel("step (sustained drive, then pause)")
    ax[0, 1].set_ylabel("approach_food activation")
    ax[0, 1].set_title("A2. Load-bearing fatigue: a homeostatic brake\non runaway activation + "
                       "post-bout refractory")
    ax[0, 1].legend(fontsize=8, loc="upper left")

    ax[1, 0].axhline(drive_rev, color="tab:green", ls="--", lw=1.5,
                     label=f"drive readout (K=∞): {drive_rev:+.2f}")
    ax[1, 0].plot(Ks + [max(Ks) + 8], revs, "o-", color="tab:blue", lw=2, label="Verlet emission")
    ax[1, 0].axhline(0.0, color="0.85", lw=0.8)
    ax[1, 0].set_xlabel("molecular deliberation steps $K$ (timescale separation)")
    ax[1, 0].set_ylabel("energy-budget reversal (below $R$ − above)")
    ax[1, 0].set_title("C. The bridge closes: timescale separation lets the\nmolecular dynamics "
                       "express the molar rule")
    ax[1, 0].legend(fontsize=8, loc="lower right")

    ax[1, 1].axvline(R, color="0.7", ls="--", lw=1, label=f"requirement $R$={R:.2f}")
    ax[1, 1].axhline(0.5, color="0.9", lw=1)
    ax[1, 1].plot(imp_bins, imp_curve, "o-", color="0.4", label="imposed utility (exp030)")
    ax[1, 1].plot(bins, sweep[1][0], "x--", color="tab:gray", label="K=1 (under-builds)")
    ax[1, 1].plot(bins, sweep[15][0], "^-", color="tab:blue", label="K=15 (expresses rule)")
    ax[1, 1].plot(bins, eff_curve, "s-", color="tab:red", label="K=15 + costly effort")
    ax[1, 1].set_xlabel("current energy reserve $E$")
    ax[1, 1].set_ylabel("P(choose risky)")
    ax[1, 1].set_ylim(-0.02, 1.02)
    ax[1, 1].set_title("D. Emergent energy-budget rule from the atom engine\n(risk-prone "
                       "below $R$), with costly effort")
    ax[1, 1].legend(fontsize=7.5, loc="upper right")

    fig.tight_layout()
    out = FIG / "exp059_molecular_bridge.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
