"""exp045 -- approach B in a 2D arena: the bridge in the real spatial engine.

The molecular-molar bridge realized as a 2D forager: a SAFE patch (constant intake) and a RISKY one
(variable, matched mean) at fixed locations, a day of foraging then an overnight fast that sets the
survival requirement R = night drain, energy/death (starvation + predation), and learning from a
period-scale SURVIVAL signal credited through an eligibility trace conditioned on energy and
time-of-day. Movement uses the engine's geometry (drives pull the organism toward each patch with a
Shepard-style distance falloff; the organism heads to the higher-drive patch).

Two questions, closing the exp036/exp037 thread:

1. EMISSION. exp036 showed the per-step softmax-over-Verlet-activations choice under-builds (one
   damped-Verlet step is tiny). Does committing to a spatial approach rescue it? We compare the real
   verlet_update + softmax movement atoms against the drive readout (the movement atom's steady
   state). Finding: even in 2D the raw Verlet+softmax under-expresses (the 4-way navigation re-adds
   noise faster than commitment builds the activation); the drive readout is the faithful emission.

2. THE RULE. With the drive readout, does the 2D forager show the energy-budget rule? Yes: without
   spatial travel (intake every step from the chosen patch) it is clearly risk-prone below R. WITH
   travel (intake only on patch contact) the rule WEAKENS -- a failed risky trip wastes the round
   trip, lethal when low -- the same travel-cost effect that fully inverts the rule in 1D (exp037);
   in this 2D geometry it pushes toward inversion without, at these parameters, fully flipping.

Run:  python experiments/exp045_spatial_survival_2d.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from behavioral_md.atoms import verlet_update
from behavioral_md.chamber import ChamberConfig, run_risk_choice

FIG = Path("outputs/figures")
S = 0.10
SAFE = [(1.0, S)]
RISKY = [(0.5, 0.0), (0.5, 2 * S)]
SAFE_POS = np.array([-0.7, 0.4])
RISKY_POS = np.array([0.7, 0.4])
START = np.array([0.0, -0.6])


def run_2d(*, emission: str, travel: bool, n_org=3000, n_cycles=500, seed=0, cap=1.0, e_init=0.5,
           day_steps=44, night_steps=12, step_cost=0.024, night_cost=0.025,
           ebins=10, tbins=3, lr=0.15, decay=0.92, T=0.05,
           sens=0.3, mu=2.0, p=2.0, c=10.0, mass=1.0, dt=0.1,
           rng_range=1.2, move_speed=0.18, contact=0.2, measure_last=120,
           predation_threshold=0.55, predation_prob=0.12):
    """2D forager. emission='drive' (steady-state readout) or 'verlet' (verlet_update+softmax)."""
    rng = np.random.default_rng(seed)
    rp = np.array([o[0] for o in RISKY])
    rd = np.array([o[1] for o in RISKY])
    oi = np.arange(n_org)
    patches = np.stack([SAFE_POS, RISKY_POS])
    W = np.zeros((n_org, ebins, tbins, 2))
    elig = np.zeros((n_org, ebins, tbins, 2))
    pos = np.tile(START, (n_org, 1)).astype(float)
    m = np.zeros((n_org, 2))
    m_prev = np.zeros((n_org, 2))   # toward-safe / toward-risky atoms
    E = np.full(n_org, e_init)
    risky_feeds = np.zeros(ebins)
    feeds = np.zeros(ebins)

    def credit(maskd, target):
        if maskd.any():
            W[maskd] += lr * elig[maskd] * (target - W[maskd])
            elig[maskd] = 0.0

    def predation(e):
        return (e > predation_threshold) & (rng.random(n_org) < predation_prob)

    def dead_reset(dead):
        credit(dead, 0.0)
        E[dead] = e_init
        pos[dead] = START
        m[dead] = 0.0
        m_prev[dead] = 0.0

    for cyc in range(n_cycles):
        measuring = cyc >= n_cycles - measure_last
        for ds in range(day_steps):
            tb = min(int(ds / day_steps * tbins), tbins - 1)
            eb = np.clip((E / cap * ebins).astype(int), 0, ebins - 1)
            gain = mu * np.clip(1.0 - E / cap, 0.0, 1.0) ** p
            diff = patches[None, :, :] - pos[:, None, :]
            dist = np.linalg.norm(diff, axis=2) + 1e-9
            unit = diff / dist[:, :, None]
            drive = (sens + W[oi, eb, tb, :] + gain[:, None]) * np.exp(-dist / rng_range)  # [O,2]
            if emission == "verlet":
                vel = (m - m_prev) / dt
                m_new = np.clip(verlet_update(m, m_prev, drive - c * vel, mass, dt), -10.0, 10.0)
                m_prev, m = m, m_new
                q = m / T
            else:  # drive readout (steady state of the damped accumulator)
                q = drive / T
            q -= q.max(axis=1, keepdims=True)
            ez = np.exp(q)
            choose_risky = rng.random(n_org) < ez[:, 1] / ez.sum(axis=1)
            target = np.where(choose_risky[:, None], unit[:, 1, :], unit[:, 0, :])
            pos = np.clip(pos + move_speed * target, -1.0, 1.0)
            elig *= decay
            elig[oi, eb, tb, choose_risky.astype(int)] += 1.0
            d2 = np.linalg.norm(patches[None, :, :] - pos[:, None, :], axis=2)
            at_safe = d2[:, 0] < contact
            at_risky = d2[:, 1] < contact
            ridx = (rng.random(n_org)[:, None] < np.cumsum(rp)[None, :]).argmax(axis=1)
            if travel:
                intake = np.zeros(n_org)
                intake[at_safe] = S
                intake[at_risky] = rd[ridx][at_risky]
                fed = at_safe | at_risky
                got_risky = at_risky
            else:
                intake = np.where(choose_risky, rd[ridx], S)
                fed = np.ones(n_org, bool)
                got_risky = choose_risky
            if measuring and fed.any():
                fe = eb[fed]
                np.add.at(feeds, fe, 1.0)
                np.add.at(risky_feeds, fe, got_risky[fed].astype(float))
            E = np.clip(E + intake - step_cost, 0.0, cap)
            if travel:
                pos[fed] = START
                m[fed] = 0.0
                m_prev[fed] = 0.0
            dead_reset((E <= 0.0) | predation(E))

        for _ in range(night_steps):
            E = np.clip(E - night_cost, 0.0, cap)
            elig *= decay
            dead_reset((E <= 0.0) | predation(E))
        credit(np.ones(n_org, bool), 1.0)

    bins = (np.arange(ebins) + 0.5) / ebins * cap
    curve = np.divide(risky_feeds, feeds, out=np.full(ebins, np.nan), where=feeds > 0)
    return curve, bins


def _rev(curve, bins, R):
    return np.nanmean(curve[bins < R]) - np.nanmean(curve[(bins >= R) & (bins < 0.9)])


def main() -> None:
    R = 0.025 * 12
    cfg = ChamberConfig(temperature=0.02)
    imp = run_risk_choice(SAFE, RISKY, cfg, 4000, 1000, seed=0, cost=0.10, e_init=0.5, e_req=R)
    imp_curve = np.asarray(imp["risky_by_energy"], float)
    imp_bins = np.asarray(imp["energy_bins"], float)

    drive_nt, bins = run_2d(emission="drive", travel=False, seed=0)
    drive_tr, _ = run_2d(emission="drive", travel=True, seed=0)
    verlet_nt, _ = run_2d(emission="verlet", travel=False, seed=0)

    print(f"2D real-engine forager, P(feed risky) reversal (below - above R={R:.2f}):\n")
    print(f"  imposed reference (exp030)          {_rev(imp_curve, imp_bins, R):+.3f}")
    print(f"  drive readout, no travel  -> RULE   {_rev(drive_nt, bins, R):+.3f}")
    print(f"  drive readout, with travel-> WEAKEN {_rev(drive_tr, bins, R):+.3f}")
    print(f"  raw verlet+softmax, no travel       {_rev(verlet_nt, bins, R):+.3f}  "
          f"(under-builds -- exp036 limit persists in 2D)")

    plt.figure(figsize=(7.2, 4.5))
    plt.axvline(R, color="0.7", ls="--", lw=1, label=f"requirement R={R:.2f}")
    plt.axhline(0.5, color="0.9", lw=1)
    plt.plot(imp_bins, imp_curve, "o-", color="tab:blue", label="imposed utility (exp030)")
    plt.plot(bins, drive_nt, "^-", color="tab:green", label="drive readout, no travel: rule")
    plt.plot(bins, drive_tr, "s-", color="tab:red", label="drive readout + travel: weakened")
    plt.plot(bins, verlet_nt, "x--", color="0.6", label="raw verlet+softmax: under-builds")
    plt.xlabel("current energy reserve E")
    plt.ylabel("P(feed risky)")
    plt.title("exp045: approach B in a 2D arena (real engine geometry)")
    plt.ylim(-0.02, 1.02)
    plt.legend(fontsize=8)
    plt.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    out = FIG / "exp045_spatial_survival_2d.png"
    plt.savefig(out, dpi=130)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
