"""exp040 -- behavioral contrast: does responding in an unchanged component shift?

In a two-component multiple schedule, behavioral contrast is a change in responding in one component
when the OTHER component's schedule changes, in the opposite direction (positive contrast: rate in
the unchanged component rises when the other component is worsened, e.g. extinguished).

This tests whether the engine produces it, and isolates the route. The chamber's per-component
response value and Pavlovian context are LOCAL (each updates only while its component is present,
from its own reinforcement) -- there is no relative-rate or cross-component term -- so
contrast is not expected. The one channel shared across components is energy (deprivation): a
worsened component lowers total intake and raises motivation, lifting responding everywhere.
We separate them by clamping energy (motivation fixed -> associative route) versus leaving it
free (shared-deprivation route).

Protocol: both components on VI baseline, then component B is extinguished while A is unchanged.
Report A's rate (the contrast measure) and B's rate, baseline vs phase 2.

Run:  python experiments/exp040_behavioral_contrast.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from behavioral_md.chamber import ChamberConfig, run_contrast

FIG = Path("outputs/figures")
VI_BASELINE = 10.0
N_ORG = 400
COMP_STEPS = 300
N_BASELINE = 15
N_PHASE2 = 12
SEED = 0


def _config() -> ChamberConfig:
    return ChamberConfig(
        motiv_strength=2.0, energy_init=0.5, emission_bias=1.2, temperature=0.5,
        ctx_drive_gain=0.8, momentum_mass_gain=0.0, reinf_asymptote=1.0,
    )


def _rates(res: dict) -> dict:
    """Mean press rate over the last 3 sessions of each phase, per component."""
    pr = res["press_rate"]
    nb = res["n_baseline"]
    a, b = res["other"], res["changed"]
    return {
        "A_base": pr[nb - 3:nb, a].mean(), "A_phase2": pr[nb:, a][-3:].mean(),
        "B_base": pr[nb - 3:nb, b].mean(), "B_phase2": pr[nb:, b][-3:].mean(),
    }


def main() -> None:
    cfg = _config()
    rows = {}
    for clamp in (True, False):
        res = run_contrast(VI_BASELINE, cfg, N_ORG, COMP_STEPS, N_BASELINE, N_PHASE2,
                           changed=1, manipulation="extinction", clamp_energy=clamp, seed=SEED)
        rows[clamp] = _rates(res)

    print("Behavioral contrast: extinguish component B after baseline; does A shift?\n")
    print(f"{'condition':28s} A_base  A_phase2  A_ratio   B_base  B_phase2")
    for clamp, label in [(True, "energy clamped (associative)"),
                         (False, "energy free (deprivation)")]:
        r = rows[clamp]
        ratio = r["A_phase2"] / r["A_base"] if r["A_base"] > 0 else float("nan")
        print(f"  {label:26s} {r['A_base']:.3f}   {r['A_phase2']:.3f}    {ratio:.2f}     "
              f"{r['B_base']:.3f}   {r['B_phase2']:.3f}")
    print("\n  A_ratio > 1 = positive contrast in the unchanged component.")
    print("  Associative route (clamped) is expected ~1.0 (no cross-component term);")
    print("  any contrast comes from the shared-deprivation route (energy free).")

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    x = np.arange(2)
    w = 0.35
    for off, clamp, lab, col in [(-w / 2, True, "energy clamped (associative)", "0.6"),
                                 (w / 2, False, "energy free (deprivation)", "tab:red")]:
        r = rows[clamp]
        ax.bar(x + off, [r["A_base"], r["A_phase2"]], w, color=col, label=lab)
    ax.set_xticks(x)
    ax.set_xticklabels(["A baseline", "A after B extinguished"])
    ax.set_ylabel("press rate in component A (unchanged)")
    ax.set_title("exp040: behavioral contrast only via shared deprivation, not the local mechanism")
    ax.legend(fontsize=9)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    out = FIG / "exp040_behavioral_contrast.png"
    fig.savefig(out, dpi=130)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
