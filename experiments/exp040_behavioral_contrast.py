"""exp040 -- behavioral contrast from the shared energy budget (no relative-rate term installed).

In a two-component multiple schedule, behavioral contrast is a change in responding in one component
when the OTHER component's schedule changes. Here it is NOT installed as a relative-rate term; it
emerges because both components feed one body's energy reserve and the drive carries a CONVEX hunger
term, motiv_strength * (1 - E/E_cap)**deficit_exponent (the energy-budget marginal value of energy).
Worsen one component, total intake falls, the organism gets hungrier and works harder in the
still-paying component (positive contrast). Enrich one component, the organism is sated and works
less elsewhere (negative contrast). Nothing relative is computed; it follows from one shared body.

The convexity makes a prediction: g(E) rises steeply toward starvation but flattens near satiation,
so WHERE the baseline sits matters. Near satiation only worsening moves behavior (positive contrast
is robust); a hungry baseline is needed for enriching to bite (negative contrast). We show each at
the baseline that exposes it, and knock the effect out with motiv_strength = 0 -- with the convex
hunger term removed, the per-component value/context is purely local and there is no contrast. That
knockout is the evidence the effect is the shared hunger term, not anything installed.

Scope: this is the molar/energy account (current reserve). Anticipatory contrast, where responding
depends on the value of the UPCOMING component, is a learned sequential association -- see exp041.

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
ARGS = dict(n_org=400, comp_steps=300, n_baseline=15, n_phase2=12, changed=1,
            clamp_energy=False, seed=0)


def _cfg(motiv_strength: float, food_energy: float) -> ChamberConfig:
    return ChamberConfig(motiv_strength=motiv_strength, energy_init=0.5, emission_bias=1.2,
                         temperature=0.5, ctx_drive_gain=0.8, food_energy=food_energy,
                         deficit_exponent=2.0)


def _a_rates(res: dict) -> tuple[float, float]:
    """Press rate in the UNCHANGED component A, last 3 sessions of baseline vs phase 2."""
    pr, nb, a = res["press_rate"], res["n_baseline"], res["other"]
    return pr[nb - 3:nb, a].mean(), pr[nb:, a][-3:].mean()


def main() -> None:
    # Positive contrast at a sated baseline (worsen B); negative at a hungry baseline (enrich B).
    pos = {ms: _a_rates(run_contrast(14.0, _cfg(ms, 0.15), manipulation="extinction", **ARGS))
           for ms in (1.5, 0.0)}
    neg = {ms: _a_rates(run_contrast(20.0, _cfg(ms, 0.06), manipulation="enrich",
                                     vi_phase2=4.0, **ARGS))
           for ms in (1.5, 0.0)}

    print("Behavioral contrast in component A (unchanged) from the shared energy budget.\n")
    print(f"{'condition':34s} A_base  A_phase2  ratio")
    for label, (b, p) in [("positive (worsen B), hunger on", pos[1.5]),
                          ("positive (worsen B), hunger OFF", pos[0.0]),
                          ("negative (enrich B), hunger on", neg[1.5]),
                          ("negative (enrich B), hunger OFF", neg[0.0])]:
        print(f"  {label:32s} {b:.3f}   {p:.3f}    {p / b:.2f}")
    print("\n  Positive contrast (ratio > 1) at a sated baseline; negative (ratio < 1) at a hungry")
    print("  one; both vanish with hunger off -> the convex shared-hunger term carries it, not a")
    print("  relative-rate term. The asymmetry is the convex marginal value of energy.")

    fig, ax = plt.subplots(figsize=(8, 4.3))
    groups = ["positive contrast\n(worsen B, sated)", "negative contrast\n(enrich B, hungry)"]
    on = [pos[1.5][1] / pos[1.5][0], neg[1.5][1] / neg[1.5][0]]
    off = [pos[0.0][1] / pos[0.0][0], neg[0.0][1] / neg[0.0][0]]
    x = np.arange(2)
    ax.bar(x - 0.18, on, 0.36, color="tab:red", label="hunger on")
    ax.bar(x + 0.18, off, 0.36, color="0.7", label="hunger off (knockout)")
    ax.axhline(1.0, color="0.5", ls="--", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=9)
    ax.set_ylabel("A rate ratio (phase 2 / baseline)")
    ax.set_title("exp040: behavioral contrast from convex shared hunger, not an installed term")
    ax.legend(fontsize=9)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    out = FIG / "exp040_behavioral_contrast.png"
    fig.savefig(out, dpi=130)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
