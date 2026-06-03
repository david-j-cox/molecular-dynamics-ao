"""Four processes, one phenomenon: a model-mimicry study of resurgence.

Resurgence -- recovery of a previously extinguished target response when a more
recently reinforced alternative is itself extinguished -- is produced in this engine
by (at least) four mechanistically distinct processes, all running through the
IDENTICAL three-phase preparation (chamber.run_resurgence):

  1. local choice (single)  -- delta-rule response value + softmax (local matching).
  2. behavioral momentum    -- reinforcement-history mass slows the target's decay.
  3. dual exc/inhibitory    -- extinction builds a separate inhibition; excitation
                               (and thus a latent target strength) is preserved.
  4. resurgence as choice   -- molar: a temporally-weighted reinforcement value with
                               matching allocation (Shahan & Craig, 2017).

This script (a) shows all four reproduce the canonical resurgence curve -- so the basic
result cannot discriminate them (model mimicry) -- and (b) runs two parametric
dissociations (alternative-reinforcement rate in phase 2; target-reinforcement rate in
phase 1) where the mechanisms make DIFFERENT predictions, motivating the diagnostic
experiments catalogued in README.md.

Run:   python studies/resurgence_mechanisms/compare_mechanisms.py
Saves: studies/resurgence_mechanisms/figures/*.png
       studies/resurgence_mechanisms/figures/summary.json
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from behavioral_md.chamber import ChamberConfig, run_resurgence  # noqa: E402

FIG = Path(__file__).parent / "figures"
N_ORG = 500
PHASE_STEPS = 2500
BLOCK = 50
R_OTHER = 0.2
SEED = 0

# Shared knobs for the local/associative arms (single, momentum, dual).
_LOCAL = dict(learning_rate=0.10, value_extinction=0.02, approach_gain=4.0,
              temperature=0.5, act_tau=3.0)

# The four mechanisms as ChamberConfig kwargs (same preparation, different process).
MECHANISMS = {
    "local choice": dict(value_rule="single", momentum_mass_gain=0.0, **_LOCAL),
    "behavioral momentum": dict(value_rule="single", momentum_mass_gain=8.0, **_LOCAL),
    "dual exc/inhib": dict(value_rule="dual", inhib_rate=0.06, inhib_relax=0.12,
                           inhib_passive_decay=0.005, **_LOCAL),
    "resurgence as choice": dict(value_rule="rac", rac_tau=500.0, rac_bump=0.04,
                                 rac_sensitivity=1.0, rac_floor=0.1),
}

plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                     "axes.labelsize": 13, "xtick.labelsize": 10,
                     "ytick.labelsize": 10, "legend.fontsize": 10})
_STYLES = [("-", "o", "black"), ("--", "s", "0.35"), ("-.", "^", "0.5"), (":", "D", "0.0")]


def _run(kw, vi_r1=5.0, vi_r2=5.0, control=False, n_org=N_ORG, phase_steps=PHASE_STEPS):
    return run_resurgence(ChamberConfig(**kw), n_org, phase_steps, seed=SEED,
                          vi_r1=vi_r1, vi_r2=vi_r2, r_other=R_OTHER, block=BLOCK,
                          control_reinforce_r2=control)


def _resurgence_index(res) -> tuple[float, float]:
    """(suppressed level entering test, peak target responding during test)."""
    pb = res["phase_blocks"]
    r1 = np.asarray(res["r1"])
    return float(r1[2 * pb - 3:2 * pb].mean()), float(r1[2 * pb:].max())


def figure_mimicry() -> dict:
    """2x2: all four mechanisms reproduce the canonical resurgence curve."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    summary = {}
    for ax, (name, kw) in zip(axes.ravel(), MECHANISMS.items(), strict=True):
        res = _run(kw)
        pb = res["phase_blocks"]
        r1, r2 = np.asarray(res["r1"]), np.asarray(res["r2"])
        b = np.arange(len(r1))
        ax.plot(b, r1, color="black", lw=1.6, label="target (R1)")
        ax.plot(b, r2, color="0.55", ls="--", lw=1.4, label="alternative (R2)")
        for i in (1, 2):
            ax.axvline(i * pb - 0.5, color="0.7", ls=":", lw=0.9)
        end_p2, peak_p3 = _resurgence_index(res)
        ax.set_title(f"{name}   (resurgence +{peak_p3 - end_p2:.2f})", fontsize=12)
        ax.set_ylim(0, 1.0)
        summary[name] = {"end_phase2": end_p2, "peak_phase3": peak_p3,
                         "resurgence": peak_p3 - end_p2}
    for ax in axes[-1]:
        ax.set_xlabel("block")
    for ax in axes[:, 0]:
        ax.set_ylabel("response allocation")
    axes[0, 0].legend(loc="upper right", frameon=False)
    fig.suptitle("Four processes, one phenomenon: resurgence under an identical preparation",
                 fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path = FIG / "mimicry_four_mechanisms.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return summary


def figure_dissociation() -> dict:
    """Two parametric sweeps where the mechanisms' predictions diverge.

    Left:  resurgence magnitude vs ALTERNATIVE reinforcement rate (phase-2 richness),
           target training held fixed. Right: vs TARGET reinforcement rate (phase-1
           richness), alternative held fixed. Reinforcement rate ~ 1/VI; we sweep VI
           and plot against 1/VI so the x-axis is 'richer to the right'.
    """
    vis = [40.0, 20.0, 10.0, 5.0, 3.0]
    rates = [1.0 / v for v in vis]
    out = {"alt_rate_sweep": {}, "target_rate_sweep": {}}

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for (sty, mk, col), (name, kw) in zip(_STYLES, MECHANISMS.items(), strict=True):
        alt = []
        for v in vis:                                   # vary alternative (phase-2) VI
            e, pk = _resurgence_index(_run(kw, vi_r1=5.0, vi_r2=v))
            alt.append(pk - e)
        tgt = []
        for v in vis:                                   # vary target (phase-1) VI
            e, pk = _resurgence_index(_run(kw, vi_r1=v, vi_r2=5.0))
            tgt.append(pk - e)
        out["alt_rate_sweep"][name] = alt
        out["target_rate_sweep"][name] = tgt
        axes[0].plot(rates, alt, ls=sty, marker=mk, color=col, label=name)
        axes[1].plot(rates, tgt, ls=sty, marker=mk, color=col, label=name)

    axes[0].set_title("vary ALTERNATIVE reinforcement rate\n(phase 2; target training fixed)")
    axes[0].set_xlabel("alternative reinforcement rate (1/VI)")
    axes[1].set_title("vary TARGET reinforcement rate\n(phase 1; alternative fixed)")
    axes[1].set_xlabel("target reinforcement rate (1/VI)")
    for ax in axes:
        ax.set_ylabel("resurgence magnitude")
        ax.axhline(0, color="0.85", lw=0.6)
    axes[1].legend(loc="upper left", frameon=False, bbox_to_anchor=(1.02, 1.0))
    fig.tight_layout()
    path = FIG / "dissociation_reinforcement_rate.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    mimicry = figure_mimicry()
    print("Resurgence under the identical preparation (model mimicry):")
    for name, d in mimicry.items():
        print(f"  {name:22s} suppressed={d['end_phase2']:.3f} -> peak={d['peak_phase3']:.3f}"
              f"   resurgence={d['resurgence']:+.3f}")

    dissoc = figure_dissociation()
    print("\nDissociation 1 -- resurgence vs ALTERNATIVE reinforcement rate (rich -> right):")
    for name, vals in dissoc["alt_rate_sweep"].items():
        print(f"  {name:22s} " + "  ".join(f"{x:+.2f}" for x in vals))
    print("Dissociation 2 -- resurgence vs TARGET reinforcement rate (rich -> right):")
    for name, vals in dissoc["target_rate_sweep"].items():
        print(f"  {name:22s} " + "  ".join(f"{x:+.2f}" for x in vals))

    (FIG / "summary.json").write_text(json.dumps(
        {"mimicry": mimicry, "dissociation": dissoc,
         "alt_vis": [40, 20, 10, 5, 3], "target_vis": [40, 20, 10, 5, 3]}, indent=2))
    print(f"\nSaved figures + summary.json under {FIG}")


if __name__ == "__main__":
    main()
