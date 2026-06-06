"""exp047 -- VR >> VI from molar feedback sensitivity (Baum's correlation-based law of effect).

Variable-ratio (VR) schedules sustain higher response rates than variable-interval (VI) schedules
matched for reinforcement rate. The per-press value rule alone (an unreinforced press erodes the
press value) underproduces this: at matched reinforcement rate the response rate is dominated by the
energy deficit and the VR and VI press-rate curves nearly overlap (this experiment, feedback off).

The missing ingredient is sensitivity to the molar FEEDBACK FUNCTION -- how reinforcement rate
depends on the organism's own response rate. The chamber estimates the recent response->reinforcer
regression slope (cov(window rate, window reinf) / var(window rate), EMA-smoothed) and the positive
part of that slope boosts the press drive (`feedback_gain`). On VR, pressing more yields
proportionally more reinforcement (positive slope -> drive boosted -> high rate). On VI,
reinforcement is rate-independent and the homeostatic feed-pause cycle even makes the raw
correlation negative, so the clipped slope gives no boost -> lower rate. With the feedback term on,
the VR press-rate curve lies well above the VI curve at matched reinforcement rate (~1.7x vs ~1.0x).

Run:  python experiments/exp047_vr_vi_feedback.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from behavioral_md.chamber import ChamberConfig, run_chamber

FIG = Path("outputs/figures")
VR_NS = [12, 18, 28, 45, 80]
VI_TS = [15, 22, 35, 60, 110]
N_ORG = 600
N_STEPS = 6000
WARM = 2500


def _cfg(feedback_gain: float) -> ChamberConfig:
    return ChamberConfig(motiv_strength=0.6, energy_init=0.6, emission_bias=1.0, temperature=0.5,
                         value_extinction=0.03, feedback_gain=feedback_gain)


def _rate(sched, param, cfg):
    r = run_chamber(sched, param, cfg, N_ORG, N_STEPS, seed=0)
    return r["presses"][WARM:].mean(), r["reinforced"][WARM:].mean()


def _curves(cfg):
    vr = [(_rate("VR", n, cfg)) for n in VR_NS]
    vi = [(_rate("VI", t, cfg)) for t in VI_TS]
    return vr, vi


def _matched_ratio(vr, vi):
    """VR press rate / VI press rate at the VI reinforcement rate closest to each VR point."""
    vi_r = np.array([r for _, r in vi])
    vi_p = np.array([p for p, _ in vi])
    ratios = []
    for p_vr, r_vr in vr:
        j = int(np.argmin(np.abs(vi_r - r_vr)))
        if abs(vi_r[j] - r_vr) < 0.004:
            ratios.append(p_vr / vi_p[j])
    return float(np.mean(ratios)) if ratios else float("nan")


def main() -> None:
    off = _curves(_cfg(0.0))
    on = _curves(_cfg(50.0))
    print("VR vs VI press rate at matched reinforcement rate (VR/VI > 1 is the VR>VI effect):")
    print(f"  feedback OFF: mean VR/VI = {_matched_ratio(*off):.2f}")
    print(f"  feedback ON : mean VR/VI = {_matched_ratio(*on):.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, (vr, vi), title in [(axes[0], off, "feedback off (per-press value only)"),
                                (axes[1], on, "feedback on (response-reinforcer correlation)")]:
        rr_vr = [r for _, r in vr]
        pp_vr = [p for p, _ in vr]
        rr_vi = [r for _, r in vi]
        pp_vi = [p for p, _ in vi]
        ax.plot(rr_vr, pp_vr, "o-", color="tab:red", label="VR")
        ax.plot(rr_vi, pp_vi, "s-", color="tab:blue", label="VI")
        ax.set_xlabel("reinforcement rate (per step)")
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=9)
    axes[0].set_ylabel("press rate (per step)")
    fig.suptitle("exp047: VR >> VI emerges from molar feedback sensitivity", fontsize=13)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    out = FIG / "exp047_vr_vi_feedback.png"
    fig.savefig(out, dpi=130)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
