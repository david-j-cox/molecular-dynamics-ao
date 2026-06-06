"""exp041 -- anticipatory contrast: a learned predicted-income term, energy-budget sign.

Anticipatory contrast (Flaherty, Williams): responding in a component depends on the value of the
UPCOMING component. It is learned -- the animal discovers over sessions that this context precedes a
richer or leaner one -- not foreseen. The puzzle is the SIGN: animals respond LESS in a component
that precedes a richer one, which a plain conditioned-value term gets backwards (a stimulus before
reward should gain positive value -> more responding).

The energy budget gives the right sign. Each component learns, by temporal credit over the
experienced sequence, the income of the component that follows it (chamber antic_lr); that predicted
upcoming income DISCOUNTS present hunger (chamber antic_discount). If a feast is coming, current
urgency drops -> work less now (negative contrast before a rich component); if famine is coming,
urgency stays high -> work more now. Same convex survival logic as exp040, but on PREDICTED rather
than current energy.

To isolate the anticipatory route from the current-reserve route of exp040, energy is CLAMPED here
(current hunger fixed), so any shift in the unchanged component is purely anticipatory. Knocking out
the discount (antic_discount = 0) removes it entirely -- the evidence it is the learned predicted
income, nothing installed.

Run:  python experiments/exp041_anticipatory_contrast.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from behavioral_md.chamber import ChamberConfig, run_contrast

FIG = Path("outputs/figures")
ARGS = dict(n_org=400, comp_steps=300, n_baseline=20, n_phase2=15, changed=1,
            clamp_energy=True, seed=0)   # energy clamped -> isolates the anticipatory route


def _cfg(antic_discount: float) -> ChamberConfig:
    return ChamberConfig(motiv_strength=2.0, energy_init=0.5, emission_bias=1.2, temperature=0.5,
                         ctx_drive_gain=0.8, antic_lr=0.2, antic_discount=antic_discount)


def _ratio(res: dict) -> tuple[float, float]:
    pr, nb, a = res["press_rate"], res["n_baseline"], res["other"]
    return pr[nb - 3:nb, a].mean(), pr[nb:, a][-3:].mean()


def main() -> None:
    # lean B coming (extinguish) -> A should rise; rich B coming (enrich) -> A should fall.
    lean = {d: _ratio(run_contrast(14.0, _cfg(d), manipulation="extinction", **ARGS))
            for d in (0.4, 0.0)}
    rich = {d: _ratio(run_contrast(14.0, _cfg(d), manipulation="enrich", vi_phase2=3.5, **ARGS))
            for d in (0.4, 0.0)}

    print("Anticipatory contrast in component A (energy clamped -> purely anticipatory).\n")
    print(f"{'condition':40s} A_base  A_phase2  ratio")
    for label, (b, p) in [("lean B coming (extinguish), discount on", lean[0.4]),
                          ("lean B coming (extinguish), discount OFF", lean[0.0]),
                          ("rich B coming (enrich), discount on", rich[0.4]),
                          ("rich B coming (enrich), discount OFF", rich[0.0])]:
        print(f"  {label:38s} {b:.3f}   {p:.3f}    {p / b:.2f}")
    print("\n  Respond more before a lean component, less before a rich one (ratio >1 / <1), both")
    print("  gone with the discount off. Correct anticipatory sign from the energy budget, learned")
    print("  by temporal credit over the sequence -- with energy clamped, purely anticipatory.")

    fig, ax = plt.subplots(figsize=(8, 4.3))
    groups = ["lean component coming\n(respond more)", "rich component coming\n(respond less)"]
    on = [lean[0.4][1] / lean[0.4][0], rich[0.4][1] / rich[0.4][0]]
    off = [lean[0.0][1] / lean[0.0][0], rich[0.0][1] / rich[0.0][0]]
    x = np.arange(2)
    ax.bar(x - 0.18, on, 0.36, color="tab:purple", label="predicted-income discount on")
    ax.bar(x + 0.18, off, 0.36, color="0.7", label="discount off (knockout)")
    ax.axhline(1.0, color="0.5", ls="--", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=9)
    ax.set_ylabel("A rate ratio (phase 2 / baseline)")
    ax.set_title("exp041: anticipatory contrast from learned predicted income (energy clamped)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    out = FIG / "exp041_anticipatory_contrast.png"
    fig.savefig(out, dpi=130)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
