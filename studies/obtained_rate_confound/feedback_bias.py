"""Obtained punishment rates are endogenous: a feedback artifact that can flip the
sign of a fitted punishment sensitivity.

A punisher is collected only when the punished response is EMITTED. But punishment
suppresses that very response, so as the SCHEDULED punishment rate rises the OBTAINED
punishment rate first rises, then COLLAPSES (an inverted-U) once allocation dies. The
obtained rate is therefore a function of the behavior it is used to predict. Regressing
log response allocation on log OBTAINED punishment ratio -- the standard concatenated-
matching-law fit -- is thus endogenous and can recover a punishment sensitivity of the
WRONG SIGN. Using the SCHEDULED (programmed) rate, which is exogenous, recovers the
correct monotonic suppression.

Single target prep (chamber.run_punishment_choice): two responses equally reinforced
(VI 5), the target punished at a rising scheduled rate, the alternative unpunished.

Run:   python studies/obtained_rate_confound/feedback_bias.py
Saves: studies/obtained_rate_confound/figures/*.png + summary.json
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from behavioral_md.chamber import ChamberConfig, run_punishment_choice  # noqa: E402

FIG = Path(__file__).parent / "figures"
INF = float("inf")
N_ORG, N_STEPS, SEED = 800, 6000, 0
PUN_VIS = [40.0, 25.0, 15.0, 10.0, 7.0, 5.0, 4.0]   # scheduled punishment VIs on the target
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                     "axes.labelsize": 12, "xtick.labelsize": 10,
                     "ytick.labelsize": 10, "legend.fontsize": 10,
                     "font.family": "serif",
                     "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                     "mathtext.fontset": "stix"})


def sweep() -> dict:
    cfg = ChamberConfig(pun_tau=800.0, pun_bump=0.04, pun_floor=0.1, pun_c=1.0)
    sched, obtained, log_br = [], [], []
    for vp in PUN_VIS:
        res = run_punishment_choice("subtractive", cfg, N_ORG, N_STEPS,
                                    vi_reinf=[5.0, 5.0], vi_punish=[vp, INF], seed=SEED)
        e = res["emit"].sum(0)
        p1_obt = res["punished"].sum(0)[0] / res["steps"] / N_ORG
        sched.append(1.0 / vp)
        obtained.append(float(p1_obt))
        log_br.append(float(np.log(e[0] / e[1])))     # log(B_target / B_alt)
    return {"scheduled": sched, "obtained": obtained, "log_BR": log_br}


def _slope(x, y) -> float:
    return float(np.polyfit(np.log(x), y, 1)[0])


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    d = sweep()
    sched = np.array(d["scheduled"])
    obt = np.array(d["obtained"])
    logbr = np.array(d["log_BR"])

    # Recovered punishment sensitivity a_p = -slope of log(B1/B2) on log(rate).
    a_sched = -_slope(sched, logbr)
    a_obt = -_slope(obt, logbr)
    peak = int(np.argmax(obt))

    # Figure 1: the feedback function -- obtained vs scheduled punisher rate (inverted-U).
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    ax.plot(sched, obt, color="black", marker="o", ms=5)
    ax.set_xlabel("scheduled punishment rate on target (1/VI)")
    ax.set_ylabel("obtained punisher rate")
    ax.set_ylim(bottom=0)
    f1 = FIG / "feedback_function.png"
    fig.savefig(f1, dpi=130, bbox_inches="tight")
    plt.close(fig)

    # Figure 2: the fit -- log allocation vs scheduled (monotonic) vs obtained (folded).
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.6))
    axL.plot(np.log(sched), logbr, color="black", marker="o", ms=5)
    axL.set_xlabel("log scheduled punishment rate")
    axL.set_ylabel("log(B_target / B_alt)")
    axR.plot(np.log(obt), logbr, color="0.4", marker="s", ms=5)
    for i in range(len(obt) - 1):                     # arrows trace the sweep order
        axR.annotate("", xy=(np.log(obt[i + 1]), logbr[i + 1]),
                     xytext=(np.log(obt[i]), logbr[i]),
                     arrowprops=dict(arrowstyle="->", color="0.6", lw=0.8))
    axR.set_xlabel("log obtained punisher rate")
    axR.set_ylabel("log(B_target / B_alt)")
    fig.tight_layout()
    f2 = FIG / "fit_scheduled_vs_obtained.png"
    fig.savefig(f2, dpi=130, bbox_inches="tight")
    plt.close(fig)

    (FIG / "summary.json").write_text(json.dumps(
        {**d, "recovered_a_p_scheduled": a_sched, "recovered_a_p_obtained": a_obt}, indent=2))

    print("Scheduled vs obtained punisher rate (the feedback):")
    for s, o, lb in zip(sched, obt, logbr, strict=True):
        print(f"  scheduled={s:.4f}  obtained={o:.4f}  log(B_t/B_a)={lb:+.2f}")
    print(f"\nObtained rate peaks at scheduled={sched[peak]:.4f} then collapses (inverted-U).")
    print("Recovered punishment sensitivity a_p:")
    print(f"  on SCHEDULED rate: {a_sched:+.2f}  (correct -- suppression)")
    print(f"  on OBTAINED rate:  {a_obt:+.2f}  (WRONG SIGN -- the artifact)")
    print(f"\nSaved {f1}\nSaved {f2}\nSaved {FIG/'summary.json'}")


if __name__ == "__main__":
    main()
