"""Experiment 015 -- operant chamber: schedules of reinforcement.

Runs the single-response operant chamber under FR, VR, FI, VI and reports
steady-state response rate, obtained reinforcement rate, and the FI
within-interval response pattern (the scallop test).

Finding (documented in the lab notebook): with a restoring force the chamber
produces GRADED, reinforcement-maintained pressing, but it does NOT reproduce the
molecular schedule signatures -- response rates are ~equal across schedules at
matched reinforcement, and FI is flat (no scallop). Those signatures (VR>VI rate
difference, FI scallop, FR break-and-run) require molecular mechanisms this molar
value/energy model lacks: differential reinforcement of inter-response times and
temporal discrimination of elapsed time. See the lab notebook for the fork.

Run:  python -m experiments.exp015_operant_chamber
"""

from __future__ import annotations

import numpy as np

from behavioral_md.chamber import ChamberConfig, run_chamber

N_ORG, N_STEPS = 400, 8000
SCHEDULES = [("VR", 20), ("VI", 20), ("FR", 20), ("FI", 20)]


def main() -> None:
    cfg = ChamberConfig()
    warm = N_STEPS // 2
    print(f"{N_ORG} organisms x {N_STEPS} steps per schedule\n")
    print(f"{'schedule':>8} {'param':>6} {'resp_rate':>10} {'reinf_rate':>11}")
    fi = None
    for sched, param in SCHEDULES:
        r = run_chamber(sched, param, cfg, N_ORG, N_STEPS, seed=0)
        rr = r["presses"][warm:].mean()
        fr = r["reinforced"][warm:].mean()
        print(f"{sched:>8} {param:>6} {rr:>10.3f} {fr:>11.4f}")
        if sched == "FI":
            fi = (r, param)

    r, param = fi
    pr, ts = r["presses"][warm:], r["time_since_reinf"][warm:]
    frac = np.clip(ts / param, 0.0, 1.0)
    bins = np.linspace(0, 1, 6)
    rates = [pr[(frac >= bins[i]) & (frac < bins[i + 1])].mean() for i in range(5)]
    print("\nFI within-interval response rate by elapsed fraction (scallop test):")
    for i, rate in enumerate(rates):
        print(f"  {bins[i]:.1f}-{bins[i+1]:.1f}: {rate:.3f}")
    scallop = rates[-1] - rates[0]
    print(f"\n  scallop index (last - first) = {scallop:+.3f} "
          f"({'flat -- no scallop' if abs(scallop) < 0.05 else 'scalloped'})")


if __name__ == "__main__":
    main()
