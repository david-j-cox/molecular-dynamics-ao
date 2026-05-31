"""Matplotlib visualizations of organism behavior and learning.

House style (see also the project plot-style note): despined top/right, no plot
titles, axis labels at fontsize 18 with labelpad 12, ticks at 12, black-and-white
(series distinguished by linestyle/marker), legends at fontsize 14 placed outside
the axes, and markers + 95% confidence intervals on aggregate plots so
variability is visible. A non-interactive backend is used so figures render
headless.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

plt.rcParams.update(
    {
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelsize": 18,
        "axes.labelpad": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 14,
        "figure.autolayout": False,
    }
)

# Black-and-white series cycle: vary linestyle + marker, not color.
_BW_CYCLE = [
    ("-", "o"),
    ("--", "s"),
    ("-.", "^"),
    (":", "D"),
    ("-", "v"),
    ("--", "P"),
]
# Source markers for the trajectory map (grayscale + distinct shapes).
_SOURCE_STYLE = {
    "food": ("Food", "h", "white"),   # hexagon (distinct from the circle endpoints)
    "danger": ("Danger", "X", "black"),
    "light": ("Light", "*", "0.6"),
    "cue": ("Cue", "s", "0.85"),
}


def _save(fig: plt.Figure, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def _legend_outside(ax: plt.Axes, **kw) -> None:
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0,
              frameon=False, **kw)


def _mean_ci(values: np.ndarray, axis: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Mean and 95% CI half-width across ``axis`` (normal approx, 1.96*SEM)."""
    mean = np.nanmean(values, axis=axis)
    n = np.sum(~np.isnan(values), axis=axis)
    sem = np.nanstd(values, axis=axis, ddof=1) / np.sqrt(np.maximum(n, 1))
    return mean, 1.96 * sem


def _band(ax: plt.Axes, x, mean, ci, label: str, ls: str, marker: str) -> None:
    ax.plot(x, mean, color="black", ls=ls, marker=marker, ms=4, markevery=max(1, len(x) // 20),
            label=label)
    ax.fill_between(x, mean - ci, mean + ci, color="0.7", alpha=0.5, lw=0)


# --- Single-realization plots (one life; no CI) ----------------------------- #
def _episode_slice(log: pd.DataFrame, episode: int) -> pd.DataFrame:
    return log[log["episode"] == episode]


def plot_trajectory(log: pd.DataFrame, episode: int, path: str | Path) -> Path:
    """Path of the organism through the arena for one life.

    Direction of travel is shown with sparse, non-overlapping arrows along the
    route (no time colorbar); start and end are marked, sources are large
    black-and-white markers.
    """
    steps = _episode_slice(log, episode).drop_duplicates("timestep").sort_values("timestep")
    x = steps["x"].to_numpy()
    y = steps["y"].to_numpy()

    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.plot(x, y, "-", color="0.65", lw=0.8, alpha=0.8, zorder=1)

    # Sparse arrows showing the LOCAL direction of travel along the route (each
    # arrow is the single-step move at a sampled point, so it follows the path
    # rather than cutting across the wandering).
    if len(x) > 2:
        stride = max(1, len(x) // 18)
        idx = np.arange(0, len(x) - 1, stride)
        dx, dy = x[idx + 1] - x[idx], y[idx + 1] - y[idx]
        moved = (dx != 0) | (dy != 0)
        ax.quiver(
            x[idx][moved], y[idx][moved], dx[moved], dy[moved],
            angles="xy", scale_units="xy", scale=1.0, color="black",
            width=0.008, headwidth=4, headlength=5, zorder=2,
        )

    ax.plot(x[0], y[0], "o", mfc="black", mec="black", ms=14, label="Start", zorder=5)
    ax.plot(x[-1], y[-1], "o", mfc="white", mec="black", mew=1.5, ms=14, label="End", zorder=5)
    for key, (label, marker, facecolor) in _SOURCE_STYLE.items():
        cx = f"{key}_x"
        if cx in steps:
            ax.scatter(steps[cx].iloc[0], steps[f"{key}_y"].iloc[0], marker=marker,
                       s=320, facecolors=facecolor, edgecolors="black", linewidths=1.4,
                       label=label, zorder=4)
    ax.set_xlabel("X position")
    ax.set_ylabel("Y position")
    ax.set_aspect("equal")
    _legend_outside(ax)
    return _save(fig, path)


def plot_energy(log: pd.DataFrame, episode: int, path: str | Path) -> Path:
    """Energy reserve over a single life, marking death if it occurred."""
    steps = _episode_slice(log, episode).drop_duplicates("timestep").sort_values("timestep")
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.plot(steps["timestep"], steps["energy"], color="black", lw=1.3)
    if not bool(steps["alive"].iloc[-1]):
        t_death = steps["timestep"].iloc[-1]
        ax.axvline(t_death, color="black", ls="--", lw=1)
        ax.text(t_death, ax.get_ylim()[1] * 0.9, " Death", fontsize=12)
    ax.set_xlabel("Time (steps)")
    ax.set_ylabel("Energy reserve")
    ax.set_ylim(bottom=0)
    return _save(fig, path)


def plot_occupancy_landscape(log: pd.DataFrame, episode: int, path: str | Path) -> Path:
    """3D occupancy "landscape": height at each cell = time the organism spent there.

    Camping shows as a tall peak. Sources are marked above the surface.
    """
    steps = _episode_slice(log, episode).drop_duplicates("timestep").sort_values("timestep")
    xs = steps["x"].to_numpy().astype(int)
    ys = steps["y"].to_numpy().astype(int)
    g = int(max(xs.max(), ys.max())) + 1
    dwell = np.zeros((g, g))
    for xi, yi in zip(xs, ys, strict=True):
        dwell[xi, yi] += 1.0
    gx, gy = np.meshgrid(np.arange(g), np.arange(g), indexing="ij")

    fig = plt.figure(figsize=(7.5, 6))
    ax = fig.add_subplot(projection="3d")
    # Filled surface for the relief, plus a black wireframe so every cell (incl.
    # the z=0 floor) is visible as a mesh.
    ax.plot_surface(gx, gy, dwell, cmap="Greys", edgecolor="black", linewidth=0.35,
                    alpha=0.85, rstride=1, cstride=1, antialiased=True)
    ax.plot_wireframe(gx, gy, dwell, color="black", linewidth=0.4,
                      rstride=1, cstride=1, alpha=0.6)

    top = dwell.max() * 1.08 if dwell.max() > 0 else 1.0
    for key, (label, marker, facecolor) in _SOURCE_STYLE.items():
        cx = f"{key}_x"
        if cx in steps:
            sx, sy = int(steps[cx].iloc[0]), int(steps[f"{key}_y"].iloc[0])
            ax.scatter([sx], [sy], [top], marker=marker, s=140, facecolors=facecolor,
                       edgecolors="black", linewidths=1.2, label=label, depthshade=False)

    ax.set_xlabel("X position", fontsize=16, labelpad=10)
    ax.set_ylabel("Y position", fontsize=16, labelpad=10)
    ax.set_zlabel("Time spent (steps)", fontsize=16, labelpad=10)
    ax.tick_params(labelsize=11)
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 0.9), frameon=False, fontsize=14)
    ax.view_init(elev=42, azim=-130)
    return _save(fig, path)


def plot_acquisition_latency(summaries: pd.DataFrame, path: str | Path) -> Path:
    """Latency to first food vs. life (mean +/- 95% CI over agents).

    ``summaries`` must have columns: seed, episode, latency.
    """
    episodes, mat = _pivot(summaries, "latency")
    mean, ci = _mean_ci(mat, axis=0)
    fig, ax = plt.subplots(figsize=(8, 4))
    _band(ax, episodes, mean, ci, "Latency", "-", "o")
    ax.set_xlabel("Life (episode)")
    ax.set_ylabel("Latency to food (steps)")
    ax.set_ylim(bottom=0)
    return _save(fig, path)


def plot_force_decomposition(
    log: pd.DataFrame, episode: int, atom_name: str, path: str | Path
) -> Path:
    """Decompose one drive atom's force into its components over a single life.

    Shows sensory / history / motivational / coupling and the net total, so the
    learned (history) contribution is visible against the fixed innate (sensory)
    one.
    """
    a = _episode_slice(log, episode)
    a = a[a["atom_name"] == atom_name].sort_values("timestep")
    t = a["timestep"]
    comps = [
        ("force_sensory", "Sensory (innate)"),
        ("force_history", "History (learned)"),
        ("force_motivational", "Motivational (energy)"),
        ("force_coupling", "Coupling"),
        ("atom_force", "Net"),
    ]
    fig, ax = plt.subplots(figsize=(8, 4))
    for (col, label), (ls, marker) in zip(comps, _BW_CYCLE, strict=False):
        if col in a:
            ax.plot(t, a[col], color="black", ls=ls, lw=1.0, marker=marker, ms=3,
                    markevery=max(1, len(a) // 18), label=label)
    ax.axhline(0, color="0.6", lw=0.5)
    ax.set_xlabel("Time (steps)")
    ax.set_ylabel("Force")
    _legend_outside(ax)
    return _save(fig, path)


_FORCE_COMPONENTS = [
    ("force_sensory", "Sensory (innate)"),
    ("force_history", "History (learned)"),
    ("force_motivational", "Motivational (energy)"),
    ("force_coupling", "Coupling"),
    ("atom_force", "Net"),
]


def plot_force_decomposition_grid(
    log: pd.DataFrame,
    episode: int,
    atom_names: list[str],
    path: str | Path,
    ncols: int = 3,
) -> Path:
    """One force-decomposition panel per atom, in a shared grid with one legend."""
    ep = _episode_slice(log, episode)
    n = len(atom_names)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 2.6 * nrows),
                             sharex=True, constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    handles_labels = None

    for ax, name in zip(axes, atom_names, strict=False):
        a = ep[ep["atom_name"] == name].sort_values("timestep")
        t = a["timestep"]
        cols = [c for c, _ in _FORCE_COMPONENTS if c in a]
        # Per-panel normalization by max magnitude: force is unitless, so this
        # puts each atom on [-1, 1] (or [0, 1] if all-positive) while preserving
        # the relative sizes of the components within the atom.
        scale = float(np.nanmax(np.abs(a[cols].to_numpy()))) if cols else 1.0
        scale = scale if scale > 1e-9 else 1.0
        any_negative = bool((a[cols].to_numpy() < -1e-9).any()) if cols else False
        for (col, label), (ls, marker) in zip(_FORCE_COMPONENTS, _BW_CYCLE, strict=False):
            if col in a:
                ax.plot(t, a[col] / scale, color="black", ls=ls, lw=0.9, marker=marker,
                        ms=4, markevery=max(1, len(a) // 12), label=label)
        ax.axhline(0, color="0.6", lw=0.4)
        ax.set_ylim(-1.08, 1.08) if any_negative else ax.set_ylim(-0.04, 1.08)
        ax.spines.top.set_visible(False)
        ax.spines.right.set_visible(False)
        # In-panel atom label (upper right), prettified to readable text.
        ax.text(0.97, 0.95, name.replace("_", " ").capitalize(), transform=ax.transAxes,
                va="top", ha="right", fontsize=12, fontweight="bold")
        if handles_labels is None:
            handles_labels = ax.get_legend_handles_labels()

    for ax in axes[n:]:  # hide unused cells
        ax.set_visible(False)
    for i, ax in enumerate(axes[:n]):
        if i % ncols == 0:  # left column
            ax.set_ylabel("Normalized\nForce")
        if i + ncols >= n:  # no visible panel below -> bottom edge
            ax.set_xlabel("Time (steps)")

    if handles_labels:
        fig.legend(*handles_labels, loc="center left", bbox_to_anchor=(1.0, 0.5),
                   frameon=False, fontsize=14)
    return _save(fig, path)


def plot_food_biomass(log: pd.DataFrame, episode: int, path: str | Path) -> Path:
    """Food-patch biomass over one life (shows the deplete/regrow VI dynamic)."""
    steps = _episode_slice(log, episode).drop_duplicates("timestep").sort_values("timestep")
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.plot(steps["timestep"], steps["food_biomass"], color="black", lw=1.2)
    ax.set_xlabel("Time (steps)")
    ax.set_ylabel("Food biomass")
    ax.set_ylim(bottom=0)
    return _save(fig, path)


def plot_atom_series(
    log: pd.DataFrame,
    episode: int,
    column: str,
    atom_names: list[str],
    path: str | Path,
    ylabel: str,
    normalize: bool = False,
) -> Path:
    """Time series of a per-atom quantity (activation / force / hw_*) for one life.

    With ``normalize=True`` all series are divided by the shared max magnitude
    (force is unitless), putting them on [-1, 1] for comparison.
    """
    ep = _episode_slice(log, episode)
    series = {n: ep[ep["atom_name"] == n].sort_values("timestep") for n in atom_names}
    scale = 1.0
    if normalize:
        vals = np.concatenate([s[column].to_numpy() for s in series.values()])
        scale = float(np.nanmax(np.abs(vals))) if len(vals) else 1.0
        scale = scale if scale > 1e-9 else 1.0
        ylabel = f"Normalized\n{ylabel}"
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for (name, a), (ls, marker) in zip(series.items(), _BW_CYCLE, strict=False):
        ax.plot(a["timestep"], a[column] / scale, color="black", ls=ls, lw=1.0,
                marker=marker, ms=4, markevery=max(1, len(a) // 18), label=name)
    ax.set_xlabel("Time (steps)")
    ax.set_ylabel(ylabel)
    _legend_outside(ax, ncol=1)
    return _save(fig, path)


# --- Aggregate plots (across seeds; mean + 95% CI) -------------------------- #
def _pivot(summaries: pd.DataFrame, value_col: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (episodes, matrix[seed, episode]) for a summary column."""
    wide = summaries.pivot(index="seed", columns="episode", values=value_col)
    return wide.columns.to_numpy(), wide.to_numpy()


def plot_learning_curve(summaries: pd.DataFrame, path: str | Path) -> Path:
    """Across-lives acquisition curves (mean +/- 95% CI over seeds).

    ``summaries`` must have columns: seed, episode, n_consumed, steps, latency.
    """
    s = summaries.copy()
    s["contact_rate"] = s["n_consumed"] / s["steps"].clip(lower=1)

    panels = [
        ("contact_rate", "Food-contact rate"),
        ("steps", "Steps survived"),
        ("latency", "Latency (steps)"),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(8, 8.5), sharex=True, constrained_layout=True)
    for ax, (col, ylabel) in zip(axes, panels, strict=True):
        episodes, mat = _pivot(s, col)
        mean, ci = _mean_ci(mat, axis=0)
        _band(ax, episodes, mean, ci, ylabel, "-", "o")
        ax.set_ylabel(ylabel)
    axes[-1].set_xlabel("Life (episode)")
    return _save(fig, path)


def plot_extinction(summaries: pd.DataFrame, transition: int, path: str | Path) -> Path:
    """Train -> extinction across lives (mean +/- 95% CI over agents).

    ``summaries`` needs columns: seed, episode, hw_food, steps_at_food. A dotted
    line marks the reinforced -> extinction transition.
    """
    panels = [("hw_food", "Learned food weight"), ("steps_at_food", "Steps at food")]
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True, constrained_layout=True)
    for ax, (col, ylabel) in zip(axes, panels, strict=True):
        episodes, mat = _pivot(summaries, col)
        mean, ci = _mean_ci(mat, axis=0)
        _band(ax, episodes, mean, ci, ylabel, "-", "o")
        ax.axvline(transition - 0.5, color="black", ls=":", lw=1.2)
        ax.set_ylabel(ylabel)
        ax.set_ylim(bottom=0)
    axes[-1].set_xlabel("Life (episode)")
    return _save(fig, path)


def plot_generalization_gradient(
    values: np.ndarray,
    responses: np.ndarray,
    trained_value: float,
    path: str | Path,
    s_minus: float | None = None,
) -> Path:
    """Generalization gradient: conditioned response vs cue value (mean +/- 95% CI).

    ``responses`` is an (agents, values) array of probed conditioned responses. A
    dotted line marks S+ (``trained_value``); if ``s_minus`` is given (a
    discrimination/peak-shift test), a dashed line marks S- and a solid line marks
    the empirical peak, so any peak shift away from S- is visible.
    """
    mean, ci = _mean_ci(responses, axis=0)
    fig, ax = plt.subplots(figsize=(8, 4))
    _band(ax, values, mean, ci, "response", "-", "o")
    span = float(values[-1] - values[0]) or 1.0

    def mark(v: float, ls: str, label: str) -> None:
        ax.axvline(v, color="black", ls=ls, lw=1.2)
        # Label just right of the line (left if near the right edge) to avoid
        # overlapping the curve or other labels.
        right = v < values[0] + 0.85 * span
        ax.annotate(
            label, xy=(v, 1.0), xycoords=("data", "axes fraction"),
            xytext=(4 if right else -4, -4), textcoords="offset points",
            ha="left" if right else "right", va="top", fontsize=12,
            annotation_clip=False,
        )

    mark(trained_value, ":", "S+")
    if s_minus is not None:
        mark(float(values[int(np.argmax(mean))]), "-", "peak")
        mark(s_minus, "--", "S-")
    ax.set_xlabel("Test cue value")
    ax.set_ylabel("Conditioned Response")
    ax.set_ylim(bottom=0)
    return _save(fig, path)


def plot_survival_curve(steps: np.ndarray, frac_alive: np.ndarray, path: str | Path) -> Path:
    """Fraction of organisms still alive vs. time within a life (pooled)."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps, frac_alive, color="black", lw=1.5)
    ax.set_xlabel("Time (steps)")
    ax.set_ylabel("Fraction Surviving")
    ax.set_ylim(0, 1.02)
    return _save(fig, path)


def plot_time_to_death(times: np.ndarray, n_steps: int, path: str | Path) -> Path:
    """Distribution of time-to-death (organisms that died)."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(times, bins=30, range=(0, n_steps), color="0.5", edgecolor="black")
    ax.set_xlabel("Time of Death (steps)")
    ax.set_ylabel("Count")
    return _save(fig, path)


def plot_mortality_by_life(mortality: dict, path: str | Path) -> Path:
    """Death rate and its cause breakdown across lives (does learning reduce death?)."""
    n = len(mortality["death_rate"])
    lives = np.arange(n)
    fig, ax = plt.subplots(figsize=(8, 4))
    for key, ls, marker, label in [
        ("death_rate", "-", "o", "Total"),
        ("starvation_rate", "--", "s", "Starvation"),
        ("danger_rate", "-.", "^", "Danger"),
    ]:
        ax.plot(lives, mortality[key], color="black", ls=ls, marker=marker, ms=4,
                markevery=max(1, n // 20), label=label)
    ax.set_xlabel("Life (episode)")
    ax.set_ylabel("Death Rate")
    ax.set_ylim(bottom=0)
    _legend_outside(ax)
    return _save(fig, path)


def plot_matching(log_R: np.ndarray, log_B: np.ndarray, a: float, log_b: float,
                  path: str | Path,
                  xlabel: str = "log(R$_L$/R$_R$)",
                  ylabel: str = "log(B$_L$/B$_R$)") -> Path:
    """Generalized matching law: log behavior ratio vs log reinforcement ratio.

    Points are individual organisms (and alternatives) across schedules; the solid
    line is the GML fit, the dotted line perfect matching (slope 1 through 0).
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(log_R, log_B, s=10, facecolors="none", edgecolors="0.4", linewidths=0.5)
    lim = float(np.nanmax(np.abs([log_R.min(), log_R.max(), log_B.min(), log_B.max()])))
    xs = np.array([-lim, lim])
    ax.plot(xs, xs, ls=":", color="0.5", lw=1.0, label="Perfect matching")
    ax.plot(xs, a * xs + log_b, color="black", lw=1.6,
            label=f"Fit: a={a:.2f}, log b={log_b:+.2f}")
    ax.axhline(0, color="0.8", lw=0.5)
    ax.axvline(0, color="0.8", lw=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_aspect("equal")
    _legend_outside(ax)
    return _save(fig, path)


def plot_matching_cod(separations, a_mean, a_ci, path: str | Path) -> Path:
    """Matching sensitivity (GML slope a) vs. changeover delay (patch separation)."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    _band(ax, np.asarray(separations), np.asarray(a_mean), np.asarray(a_ci),
          "sensitivity", "-", "o")
    ax.axhline(1.0, color="0.5", ls=":", lw=1.0)
    ax.text(np.asarray(separations).max(), 1.0, " perfect matching", fontsize=11, va="bottom",
            ha="right")
    ax.set_xlabel("Patch Separation (travel steps = COD)")
    ax.set_ylabel("Matching Sensitivity (a)")
    ax.set_ylim(bottom=0)
    return _save(fig, path)


def plot_demand(unit_price: np.ndarray, consumption: np.ndarray, path: str | Path) -> Path:
    """Behavioral-economic demand curve: consumption vs unit price (log-log)."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(unit_price, consumption, color="black", ls="-", marker="o", ms=5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Unit Price (responses x effort / magnitude)")
    ax.set_ylabel("Consumption (reinforcers / step)")
    return _save(fig, path)


def plot_weight_acquisition(
    weights: pd.DataFrame,
    path: str | Path,
    series: tuple[tuple[str, str], ...] = (
        ("approach_food", "hw_food"),
        ("avoid_danger", "hw_danger"),
    ),
) -> Path:
    """End-of-life learned drive weights vs. life (mean +/- 95% CI over seeds).

    ``weights`` must have columns: seed, episode, <atom>.<channel> values, i.e.
    one column per (atom_name, channel) pair named ``f"{atom}.{channel}"``.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    for (atom, channel), (ls, marker) in zip(series, _BW_CYCLE, strict=False):
        col = f"{atom}.{channel}"
        episodes, mat = _pivot(weights, col)
        mean, ci = _mean_ci(mat, axis=0)
        _band(ax, episodes, mean, ci, f"{atom} ({channel})", ls, marker)
    ax.axhline(0, color="0.5", lw=0.6)
    ax.set_xlabel("Life (episode)")
    ax.set_ylabel("Learned weight")
    _legend_outside(ax)
    return _save(fig, path)
