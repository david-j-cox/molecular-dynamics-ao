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
    ax.plot_surface(gx, gy, dwell, cmap="Greys", edgecolor="0.5", linewidth=0.2,
                    alpha=0.9, rstride=1, cstride=1)

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
) -> Path:
    """Time series of a per-atom quantity (activation / force / hw_*) for one life."""
    ep = _episode_slice(log, episode)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for name, (ls, marker) in zip(atom_names, _BW_CYCLE, strict=False):
        a = ep[ep["atom_name"] == name].sort_values("timestep")
        ax.plot(a["timestep"], a[column], color="black", ls=ls, lw=1.0,
                marker=marker, ms=3, markevery=max(1, len(a) // 18), label=name)
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
