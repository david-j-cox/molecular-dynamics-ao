"""Matplotlib visualizations of organism behavior and learning.

All functions take the long-format timestep log (and/or the episode-summary
DataFrame) produced by :mod:`behavioral_md.simulation` and write a figure to
disk. A non-interactive backend is used so figures render headless.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

_SOURCE_STYLE = {
    "food": ("food", "#3cc85a", "o"),
    "danger": ("danger", "#dc3c3c", "X"),
    "light": ("light", "#f0dc50", "*"),
    "cue": ("cue", "#5a8ce6", "s"),
}


def _save(fig: plt.Figure, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def _episode_slice(log: pd.DataFrame, episode: int) -> pd.DataFrame:
    return log[log["episode"] == episode]


def plot_trajectory(log: pd.DataFrame, episode: int, path: str | Path) -> Path:
    """Path of the organism through the arena for one life, with sources."""
    ep = _episode_slice(log, episode)
    # One row per timestep (collapse the per-atom long format).
    steps = ep.drop_duplicates("timestep").sort_values("timestep")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(steps["x"], steps["y"], "-", color="#888", lw=0.8, alpha=0.8, zorder=1)
    sc = ax.scatter(
        steps["x"], steps["y"], c=steps["timestep"], cmap="viridis", s=12, zorder=2
    )
    fig.colorbar(sc, ax=ax, label="timestep")
    ax.plot(steps["x"].iloc[0], steps["y"].iloc[0], "ko", ms=10, label="start", zorder=3)

    for key, (label, color, marker) in _SOURCE_STYLE.items():
        col_x = f"{key}_x"
        if col_x in steps:
            ax.scatter(
                steps[col_x].iloc[0], steps[f"{key}_y"].iloc[0],
                c=color, marker=marker, s=160, edgecolors="k", label=label, zorder=4,
            )
    ax.set(xlabel="x", ylabel="y", title=f"Trajectory (episode {episode})")
    ax.set_aspect("equal")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    return _save(fig, path)


def plot_energy(log: pd.DataFrame, episode: int, path: str | Path) -> Path:
    """Energy reserve over a single life, marking death if it occurred."""
    steps = _episode_slice(log, episode).drop_duplicates("timestep").sort_values("timestep")
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.plot(steps["timestep"], steps["energy"], color="#2a7", lw=1.3)
    ax.fill_between(steps["timestep"], 0, steps["energy"], color="#2a7", alpha=0.15)
    if not bool(steps["alive"].iloc[-1]):
        t_death = steps["timestep"].iloc[-1]
        ax.axvline(t_death, color="#c33", ls="--", lw=1)
        ax.text(t_death, ax.get_ylim()[1] * 0.9, " death", color="#c33", fontsize=9)
    ax.set(xlabel="timestep", ylabel="energy reserve", title=f"Energy (episode {episode})")
    ax.set_ylim(bottom=0)
    return _save(fig, path)


def plot_atom_series(
    log: pd.DataFrame,
    episode: int,
    column: str,
    atom_names: list[str],
    path: str | Path,
    ylabel: str | None = None,
) -> Path:
    """Time series of a per-atom quantity (activation / force / hw_*) for one life."""
    ep = _episode_slice(log, episode)
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for name in atom_names:
        a = ep[ep["atom_name"] == name].sort_values("timestep")
        ax.plot(a["timestep"], a[column], lw=1.0, label=name)
    ax.set(xlabel="timestep", ylabel=ylabel or column, title=f"{column} (episode {episode})")
    ax.legend(fontsize=8, ncol=2, framealpha=0.9)
    return _save(fig, path)


def plot_learning_curve(summary: pd.DataFrame, path: str | Path) -> Path:
    """Across-lives acquisition curve: contact rate, survival, first-food latency."""
    s = summary.sort_values("episode").copy()
    s["contact_rate"] = s["n_consumed"] / s["steps"].clip(lower=1)

    fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
    axes[0].plot(s["episode"], s["contact_rate"], color="#2a7", marker=".")
    axes[0].set_ylabel("food-contact rate")
    axes[1].plot(s["episode"], s["steps"], color="#37a", marker=".")
    axes[1].set_ylabel("steps survived")
    axes[2].plot(s["episode"], s["latency"], color="#a63", marker=".")
    axes[2].set_ylabel("latency to first food")
    axes[2].set_xlabel("life (episode)")
    axes[0].set_title("Acquisition across lives")
    return _save(fig, path)


def plot_weight_acquisition(
    log: pd.DataFrame,
    path: str | Path,
    series: tuple[tuple[str, str], ...] = (
        ("approach_food", "hw_food"),
        ("avoid_danger", "hw_danger"),
    ),
) -> Path:
    """End-of-life learned drive weights vs. life (the learning history forming)."""
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for atom_name, column in series:
        a = log[log["atom_name"] == atom_name]
        # last row per episode = weight at end of that life.
        last = a.sort_values("timestep").groupby("episode").tail(1).sort_values("episode")
        ax.plot(last["episode"], last[column], marker=".", label=f"{atom_name}.{column}")
    ax.axhline(0, color="#999", lw=0.6)
    ax.set(xlabel="life (episode)", ylabel="learned weight",
           title="Learning history across lives")
    ax.legend(fontsize=8)
    return _save(fig, path)
