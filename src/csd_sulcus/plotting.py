from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .analysis import ConditionComparison, electrode_arrival
from .model import SimulationOutput


def _save(fig: plt.Figure, output_path: str | Path | None) -> None:
    if output_path is None:
        return
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")


def plot_coupling_and_arrivals(
    comparison: ConditionComparison,
    output_path: str | Path | None = None,
) -> plt.Figure:
    arr_control_s = comparison.control.arr / comparison.fixed_scale
    arr_dipole_s = comparison.dipole.arr / comparison.fixed_scale
    delay = arr_dipole_s - arr_control_s

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))

    im0 = ax[0].imshow(comparison.dipole.g_field.T, origin="lower", cmap="viridis")
    ax[0].set_title("g(x, y) dipole")
    plt.colorbar(im0, ax=ax[0])

    im1 = ax[1].imshow(arr_dipole_s.T, origin="lower", cmap="magma")
    ax[1].set_title("arrival dipole (scaled s)")
    plt.colorbar(im1, ax=ax[1])

    im2 = ax[2].imshow(delay.T, origin="lower", cmap="coolwarm")
    ax[2].set_title("delta arrival (dipole - control)")
    plt.colorbar(im2, ax=ax[2])

    for axis in ax:
        axis.set_xticks([])
        axis.set_yticks([])

    fig.tight_layout()
    _save(fig, output_path)
    return fig


def plot_wavefront_snapshots(
    control: SimulationOutput,
    dipole: SimulationOutput,
    output_path: str | Path | None = None,
    scale_bar_mm: float = 5.0,
) -> plt.Figure:
    if control.snapshot_fields.size == 0 or dipole.snapshot_fields.size == 0:
        raise ValueError("Wavefront plotting requires snapshot times to be requested during the simulation.")

    nrows = len(control.snapshot_times)
    fig, ax = plt.subplots(
        nrows=nrows,
        ncols=2,
        figsize=(8, 3.2 * nrows),
        constrained_layout=True,
    )
    ax = np.atleast_2d(ax)

    for row in range(nrows):
        im = ax[row, 0].imshow(control.snapshot_fields[row].T, origin="lower", vmin=0, vmax=1, cmap="viridis")
        ax[row, 0].contour(control.sulc_mask.T, levels=[0.5], colors="white", linewidths=0.8)
        ax[row, 0].set_title(f"Control  t = {control.snapshot_times[row]:.1f} s")

        ax[row, 1].imshow(dipole.snapshot_fields[row].T, origin="lower", vmin=0, vmax=1, cmap="viridis")
        ax[row, 1].contour(dipole.sulc_mask.T, levels=[0.5], colors="white", linewidths=0.8)
        ax[row, 1].set_title(f"Dipole  t = {dipole.snapshot_times[row]:.1f} s")
        ax[row, 0].set_ylabel("y")

    for axis in ax.flat:
        axis.set_xticks([])
        axis.set_yticks([])

    cbar = fig.colorbar(im, ax=ax.ravel().tolist(), shrink=0.85, pad=0.02)
    cbar.set_label("Activator variable u", rotation=90)

    bar_px = int(scale_bar_mm / control.params.dx)
    x0 = 10
    y0 = 10
    ax[-1, 0].plot([x0, x0 + bar_px], [y0, y0], "w-", lw=3)
    ax[-1, 0].text(
        x0 + bar_px / 2,
        y0 + 4,
        f"{scale_bar_mm:.0f} mm",
        color="white",
        ha="center",
        va="bottom",
        fontsize=9,
    )

    _save(fig, output_path)
    return fig


def plot_virtual_electrode_arrivals(
    comparison: ConditionComparison,
    e1: tuple[float, float],
    e2: tuple[float, float],
    radius_mm: float,
    output_path: str | Path | None = None,
) -> plt.Figure:
    p = comparison.control.params

    tC1 = electrode_arrival(comparison.control.arr, p, e1, radius_mm) / comparison.fixed_scale
    tC2 = electrode_arrival(comparison.control.arr, p, e2, radius_mm) / comparison.fixed_scale
    tD1 = electrode_arrival(comparison.dipole.arr, p, e1, radius_mm) / comparison.fixed_scale
    tD2 = electrode_arrival(comparison.dipole.arr, p, e2, radius_mm) / comparison.fixed_scale

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([1, 2], [tC1, tC2], "o-", lw=2, label="Control")
    ax.plot([1, 2], [tD1, tD2], "o--", lw=2, label="Dipole")
    ax.set_xticks([1, 2], ["E1", "E2"])
    ax.set_ylabel("Arrival time (s)")
    ax.set_title("Virtual electrode arrival times")
    ax.legend()
    fig.tight_layout()

    _save(fig, output_path)
    return fig


def plot_velocity_vs_coupling(
    sweep_rows: list[dict[str, float]],
    control_speed_mm_min: float,
    output_path: str | Path | None = None,
) -> plt.Figure:
    gvals = [row["g_sulcus_min"] for row in sweep_rows]
    vvals = [row["speed_mm_min"] for row in sweep_rows]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(gvals, vvals, "o-", lw=2)
    ax.axhline(control_speed_mm_min, color="k", ls="--", label="Control")
    ax.set_xlabel("g_sulcus")
    ax.set_ylabel("Velocity (mm/min)")
    ax.set_title("Sulcal coupling reduction slows CSD propagation")
    ax.legend()
    fig.tight_layout()

    _save(fig, output_path)
    return fig


def plot_theory_vs_observed_sweep(
    sweep_rows: list[dict[str, float]],
    output_path: str | Path | None = None,
) -> plt.Figure:
    gvals = [row["g_sulcus_min"] for row in sweep_rows]
    observed = [row["speed_mm_min"] for row in sweep_rows]
    theory = [row["theory_speed_mm_min"] for row in sweep_rows]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(gvals, observed, "o-", lw=2, label="Observed")
    ax.plot(gvals, theory, "s--", lw=2, label="Theory: v0*sqrt(g)")
    ax.set_xlabel("g_sulcus")
    ax.set_ylabel("Velocity (mm/min)")
    ax.set_title("Observed vs theory-predicted slowing")
    ax.legend()
    fig.tight_layout()

    _save(fig, output_path)
    return fig


def plot_local_speed_triptych(
    control_speed_map: np.ndarray,
    case_a_speed_map: np.ndarray,
    case_b_speed_map: np.ndarray,
    sulc_mask: np.ndarray,
    output_path: str | Path | None = None,
    labels: tuple[str, str, str] = ("Control local speed", "Case A local speed", "Case B local speed"),
) -> plt.Figure:
    fig, ax = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    datasets = [
        (control_speed_map, labels[0]),
        (case_a_speed_map, labels[1]),
        (case_b_speed_map, labels[2]),
    ]

    finite_values = [field[np.isfinite(field)] for field, _ in datasets if np.isfinite(field).any()]
    vmax = np.nanpercentile(np.concatenate(finite_values), 95) if finite_values else 1.0
    for axis, (field, title) in zip(ax, datasets):
        im = axis.imshow(field.T, origin="lower", cmap="viridis", vmin=0, vmax=vmax)
        axis.contour(sulc_mask.T, levels=[0.5], colors="white", linewidths=0.8)
        axis.set_title(title)
        axis.set_xticks([])
        axis.set_yticks([])
    fig.colorbar(im, ax=ax.ravel().tolist(), shrink=0.8, pad=0.02, label="Speed (mm/min)")
    _save(fig, output_path)
    return fig


def plot_profile_heatmaps(
    rows: list[dict[str, float]],
    profiles: list[str],
    widths: list[float],
    gmins: list[float],
    metric_key: str,
    value_label: str,
    output_path: str | Path | None = None,
    cmap: str = "viridis",
) -> plt.Figure:
    fig, ax = plt.subplots(1, len(profiles), figsize=(6 * len(profiles), 4), constrained_layout=True)
    ax = np.atleast_1d(ax)

    for axis, profile in zip(ax, profiles):
        matrix = np.full((len(widths), len(gmins)), np.nan)
        for row in rows:
            if row["g_profile"] != profile:
                continue
            i = widths.index(row["sulcus_width_mm"])
            j = gmins.index(row["g_sulcus_min"])
            matrix[i, j] = row[metric_key]

        im = axis.imshow(matrix, origin="lower", aspect="auto", cmap=cmap)
        axis.set_xticks(range(len(gmins)), [f"{g:.2f}" for g in gmins])
        axis.set_yticks(range(len(widths)), [f"{w:.1f}" for w in widths])
        axis.set_xlabel("g_sulcus")
        axis.set_ylabel("Sulcus width (mm)")
        axis.set_title(profile.replace("-", " ").title())

    fig.colorbar(im, ax=ax.ravel().tolist(), shrink=0.85, pad=0.02, label=value_label)
    _save(fig, output_path)
    return fig
