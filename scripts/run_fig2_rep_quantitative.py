from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib
import numpy as np
from scipy.sparse import csgraph

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from csd_sulcus.surface_io import generate_folded_strip_mesh
from csd_sulcus.surface_mechanistic import (
    mechanistic_surface_arrival_speed_mm_min,
    run_mechanistic_surface_simulation,
)
from run_surface_mechanistic_study import REPRESENTATIVE_CASES


FIGURE_TITLE = "Quantitative readout of the representative folded-surface slowdown"

DISPLAY_VALUES = {
    "representative": {
        "speed_no_dipole": "2.558 mm/min",
        "speed_dipole": "2.476 mm/min",
        "speed_delta": "-0.082 mm/min (-3.21%)",
        "delay_no_dipole": "130.5 s",
        "delay_dipole": "134.7 s",
        "delay_delta": "+4.2 s",
        "ve_no_dipole": "18.99 mV",
        "ve_dipole": "19.50 mV",
        "ve_delta": "+0.51 mV",
    },
    "flat_control": {
        "speed_no_dipole": "3.375 mm/min",
        "speed_dipole": "3.375 mm/min",
        "delay_no_dipole": "69.02 s",
        "delay_dipole": "69.02 s",
    },
}

TRACE_STYLE = {
    "no_dipole": dict(color="black", lw=1.8, ls="-"),
    "dipole_enabled": dict(color="0.45", lw=1.8, ls="--"),
    "no_dipole_e1": dict(color="0.20", lw=1.2, ls="-"),
    "dipole_enabled_e1": dict(color="0.60", lw=1.2, ls="--"),
}

MARKER_STYLE = {
    "no_dipole": dict(s=30, facecolor="black", edgecolor="black", linewidth=0.8, zorder=5),
    "dipole_enabled": dict(s=30, facecolor="white", edgecolor="0.45", linewidth=1.0, zorder=6),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the new representative quantitative Figure 2.")
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=ROOT / "outputs" / "surface_mechanistic_study" / "mechanistic_study_summary.json",
        help="Current mechanistic study summary used as the authoritative representative setup.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=ROOT / "manuscript" / "figures" / "fig2_rep_quantitative",
        help="Output path without extension; the script writes both PDF and PNG.",
    )
    parser.add_argument(
        "--trace-dt",
        type=float,
        default=0.25,
        help="Sampling interval in seconds for the representative trace export.",
    )
    return parser.parse_args()


def _load_summary(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _representative_row(summary: dict[str, object], case_label: str) -> dict[str, object]:
    rows = summary["representative_rows"]
    return next(row for row in rows if row["case_label"] == case_label)


def _panel_label(ax, label: str) -> None:
    ax.text(
        -0.08,
        1.02,
        label,
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        va="top",
        ha="left",
    )


def _shortest_path_vertices(graph, start_vertex: int, end_vertex: int) -> np.ndarray:
    _dist, predecessors = csgraph.dijkstra(
        graph,
        directed=False,
        indices=int(start_vertex),
        return_predecessors=True,
    )
    path = [int(end_vertex)]
    cursor = int(end_vertex)
    while cursor != int(start_vertex):
        cursor = int(predecessors[cursor])
        if cursor < 0:
            raise RuntimeError("Could not reconstruct the representative cross-sulcal path.")
        path.append(cursor)
    path.reverse()
    return np.asarray(path, dtype=int)


def _path_distance(vertices: np.ndarray) -> np.ndarray:
    distance = np.zeros(vertices.shape[0], dtype=float)
    if vertices.shape[0] > 1:
        distance[1:] = np.cumsum(np.linalg.norm(np.diff(vertices, axis=0), axis=1))
    return distance


def _format_close(actual: float, expected: float, tol: float, label: str) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=tol):
        raise RuntimeError(f"{label} mismatch: expected {expected:.6f}, got {actual:.6f}")


def _representative_mesh_and_vertices(summary: dict[str, object]) -> tuple[object, int, int, int]:
    baseline = _representative_row(summary, "mechanistic_multion_baseline")
    mesh = generate_folded_strip_mesh(
        nx=64,
        ny=28,
        length_mm=22.0,
        width_mm=10.0,
        fold_depth_mm=float(baseline["fold_depth_mm"]),
        fold_sigma_mm=float(baseline["fold_sigma_mm"]),
    )
    stimulus_vertex = int(baseline["stimulus_vertex"])
    e1_vertex = int(baseline["electrode_1_vertex"])
    e2_vertex = int(baseline["electrode_2_vertex"])
    return mesh, stimulus_vertex, e1_vertex, e2_vertex


def _run_representative_outputs(
    summary: dict[str, object],
    trace_dt: float,
) -> tuple[object, object, object, int, int, int]:
    mesh, stimulus_vertex, e1_vertex, e2_vertex = _representative_mesh_and_vertices(summary)
    baseline_row = _representative_row(summary, "mechanistic_multion_baseline")
    dipole_row = _representative_row(summary, "mechanistic_multion_dipole")
    trace_times = np.arange(0.0, 220.0 + 0.5 * trace_dt, trace_dt, dtype=float)
    baseline_output = run_mechanistic_surface_simulation(
        mesh,
        REPRESENTATIVE_CASES["mechanistic_multion_baseline"],
        stimulus_vertex=stimulus_vertex,
        snapshot_times=trace_times,
    )
    dipole_output = run_mechanistic_surface_simulation(
        mesh,
        REPRESENTATIVE_CASES["mechanistic_multion_dipole"],
        stimulus_vertex=stimulus_vertex,
        snapshot_times=trace_times,
    )

    baseline_speed = mechanistic_surface_arrival_speed_mm_min(baseline_output, e1_vertex, e2_vertex, radius_mm=1.0)
    dipole_speed = mechanistic_surface_arrival_speed_mm_min(dipole_output, e1_vertex, e2_vertex, radius_mm=1.0)
    _format_close(baseline_speed, float(baseline_row["arrival_speed_mm_min"]), 1e-6, "Baseline speed")
    _format_close(dipole_speed, float(dipole_row["arrival_speed_mm_min"]), 1e-6, "Dipole speed")
    _format_close(
        float(baseline_output.arrival_times[e2_vertex] - baseline_output.arrival_times[e1_vertex]),
        float(baseline_row["cross_fold_delay_s"]),
        1e-6,
        "Baseline delay",
    )
    _format_close(
        float(dipole_output.arrival_times[e2_vertex] - dipole_output.arrival_times[e1_vertex]),
        float(dipole_row["cross_fold_delay_s"]),
        1e-6,
        "Dipole delay",
    )
    return mesh, baseline_output, dipole_output, stimulus_vertex, e1_vertex, e2_vertex


def _sulcal_span(path_distance: np.ndarray, path_depth: np.ndarray) -> tuple[float, float]:
    depth_threshold = 0.5 * float(np.max(path_depth))
    mask = path_depth >= depth_threshold
    if not np.any(mask):
        return float(path_distance[0]), float(path_distance[-1])
    idx = np.where(mask)[0]
    return float(path_distance[idx[0]]), float(path_distance[idx[-1]])


def _trace_value(times: np.ndarray, values: np.ndarray, query_time: float) -> float:
    return float(np.interp(query_time, times, values))


def _panel_a(ax, times: np.ndarray, baseline_output, dipole_output, e1_vertex: int, e2_vertex: int) -> None:
    k_b_e1 = baseline_output.snapshot_potassium_e[:, e1_vertex]
    k_b_e2 = baseline_output.snapshot_potassium_e[:, e2_vertex]
    k_d_e1 = dipole_output.snapshot_potassium_e[:, e1_vertex]
    k_d_e2 = dipole_output.snapshot_potassium_e[:, e2_vertex]

    ax.plot(times, k_b_e1, **TRACE_STYLE["no_dipole_e1"])
    ax.plot(times, k_d_e1, **TRACE_STYLE["dipole_enabled_e1"])
    ax.plot(times, k_b_e2, **TRACE_STYLE["no_dipole"])
    ax.plot(times, k_d_e2, **TRACE_STYLE["dipole_enabled"])

    arrival_b_e1 = float(baseline_output.arrival_times[e1_vertex])
    arrival_b_e2 = float(baseline_output.arrival_times[e2_vertex])
    arrival_d_e1 = float(dipole_output.arrival_times[e1_vertex])
    arrival_d_e2 = float(dipole_output.arrival_times[e2_vertex])

    for x_time, curve, style_key in (
        (arrival_b_e1, k_b_e1, "no_dipole"),
        (arrival_b_e2, k_b_e2, "no_dipole"),
        (arrival_d_e1, k_d_e1, "dipole_enabled"),
        (arrival_d_e2, k_d_e2, "dipole_enabled"),
    ):
        y_val = _trace_value(times, curve, x_time)
        ax.scatter(x_time, y_val, **MARKER_STYLE[style_key])
        ax.axvline(x_time, color="0.80", lw=0.75, ls=":")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Extracellular K+ (mM)")
    ax.set_xlim(0.0, float(times[-1]))
    ax.set_ylim(2.0, max(float(np.max(k_b_e2)), float(np.max(k_d_e2))) + 1.6)
    legend_handles = [
        Line2D([0], [0], **TRACE_STYLE["no_dipole"]),
        Line2D([0], [0], **TRACE_STYLE["dipole_enabled"]),
    ]
    legend_labels = [
        f"E2 no-dipole ({DISPLAY_VALUES['representative']['delay_no_dipole']})",
        f"E2 dipole-enabled ({DISPLAY_VALUES['representative']['delay_dipole']})",
    ]
    legend = ax.legend(
        legend_handles,
        legend_labels,
        loc="lower left",
        bbox_to_anchor=(0.36, 0.13),
        frameon=True,
        framealpha=0.95,
        fontsize=7.6,
        title=f"E2 traces\nΔ delay = {DISPLAY_VALUES['representative']['delay_delta']}",
        title_fontsize=7.9,
        borderpad=0.5,
        handlelength=2.4,
    )
    legend.get_frame().set_edgecolor("0.80")
    ax.grid(alpha=0.20, lw=0.5)
    _panel_label(ax, "A")


def _panel_b(ax, mesh, baseline_output, dipole_output, e1_vertex: int, e2_vertex: int) -> None:
    path_vertices = _shortest_path_vertices(baseline_output.operators.graph, e1_vertex, e2_vertex)
    path_coords = mesh.vertices[path_vertices]
    path_distance = _path_distance(path_coords)
    baseline_arrival = baseline_output.arrival_times[path_vertices]
    dipole_arrival = dipole_output.arrival_times[path_vertices]
    path_depth = np.asarray(mesh.sulcal_depth[path_vertices], dtype=float)
    span_start, span_end = _sulcal_span(path_distance, path_depth)

    ax.axvspan(span_start, span_end, color="0.90", zorder=0)
    ax.fill_between(path_distance, baseline_arrival, dipole_arrival, color="0.75", alpha=0.35, zorder=1)
    ax.plot(path_distance, baseline_arrival, **TRACE_STYLE["no_dipole"])
    ax.plot(path_distance, dipole_arrival, **TRACE_STYLE["dipole_enabled"])
    ax.scatter([path_distance[0], path_distance[-1]], [baseline_arrival[0], baseline_arrival[-1]], **MARKER_STYLE["no_dipole"])
    ax.scatter([path_distance[0], path_distance[-1]], [dipole_arrival[0], dipole_arrival[-1]], **MARKER_STYLE["dipole_enabled"])

    ax.set_xlabel("Geodesic distance along E1-E2 path (mm)")
    ax.set_ylabel("Front arrival time (s)")
    ax.set_xlim(0.0, float(path_distance[-1]))
    ax.set_ylim(
        min(float(np.nanmin(baseline_arrival)), float(np.nanmin(dipole_arrival))) - 3.0,
        max(float(np.nanmax(baseline_arrival)), float(np.nanmax(dipole_arrival))) + 5.5,
    )
    ax.text(0.03, 0.96, f"No-dipole: {DISPLAY_VALUES['representative']['speed_no_dipole']}", transform=ax.transAxes, ha="left", va="top", fontsize=8.5)
    ax.text(0.03, 0.89, f"Dipole-enabled: {DISPLAY_VALUES['representative']['speed_dipole']}", transform=ax.transAxes, ha="left", va="top", fontsize=8.5)
    ax.text(0.03, 0.82, f"Δ speed: {DISPLAY_VALUES['representative']['speed_delta']}", transform=ax.transAxes, ha="left", va="top", fontsize=8.5)
    ax.text(
        0.5 * (span_start + span_end),
        ax.get_ylim()[0] + 2.2,
        "sulcal segment",
        ha="center",
        va="bottom",
        fontsize=8.0,
        color="0.25",
    )
    ax.text(path_distance[-1] - 0.10, baseline_arrival[-1] - 1.7, "no-dipole", ha="right", va="top", fontsize=8.2)
    ax.text(path_distance[-1] - 0.10, dipole_arrival[-1] + 1.0, "dipole-enabled", ha="right", va="bottom", fontsize=8.2)
    ax.grid(alpha=0.20, lw=0.5)
    _panel_label(ax, "B")


def _panel_c(parent_spec, summary: dict[str, object]):
    ax = plt.subplot(parent_spec)
    ax.set_axis_off()
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    _panel_label(ax, "C")

    x_metric = 0.04
    x_no = 0.38
    x_dip = 0.57
    x_delta = 0.78

    ax.text(x_metric, 0.88, "Readout", fontsize=9.2, fontweight="bold", ha="left", va="center")
    ax.text(x_no, 0.88, "No-dipole", fontsize=9.2, fontweight="bold", ha="center", va="center")
    ax.text(x_dip, 0.88, "Dipole-enabled", fontsize=9.2, fontweight="bold", ha="center", va="center")
    ax.text(x_delta, 0.88, "Change", fontsize=9.2, fontweight="bold", ha="center", va="center")
    ax.plot([0.04, 0.96], [0.82, 0.82], color="0.75", lw=0.8)

    rows = [
        ("Cross-fold speed", "2.558 mm/min", "2.476 mm/min", "-0.082 mm/min (-3.21%)"),
        ("E1-E2 delay", "130.5 s", "134.7 s", "+4.2 s"),
        ("Max |Ve|", "18.99 mV", "19.50 mV", "+0.51 mV"),
    ]
    y_positions = [0.68, 0.52, 0.36]
    for (label, no_val, dip_val, delta_val), y in zip(rows, y_positions):
        ax.text(x_metric, y, label, fontsize=9.2, ha="left", va="center")
        ax.text(x_no, y, no_val, fontsize=9.0, ha="center", va="center")
        ax.text(x_dip, y, dip_val, fontsize=9.0, ha="center", va="center")
        ax.text(x_delta, y, delta_val, fontsize=9.0, ha="center", va="center")
        ax.plot([0.04, 0.96], [y - 0.08, y - 0.08], color="0.92", lw=0.8)

    return ax


def build_figure(summary: dict[str, object], trace_dt: float) -> plt.Figure:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 10,
            "axes.labelsize": 10,
            "xtick.labelsize": 8.7,
            "ytick.labelsize": 8.7,
            "font.family": "DejaVu Sans",
        }
    )

    mesh, baseline_output, dipole_output, _stimulus_vertex, e1_vertex, e2_vertex = _run_representative_outputs(
        summary,
        trace_dt,
    )
    fig = plt.figure(figsize=(11.4, 7.5))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.18, 0.94], height_ratios=[1.10, 0.94])

    panel_a = fig.add_subplot(grid[0, 0])
    _panel_a(panel_a, baseline_output.snapshot_times, baseline_output, dipole_output, e1_vertex, e2_vertex)

    panel_b = fig.add_subplot(grid[0, 1])
    _panel_b(panel_b, mesh, baseline_output, dipole_output, e1_vertex, e2_vertex)

    _panel_c(grid[1, :], summary)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.09, wspace=0.12, hspace=0.20)
    return fig


def main() -> None:
    args = parse_args()
    summary = _load_summary(args.summary_json.resolve())
    fig = build_figure(summary, trace_dt=float(args.trace_dt))
    output_prefix = args.output_prefix.resolve()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".pdf"))
    fig.savefig(output_prefix.with_suffix(".png"), dpi=300)
    plt.close(fig)
    print(f"Saved {output_prefix.with_suffix('.pdf')}")
    print(f"Saved {output_prefix.with_suffix('.png')}")


if __name__ == "__main__":
    main()
