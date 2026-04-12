import argparse
import csv
import dataclasses as dc
import json
import math
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
from scipy.sparse import csgraph
from scipy.stats import spearmanr, wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

from csd_sulcus.atlas_patch import prepare_atlas_patch_pair
from csd_sulcus.surface_io import generate_folded_strip_mesh
from csd_sulcus.surface_mechanistic import (
    MechanisticSurfaceParams,
    mechanistic_edge_speed_stats,
    mechanistic_surface_arrival_speed_mm_min,
    run_mechanistic_surface_simulation,
)
from run_surface_representative import choose_auto_vertices


REPRESENTATIVE_CASES = {
    'mechanistic_multion_baseline': MechanisticSurfaceParams(
        final_t_end=220.0,
        enable_dipole_alignment=False,
        enable_vascular_feedback=False,
    ),
    'mechanistic_multion_dipole': MechanisticSurfaceParams(
        final_t_end=220.0,
        enable_dipole_alignment=True,
        enable_vascular_feedback=False,
    ),
    'mechanistic_multion_full_coupled': MechanisticSurfaceParams(
        final_t_end=220.0,
        enable_dipole_alignment=True,
        enable_vascular_feedback=True,
    ),
}


def _safe_float(value: float) -> float | None:
    value = float(value)
    if math.isfinite(value):
        return value
    return None


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_ready(item) for item in value.tolist()]
    if isinstance(value, (np.floating, np.integer)):
        return _safe_float(float(value))
    if isinstance(value, float):
        return _safe_float(value)
    return value


def _compute_case_metrics(output, e1_vertex: int, e2_vertex: int, roi_radius_mm: float = 1.0) -> dict[str, float | bool | None]:
    edge_stats = mechanistic_edge_speed_stats(output)
    e1_arrival = _safe_float(output.arrival_times[e1_vertex])
    e2_arrival = _safe_float(output.arrival_times[e2_vertex])
    cross_fold_delay = None
    if e1_arrival is not None and e2_arrival is not None:
        cross_fold_delay = e2_arrival - e1_arrival
    return {
        'dt_used_s': _safe_float(output.dt_used),
        'arrival_speed_mm_min': _safe_float(
            mechanistic_surface_arrival_speed_mm_min(output, e1_vertex, e2_vertex, radius_mm=roi_radius_mm)
        ),
        'e1_arrival_s': e1_arrival,
        'e2_arrival_s': e2_arrival,
        'cross_fold_delay_s': _safe_float(cross_fold_delay) if cross_fold_delay is not None else None,
        'median_edge_speed_mm_min': _safe_float(edge_stats['median_edge_speed_mm_min']),
        'deep_edge_speed_mm_min': _safe_float(edge_stats['deep_edge_speed_mm_min']),
        'shallow_edge_speed_mm_min': _safe_float(edge_stats['shallow_edge_speed_mm_min']),
        'crossed_fraction': _safe_float(np.mean(np.isfinite(output.arrival_times))),
        'field_reference_mV': _safe_float(output.params.field_reference_mV),
        'max_abs_potential_hat': _safe_float(np.nanmax(np.abs(output.electric_potential))),
        'max_abs_potential_mV': _safe_float(output.params.field_reference_mV * np.nanmax(np.abs(output.electric_potential))),
        'max_potassium_e_mM': _safe_float(np.nanmax(output.potassium_e)),
        'min_sodium_e_mM': _safe_float(np.nanmin(output.sodium_e)),
        'max_swelling_au': _safe_float(np.nanmax(output.swelling)),
        'min_oxygen_au': _safe_float(np.nanmin(output.oxygen)),
        'min_perfusion_au': _safe_float(np.nanmin(output.perfusion)),
        'dipole_alignment': bool(output.params.enable_dipole_alignment),
        'vascular_feedback': bool(output.params.enable_vascular_feedback),
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open('r', newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def _representative_case_display_name(case_label: str) -> str:
    mapping = {
        'mechanistic_multion_baseline': 'Baseline mechanistic',
        'mechanistic_multion_dipole': 'Dipole-aligned mechanistic',
        'mechanistic_multion_full_coupled': 'Full coupled sensitivity',
    }
    return mapping.get(case_label, case_label.replace('_', ' ').title())


def _write_table_s2_from_representative_csv(csv_path: Path, output_path: Path) -> None:
    rows = _read_csv_rows(csv_path)
    if not rows:
        raise RuntimeError(f'No representative rows were found in {csv_path}.')

    lines = [
        r'\begin{table}[t]',
        r'\centering',
        r'\caption{Exact representative-run outputs underlying manuscript Table~1. Values are copied directly from the saved final representative run, not from a rerun.}',
        r'\label{tab:s2_exact_representative_run}',
        r'\scriptsize',
        r'\resizebox{\textwidth}{!}{%',
        r'\begin{tabular}{@{}lrrrrrrrrrr@{}}',
        r'\toprule',
        r'Case & $dt$ (s) & Speed (mm/min) & E1 arrival (s) & E2 arrival (s) & Delay (s) & Max $|V_e|$ (mV) & Max $K_e$ (mM) & Max swelling & Min oxygen & Min perfusion \\',
        r'\midrule',
    ]

    for row in rows:
        lines.append(
            ' & '.join(
                [
                    _representative_case_display_name(row['case_label']),
                    row['dt_used_s'],
                    row['arrival_speed_mm_min'],
                    row['e1_arrival_s'],
                    row['e2_arrival_s'],
                    row['cross_fold_delay_s'],
                    row['max_abs_potential_mV'],
                    row['max_potassium_e_mM'],
                    row['max_swelling_au'],
                    row['min_oxygen_au'],
                    row['min_perfusion_au'],
                ]
            )
            + r' \\'
        )

    lines.extend(
        [
            r'\bottomrule',
            r'\end{tabular}%',
            r'}',
            r'\end{table}',
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def _bootstrap_mean_ci(values: np.ndarray, *, seed: int, n_resamples: int = 4000) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(n_resamples, values.size))
    samples = values[indices].mean(axis=1)
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return float(lo), float(hi)


def _wilcoxon_greater(x: np.ndarray, y: np.ndarray | None = None) -> dict[str, float | None]:
    x = np.asarray(x, dtype=float)
    if y is None:
        y = np.zeros_like(x)
    else:
        y = np.asarray(y, dtype=float)
    if x.size == 0 or np.allclose(x, y):
        return {'statistic': None, 'pvalue': None}
    result = wilcoxon(x, y, alternative='greater')
    return {'statistic': float(result.statistic), 'pvalue': float(result.pvalue)}


def _paired_statistics(rows: list[dict]) -> dict[str, object]:
    folded = [row for row in rows if row['fold_depth_mm'] > 0.0]
    flat = [row for row in rows if row['fold_depth_mm'] == 0.0]

    baseline_speed = np.asarray([row['baseline_arrival_speed_mm_min'] for row in folded], dtype=float)
    dipole_speed = np.asarray([row['dipole_arrival_speed_mm_min'] for row in folded], dtype=float)
    baseline_delay = np.asarray([row['baseline_cross_fold_delay_s'] for row in folded], dtype=float)
    dipole_delay = np.asarray([row['dipole_cross_fold_delay_s'] for row in folded], dtype=float)
    baseline_deep = np.asarray([row['baseline_deep_edge_speed_mm_min'] for row in folded], dtype=float)
    dipole_deep = np.asarray([row['dipole_deep_edge_speed_mm_min'] for row in folded], dtype=float)

    speed_slowdown = baseline_speed - dipole_speed
    delay_increase = dipole_delay - baseline_delay
    deep_edge_slowdown = baseline_deep - dipole_deep
    relative_slowdown = 100.0 * speed_slowdown / baseline_speed
    fold_depths = np.asarray([row['fold_depth_mm'] for row in folded], dtype=float)
    fold_severity = np.asarray([row['fold_depth_mm'] / row['fold_sigma_mm'] for row in folded], dtype=float)

    depth_vs_slowdown = spearmanr(fold_depths, speed_slowdown)
    severity_vs_slowdown = spearmanr(fold_severity, speed_slowdown)
    flat_control = flat[0] if flat else None

    speed_ci = _bootstrap_mean_ci(speed_slowdown, seed=7)
    delay_ci = _bootstrap_mean_ci(delay_increase, seed=11)

    return {
        'n_folded_geometries': int(len(folded)),
        'n_flat_controls': int(len(flat)),
        'folded_cases_slower_with_dipole': int(np.sum(speed_slowdown > 0.0)),
        'mean_speed_slowdown_mm_min': float(np.mean(speed_slowdown)),
        'speed_slowdown_mm_min_ci95': [speed_ci[0], speed_ci[1]],
        'sd_speed_slowdown_mm_min': float(np.std(speed_slowdown, ddof=1)),
        'mean_relative_speed_slowdown_pct': float(np.mean(relative_slowdown)),
        'mean_delay_increase_s': float(np.mean(delay_increase)),
        'delay_increase_s_ci95': [delay_ci[0], delay_ci[1]],
        'sd_delay_increase_s': float(np.std(delay_increase, ddof=1)),
        'mean_deep_edge_speed_slowdown_mm_min': float(np.mean(deep_edge_slowdown)),
        'sd_deep_edge_speed_slowdown_mm_min': float(np.std(deep_edge_slowdown, ddof=1)),
        'speed_wilcoxon': _wilcoxon_greater(baseline_speed, dipole_speed),
        'delay_wilcoxon': _wilcoxon_greater(dipole_delay, baseline_delay),
        'deep_edge_wilcoxon': _wilcoxon_greater(baseline_deep, dipole_deep),
        'depth_vs_slowdown_spearman': {
            'rho': float(depth_vs_slowdown.statistic),
            'pvalue': float(depth_vs_slowdown.pvalue),
        },
        'severity_vs_slowdown_spearman': {
            'rho': float(severity_vs_slowdown.statistic),
            'pvalue': float(severity_vs_slowdown.pvalue),
        },
        'flat_control': flat_control,
    }


def _save_geometry_figure(rows: list[dict], output_path: Path) -> None:
    folded = [row for row in rows if row['fold_depth_mm'] > 0.0]
    flat = [row for row in rows if row['fold_depth_mm'] == 0.0][0]
    depths = np.asarray([row['fold_depth_mm'] for row in folded], dtype=float)
    severity = np.asarray([row['fold_depth_mm'] / row['fold_sigma_mm'] for row in folded], dtype=float)
    slowdown = np.asarray([row['speed_slowdown_mm_min'] for row in folded], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    cmap = plt.get_cmap('viridis')
    depth_norm = plt.Normalize(vmin=float(np.min(depths)), vmax=float(np.max(depths)))

    for row in folded:
        color = cmap(depth_norm(row['fold_depth_mm']))
        axes[0].plot(
            [0, 1],
            [row['baseline_arrival_speed_mm_min'], row['dipole_arrival_speed_mm_min']],
            color=color,
            marker='o',
            linewidth=1.4,
            alpha=0.9,
        )
    axes[0].set_xticks([0, 1], ['No dipole', 'Sulcal dipole'])
    axes[0].set_ylabel('Cross-fold speed (mm/min)')
    axes[0].set_title('Folded geometries')
    axes[0].grid(alpha=0.25)

    scatter = axes[1].scatter(severity, slowdown, c=depths, cmap=cmap, s=52, edgecolor='black', linewidth=0.4)
    if severity.size >= 2:
        line_x = np.linspace(float(np.min(severity)), float(np.max(severity)), 100)
        line_y = np.polyval(np.polyfit(severity, slowdown, deg=1), line_x)
        axes[1].plot(line_x, line_y, color='black', linestyle='--', linewidth=1.2)
    axes[1].scatter(
        [0.0],
        [flat['speed_slowdown_mm_min']],
        color='white',
        edgecolor='black',
        s=72,
        marker='s',
        linewidth=0.8,
        label='Flat control',
    )
    axes[1].axhline(0.0, color='black', linewidth=0.8, alpha=0.45)
    axes[1].set_xlabel('Fold severity (depth / sigma)')
    axes[1].set_ylabel('Dipole-induced slowing (mm/min)')
    axes[1].set_title('Stronger folds slow more')
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc='upper left', frameon=False)

    colorbar = fig.colorbar(scatter, ax=axes[1], fraction=0.046, pad=0.02)
    colorbar.set_label('Fold depth (mm)')
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _save_representative_figure(rows: list[dict], output_path: Path) -> None:
    labels = ['Baseline', 'Dipole', 'Full coupled']
    speed = np.asarray([row['arrival_speed_mm_min'] for row in rows], dtype=float)
    delay = np.asarray([row['cross_fold_delay_s'] for row in rows], dtype=float)
    potential = np.asarray([row['max_abs_potential_mV'] for row in rows], dtype=float)
    colors = ['#4c78a8', '#e45756', '#72b7b2']

    fig, axes = plt.subplots(1, 3, figsize=(11.2, 4.0), constrained_layout=True)
    x = np.arange(len(labels))

    axes[0].bar(x, speed, color=colors, edgecolor='black', linewidth=0.5)
    axes[0].set_xticks(x, labels, rotation=15)
    axes[0].set_ylabel('Cross-fold speed (mm/min)')
    axes[0].set_title('Representative speed')
    axes[0].grid(axis='y', alpha=0.25)

    axes[1].bar(x, delay, color=colors, edgecolor='black', linewidth=0.5)
    axes[1].set_xticks(x, labels, rotation=15)
    axes[1].set_ylabel('E1-E2 delay (s)')
    axes[1].set_title('Representative delay')
    axes[1].grid(axis='y', alpha=0.25)

    axes[2].bar(x, potential, color=colors, edgecolor='black', linewidth=0.5)
    axes[2].set_xticks(x, labels, rotation=15)
    axes[2].set_ylabel('Max absolute potential (mV)')
    axes[2].set_title('Extracellular field amplitude')
    axes[2].grid(axis='y', alpha=0.25)

    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _project_vertices_2d(vertices: np.ndarray) -> np.ndarray:
    centered = np.asarray(vertices, dtype=float) - np.mean(vertices, axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    return centered @ vh[:2].T


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
            raise RuntimeError('Could not reconstruct the requested geodesic path.')
        path.append(cursor)
    path.reverse()
    return np.asarray(path, dtype=int)


def _select_propagation_times(baseline_output, dipole_output, e1_vertex: int, e2_vertex: int) -> np.ndarray:
    anchors = np.asarray(
        [
            baseline_output.arrival_times[e1_vertex],
            baseline_output.arrival_times[e2_vertex],
            dipole_output.arrival_times[e1_vertex],
            dipole_output.arrival_times[e2_vertex],
        ],
        dtype=float,
    )
    anchors = np.sort(anchors[np.isfinite(anchors)])
    if anchors.size < 2:
        return np.asarray([60.0, 100.0, 140.0], dtype=float)
    return np.asarray(
        [
            0.60 * anchors[0],
            0.55 * (anchors[1] + anchors[-1]),
            0.92 * anchors[-1],
        ],
        dtype=float,
    )


def _render_snapshot_panel(ax, mesh, field: np.ndarray, title: str, *, vmin: float, vmax: float) -> None:
    projected = _project_vertices_2d(mesh.vertices)
    triangulation = mtri.Triangulation(projected[:, 0], projected[:, 1], triangles=mesh.faces)
    artist = ax.tripcolor(triangulation, field, shading='gouraud', cmap='magma', vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')
    ax.set_frame_on(False)
    return artist


def _save_propagation_figure(output_path: Path) -> dict[str, object]:
    mesh = generate_folded_strip_mesh(nx=64, ny=28, length_mm=22.0, width_mm=10.0, fold_depth_mm=2.4, fold_sigma_mm=1.5)
    stimulus_vertex, e1_vertex, e2_vertex = choose_auto_vertices(mesh)
    baseline_params = REPRESENTATIVE_CASES['mechanistic_multion_baseline']
    dipole_params = REPRESENTATIVE_CASES['mechanistic_multion_dipole']

    baseline_probe = run_mechanistic_surface_simulation(mesh, baseline_params, stimulus_vertex=stimulus_vertex)
    dipole_probe = run_mechanistic_surface_simulation(mesh, dipole_params, stimulus_vertex=stimulus_vertex)
    snapshot_times = _select_propagation_times(baseline_probe, dipole_probe, e1_vertex, e2_vertex)
    trace_times = np.linspace(0.0, baseline_params.final_t_end, 121)

    baseline_output = run_mechanistic_surface_simulation(
        mesh,
        baseline_params,
        stimulus_vertex=stimulus_vertex,
        snapshot_times=trace_times,
    )
    dipole_output = run_mechanistic_surface_simulation(
        mesh,
        dipole_params,
        stimulus_vertex=stimulus_vertex,
        snapshot_times=trace_times,
    )

    path_vertices = _shortest_path_vertices(baseline_output.operators.graph, e1_vertex, e2_vertex)
    path_coords = mesh.vertices[path_vertices]
    path_distance = np.zeros(path_vertices.size, dtype=float)
    if path_vertices.size > 1:
        path_distance[1:] = np.cumsum(np.linalg.norm(np.diff(path_coords, axis=0), axis=1))

    snapshot_indices = np.asarray([int(np.argmin(np.abs(trace_times - time_s))) for time_s in snapshot_times], dtype=int)
    field_max = float(
        max(
            np.nanmax(baseline_output.snapshot_potassium_e[snapshot_indices]),
            np.nanmax(dipole_output.snapshot_potassium_e[snapshot_indices]),
        )
    )
    field_min = float(
        min(
            np.nanmin(baseline_output.snapshot_potassium_e[snapshot_indices]),
            np.nanmin(dipole_output.snapshot_potassium_e[snapshot_indices]),
        )
    )

    fig = plt.figure(figsize=(14.5, 7.6), constrained_layout=True)
    grid = fig.add_gridspec(2, 4, width_ratios=[1.0, 1.0, 1.0, 1.18])

    for row_idx, (label, output) in enumerate(
        [('No dipole', baseline_output), ('Sulcal dipole', dipole_output)]
    ):
        color_artist = None
        for col_idx, snap_idx in enumerate(snapshot_indices):
            ax = fig.add_subplot(grid[row_idx, col_idx])
            color_artist = _render_snapshot_panel(
                ax,
                mesh,
                output.snapshot_potassium_e[snap_idx],
                f'{label}\nt = {trace_times[snap_idx]:.0f} s',
                vmin=field_min,
                vmax=field_max,
            )
            if row_idx == 1:
                ax.set_xlabel(' ')
        cax = fig.add_subplot(grid[row_idx, 3])
        kymograph = output.snapshot_potassium_e[:, path_vertices].T
        image = cax.imshow(
            kymograph,
            origin='lower',
            aspect='auto',
            cmap='magma',
            vmin=field_min,
            vmax=field_max,
            extent=[trace_times[0], trace_times[-1], path_distance[0], path_distance[-1]],
        )
        cax.set_title(f'{label}\nCross-sulcal kymograph')
        cax.set_xlabel('Time (s)')
        cax.set_ylabel('Geodesic distance (mm)')
        cax.axhline(path_distance[0], color='white', linestyle=':', linewidth=0.8, alpha=0.6)
        cax.axhline(path_distance[-1], color='white', linestyle=':', linewidth=0.8, alpha=0.6)
        fig.colorbar(image, ax=cax, fraction=0.046, pad=0.02, label='Extracellular K+ (mM)')

    fig.suptitle('Mechanistic wave propagation on the representative folded surface', fontsize=14)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    return {
        'stimulus_vertex': int(stimulus_vertex),
        'electrode_1_vertex': int(e1_vertex),
        'electrode_2_vertex': int(e2_vertex),
        'snapshot_times_s': [float(trace_times[idx]) for idx in snapshot_indices],
        'path_length_mm': float(path_distance[-1]),
    }


def _save_atlas_patch_figure(patch_pair, rows: list[dict], output_path: Path) -> None:
    atlas_projection = _project_vertices_2d(patch_pair.atlas_mesh.vertices)
    triangulation = mtri.Triangulation(
        atlas_projection[:, 0],
        atlas_projection[:, 1],
        triangles=patch_pair.atlas_mesh.faces,
    )
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), constrained_layout=True)

    artist = axes[0].tripcolor(
        triangulation,
        patch_pair.atlas_mesh.sulcal_depth,
        shading='gouraud',
        cmap='cividis',
    )
    axes[0].scatter(
        atlas_projection[patch_pair.sulcal_roi_mask, 0],
        atlas_projection[patch_pair.sulcal_roi_mask, 1],
        s=7,
        color='#e45756',
        alpha=0.65,
        label='Sulcal patch',
    )
    axes[0].scatter(
        atlas_projection[patch_pair.flat_roi_mask, 0],
        atlas_projection[patch_pair.flat_roi_mask, 1],
        s=7,
        color='#4c78a8',
        alpha=0.65,
        label='Flatter patch',
    )
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    axes[0].set_aspect('equal')
    axes[0].legend(loc='lower left', frameon=False)
    fig.colorbar(artist, ax=axes[0], fraction=0.046, pad=0.02, label='Normalized sulcal depth')

    x = np.arange(len(rows))
    baseline = np.asarray([row['baseline_traversal_speed_mm_min'] for row in rows], dtype=float)
    dipole = np.asarray([row['dipole_traversal_speed_mm_min'] for row in rows], dtype=float)
    for idx, row in enumerate(rows):
        axes[1].plot([0, 1], [baseline[idx], dipole[idx]], marker='o', linewidth=1.8, label=row['patch_label'])
    axes[1].set_xticks([0, 1], ['No dipole', 'Sulcal dipole'])
    axes[1].set_ylabel('Stimulus-to-downstream speed (mm/min)')
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)

    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _representative_rows(output_dir: Path) -> list[dict]:
    mesh = generate_folded_strip_mesh(nx=64, ny=28, length_mm=22.0, width_mm=10.0, fold_depth_mm=2.4, fold_sigma_mm=1.5)
    stimulus_vertex, e1_vertex, e2_vertex = choose_auto_vertices(mesh)
    rows: list[dict] = []
    for label, params in REPRESENTATIVE_CASES.items():
        output = run_mechanistic_surface_simulation(mesh, params, stimulus_vertex=stimulus_vertex)
        metrics = _compute_case_metrics(output, e1_vertex, e2_vertex)
        rows.append(
            {
                'case_label': label,
                'case_title': label.replace('_', ' '),
                'mesh_vertices': mesh.n_vertices,
                'mesh_faces': mesh.n_faces,
                'fold_depth_mm': float(mesh.metadata.get('fold_depth_mm', 2.4)),
                'fold_sigma_mm': float(mesh.metadata.get('fold_sigma_mm', 1.5)),
                'stimulus_vertex': int(stimulus_vertex),
                'electrode_1_vertex': int(e1_vertex),
                'electrode_2_vertex': int(e2_vertex),
                **metrics,
            }
        )
    _write_csv(output_dir / 'mechanistic_representative_summary.csv', rows)
    return rows


def _atlas_patch_rows(output_dir: Path, atlas_cache_dir: Path) -> tuple[list[dict], dict[str, object]]:
    patch_pair = prepare_atlas_patch_pair(atlas_cache_dir, patch_radius_mm=12.0, min_separation_mm=30.0)
    base_params = MechanisticSurfaceParams(final_t_end=180.0, enable_vascular_feedback=False)

    rows: list[dict] = []
    for patch in (patch_pair.sulcal_patch, patch_pair.flat_patch):
        baseline_output = run_mechanistic_surface_simulation(
            patch.mesh,
            dc.replace(base_params, enable_dipole_alignment=False),
            stimulus_vertex=patch.stimulus_vertex,
        )
        dipole_output = run_mechanistic_surface_simulation(
            patch.mesh,
            dc.replace(base_params, enable_dipole_alignment=True),
            stimulus_vertex=patch.stimulus_vertex,
        )
        baseline = _compute_case_metrics(baseline_output, patch.electrode_1_vertex, patch.electrode_2_vertex)
        dipole = _compute_case_metrics(dipole_output, patch.electrode_1_vertex, patch.electrode_2_vertex)
        baseline_traversal_speed = mechanistic_surface_arrival_speed_mm_min(
            baseline_output,
            patch.stimulus_vertex,
            patch.electrode_2_vertex,
            radius_mm=1.0,
        )
        dipole_traversal_speed = mechanistic_surface_arrival_speed_mm_min(
            dipole_output,
            patch.stimulus_vertex,
            patch.electrode_2_vertex,
            radius_mm=1.0,
        )
        baseline_downstream_arrival = _safe_float(baseline_output.arrival_times[patch.electrode_2_vertex])
        dipole_downstream_arrival = _safe_float(dipole_output.arrival_times[patch.electrode_2_vertex])
        rows.append(
            {
                'patch_label': patch.label,
                'patch_vertices': patch.mesh.n_vertices,
                'patch_faces': patch.mesh.n_faces,
                'mean_sulcal_depth': _safe_float(np.mean(patch.mesh.sulcal_depth)),
                'mean_thickness_mm': _safe_float(np.mean(patch.mesh.thickness)),
                'baseline_traversal_speed_mm_min': _safe_float(baseline_traversal_speed),
                'dipole_traversal_speed_mm_min': _safe_float(dipole_traversal_speed),
                'baseline_downstream_arrival_s': baseline_downstream_arrival,
                'dipole_downstream_arrival_s': dipole_downstream_arrival,
                'baseline_inner_delay_s': baseline['cross_fold_delay_s'],
                'dipole_inner_delay_s': dipole['cross_fold_delay_s'],
                'speed_slowdown_mm_min': _safe_float(baseline_traversal_speed - dipole_traversal_speed),
                'delay_increase_s': _safe_float(dipole_downstream_arrival - baseline_downstream_arrival),
                'baseline_max_abs_potential_mV': baseline['max_abs_potential_mV'],
                'dipole_max_abs_potential_mV': dipole['max_abs_potential_mV'],
            }
        )

    _write_csv(output_dir / 'mechanistic_atlas_patch_check.csv', rows)
    figure_path = output_dir / 'mechanistic_atlas_patch_qc.png'
    _save_atlas_patch_figure(patch_pair, rows, figure_path)
    sulcal_row = next(row for row in rows if row['patch_label'] == 'atlas_sulcal_patch')
    flat_row = next(row for row in rows if row['patch_label'] == 'atlas_flat_patch')
    summary = {
        'sulcal_patch_slowdown_mm_min': _safe_float(sulcal_row['speed_slowdown_mm_min']),
        'flat_patch_slowdown_mm_min': _safe_float(flat_row['speed_slowdown_mm_min']),
        'sulcal_patch_delay_increase_s': _safe_float(sulcal_row['delay_increase_s']),
        'flat_patch_delay_increase_s': _safe_float(flat_row['delay_increase_s']),
        'effect_preserved_on_sulcal_patch': bool(float(sulcal_row['speed_slowdown_mm_min']) > 0.0),
        'flat_patch_effect_smaller': bool(float(sulcal_row['speed_slowdown_mm_min']) > float(flat_row['speed_slowdown_mm_min'])),
        'figure_path': str(figure_path),
        'atlas_source_dir': str(patch_pair.atlas_source_dir) if patch_pair.atlas_source_dir is not None else None,
    }
    return rows, summary


def _geometry_sweep_rows(output_dir: Path) -> tuple[list[dict], dict[str, object]]:
    rows: list[dict] = []
    geometries = [(0.0, 1.5)] + [(depth, sigma) for depth in (1.2, 1.8, 2.4, 3.0) for sigma in (1.1, 1.5, 1.9)]
    base_params = MechanisticSurfaceParams(final_t_end=210.0, enable_vascular_feedback=False)

    for fold_depth_mm, fold_sigma_mm in geometries:
        mesh = generate_folded_strip_mesh(
            nx=52,
            ny=24,
            length_mm=22.0,
            width_mm=10.0,
            fold_depth_mm=fold_depth_mm,
            fold_sigma_mm=fold_sigma_mm,
        )
        stimulus_vertex, e1_vertex, e2_vertex = choose_auto_vertices(mesh)
        baseline_output = run_mechanistic_surface_simulation(
            mesh,
            dc.replace(base_params, enable_dipole_alignment=False),
            stimulus_vertex=stimulus_vertex,
        )
        dipole_output = run_mechanistic_surface_simulation(
            mesh,
            dc.replace(base_params, enable_dipole_alignment=True),
            stimulus_vertex=stimulus_vertex,
        )
        baseline = _compute_case_metrics(baseline_output, e1_vertex, e2_vertex)
        dipole = _compute_case_metrics(dipole_output, e1_vertex, e2_vertex)
        rows.append(
            {
                'fold_depth_mm': float(fold_depth_mm),
                'fold_sigma_mm': float(fold_sigma_mm),
                'fold_severity': float(fold_depth_mm / fold_sigma_mm if fold_sigma_mm > 0.0 else 0.0),
                'mesh_vertices': mesh.n_vertices,
                'mesh_faces': mesh.n_faces,
                'baseline_arrival_speed_mm_min': baseline['arrival_speed_mm_min'],
                'dipole_arrival_speed_mm_min': dipole['arrival_speed_mm_min'],
                'baseline_cross_fold_delay_s': baseline['cross_fold_delay_s'],
                'dipole_cross_fold_delay_s': dipole['cross_fold_delay_s'],
                'baseline_deep_edge_speed_mm_min': baseline['deep_edge_speed_mm_min'],
                'dipole_deep_edge_speed_mm_min': dipole['deep_edge_speed_mm_min'],
                'baseline_max_abs_potential_mV': baseline['max_abs_potential_mV'],
                'dipole_max_abs_potential_mV': dipole['max_abs_potential_mV'],
                'speed_slowdown_mm_min': _safe_float(
                    baseline['arrival_speed_mm_min'] - dipole['arrival_speed_mm_min']
                ),
                'delay_increase_s': _safe_float(
                    dipole['cross_fold_delay_s'] - baseline['cross_fold_delay_s']
                ),
                'deep_edge_speed_slowdown_mm_min': _safe_float(
                    baseline['deep_edge_speed_mm_min'] - dipole['deep_edge_speed_mm_min']
                ),
            }
        )

    _write_csv(output_dir / 'mechanistic_geometry_sweep.csv', rows)
    summary = _paired_statistics(rows)
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(description='Run the stronger mechanistic surface electrodiffusion study.')
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=ROOT / 'outputs' / 'surface_mechanistic_study',
        help='Directory for CSV, JSON, and figure outputs.',
    )
    parser.add_argument(
        '--atlas-cache-dir',
        type=Path,
        default=ROOT / 'outputs' / 'atlas_cache',
        help='Directory for downloaded atlas assets used in the atlas patch sanity check.',
    )
    parser.add_argument(
        '--representative-csv',
        type=Path,
        default=None,
        help='Existing representative CSV to use when regenerating Table S2 without rerunning the study.',
    )
    parser.add_argument(
        '--table-s2-only',
        action='store_true',
        help='Regenerate manuscript Table S2 from an existing representative CSV and exit.',
    )
    parser.add_argument(
        '--table-s2-output',
        type=Path,
        default=ROOT / 'manuscript' / 'table_s2_exact_representative_run.tex',
        help='Destination for the manuscript-ready Table S2 LaTeX file.',
    )
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    representative_csv = (
        args.representative_csv.resolve()
        if args.representative_csv is not None
        else output_dir / 'mechanistic_representative_summary.csv'
    )
    table_s2_output = args.table_s2_output.resolve()

    if args.table_s2_only:
        _write_table_s2_from_representative_csv(representative_csv, table_s2_output)
        print(f'Regenerated Table S2 from {representative_csv}')
        print(f'Saved {table_s2_output}')
        return

    start = time.perf_counter()

    representative_rows = _representative_rows(output_dir)
    geometry_rows, geometry_summary = _geometry_sweep_rows(output_dir)
    atlas_rows, atlas_summary = _atlas_patch_rows(output_dir, args.atlas_cache_dir.resolve())
    figure_path = output_dir / 'mechanistic_sulcal_slowing.png'
    representative_figure_path = output_dir / 'mechanistic_representative_summary.png'
    propagation_figure_path = output_dir / 'mechanistic_wave_propagation.png'
    _save_geometry_figure(geometry_rows, figure_path)
    _save_representative_figure(representative_rows, representative_figure_path)
    propagation_summary = _save_propagation_figure(propagation_figure_path)
    _write_table_s2_from_representative_csv(
        output_dir / 'mechanistic_representative_summary.csv',
        table_s2_output,
    )

    summary = {
        'representative_rows': representative_rows,
        'geometry_summary': geometry_summary,
        'atlas_patch_rows': atlas_rows,
        'atlas_patch_summary': atlas_summary,
        'figure_path': str(figure_path),
        'representative_figure_path': str(representative_figure_path),
        'propagation_figure_path': str(propagation_figure_path),
        'propagation_summary': propagation_summary,
        'runtime_s': float(time.perf_counter() - start),
    }
    with (output_dir / 'mechanistic_study_summary.json').open('w', encoding='utf-8') as handle:
        json.dump(_json_ready(summary), handle, indent=2)

    print(f'Mechanistic study complete: {output_dir}')
    print(
        'Representative folded case: '
        f"baseline={representative_rows[0]['arrival_speed_mm_min']:.3f} mm/min, "
        f"dipole={representative_rows[1]['arrival_speed_mm_min']:.3f} mm/min"
    )
    print(
        'Folded geometry sweep: '
        f"{geometry_summary['folded_cases_slower_with_dipole']}/{geometry_summary['n_folded_geometries']} "
        'folded geometries slower with dipole alignment'
    )
    print(
        'Mean slowdown: '
        f"{geometry_summary['mean_speed_slowdown_mm_min']:.3f} mm/min "
        f"({geometry_summary['mean_relative_speed_slowdown_pct']:.2f}%)"
    )
    print(
        'Mean delay increase: '
        f"{geometry_summary['mean_delay_increase_s']:.2f} s"
    )
    print(
        'Atlas sanity check: '
        f"sulcal slowdown={atlas_summary['sulcal_patch_slowdown_mm_min']:.3f} mm/min, "
        f"flat slowdown={atlas_summary['flat_patch_slowdown_mm_min']:.3f} mm/min"
    )


if __name__ == '__main__':
    main()
