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
from scipy.stats import spearmanr, wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))

matplotlib.use('Agg')
import matplotlib.pyplot as plt

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
        'max_abs_potential_au': _safe_float(np.nanmax(np.abs(output.electric_potential))),
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
    potential = np.asarray([row['max_abs_potential_au'] for row in rows], dtype=float)
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
    axes[2].set_ylabel('Max absolute potential (a.u.)')
    axes[2].set_title('Extracellular field amplitude')
    axes[2].grid(axis='y', alpha=0.25)

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
                'baseline_max_abs_potential_au': baseline['max_abs_potential_au'],
                'dipole_max_abs_potential_au': dipole['max_abs_potential_au'],
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
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()

    representative_rows = _representative_rows(output_dir)
    geometry_rows, geometry_summary = _geometry_sweep_rows(output_dir)
    figure_path = output_dir / 'mechanistic_sulcal_slowing.png'
    representative_figure_path = output_dir / 'mechanistic_representative_summary.png'
    _save_geometry_figure(geometry_rows, figure_path)
    _save_representative_figure(representative_rows, representative_figure_path)

    summary = {
        'representative_rows': representative_rows,
        'geometry_summary': geometry_summary,
        'figure_path': str(figure_path),
        'representative_figure_path': str(representative_figure_path),
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


if __name__ == '__main__':
    main()
