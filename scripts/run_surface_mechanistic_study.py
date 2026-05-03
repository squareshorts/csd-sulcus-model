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
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

from csd_sulcus.atlas_patch import prepare_atlas_multi_patch_panel
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
        'swelling_cap_au': _safe_float(output.params.swelling_target_max),
        'swelling_cap_fraction': _safe_float(np.nanmax(output.swelling) / max(output.params.swelling_target_max, 1e-6)),
        'min_oxygen_au': _safe_float(np.nanmin(output.oxygen)),
        'min_perfusion_au': _safe_float(np.nanmin(output.perfusion)),
        'dipole_alignment': bool(output.params.enable_dipole_alignment),
        'vascular_feedback': bool(output.params.enable_vascular_feedback),
        'dipole_kernel_mode': str(output.params.dipole_kernel_mode),
    }


def _safe_difference(lhs: float | None, rhs: float | None) -> float | None:
    if lhs is None or rhs is None:
        return None
    return _safe_float(float(lhs) - float(rhs))


def _paired_outputs(
    mesh,
    base_params: MechanisticSurfaceParams,
    stimulus_vertex: int,
    e1_vertex: int,
    e2_vertex: int,
    *,
    comparison_params: MechanisticSurfaceParams | None = None,
    snapshot_times=(),
):
    baseline_params = dc.replace(base_params, enable_dipole_alignment=False, dipole_kernel_mode='aligned')
    if comparison_params is None:
        comparison_params = dc.replace(base_params, enable_dipole_alignment=True, dipole_kernel_mode='aligned')
    baseline_output = run_mechanistic_surface_simulation(
        mesh,
        baseline_params,
        stimulus_vertex=stimulus_vertex,
        snapshot_times=snapshot_times,
    )
    comparison_output = run_mechanistic_surface_simulation(
        mesh,
        comparison_params,
        stimulus_vertex=stimulus_vertex,
        snapshot_times=snapshot_times,
    )
    baseline_metrics = _compute_case_metrics(baseline_output, e1_vertex, e2_vertex)
    comparison_metrics = _compute_case_metrics(comparison_output, e1_vertex, e2_vertex)
    return baseline_output, comparison_output, baseline_metrics, comparison_metrics


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
    axes[0].grid(axis='y', alpha=0.25)

    axes[1].bar(x, delay, color=colors, edgecolor='black', linewidth=0.5)
    axes[1].set_xticks(x, labels, rotation=15)
    axes[1].set_ylabel('E1-E2 delay (s)')
    axes[1].grid(axis='y', alpha=0.25)

    axes[2].bar(x, potential, color=colors, edgecolor='black', linewidth=0.5)
    axes[2].set_xticks(x, labels, rotation=15)
    axes[2].set_ylabel('Max absolute potential (mV)')
    axes[2].grid(axis='y', alpha=0.25)

    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _sensitivity_parameter_specs() -> list[tuple[str, str, list[float]]]:
    return [
        ('field_reference_mV', r'$V_0$ (mV)', [3.0, 5.0, 7.0]),
        ('electrodiffusion_mobility_fraction', r'$m_{\mathrm{ed}}$', [0.25, 0.50, 0.75]),
        ('dipole_screening_length_mm', r'$\ell_d$ (mm)', [2.5, 4.0, 5.5]),
        ('dipole_cutoff_mm', r'$d_c$ (mm)', [8.0, 12.0, 16.0]),
        ('dipole_field_gain', r'$\gamma_d$ (a.u.)', [1.00, 1.50, 2.00]),
    ]


def _sensitivity_geometry_specs() -> list[tuple[str, float, float]]:
    return [
        ('flat_control', 0.0, 1.5),
        ('low_severity', 1.2, 1.9),
        ('representative', 2.4, 1.5),
        ('high_severity', 3.0, 1.1),
    ]


def _is_monotone_non_decreasing(values: list[float], tolerance: float = 1e-6) -> bool:
    arr = np.asarray(values, dtype=float)
    if arr.size < 2 or not np.all(np.isfinite(arr)):
        return False
    return bool(np.all(np.diff(arr) >= -tolerance))


def _sensitivity_rows(output_dir: Path) -> tuple[list[dict], list[dict], dict[str, object], Path]:
    output_path = output_dir / 'mechanistic_dipole_sensitivity.csv'
    summary_path = output_dir / 'mechanistic_dipole_sensitivity_summary.csv'
    base_params = MechanisticSurfaceParams(final_t_end=190.0, enable_vascular_feedback=False)

    rows: list[dict] = []
    summary_rows: list[dict] = []
    for parameter_name, parameter_label, values in _sensitivity_parameter_specs():
        for parameter_value in values:
            setting_rows: list[dict] = []
            for geometry_label, fold_depth_mm, fold_sigma_mm in _sensitivity_geometry_specs():
                mesh = generate_folded_strip_mesh(
                    nx=52,
                    ny=24,
                    length_mm=22.0,
                    width_mm=10.0,
                    fold_depth_mm=fold_depth_mm,
                    fold_sigma_mm=fold_sigma_mm,
                )
                stimulus_vertex, e1_vertex, e2_vertex = choose_auto_vertices(mesh)
                tuned_params = dc.replace(base_params, **{parameter_name: parameter_value})
                _, _, baseline, dipole = _paired_outputs(
                    mesh,
                    tuned_params,
                    stimulus_vertex,
                    e1_vertex,
                    e2_vertex,
                )
                row = {
                    'parameter_name': parameter_name,
                    'parameter_label': parameter_label,
                    'parameter_value': float(parameter_value),
                    'geometry_label': geometry_label,
                    'fold_depth_mm': float(fold_depth_mm),
                    'fold_sigma_mm': float(fold_sigma_mm),
                    'fold_severity': float(fold_depth_mm / fold_sigma_mm if fold_sigma_mm > 0.0 else 0.0),
                    'baseline_arrival_speed_mm_min': baseline['arrival_speed_mm_min'],
                    'dipole_arrival_speed_mm_min': dipole['arrival_speed_mm_min'],
                    'baseline_cross_fold_delay_s': baseline['cross_fold_delay_s'],
                    'dipole_cross_fold_delay_s': dipole['cross_fold_delay_s'],
                    'speed_slowdown_mm_min': _safe_difference(
                        baseline['arrival_speed_mm_min'],
                        dipole['arrival_speed_mm_min'],
                    ),
                    'delay_increase_s': _safe_difference(
                        dipole['cross_fold_delay_s'],
                        baseline['cross_fold_delay_s'],
                    ),
                    'baseline_max_abs_potential_mV': baseline['max_abs_potential_mV'],
                    'dipole_max_abs_potential_mV': dipole['max_abs_potential_mV'],
                    'baseline_swelling_cap_fraction': baseline['swelling_cap_fraction'],
                    'dipole_swelling_cap_fraction': dipole['swelling_cap_fraction'],
                }
                rows.append(row)
                setting_rows.append(row)

            by_geometry = {row['geometry_label']: row for row in setting_rows}
            folded_slowdown = [
                float(by_geometry[label]['speed_slowdown_mm_min'])
                for label in ('low_severity', 'representative', 'high_severity')
            ]
            summary_rows.append(
                {
                    'parameter_name': parameter_name,
                    'parameter_label': parameter_label,
                    'parameter_value': float(parameter_value),
                    'all_folded_slower': bool(all(value > 0.0 for value in folded_slowdown)),
                    'flat_abs_slowdown_mm_min': _safe_float(abs(float(by_geometry['flat_control']['speed_slowdown_mm_min']))),
                    'severity_monotone': _is_monotone_non_decreasing(folded_slowdown),
                    'low_severity_slowdown_mm_min': by_geometry['low_severity']['speed_slowdown_mm_min'],
                    'representative_slowdown_mm_min': by_geometry['representative']['speed_slowdown_mm_min'],
                    'high_severity_slowdown_mm_min': by_geometry['high_severity']['speed_slowdown_mm_min'],
                }
            )

    _write_csv(output_path, rows)
    _write_csv(summary_path, summary_rows)
    figure_path = output_dir / 'mechanistic_dipole_sensitivity.png'
    return rows, summary_rows, {
        'n_rows': int(len(rows)),
        'all_settings_folded_positive': bool(all(row['all_folded_slower'] for row in summary_rows)),
        'max_flat_abs_slowdown_mm_min': _safe_float(max(float(row['flat_abs_slowdown_mm_min']) for row in summary_rows)),
        'monotone_settings': int(sum(bool(row['severity_monotone']) for row in summary_rows)),
        'n_settings': int(len(summary_rows)),
        'figure_path': str(figure_path),
    }, figure_path


def _save_sensitivity_figure(rows: list[dict], output_path: Path) -> None:
    parameter_specs = _sensitivity_parameter_specs()
    geometry_styles = {
        'flat_control': dict(color='black', marker='s', linestyle=':', label='Flat control'),
        'low_severity': dict(color='#4c78a8', marker='o', linestyle='-', label='Low severity'),
        'representative': dict(color='#f58518', marker='o', linestyle='-', label='Representative'),
        'high_severity': dict(color='#e45756', marker='o', linestyle='-', label='High severity'),
    }
    fig, axes = plt.subplots(2, 3, figsize=(12.0, 6.8), constrained_layout=True)
    axes = axes.ravel()

    for axis, (parameter_name, parameter_label, values) in zip(axes, parameter_specs):
        parameter_rows = [row for row in rows if row['parameter_name'] == parameter_name]
        for geometry_label, style in geometry_styles.items():
            geometry_rows = [row for row in parameter_rows if row['geometry_label'] == geometry_label]
            geometry_rows.sort(key=lambda row: row['parameter_value'])
            axis.plot(
                [row['parameter_value'] for row in geometry_rows],
                [row['speed_slowdown_mm_min'] for row in geometry_rows],
                color=style['color'],
                marker=style['marker'],
                linestyle=style['linestyle'],
                linewidth=1.6,
                label=style['label'],
            )
        axis.axhline(0.0, color='black', linewidth=0.8, alpha=0.45)
        axis.set_title(parameter_label)
        axis.set_xlabel(parameter_label)
        axis.set_ylabel('Slowdown (mm/min)')
        axis.grid(alpha=0.25)

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=style['color'],
            marker=style['marker'],
            linestyle=style['linestyle'],
            linewidth=1.6,
            label=style['label'],
        )
        for style in geometry_styles.values()
    ]
    axes[-1].legend(handles=legend_handles, loc='center', frameon=False)
    axes[-1].axis('off')
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _null_model_rows(output_dir: Path) -> tuple[list[dict], dict[str, object], Path]:
    mesh = generate_folded_strip_mesh(nx=64, ny=28, length_mm=22.0, width_mm=10.0, fold_depth_mm=2.4, fold_sigma_mm=1.5)
    stimulus_vertex, e1_vertex, e2_vertex = choose_auto_vertices(mesh)
    base_params = MechanisticSurfaceParams(final_t_end=210.0, enable_vascular_feedback=False)
    variant_specs = [
        ('aligned', 'Aligned dipole', dc.replace(base_params, enable_dipole_alignment=True, dipole_kernel_mode='aligned')),
        ('distance_only', 'Distance-only kernel', dc.replace(base_params, enable_dipole_alignment=True, dipole_kernel_mode='distance_only')),
        ('scrambled_normals', 'Scrambled-normal kernel', dc.replace(base_params, enable_dipole_alignment=True, dipole_kernel_mode='scrambled_normals')),
    ]

    baseline_output = run_mechanistic_surface_simulation(
        mesh,
        dc.replace(base_params, enable_dipole_alignment=False, dipole_kernel_mode='aligned'),
        stimulus_vertex=stimulus_vertex,
    )
    baseline_metrics = _compute_case_metrics(baseline_output, e1_vertex, e2_vertex)

    rows: list[dict] = [
        {
            'kernel_mode': 'baseline',
            'kernel_title': 'No dipole',
            'arrival_speed_mm_min': baseline_metrics['arrival_speed_mm_min'],
            'cross_fold_delay_s': baseline_metrics['cross_fold_delay_s'],
            'max_abs_potential_mV': baseline_metrics['max_abs_potential_mV'],
            'speed_slowdown_mm_min': 0.0,
            'delay_increase_s': 0.0,
            'swelling_cap_fraction': baseline_metrics['swelling_cap_fraction'],
        }
    ]
    for kernel_mode, kernel_title, comparison_params in variant_specs:
        output = run_mechanistic_surface_simulation(
            mesh,
            comparison_params,
            stimulus_vertex=stimulus_vertex,
        )
        metrics = _compute_case_metrics(output, e1_vertex, e2_vertex)
        rows.append(
            {
                'kernel_mode': kernel_mode,
                'kernel_title': kernel_title,
                'arrival_speed_mm_min': metrics['arrival_speed_mm_min'],
                'cross_fold_delay_s': metrics['cross_fold_delay_s'],
                'max_abs_potential_mV': metrics['max_abs_potential_mV'],
                'speed_slowdown_mm_min': _safe_difference(
                    baseline_metrics['arrival_speed_mm_min'],
                    metrics['arrival_speed_mm_min'],
                ),
                'delay_increase_s': _safe_difference(
                    metrics['cross_fold_delay_s'],
                    baseline_metrics['cross_fold_delay_s'],
                ),
                'swelling_cap_fraction': metrics['swelling_cap_fraction'],
            }
        )

    _write_csv(output_dir / 'mechanistic_null_models.csv', rows)
    figure_path = output_dir / 'mechanistic_null_models.png'
    aligned = next(row for row in rows if row['kernel_mode'] == 'aligned')
    distance_only = next(row for row in rows if row['kernel_mode'] == 'distance_only')
    scrambled = next(row for row in rows if row['kernel_mode'] == 'scrambled_normals')
    summary = {
        'aligned_slowdown_mm_min': aligned['speed_slowdown_mm_min'],
        'distance_only_slowdown_mm_min': distance_only['speed_slowdown_mm_min'],
        'scrambled_normals_slowdown_mm_min': scrambled['speed_slowdown_mm_min'],
        'aligned_delay_increase_s': aligned['delay_increase_s'],
        'aligned_stronger_than_distance_only': bool(float(aligned['speed_slowdown_mm_min']) > float(distance_only['speed_slowdown_mm_min'])),
        'aligned_stronger_than_scrambled_normals': bool(float(aligned['speed_slowdown_mm_min']) > float(scrambled['speed_slowdown_mm_min'])),
        'figure_path': str(figure_path),
    }
    return rows, summary, figure_path


def _save_null_model_figure(rows: list[dict], output_path: Path) -> None:
    variant_rows = [row for row in rows if row['kernel_mode'] != 'baseline']
    labels = [row['kernel_title'] for row in variant_rows]
    slowdown = np.asarray([row['speed_slowdown_mm_min'] for row in variant_rows], dtype=float)
    delay = np.asarray([row['delay_increase_s'] for row in variant_rows], dtype=float)
    x = np.arange(len(labels))

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0), constrained_layout=True)
    axes[0].bar(x, slowdown, color=['#e45756', '#72b7b2', '#4c78a8'], edgecolor='black', linewidth=0.5)
    axes[0].set_xticks(x, labels, rotation=15)
    axes[0].set_ylabel('Speed slowdown (mm/min)')
    axes[0].axhline(0.0, color='black', linewidth=0.8, alpha=0.45)
    axes[0].grid(axis='y', alpha=0.25)

    axes[1].bar(x, delay, color=['#e45756', '#72b7b2', '#4c78a8'], edgecolor='black', linewidth=0.5)
    axes[1].set_xticks(x, labels, rotation=15)
    axes[1].set_ylabel('Delay increase (s)')
    axes[1].axhline(0.0, color='black', linewidth=0.8, alpha=0.45)
    axes[1].grid(axis='y', alpha=0.25)

    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _crossing_time(times: np.ndarray, values: np.ndarray, threshold: float) -> float | None:
    above = np.asarray(values >= threshold, dtype=bool)
    if not np.any(above):
        return None
    idx = int(np.argmax(above))
    if idx == 0:
        return float(times[0])
    t0, t1 = float(times[idx - 1]), float(times[idx])
    v0, v1 = float(values[idx - 1]), float(values[idx])
    if math.isclose(v0, v1, rel_tol=0.0, abs_tol=1e-12):
        return t1
    frac = (threshold - v0) / (v1 - v0)
    return float(t0 + frac * (t1 - t0))


def _trace_shape_metrics(times: np.ndarray, values: np.ndarray) -> dict[str, float | None]:
    baseline = float(np.mean(values[: min(5, values.size)]))
    centered = np.asarray(values, dtype=float) - baseline
    if centered.size == 0 or not np.any(np.isfinite(centered)):
        return {
            'baseline': _safe_float(baseline),
            'peak_shift': None,
            'peak_time_s': None,
            'rise_10_90_s': None,
        }
    peak_idx = int(np.argmax(np.abs(centered)))
    peak_shift = float(centered[peak_idx])
    if math.isclose(peak_shift, 0.0, rel_tol=0.0, abs_tol=1e-12):
        return {
            'baseline': _safe_float(baseline),
            'peak_shift': 0.0,
            'peak_time_s': _safe_float(times[peak_idx]),
            'rise_10_90_s': 0.0,
        }
    oriented = centered * np.sign(peak_shift)
    magnitude = abs(peak_shift)
    t10 = _crossing_time(times, oriented, 0.10 * magnitude)
    t90 = _crossing_time(times, oriented, 0.90 * magnitude)
    rise = None if t10 is None or t90 is None else _safe_float(t90 - t10)
    return {
        'baseline': _safe_float(baseline),
        'peak_shift': _safe_float(peak_shift),
        'peak_time_s': _safe_float(times[peak_idx]),
        'rise_10_90_s': rise,
    }


def _waveform_rows(output_dir: Path) -> tuple[list[dict], list[dict], dict[str, object], Path]:
    trace_times = np.arange(0.0, 220.0 + 0.25, 0.5, dtype=float)
    base_params = MechanisticSurfaceParams(final_t_end=220.0, enable_vascular_feedback=False)
    geometry_specs = [
        ('representative_folded', 2.4, 1.5),
        ('flat_control', 0.0, 1.5),
    ]

    trace_rows: list[dict] = []
    summary_rows: list[dict] = []
    for geometry_label, fold_depth_mm, fold_sigma_mm in geometry_specs:
        mesh = generate_folded_strip_mesh(
            nx=64,
            ny=28,
            length_mm=22.0,
            width_mm=10.0,
            fold_depth_mm=fold_depth_mm,
            fold_sigma_mm=fold_sigma_mm,
        )
        stimulus_vertex, e1_vertex, e2_vertex = choose_auto_vertices(mesh)
        baseline_output, dipole_output, baseline_metrics, dipole_metrics = _paired_outputs(
            mesh,
            base_params,
            stimulus_vertex,
            e1_vertex,
            e2_vertex,
            snapshot_times=trace_times,
        )

        case_specs = [
            ('no_dipole', baseline_output, baseline_metrics),
            ('dipole_enabled', dipole_output, dipole_metrics),
        ]
        for case_label, output, metrics in case_specs:
            for electrode_label, vertex in (('E1', e1_vertex), ('E2', e2_vertex)):
                potential_mv = output.params.field_reference_mV * output.snapshot_potential[:, vertex]
                potassium_trace = output.snapshot_potassium_e[:, vertex]
                membrane_trace = output.snapshot_voltage_mv[:, vertex]

                k_metrics = _trace_shape_metrics(trace_times, potassium_trace)
                ve_metrics = _trace_shape_metrics(trace_times, potential_mv)
                threshold_cross = _crossing_time(
                    trace_times,
                    membrane_trace,
                    output.params.arrival_voltage_threshold_mv,
                )
                summary_rows.append(
                    {
                        'geometry_label': geometry_label,
                        'case_label': case_label,
                        'electrode_label': electrode_label,
                        'arrival_time_s': _safe_float(output.arrival_times[vertex]),
                        'membrane_threshold_cross_s': threshold_cross,
                        'k_peak_shift_mM': k_metrics['peak_shift'],
                        'k_peak_time_s': k_metrics['peak_time_s'],
                        'k_rise_10_90_s': k_metrics['rise_10_90_s'],
                        'dc_peak_shift_mV': ve_metrics['peak_shift'],
                        'dc_peak_time_s': ve_metrics['peak_time_s'],
                        'dc_rise_10_90_s': ve_metrics['rise_10_90_s'],
                        'max_abs_potential_mV': metrics['max_abs_potential_mV'],
                    }
                )
                for time_s, potassium_e, potential_value, membrane_voltage in zip(
                    trace_times,
                    potassium_trace,
                    potential_mv,
                    membrane_trace,
                ):
                    trace_rows.append(
                        {
                            'geometry_label': geometry_label,
                            'case_label': case_label,
                            'electrode_label': electrode_label,
                            'time_s': _safe_float(time_s),
                            'potassium_e_mM': _safe_float(potassium_e),
                            'extracellular_potential_mV': _safe_float(potential_value),
                            'membrane_voltage_mV': _safe_float(membrane_voltage),
                        }
                    )

    _write_csv(output_dir / 'mechanistic_virtual_electrode_traces.csv', trace_rows)
    _write_csv(output_dir / 'mechanistic_virtual_electrode_summary.csv', summary_rows)
    figure_path = output_dir / 'mechanistic_virtual_electrode_waveforms.png'
    representative_e2 = [
        row
        for row in summary_rows
        if row['geometry_label'] == 'representative_folded' and row['electrode_label'] == 'E2'
    ]
    flat_e2 = [
        row
        for row in summary_rows
        if row['geometry_label'] == 'flat_control' and row['electrode_label'] == 'E2'
    ]
    summary = {
        'representative_e2_cases': representative_e2,
        'flat_e2_cases': flat_e2,
        'figure_path': str(figure_path),
    }
    return trace_rows, summary_rows, summary, figure_path


def _save_waveform_figure(trace_rows: list[dict], output_path: Path) -> None:
    def _filter_trace(geometry_label: str, case_label: str):
        filtered = [
            row for row in trace_rows
            if row['geometry_label'] == geometry_label
            and row['case_label'] == case_label
            and row['electrode_label'] == 'E2'
        ]
        filtered.sort(key=lambda row: row['time_s'])
        times = np.asarray([row['time_s'] for row in filtered], dtype=float)
        potassium = np.asarray([row['potassium_e_mM'] for row in filtered], dtype=float)
        potential = np.asarray([row['extracellular_potential_mV'] for row in filtered], dtype=float)
        return times, potassium, potential

    fig, axes = plt.subplots(2, 2, figsize=(11.2, 6.2), sharex=True, constrained_layout=True)
    for col, geometry_label in enumerate(('representative_folded', 'flat_control')):
        geometry_title = 'Representative folded' if geometry_label == 'representative_folded' else 'Flat control'
        for case_label, color, linestyle in (
            ('no_dipole', '#4c78a8', '-'),
            ('dipole_enabled', '#e45756', '--'),
        ):
            times, potassium, potential = _filter_trace(geometry_label, case_label)
            axes[0, col].plot(times, potassium, color=color, linestyle=linestyle, linewidth=1.8, label=case_label.replace('_', ' '))
            axes[1, col].plot(times, potential, color=color, linestyle=linestyle, linewidth=1.8, label=case_label.replace('_', ' '))
        axes[0, col].set_title(geometry_title)
        axes[0, col].set_ylabel('Extracellular K+ (mM)')
        axes[1, col].set_ylabel('Extracellular potential (mV)')
        axes[1, col].set_xlabel('Time (s)')
        axes[0, col].grid(alpha=0.25)
        axes[1, col].grid(alpha=0.25)
        axes[1, col].axhline(0.0, color='black', linewidth=0.8, alpha=0.45)
        axes[0, col].legend(loc='upper right', frameon=False)

    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _convergence_rows(output_dir: Path) -> tuple[list[dict], dict[str, object]]:
    output_path = output_dir / 'mechanistic_convergence.csv'
    study_specs = [
        ('mesh', 'coarse', 56, 24, 4.0),
        ('mesh', 'reference', 64, 28, 4.0),
        ('mesh', 'fine', 72, 32, 4.0),
        ('time_step', 'reference_auto_dt', 64, 28, 4.0),
        ('time_step', 'reference_half_dt', 64, 28, 8.0),
    ]
    geometry_specs = [
        ('representative_folded', 2.4, 1.5),
        ('flat_control', 0.0, 1.5),
    ]

    reference_vertices: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for geometry_label, fold_depth_mm, fold_sigma_mm in geometry_specs:
        reference_mesh = generate_folded_strip_mesh(
            nx=64,
            ny=28,
            length_mm=22.0,
            width_mm=10.0,
            fold_depth_mm=fold_depth_mm,
            fold_sigma_mm=fold_sigma_mm,
        )
        stimulus_vertex, e1_vertex, e2_vertex = choose_auto_vertices(reference_mesh)
        reference_vertices[geometry_label] = (
            np.asarray(reference_mesh.vertices[stimulus_vertex], dtype=float),
            np.asarray(reference_mesh.vertices[e1_vertex], dtype=float),
            np.asarray(reference_mesh.vertices[e2_vertex], dtype=float),
        )

    rows: list[dict] = []
    for study_axis, resolution_label, nx, ny, dt_scale in study_specs:
        for geometry_label, fold_depth_mm, fold_sigma_mm in geometry_specs:
            mesh = generate_folded_strip_mesh(
                nx=nx,
                ny=ny,
                length_mm=22.0,
                width_mm=10.0,
                fold_depth_mm=fold_depth_mm,
                fold_sigma_mm=fold_sigma_mm,
            )
            stim_coord, e1_coord, e2_coord = reference_vertices[geometry_label]
            stimulus_vertex = int(np.argmin(np.linalg.norm(mesh.vertices - stim_coord[None, :], axis=1)))
            e1_vertex = int(np.argmin(np.linalg.norm(mesh.vertices - e1_coord[None, :], axis=1)))
            e2_vertex = int(np.argmin(np.linalg.norm(mesh.vertices - e2_coord[None, :], axis=1)))
            params = MechanisticSurfaceParams(
                final_t_end=190.0,
                mechanistic_dt_scale=dt_scale,
                enable_vascular_feedback=False,
            )
            _, _, baseline, dipole = _paired_outputs(
                mesh,
                params,
                stimulus_vertex,
                e1_vertex,
                e2_vertex,
            )
            rows.append(
                {
                    'study_axis': study_axis,
                    'resolution_label': resolution_label,
                    'geometry_label': geometry_label,
                    'nx': int(nx),
                    'ny': int(ny),
                    'mechanistic_dt_scale': float(dt_scale),
                    'baseline_dt_used_s': baseline['dt_used_s'],
                    'dipole_dt_used_s': dipole['dt_used_s'],
                    'baseline_arrival_speed_mm_min': baseline['arrival_speed_mm_min'],
                    'dipole_arrival_speed_mm_min': dipole['arrival_speed_mm_min'],
                    'baseline_cross_fold_delay_s': baseline['cross_fold_delay_s'],
                    'dipole_cross_fold_delay_s': dipole['cross_fold_delay_s'],
                    'speed_slowdown_mm_min': _safe_difference(
                        baseline['arrival_speed_mm_min'],
                        dipole['arrival_speed_mm_min'],
                    ),
                    'delay_increase_s': _safe_difference(
                        dipole['cross_fold_delay_s'],
                        baseline['cross_fold_delay_s'],
                    ),
                }
            )

    _write_csv(output_path, rows)
    folded_rows = [row for row in rows if row['geometry_label'] == 'representative_folded']
    flat_rows = [row for row in rows if row['geometry_label'] == 'flat_control']
    reference_row = next(row for row in folded_rows if row['resolution_label'] == 'reference')
    reference_slowdown = float(reference_row['speed_slowdown_mm_min'])
    summary = {
        'all_folded_positive': bool(all(float(row['speed_slowdown_mm_min']) > 0.0 for row in folded_rows)),
        'max_folded_slowdown_deviation_pct': _safe_float(
            100.0
            * max(abs(float(row['speed_slowdown_mm_min']) - reference_slowdown) for row in folded_rows)
            / max(abs(reference_slowdown), 1e-6)
        ),
        'max_flat_abs_slowdown_mm_min': _safe_float(max(abs(float(row['speed_slowdown_mm_min'])) for row in flat_rows)),
    }
    return rows, summary


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

    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    return {
        'stimulus_vertex': int(stimulus_vertex),
        'electrode_1_vertex': int(e1_vertex),
        'electrode_2_vertex': int(e2_vertex),
        'snapshot_times_s': [float(trace_times[idx]) for idx in snapshot_indices],
        'path_length_mm': float(path_distance[-1]),
    }


def _save_atlas_multi_patch_figure(panel, rows: list[dict], output_path: Path) -> None:
    atlas_projection = _project_vertices_2d(panel.atlas_mesh.vertices)
    triangulation = mtri.Triangulation(
        atlas_projection[:, 0],
        atlas_projection[:, 1],
        triangles=panel.atlas_mesh.faces,
    )
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), constrained_layout=True)

    artist = axes[0].tripcolor(
        triangulation,
        panel.atlas_mesh.sulcal_depth,
        shading='gouraud',
        cmap='cividis',
    )
    for mask in panel.sulcal_roi_masks:
        axes[0].scatter(
            atlas_projection[mask, 0],
            atlas_projection[mask, 1],
            s=7,
            color='#e45756',
            alpha=0.65,
        )
    for mask in panel.flat_roi_masks:
        axes[0].scatter(
            atlas_projection[mask, 0],
            atlas_projection[mask, 1],
            s=7,
            color='#4c78a8',
            alpha=0.65,
        )

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Sulcal patches', markerfacecolor='#e45756', markersize=8),
        Line2D([0], [0], marker='o', color='w', label='Flatter patches', markerfacecolor='#4c78a8', markersize=8)
    ]
    axes[0].legend(handles=legend_elements, loc='lower left', frameon=False)
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    axes[0].set_aspect('equal')
    fig.colorbar(artist, ax=axes[0], fraction=0.046, pad=0.02, label='Normalized sulcal depth')

    import pandas as pd
    df = pd.DataFrame(rows)

    sulcal_data = df[df['patch_category'] == 'sulcal']['speed_slowdown_mm_min'].values
    flat_data = df[df['patch_category'] == 'flat']['speed_slowdown_mm_min'].values

    # Create boxplots
    axes[1].boxplot([sulcal_data, flat_data], positions=[0, 1], widths=0.4,
                    showfliers=False, patch_artist=True,
                    boxprops=dict(facecolor='white', color='black'),
                    medianprops=dict(color='black'))

    # Add jittered scatter points
    def add_jitter(x_pos, data, color):
        jitter = np.random.uniform(-0.1, 0.1, size=len(data))
        axes[1].scatter(np.full_like(data, x_pos) + jitter, data,
                        color=color, alpha=0.7, s=40, edgecolor='w')

    add_jitter(0, sulcal_data, '#e45756')
    add_jitter(1, flat_data, '#4c78a8')

    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels(['Sulcal', 'Flatter'])

    axes[1].set_xlabel('')
    axes[1].set_ylabel('Dipole-induced slowing (mm/min)')
    axes[1].grid(axis='y', alpha=0.25)
    axes[1].axhline(0.0, color='black', linewidth=0.8, alpha=0.45)

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
    panel = prepare_atlas_multi_patch_panel(
        atlas_cache_dir, n_sulcal=8, n_flat=8, patch_radius_mm=12.0, min_separation_mm=30.0
    )
    base_params = MechanisticSurfaceParams(final_t_end=180.0, enable_vascular_feedback=False)

    rows: list[dict] = []

    def _run_patch(patch, category: str):
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
            baseline_output, patch.stimulus_vertex, patch.electrode_2_vertex, radius_mm=1.0)
        dipole_traversal_speed = mechanistic_surface_arrival_speed_mm_min(
            dipole_output, patch.stimulus_vertex, patch.electrode_2_vertex, radius_mm=1.0)

        baseline_downstream_arrival = _safe_float(baseline_output.arrival_times[patch.electrode_2_vertex])
        dipole_downstream_arrival = _safe_float(dipole_output.arrival_times[patch.electrode_2_vertex])

        return {
            'patch_label': patch.label,
            'patch_category': category,
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
            'speed_slowdown_mm_min': _safe_float(baseline_traversal_speed - dipole_traversal_speed) if baseline_traversal_speed and dipole_traversal_speed else 0.0,
            'delay_increase_s': _safe_float((dipole_downstream_arrival or 0) - (baseline_downstream_arrival or 0)) if dipole_downstream_arrival and baseline_downstream_arrival else None,
            'baseline_max_abs_potential_mV': baseline['max_abs_potential_mV'],
            'dipole_max_abs_potential_mV': dipole['max_abs_potential_mV'],
        }

    for patch in panel.sulcal_patches:
        rows.append(_run_patch(patch, 'sulcal'))
    for patch in panel.flat_patches:
        rows.append(_run_patch(patch, 'flat'))

    _write_csv(output_dir / 'mechanistic_atlas_patch_check.csv', rows)
    figure_path = output_dir / 'mechanistic_atlas_patch_qc.png'
    _save_atlas_multi_patch_figure(panel, rows, figure_path)

    sulcal_slowdowns = [float(row['speed_slowdown_mm_min']) for row in rows if row['patch_category'] == 'sulcal']
    flat_slowdowns = [float(row['speed_slowdown_mm_min']) for row in rows if row['patch_category'] == 'flat']

    summary = {
        'n_sulcal': len(sulcal_slowdowns),
        'n_flat': len(flat_slowdowns),
        'mean_sulcal_patch_slowdown_mm_min': _safe_float(np.mean(sulcal_slowdowns)) if sulcal_slowdowns else 0.0,
        'mean_flat_patch_slowdown_mm_min': _safe_float(np.mean(flat_slowdowns)) if flat_slowdowns else 0.0,
        'effect_preserved_on_all_sulcal': bool(all(x > 0.0 for x in sulcal_slowdowns)) if sulcal_slowdowns else False,
        'flat_effect_smaller_mean': bool(np.mean(sulcal_slowdowns) > np.mean(flat_slowdowns)) if sulcal_slowdowns and flat_slowdowns else False,
        'sulcal_speed_slowdown_wilcoxon': _wilcoxon_greater(sulcal_slowdowns, flat_slowdowns),
        'figure_path': str(figure_path),
        'atlas_source_dir': str(panel.atlas_source_dir) if panel.atlas_source_dir is not None else None,
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
    sensitivity_rows, sensitivity_summary_rows, sensitivity_summary, sensitivity_figure_path = _sensitivity_rows(output_dir)
    null_rows, null_summary, null_figure_path = _null_model_rows(output_dir)
    waveform_trace_rows, waveform_summary_rows, waveform_summary, waveform_figure_path = _waveform_rows(output_dir)
    convergence_rows, convergence_summary = _convergence_rows(output_dir)
    figure_path = output_dir / 'mechanistic_sulcal_slowing.png'
    representative_figure_path = output_dir / 'mechanistic_representative_summary.png'
    propagation_figure_path = output_dir / 'mechanistic_wave_propagation.png'
    _save_geometry_figure(geometry_rows, figure_path)
    _save_representative_figure(representative_rows, representative_figure_path)
    _save_sensitivity_figure(sensitivity_rows, sensitivity_figure_path)
    _save_null_model_figure(null_rows, null_figure_path)
    _save_waveform_figure(waveform_trace_rows, waveform_figure_path)
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
        'sensitivity_rows': sensitivity_rows,
        'sensitivity_summary_rows': sensitivity_summary_rows,
        'sensitivity_summary': sensitivity_summary,
        'null_model_rows': null_rows,
        'null_model_summary': null_summary,
        'waveform_summary_rows': waveform_summary_rows,
        'waveform_summary': waveform_summary,
        'convergence_rows': convergence_rows,
        'convergence_summary': convergence_summary,
        'figure_path': str(figure_path),
        'representative_figure_path': str(representative_figure_path),
        'propagation_figure_path': str(propagation_figure_path),
        'sensitivity_figure_path': str(sensitivity_figure_path),
        'null_model_figure_path': str(null_figure_path),
        'waveform_figure_path': str(waveform_figure_path),
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
        'Atlas panel check: '
        f"mean sulcal slowdown={atlas_summary['mean_sulcal_patch_slowdown_mm_min']:.3f} mm/min "
        f"(n={atlas_summary['n_sulcal']}), "
        f"mean flat slowdown={atlas_summary['mean_flat_patch_slowdown_mm_min']:.3f} mm/min "
        f"(n={atlas_summary['n_flat']})"
    )
    print(
        'Sensitivity battery: '
        f"{sensitivity_summary['n_settings']} settings, "
        f"folded-positive={sensitivity_summary['all_settings_folded_positive']}, "
        f"max |flat slowdown|={sensitivity_summary['max_flat_abs_slowdown_mm_min']:.3f} mm/min"
    )
    print(
        'Null models: '
        f"aligned={null_summary['aligned_slowdown_mm_min']:.3f}, "
        f"distance-only={null_summary['distance_only_slowdown_mm_min']:.3f}, "
        f"scrambled={null_summary['scrambled_normals_slowdown_mm_min']:.3f} mm/min"
    )
    print(
        'Convergence: '
        f"folded-positive={convergence_summary['all_folded_positive']}, "
        f"max |flat slowdown|={convergence_summary['max_flat_abs_slowdown_mm_min']:.3f} mm/min"
    )


if __name__ == '__main__':
    main()
