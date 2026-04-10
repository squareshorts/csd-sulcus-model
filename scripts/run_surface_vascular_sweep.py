from __future__ import annotations

import argparse
import dataclasses as dc
import json
import sys
import time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

matplotlib.use('Agg')

from run_surface_representative import choose_auto_vertices, compute_case_metrics, load_mesh, resolve_surface_horizon, write_csv  # noqa: E402
from csd_sulcus.surface_model import SurfaceParams, edge_speed_stats, run_surface_simulation  # noqa: E402


FEEDBACK_PRESETS = {
    'weak': {
        'clearance_perfusion_gain': 0.14,
        'clearance_oxygen_gain': 0.05,
        'threshold_baseline_vulnerability_gain': 0.05,
        'threshold_constriction_gain': 0.025,
    },
    'strong': {
        'clearance_perfusion_gain': 0.28,
        'clearance_oxygen_gain': 0.12,
        'threshold_baseline_vulnerability_gain': 0.11,
        'threshold_constriction_gain': 0.055,
    },
}

CONSTRICTION_PRESETS = {
    'weak': {'a_con0': 0.16},
    'strong': {'a_con0': 0.30},
}

ANISOTROPY_PRESETS = {
    'off': {'enable_anisotropy': False, 'anisotropy_ratio': 1.0},
    'low': {'enable_anisotropy': True, 'anisotropy_ratio': 1.05},
    'high': {'enable_anisotropy': True, 'anisotropy_ratio': 1.25},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run a narrow vascular sweep for the surface families.')
    parser.add_argument('--mesh', type=Path, default=None, help='Path to an OBJ, NPZ, or GIFTI surface mesh.')
    parser.add_argument('--sulcal-depth', type=Path, default=None, help='Optional per-vertex sulcal-depth field (NPY/NPZ/GIFTI).')
    parser.add_argument('--thickness', type=Path, default=None, help='Optional per-vertex cortical-thickness field (NPY/NPZ/GIFTI).')
    parser.add_argument('--vascular-risk', type=Path, default=None, help='Optional per-vertex vascular-risk field (NPY/NPZ/GIFTI).')
    parser.add_argument('--preferred-axis', type=Path, default=None, help='Optional per-vertex preferred tangential axis (NPY).')
    parser.add_argument('--output-root', type=Path, default=None)
    parser.add_argument('--quick', action='store_true', help='Use the reduced synthetic mesh and 90 s horizon.')
    parser.add_argument('--stimulus-vertex', type=int, default=None)
    parser.add_argument('--electrode-1', type=int, default=None)
    parser.add_argument('--electrode-2', type=int, default=None)
    parser.add_argument('--roi-radius-mm', type=float, default=1.0)
    parser.add_argument('--final-t-end', type=float, default=None, help='Optional simulation horizon in seconds.')
    return parser.parse_args()


def _heatmap_values(
    rows: list[dict[str, float | int | str | bool]],
    metric_key: str,
    anisotropy_level: str,
) -> np.ndarray:
    values = np.full((len(FEEDBACK_PRESETS), len(CONSTRICTION_PRESETS)), np.nan, dtype=float)
    for row in rows:
        if str(row['anisotropy_level']) != anisotropy_level:
            continue
        feedback_index = list(FEEDBACK_PRESETS).index(str(row['feedback_level']))
        constriction_index = list(CONSTRICTION_PRESETS).index(str(row['constriction_level']))
        values[feedback_index, constriction_index] = float(row[metric_key])
    return values


def _annotation_style(image, value: float) -> dict[str, object]:
    if not np.isfinite(value):
        color = 'black'
        outline = 'white'
    else:
        rgba = image.cmap(image.norm(value))
        luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
        color = 'black' if luminance > 0.62 else 'white'
        outline = 'white' if color == 'black' else 'black'
    return {
        'color': color,
        'fontsize': 9,
        'fontweight': 'semibold',
        'path_effects': [pe.withStroke(linewidth=2.0, foreground=outline, alpha=0.9)],
    }


def plot_vascular_sweep(rows: list[dict[str, float | int | str | bool]], output_path: Path) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(13, 11), constrained_layout=True)
    anisotropy_levels = list(ANISOTROPY_PRESETS)
    feedback_labels = list(FEEDBACK_PRESETS)
    constriction_labels = list(CONSTRICTION_PRESETS)

    for col, anis_level in enumerate(anisotropy_levels):
        speed_values = _heatmap_values(rows, 'arrival_speed_mm_min', anis_level)
        delay_values = _heatmap_values(rows, 'cross_fold_delay_s', anis_level)
        perfusion_values = _heatmap_values(rows, 'min_perfusion', anis_level)

        speed_ax = axes[0, col]
        speed_im = speed_ax.imshow(speed_values, cmap='viridis', aspect='auto')
        speed_ax.set_title(f'Cross-fold speed: anisotropy {anis_level}')
        speed_ax.set_xticks(np.arange(len(constriction_labels)), constriction_labels)
        speed_ax.set_yticks(np.arange(len(feedback_labels)), feedback_labels)
        speed_ax.set_xlabel('Constriction')
        speed_ax.set_ylabel('Feedback')
        for i in range(speed_values.shape[0]):
            for j in range(speed_values.shape[1]):
                value = speed_values[i, j]
                text = 'n/a' if not np.isfinite(value) else f'{value:.2f}'
                speed_ax.text(j, i, text, ha='center', va='center', **_annotation_style(speed_im, value))
        fig.colorbar(speed_im, ax=speed_ax, shrink=0.82, label='mm/min')

        delay_ax = axes[1, col]
        delay_im = delay_ax.imshow(delay_values, cmap='magma_r', aspect='auto')
        delay_ax.set_title(f'Cross-fold delay: anisotropy {anis_level}')
        delay_ax.set_xticks(np.arange(len(constriction_labels)), constriction_labels)
        delay_ax.set_yticks(np.arange(len(feedback_labels)), feedback_labels)
        delay_ax.set_xlabel('Constriction')
        delay_ax.set_ylabel('Feedback')
        for i in range(delay_values.shape[0]):
            for j in range(delay_values.shape[1]):
                value = delay_values[i, j]
                text = 'n/a' if not np.isfinite(value) else f'{value:.1f}'
                delay_ax.text(j, i, text, ha='center', va='center', **_annotation_style(delay_im, value))
        fig.colorbar(delay_im, ax=delay_ax, shrink=0.82, label='s')

        perfusion_ax = axes[2, col]
        perfusion_im = perfusion_ax.imshow(perfusion_values, cmap='magma', aspect='auto')
        perfusion_ax.set_title(f'Min perfusion reserve: anisotropy {anis_level}')
        perfusion_ax.set_xticks(np.arange(len(constriction_labels)), constriction_labels)
        perfusion_ax.set_yticks(np.arange(len(feedback_labels)), feedback_labels)
        perfusion_ax.set_xlabel('Constriction')
        perfusion_ax.set_ylabel('Feedback')
        for i in range(perfusion_values.shape[0]):
            for j in range(perfusion_values.shape[1]):
                value = perfusion_values[i, j]
                text = 'n/a' if not np.isfinite(value) else f'{value:.2f}'
                perfusion_ax.text(j, i, text, ha='center', va='center', **_annotation_style(perfusion_im, value))
        fig.colorbar(perfusion_im, ax=perfusion_ax, shrink=0.82, label='min F')

    fig.suptitle('Narrow surface vascular sweep', fontsize=15)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    mesh = load_mesh(args)
    stimulus_vertex, auto_e1, auto_e2 = choose_auto_vertices(mesh)
    if args.stimulus_vertex is not None:
        stimulus_vertex = int(args.stimulus_vertex)
    e1_vertex = int(args.electrode_1) if args.electrode_1 is not None else auto_e1
    e2_vertex = int(args.electrode_2) if args.electrode_2 is not None else auto_e2

    output_root = args.output_root or ROOT / ('outputs/surface_vascular_sweep_quick' if args.quick else 'outputs/surface_vascular_sweep')
    output_root.mkdir(parents=True, exist_ok=True)

    base_params = SurfaceParams(enable_vascular_feedback=True)
    final_t_end = 110.0 if args.quick and args.final_t_end is None else resolve_surface_horizon(args)
    base_params = dc.replace(base_params, final_t_end=final_t_end)

    rows: list[dict[str, float | int | str | bool]] = []
    t0 = time.time()
    for feedback_level, feedback_kwargs in FEEDBACK_PRESETS.items():
        for constriction_level, constriction_kwargs in CONSTRICTION_PRESETS.items():
            for anisotropy_level, anisotropy_kwargs in ANISOTROPY_PRESETS.items():
                params = dc.replace(base_params, **feedback_kwargs, **constriction_kwargs, **anisotropy_kwargs)
                output = run_surface_simulation(mesh, params, stimulus_vertex=stimulus_vertex, snapshot_times=(15.0, 30.0, 45.0))
                case_metrics = compute_case_metrics(output, e1_vertex, e2_vertex, args.roi_radius_mm)
                edge_stats = edge_speed_stats(output)
                rows.append(
                    {
                        'feedback_level': feedback_level,
                        'constriction_level': constriction_level,
                        'anisotropy_level': anisotropy_level,
                        'case_label': 'surface_scalar_vascular' if not params.enable_anisotropy else 'surface_tensor_vascular',
                        'final_t_end_s': final_t_end,
                        'arrival_speed_mm_min': case_metrics['arrival_speed_mm_min'],
                        'e1_arrival_s': case_metrics['e1_arrival_s'],
                        'e2_arrival_s': case_metrics['e2_arrival_s'],
                        'cross_fold_delay_s': case_metrics['cross_fold_delay_s'],
                        'e1_to_e2_geodesic_mm': case_metrics['e1_to_e2_geodesic_mm'],
                        'median_edge_speed_mm_min': edge_stats['median_edge_speed_mm_min'],
                        'deep_edge_speed_mm_min': edge_stats['deep_edge_speed_mm_min'],
                        'shallow_edge_speed_mm_min': edge_stats['shallow_edge_speed_mm_min'],
                        'min_perfusion': float(np.nanmin(output.perfusion)),
                        'min_oxygen': float(np.nanmin(output.oxygen)),
                        'crossed_fraction': float(np.mean(np.isfinite(output.arrival_times))),
                        'a_con0': float(params.a_con0),
                        'anisotropy_ratio': float(params.anisotropy_ratio),
                        'clearance_perfusion_gain': float(params.clearance_perfusion_gain),
                        'clearance_oxygen_gain': float(params.clearance_oxygen_gain),
                        'threshold_baseline_vulnerability_gain': float(params.threshold_baseline_vulnerability_gain),
                        'threshold_constriction_gain': float(
                            params.vascular_excitability_gain if params.threshold_constriction_gain is None else params.threshold_constriction_gain
                        ),
                    }
                )

    elapsed = time.time() - t0
    write_csv(output_root / 'surface_vascular_sweep.csv', rows)
    summary = {
        'mesh_source': mesh.metadata.get('source', 'unknown'),
        'stimulus_vertex': stimulus_vertex,
        'electrode_1_vertex': e1_vertex,
        'electrode_2_vertex': e2_vertex,
        'final_t_end_s': final_t_end,
        'elapsed_s': round(elapsed, 2),
        'feedback_presets': FEEDBACK_PRESETS,
        'constriction_presets': CONSTRICTION_PRESETS,
        'anisotropy_presets': ANISOTROPY_PRESETS,
        'rows': rows,
    }
    (output_root / 'surface_vascular_sweep.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    plot_vascular_sweep(rows, output_root / 'surface_vascular_sweep.png')

    print(f'Surface vascular sweep complete in {elapsed:.1f}s')
    print(f'Outputs written to {output_root}')
    for row in rows:
        print(
            f"  feedback={row['feedback_level']}, constriction={row['constriction_level']}, "
            f"anisotropy={row['anisotropy_level']}: speed={row['arrival_speed_mm_min']:.3f} mm/min, "
            f"delay={row['cross_fold_delay_s']:.2f} s"
        )


if __name__ == '__main__':
    main()
