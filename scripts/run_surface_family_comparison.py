from __future__ import annotations

import argparse
import dataclasses as dc
import json
import sys
import time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

matplotlib.use('Agg')

from run_surface_representative import (  # noqa: E402
    CASE_ORDER,
    CASE_SHORT_LABELS,
    CASE_TITLES,
    CASE_DEFINITIONS,
    choose_auto_vertices,
    compute_case_metrics,
    load_mesh,
    resolve_surface_horizon,
    write_csv,
)
from csd_sulcus.surface_model import edge_speed_stats, run_surface_simulation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run the four-family surface experiment comparison.')
    parser.add_argument('--mesh', type=Path, default=None, help='Path to an OBJ, NPZ, or GIFTI surface mesh.')
    parser.add_argument('--sulcal-depth', type=Path, default=None, help='Optional per-vertex sulcal-depth field (NPY/NPZ/GIFTI).')
    parser.add_argument('--thickness', type=Path, default=None, help='Optional per-vertex cortical-thickness field (NPY/NPZ/GIFTI).')
    parser.add_argument('--vascular-risk', type=Path, default=None, help='Optional per-vertex vascular-risk field (NPY/NPZ/GIFTI).')
    parser.add_argument('--preferred-axis', type=Path, default=None, help='Optional per-vertex preferred tangential axis (NPY).')
    parser.add_argument('--output-root', type=Path, default=None)
    parser.add_argument('--quick', action='store_true', help='Use a reduced synthetic mesh and shorter run horizon for iteration.')
    parser.add_argument('--stimulus-vertex', type=int, default=None)
    parser.add_argument('--electrode-1', type=int, default=None)
    parser.add_argument('--electrode-2', type=int, default=None)
    parser.add_argument('--roi-radius-mm', type=float, default=1.0)
    parser.add_argument('--final-t-end', type=float, default=None, help='Optional simulation horizon in seconds.')
    return parser.parse_args()


def plot_family_comparison(rows: list[dict[str, float | int | str | bool]], output_path: Path) -> None:
    labels = [CASE_SHORT_LABELS[str(row['case_label'])] for row in rows]
    speed = np.asarray([float(row['arrival_speed_mm_min']) for row in rows], dtype=float)
    delay = np.asarray([float(row['cross_fold_delay_s']) for row in rows], dtype=float)
    crossed = np.asarray([float(row['crossed_fraction']) for row in rows], dtype=float)
    perfusion = np.asarray([float(row['min_perfusion']) for row in rows], dtype=float)
    oxygen = np.asarray([float(row['min_oxygen']) for row in rows], dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    x = np.arange(len(labels))
    colors = ['#4c78a8', '#72b7b2', '#f58518', '#e45756']

    axes[0, 0].bar(x, speed, color=colors)
    axes[0, 0].set_title('Cross-fold speed')
    axes[0, 0].set_ylabel('mm/min')
    axes[0, 0].set_xticks(x, labels, rotation=15, ha='right')

    axes[0, 1].bar(x, delay, color=colors)
    axes[0, 1].set_title('E1 -> E2 delay')
    axes[0, 1].set_ylabel('s')
    axes[0, 1].set_xticks(x, labels, rotation=15, ha='right')

    axes[1, 0].bar(x, crossed, color=colors)
    axes[1, 0].set_title('Recruited surface fraction')
    axes[1, 0].set_ylabel('Fraction crossed')
    axes[1, 0].set_ylim(0.0, 1.05)
    axes[1, 0].set_xticks(x, labels, rotation=15, ha='right')

    width = 0.38
    axes[1, 1].bar(x - 0.5 * width, perfusion, width=width, color='#b279a2', label='min F')
    axes[1, 1].bar(x + 0.5 * width, oxygen, width=width, color='#ff9da6', label='min O')
    axes[1, 1].set_title('Vascular minima')
    axes[1, 1].set_ylabel('Minimum value')
    axes[1, 1].set_xticks(x, labels, rotation=15, ha='right')
    axes[1, 1].legend(frameon=False)

    fig.suptitle('Four-family surface experiment comparison', fontsize=15)
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

    output_root = args.output_root or ROOT / ('outputs/surface_family_comparison_quick' if args.quick else 'outputs/surface_family_comparison')
    output_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, float | int | str | bool]] = []
    snapshot_times = (15.0, 30.0, 45.0)
    final_t_end = resolve_surface_horizon(args)

    t0 = time.time()
    for name in CASE_ORDER:
        base_params = CASE_DEFINITIONS[name]
        params = dc.replace(base_params, final_t_end=final_t_end)
        output = run_surface_simulation(mesh, params, stimulus_vertex=stimulus_vertex, snapshot_times=snapshot_times)
        edge_stats = edge_speed_stats(output)
        case_metrics = compute_case_metrics(output, e1_vertex, e2_vertex, args.roi_radius_mm)
        rows.append(
            {
                'case_label': name,
                'case_title': CASE_TITLES[name],
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
                'vascular_feedback': bool(output.params.enable_vascular_feedback),
                'anisotropy': bool(output.params.enable_anisotropy),
            }
        )

    elapsed = time.time() - t0
    write_csv(output_root / 'surface_family_comparison.csv', rows)
    summary = {
        'mesh_source': mesh.metadata.get('source', 'unknown'),
        'stimulus_vertex': stimulus_vertex,
        'electrode_1_vertex': e1_vertex,
        'electrode_2_vertex': e2_vertex,
        'final_t_end_s': final_t_end,
        'elapsed_s': round(elapsed, 2),
        'case_order': CASE_ORDER,
        'cases': rows,
    }
    (output_root / 'surface_family_comparison.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    plot_family_comparison(rows, output_root / 'surface_family_comparison.png')

    print(f'Four-family surface comparison complete in {elapsed:.1f}s')
    print(f'Outputs written to {output_root}')
    for row in rows:
        print(
            f"  {row['case_label']}: speed={row['arrival_speed_mm_min']:.3f} mm/min, "
            f"delay={row['cross_fold_delay_s']:.2f} s, crossed={row['crossed_fraction']:.3f}"
        )


if __name__ == '__main__':
    main()
