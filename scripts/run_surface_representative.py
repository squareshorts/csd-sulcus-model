from __future__ import annotations

import argparse
import csv
import dataclasses as dc
import json
import sys
import time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from scipy.sparse import csgraph

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

matplotlib.use('Agg')

from csd_sulcus.surface_io import generate_folded_strip_mesh, load_surface_mesh
from csd_sulcus.surface_model import (
    SurfaceParams,
    edge_length_graph,
    edge_speed_stats,
    geodesic_roi,
    median_arrival,
    run_surface_simulation,
    surface_arrival_speed_mm_min,
)


CASE_DEFINITIONS = {
    'surface_scalar_transport': SurfaceParams(enable_anisotropy=False, enable_vascular_feedback=False),
    'surface_tensor_transport': SurfaceParams(enable_anisotropy=True, enable_vascular_feedback=False),
    'surface_scalar_vascular': SurfaceParams(enable_anisotropy=False, enable_vascular_feedback=True),
    'surface_tensor_vascular': SurfaceParams(enable_anisotropy=True, enable_vascular_feedback=True),
}

CASE_ORDER = [
    'surface_scalar_transport',
    'surface_tensor_transport',
    'surface_scalar_vascular',
    'surface_tensor_vascular',
]

CASE_TITLES = {
    'surface_scalar_transport': 'Family 1: geometry only',
    'surface_tensor_transport': 'Family 2: geometry + anisotropy',
    'surface_scalar_vascular': 'Family 3: geometry + vascular',
    'surface_tensor_vascular': 'Family 4: geometry + anisotropy + vascular',
}

CASE_SHORT_LABELS = {
    'surface_scalar_transport': 'Geom only',
    'surface_tensor_transport': 'Geom + anis.',
    'surface_scalar_vascular': 'Geom + vasc.',
    'surface_tensor_vascular': 'Geom + anis. + vasc.',
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run the representative four-family surface-CSD scaffold.')
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


def load_mesh(args: argparse.Namespace):
    if args.mesh is not None:
        return load_surface_mesh(
            args.mesh,
            sulcal_depth_path=args.sulcal_depth,
            thickness_path=args.thickness,
            vascular_risk_path=args.vascular_risk,
            preferred_axis_path=args.preferred_axis,
        )
    if args.quick:
        return generate_folded_strip_mesh(nx=40, ny=18, length_mm=18.0, width_mm=8.0, fold_depth_mm=2.0, fold_sigma_mm=1.2)
    return generate_folded_strip_mesh(nx=64, ny=28, length_mm=22.0, width_mm=10.0, fold_depth_mm=2.4, fold_sigma_mm=1.5)


def resolve_surface_horizon(args: argparse.Namespace) -> float:
    if args.final_t_end is not None:
        return float(args.final_t_end)
    if args.quick:
        return 90.0
    return 180.0


def choose_auto_vertices(mesh) -> tuple[int, int, int]:
    centered = mesh.vertices - np.mean(mesh.vertices, axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    coords = centered @ vh[:2].T
    major = coords[:, 0]
    minor = coords[:, 1]
    depth = np.asarray(mesh.sulcal_depth, dtype=float)
    if depth.size == mesh.n_vertices and float(np.nanmax(depth) - np.nanmin(depth)) > 1e-6:
        deep_threshold = float(np.quantile(depth, 0.75))
        deep_mask = depth >= deep_threshold
        if np.sum(deep_mask) >= 3:
            fold_center = float(np.median(minor[deep_mask]))
            fold_halfwidth = float(np.quantile(np.abs(minor[deep_mask] - fold_center), 0.90))
            fold_halfwidth = max(fold_halfwidth, 0.08 * float(np.ptp(minor)), 1e-6)
            lower_bank = minor < (fold_center - 1.05 * fold_halfwidth)
            upper_bank = minor > (fold_center + 1.05 * fold_halfwidth)
            if np.sum(lower_bank) >= 3 and np.sum(upper_bank) >= 3:
                major_source = float(np.quantile(major, 0.25))
                major_electrode = float(np.quantile(major, 0.35))
                lower_minor = minor[lower_bank]
                upper_minor = minor[upper_bank]
                source_minor = float(np.quantile(lower_minor, 0.45))
                e1_minor = float(np.quantile(lower_minor, 0.75))
                e2_minor = float(np.quantile(upper_minor, 0.25))

                major_scale = max(float(np.ptp(major)), 1e-6)
                minor_scale = max(float(np.ptp(minor)), 1e-6)

                def select(mask: np.ndarray, target_major: float, target_minor: float) -> int:
                    candidates = np.where(mask)[0]
                    scores = (
                        ((major[candidates] - target_major) / major_scale) ** 2
                        + 2.5 * ((minor[candidates] - target_minor) / minor_scale) ** 2
                    )
                    return int(candidates[np.argmin(scores)])

                stimulus = select(lower_bank, major_source, source_minor)
                electrode_1 = select(lower_bank, major_electrode, e1_minor)
                electrode_2 = select(upper_bank, major_electrode, e2_minor)
                if len({stimulus, electrode_1, electrode_2}) == 3:
                    return stimulus, electrode_1, electrode_2

    mid = np.median(minor)
    band = np.quantile(np.abs(minor - mid), 0.35)
    mask = np.abs(minor - mid) <= max(float(band), 1e-6)
    if np.sum(mask) < 3:
        mask = np.ones(mesh.n_vertices, dtype=bool)

    def select_along_band(target_quantile: float) -> int:
        target = float(np.quantile(major[mask], target_quantile))
        candidates = np.where(mask)[0]
        best = candidates[np.argmin(np.abs(major[candidates] - target))]
        return int(best)

    return select_along_band(0.15), select_along_band(0.30), select_along_band(0.45)


def projection_2d(mesh) -> np.ndarray:
    centered = mesh.vertices - np.mean(mesh.vertices, axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    return centered @ vh[:2].T


def sparse_quiver_points(projected: np.ndarray, vectors: np.ndarray, target_count: int = 80) -> tuple[np.ndarray, np.ndarray]:
    n_points = int(projected.shape[0])
    if n_points <= target_count:
        idx = np.arange(n_points, dtype=int)
    else:
        step = max(1, n_points // int(target_count))
        idx = np.arange(0, n_points, step, dtype=int)
    return projected[idx], vectors[idx]


def write_csv(path: Path, rows: list[dict[str, float | int | str | bool]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def safe_float(value: float) -> float:
    return float(value) if np.isfinite(value) else float('nan')


def format_metric(value: float, digits: int = 1) -> str:
    if not np.isfinite(value):
        return 'n/a'
    return f'{value:.{digits}f}'


def family_rank_map(rows_by_name: dict[str, dict[str, float | int | str | bool]]) -> dict[str, int]:
    ranked = sorted(
        CASE_ORDER,
        key=lambda name: (
            rows_by_name[name]['arrival_speed_mm_min']
            if np.isfinite(float(rows_by_name[name]['arrival_speed_mm_min']))
            else -np.inf
        ),
        reverse=True,
    )
    return {name: idx + 1 for idx, name in enumerate(ranked)}


def describe_speed_delta(delta: float) -> str:
    if not np.isfinite(delta):
        return 'n/a'
    if abs(delta) < 0.03:
        direction = 'approximately neutral'
    elif delta > 0.0:
        direction = 'faster'
    else:
        direction = 'slower'
    return f'{delta:+.2f} mm/min ({direction})'


def compute_case_metrics(output, e1_vertex: int, e2_vertex: int, roi_radius_mm: float) -> dict[str, float]:
    roi_1 = geodesic_roi(output.operators, e1_vertex, roi_radius_mm)
    roi_2 = geodesic_roi(output.operators, e2_vertex, roi_radius_mm)
    e1_arrival = median_arrival(output.arrival_times, roi_1)
    e2_arrival = median_arrival(output.arrival_times, roi_2)
    graph = edge_length_graph(output.operators, output.mesh.n_vertices)
    distance = float(csgraph.dijkstra(graph, directed=False, indices=int(e1_vertex))[int(e2_vertex)])
    speed = surface_arrival_speed_mm_min(output, e1_vertex, e2_vertex, radius_mm=roi_radius_mm)
    delay = float(e2_arrival - e1_arrival) if np.isfinite(e1_arrival) and np.isfinite(e2_arrival) else float('nan')
    return {
        'e1_arrival_s': safe_float(e1_arrival),
        'e2_arrival_s': safe_float(e2_arrival),
        'cross_fold_delay_s': safe_float(delay),
        'e1_to_e2_geodesic_mm': safe_float(distance),
        'arrival_speed_mm_min': safe_float(speed),
    }


def render_summary_panel(ax, rows_by_name: dict[str, dict[str, float | int | str | bool]]) -> None:
    ax.set_axis_off()
    ranks = family_rank_map(rows_by_name)
    lines = ['Cross-fold results (E1 -> E2)', '']
    reference_distance = float(rows_by_name[CASE_ORDER[0]]['e1_to_e2_geodesic_mm'])
    lines.append(f'Geodesic E1-E2 distance: {format_metric(reference_distance, 2)} mm')
    lines.append('')
    lines.append('Condition                 Delay (s)   Speed (mm/min)')
    lines.append('----------------------------------------------------')
    for name in CASE_ORDER:
        row = rows_by_name[name]
        lines.append(
            f'{CASE_SHORT_LABELS[name]:<24} {format_metric(float(row["cross_fold_delay_s"]), 1):>8}   '
            f'{format_metric(float(row["arrival_speed_mm_min"]), 2):>12}'
        )

    transport_speed = float(rows_by_name['surface_scalar_transport']['arrival_speed_mm_min'])
    tensor_speed = float(rows_by_name['surface_tensor_transport']['arrival_speed_mm_min'])
    scalar_vasc_speed = float(rows_by_name['surface_scalar_vascular']['arrival_speed_mm_min'])
    tensor_vasc_speed = float(rows_by_name['surface_tensor_vascular']['arrival_speed_mm_min'])
    fastest_name = max(CASE_ORDER, key=lambda name: float(rows_by_name[name]['arrival_speed_mm_min']))
    spread = max(float(rows_by_name[name]['arrival_speed_mm_min']) for name in CASE_ORDER) - min(
        float(rows_by_name[name]['arrival_speed_mm_min']) for name in CASE_ORDER
    )
    lines.extend(
        [
            '',
            'Interpretation',
            f'1. Geometry-only transport is rank {ranks["surface_scalar_transport"]}/4 for cross-fold speed.',
            f'2. Anisotropy vs geometry only: {describe_speed_delta(tensor_speed - transport_speed)}.',
            f'3. Vascular effect on scalar branch: {describe_speed_delta(scalar_vasc_speed - transport_speed)}.',
            f'4. Vascular effect on tensor branch: {describe_speed_delta(tensor_vasc_speed - tensor_speed)}.',
            f'5. Fastest family here: {CASE_SHORT_LABELS[fastest_name]}.',
            '',
            (
                'Coupled families remain in the same regime as the transport-only families.'
                if spread < 0.20
                else 'Coupled families separate clearly from the transport-only baseline.'
            ),
            'Read this as a synthetic scaffold sanity-check, not a calibrated anatomical estimate.',
        ]
    )
    ax.text(
        0.03,
        0.98,
        '\n'.join(lines),
        va='top',
        ha='left',
        fontsize=9,
        family='monospace',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#f7f5ef', edgecolor='#c4b89b'),
    )


def plot_case_fields(
    mesh,
    case_outputs: dict[str, object],
    case_rows: list[dict[str, float | int | str | bool]],
    stimulus_vertex: int,
    e1_vertex: int,
    e2_vertex: int,
    output_path: Path,
) -> None:
    projected = projection_2d(mesh)
    triangulation = mtri.Triangulation(projected[:, 0], projected[:, 1], triangles=mesh.faces)
    fig, axes = plt.subplots(3, 3, figsize=(16, 12), constrained_layout=True)

    arrival_max = max(
        float(np.nanmax(case_outputs[name].arrival_times[np.isfinite(case_outputs[name].arrival_times)]))
        for name in CASE_ORDER
    )
    deep_threshold = float(np.quantile(np.asarray(mesh.sulcal_depth, dtype=float), 0.75))
    marker_x = projected[[stimulus_vertex, e1_vertex, e2_vertex], 0]
    marker_y = projected[[stimulus_vertex, e1_vertex, e2_vertex], 1]
    marker_style = dict(c=['white', 'red', 'orange'], s=28, edgecolors='black', linewidths=0.35, zorder=5)
    rows_by_name = {str(row['case_label']): row for row in case_rows}

    depth_plot = axes[0, 0].tripcolor(
        triangulation,
        np.asarray(mesh.sulcal_depth, dtype=float),
        shading='gouraud',
        cmap='cividis',
        vmin=0.0,
        vmax=1.0,
    )
    axes[0, 0].tricontour(
        triangulation,
        np.asarray(mesh.sulcal_depth, dtype=float),
        levels=[deep_threshold],
        colors='white',
        linewidths=1.2,
    )
    axes[0, 0].scatter(marker_x, marker_y, **marker_style)
    axes[0, 0].annotate(
        'Deep fold band',
        xy=(float(np.mean(projected[:, 0])), float(np.median(projected[:, 1]))),
        xytext=(0.03, 0.93),
        textcoords='axes fraction',
        color='white',
        fontsize=9,
        arrowprops=dict(arrowstyle='->', color='white', lw=1.0),
    )
    axes[0, 0].annotate('Near bank', xy=(0.04, 0.18), xycoords='axes fraction', color='white', fontsize=8)
    axes[0, 0].annotate('Far bank', xy=(0.04, 0.80), xycoords='axes fraction', color='white', fontsize=8)
    axes[0, 0].annotate('Source', xy=(marker_x[0], marker_y[0]), xytext=(6, 8), textcoords='offset points', color='white', fontsize=8)
    axes[0, 0].annotate('E1', xy=(marker_x[1], marker_y[1]), xytext=(6, 8), textcoords='offset points', color='white', fontsize=8)
    axes[0, 0].annotate('E2', xy=(marker_x[2], marker_y[2]), xytext=(6, 8), textcoords='offset points', color='white', fontsize=8)
    axes[0, 0].set_title('Fold scaffold and cross-fold markers')
    axes[0, 0].set_aspect('equal')
    axes[0, 0].set_xticks([])
    axes[0, 0].set_yticks([])
    fig.colorbar(depth_plot, ax=axes[0, 0], shrink=0.80, label='Normalized sulcal depth')

    arrival_axes = {
        'surface_scalar_transport': axes[0, 1],
        'surface_tensor_transport': axes[0, 2],
        'surface_scalar_vascular': axes[1, 1],
        'surface_tensor_vascular': axes[1, 2],
    }
    for name, ax in arrival_axes.items():
        output = case_outputs[name]
        field = output.arrival_times
        plot = ax.tripcolor(triangulation, field, shading='gouraud', cmap='viridis', vmin=0.0, vmax=arrival_max)
        ax.tricontour(
            triangulation,
            np.asarray(mesh.sulcal_depth, dtype=float),
            levels=[deep_threshold],
            colors='white',
            linewidths=0.9,
            alpha=0.85,
        )
        ax.scatter(marker_x, marker_y, **marker_style)
        row = rows_by_name[name]
        ax.set_title(
            f'{CASE_TITLES[name]}\n'
            f'delay={format_metric(float(row["cross_fold_delay_s"]), 1)} s, '
            f'speed={format_metric(float(row["arrival_speed_mm_min"]), 2)} mm/min'
        )
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(plot, ax=ax, shrink=0.80, label='Arrival time (s)')

    quiver_points, quiver_vectors = sparse_quiver_points(projected, np.asarray(mesh.preferred_axis, dtype=float)[:, :2], target_count=90)
    axis_plot = axes[1, 0].tripcolor(
        triangulation,
        np.asarray(mesh.sulcal_depth, dtype=float),
        shading='gouraud',
        cmap='Greys',
        vmin=0.0,
        vmax=1.0,
    )
    axes[1, 0].tricontour(
        triangulation,
        np.asarray(mesh.sulcal_depth, dtype=float),
        levels=[deep_threshold],
        colors='tab:blue',
        linewidths=1.0,
    )
    axes[1, 0].quiver(
        quiver_points[:, 0],
        quiver_points[:, 1],
        quiver_vectors[:, 0],
        quiver_vectors[:, 1],
        color='tab:orange',
        angles='xy',
        scale_units='xy',
        scale=0.7,
        width=0.003,
    )
    axes[1, 0].scatter(marker_x, marker_y, **marker_style)
    axes[1, 0].set_title('Preferred tangential axis')
    axes[1, 0].set_aspect('equal')
    axes[1, 0].set_xticks([])
    axes[1, 0].set_yticks([])
    fig.colorbar(axis_plot, ax=axes[1, 0], shrink=0.80, label='Normalized sulcal depth')

    perfusion_axes = {
        'surface_scalar_vascular': axes[2, 0],
        'surface_tensor_vascular': axes[2, 1],
    }
    perfusion_min = min(float(np.nanmin(case_outputs[name].perfusion)) for name in perfusion_axes)
    perfusion_max = max(float(np.nanmax(case_outputs[name].perfusion)) for name in perfusion_axes)
    for name, ax in perfusion_axes.items():
        output = case_outputs[name]
        plot = ax.tripcolor(
            triangulation,
            output.perfusion,
            shading='gouraud',
            cmap='magma',
            vmin=perfusion_min,
            vmax=perfusion_max,
        )
        ax.tricontour(
            triangulation,
            np.asarray(mesh.sulcal_depth, dtype=float),
            levels=[deep_threshold],
            colors='white',
            linewidths=0.9,
            alpha=0.85,
        )
        ax.scatter(marker_x, marker_y, **marker_style)
        ax.set_title(f'Perfusion reserve F: {CASE_SHORT_LABELS[name]}')
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(plot, ax=ax, shrink=0.80, label='Perfusion reserve')

    render_summary_panel(axes[2, 2], rows_by_name)

    fig.suptitle('Four surface experiment families on a synthetic folded cortical scaffold', fontsize=16)
    fig.text(
        0.5,
        0.012,
        'White contour marks the deep-fold band. Source and E1 stay on one bank; E2 sits across the fold to expose cross-sulcus slowing.',
        ha='center',
        fontsize=10,
    )
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

    output_root = args.output_root or ROOT / ('outputs/surface_representative_quick' if args.quick else 'outputs/surface_representative')
    output_root.mkdir(parents=True, exist_ok=True)

    case_outputs: dict[str, object] = {}
    rows: list[dict[str, float | int | str | bool]] = []
    snapshot_times = (15.0, 30.0, 45.0)
    final_t_end = resolve_surface_horizon(args)

    t0 = time.time()
    for name in CASE_ORDER:
        base_params = CASE_DEFINITIONS[name]
        params = dc.replace(base_params, final_t_end=final_t_end)
        output = run_surface_simulation(mesh, params, stimulus_vertex=stimulus_vertex, snapshot_times=snapshot_times)
        case_outputs[name] = output
        edge_stats = edge_speed_stats(output)
        case_metrics = compute_case_metrics(output, e1_vertex, e2_vertex, args.roi_radius_mm)
        row = {
            'case_label': name,
            'case_title': CASE_TITLES[name],
            'n_vertices': mesh.n_vertices,
            'n_faces': mesh.n_faces,
            'stimulus_vertex': stimulus_vertex,
            'electrode_1_vertex': e1_vertex,
            'electrode_2_vertex': e2_vertex,
            'final_t_end_s': final_t_end,
            'dt_used_s': output.dt_used,
            'arrival_speed_mm_min': case_metrics['arrival_speed_mm_min'],
            'e1_arrival_s': case_metrics['e1_arrival_s'],
            'e2_arrival_s': case_metrics['e2_arrival_s'],
            'cross_fold_delay_s': case_metrics['cross_fold_delay_s'],
            'e1_to_e2_geodesic_mm': case_metrics['e1_to_e2_geodesic_mm'],
            'median_edge_speed_mm_min': edge_stats['median_edge_speed_mm_min'],
            'deep_edge_speed_mm_min': edge_stats['deep_edge_speed_mm_min'],
            'shallow_edge_speed_mm_min': edge_stats['shallow_edge_speed_mm_min'],
            'deep_minus_shallow_speed_mm_min': safe_float(
                edge_stats['deep_edge_speed_mm_min'] - edge_stats['shallow_edge_speed_mm_min']
            ),
            'min_perfusion': float(np.nanmin(output.perfusion)),
            'min_oxygen': float(np.nanmin(output.oxygen)),
            'mean_baseline_reserve': float(np.nanmean(output.baseline_reserve)),
            'mean_d_perp': float(np.nanmean(output.d_perp)),
            'mean_d_parallel': float(np.nanmean(output.d_parallel)),
            'crossed_fraction': float(np.mean(np.isfinite(output.arrival_times))),
            'vascular_feedback': bool(output.params.enable_vascular_feedback),
            'anisotropy': bool(output.params.enable_anisotropy),
        }
        rows.append(row)
        np.savez(
            output_root / f'{name}_fields.npz',
            arrival_times=output.arrival_times,
            potassium=output.potassium,
            buffer_available=output.buffer_available,
            perfusion=output.perfusion,
            constriction=output.constriction,
            oxygen=output.oxygen,
            baseline_reserve=output.baseline_reserve,
            d_parallel=output.d_parallel,
            d_perp=output.d_perp,
            snapshot_times=output.snapshot_times,
            snapshot_potassium=output.snapshot_potassium,
            snapshot_perfusion=output.snapshot_perfusion,
            snapshot_oxygen=output.snapshot_oxygen,
        )

    elapsed = time.time() - t0
    write_csv(output_root / 'surface_representative_summary.csv', rows)
    summary_payload = {
        'mesh_source': mesh.metadata.get('source', 'unknown'),
        'n_vertices': mesh.n_vertices,
        'n_faces': mesh.n_faces,
        'stimulus_vertex': stimulus_vertex,
        'electrode_1_vertex': e1_vertex,
        'electrode_2_vertex': e2_vertex,
        'final_t_end_s': final_t_end,
        'elapsed_s': round(elapsed, 2),
        'case_order': CASE_ORDER,
        'cases': rows,
    }
    (output_root / 'surface_representative_summary.json').write_text(json.dumps(summary_payload, indent=2), encoding='utf-8')
    plot_case_fields(mesh, case_outputs, rows, stimulus_vertex, e1_vertex, e2_vertex, output_root / 'surface_representative_fields.png')

    print(f'Surface representative run complete in {elapsed:.1f}s')
    print(f'Outputs written to {output_root}')
    for row in rows:
        print(
            f"  {row['case_label']}: speed={row['arrival_speed_mm_min']:.3f} mm/min, "
            f"delay={row['cross_fold_delay_s']:.2f} s, minF={row['min_perfusion']:.3f}, minO={row['min_oxygen']:.3f}"
        )


if __name__ == '__main__':
    main()
