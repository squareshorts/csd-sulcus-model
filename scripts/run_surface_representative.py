from __future__ import annotations

import argparse
import csv
import dataclasses as dc
import json
import time
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

from csd_sulcus.surface_io import generate_folded_strip_mesh, load_surface_mesh
from csd_sulcus.surface_model import SurfaceParams, edge_speed_stats, run_surface_simulation, surface_arrival_speed_mm_min


CASE_DEFINITIONS = {
    'surface_scalar_transport': SurfaceParams(enable_anisotropy=False, enable_vascular_feedback=False),
    'surface_tensor_transport': SurfaceParams(enable_anisotropy=True, enable_vascular_feedback=False),
    'surface_tensor_vascular': SurfaceParams(enable_anisotropy=True, enable_vascular_feedback=True),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run the representative surface-CSD scaffold on a cortical mesh or synthetic folded strip.')
    parser.add_argument('--mesh', type=Path, default=None, help='Path to an OBJ, NPZ, or GIFTI surface mesh.')
    parser.add_argument('--sulcal-depth', type=Path, default=None, help='Optional per-vertex sulcal-depth field (NPY/NPZ/GIFTI).')
    parser.add_argument('--thickness', type=Path, default=None, help='Optional per-vertex cortical-thickness field (NPY/NPZ/GIFTI).')
    parser.add_argument('--vascular-risk', type=Path, default=None, help='Optional per-vertex vascular-risk field (NPY/NPZ/GIFTI).')
    parser.add_argument('--preferred-axis', type=Path, default=None, help='Optional per-vertex preferred tangential axis (NPY).')
    parser.add_argument('--output-root', type=Path, default=None)
    parser.add_argument('--quick', action='store_true', help='Use a reduced synthetic mesh or shorter surface run for iteration.')
    parser.add_argument('--stimulus-vertex', type=int, default=None)
    parser.add_argument('--electrode-1', type=int, default=None)
    parser.add_argument('--electrode-2', type=int, default=None)
    parser.add_argument('--roi-radius-mm', type=float, default=1.0)
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


def choose_auto_vertices(mesh) -> tuple[int, int, int]:
    centered = mesh.vertices - np.mean(mesh.vertices, axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    coords = centered @ vh[:2].T
    major = coords[:, 0]
    minor = coords[:, 1]
    mid = np.median(minor)
    band = np.quantile(np.abs(minor - mid), 0.35)
    mask = np.abs(minor - mid) <= max(float(band), 1e-6)
    if np.sum(mask) < 3:
        mask = np.ones(mesh.n_vertices, dtype=bool)

    def select(target_quantile: float) -> int:
        target = float(np.quantile(major[mask], target_quantile))
        candidates = np.where(mask)[0]
        best = candidates[np.argmin(np.abs(major[candidates] - target))]
        return int(best)

    return select(0.15), select(0.30), select(0.45)


def projection_2d(mesh) -> np.ndarray:
    centered = mesh.vertices - np.mean(mesh.vertices, axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    return centered @ vh[:2].T


def write_csv(path: Path, rows: list[dict[str, float | int | str | bool]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_case_fields(mesh, case_outputs: dict[str, object], stimulus_vertex: int, e1_vertex: int, e2_vertex: int, output_path: Path) -> None:
    projected = projection_2d(mesh)
    triangulation = mtri.Triangulation(projected[:, 0], projected[:, 1], triangles=mesh.faces)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)

    case_names = ['surface_scalar_transport', 'surface_tensor_transport', 'surface_tensor_vascular']
    arrival_max = max(
        float(np.nanmax(case_outputs[name].arrival_times[np.isfinite(case_outputs[name].arrival_times)]))
        for name in case_names
    )

    for ax, name in zip(axes.flat[:3], case_names):
        output = case_outputs[name]
        field = output.arrival_times
        plot = ax.tripcolor(triangulation, field, shading='gouraud', cmap='viridis', vmin=0.0, vmax=arrival_max)
        ax.scatter(projected[[stimulus_vertex, e1_vertex, e2_vertex], 0], projected[[stimulus_vertex, e1_vertex, e2_vertex], 1], c=['white', 'red', 'orange'], s=24)
        ax.set_title(name.replace('_', ' '))
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(plot, ax=ax, shrink=0.75, label='Arrival time (s)')

    coupled = case_outputs['surface_tensor_vascular']
    coupled_field = coupled.perfusion
    plot = axes[1, 1].tripcolor(triangulation, coupled_field, shading='gouraud', cmap='magma', vmin=float(np.nanmin(coupled_field)), vmax=float(np.nanmax(coupled_field)))
    axes[1, 1].scatter(projected[[stimulus_vertex, e1_vertex, e2_vertex], 0], projected[[stimulus_vertex, e1_vertex, e2_vertex], 1], c=['white', 'red', 'orange'], s=24)
    axes[1, 1].set_title('Coupled perfusion reserve F')
    axes[1, 1].set_aspect('equal')
    axes[1, 1].set_xticks([])
    axes[1, 1].set_yticks([])
    fig.colorbar(plot, ax=axes[1, 1], shrink=0.75, label='Perfusion reserve')

    fig.suptitle('Representative surface CSD scaffold', fontsize=14)
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

    case_outputs = {}
    rows: list[dict[str, float | int | str | bool]] = []
    snapshot_times = (15.0, 30.0, 45.0) if not args.quick else (15.0, 30.0, 45.0)

    t0 = time.time()
    for name, base_params in CASE_DEFINITIONS.items():
        params = base_params
        if args.quick:
            params = dc.replace(base_params, final_t_end=60.0)
        output = run_surface_simulation(mesh, params, stimulus_vertex=stimulus_vertex, snapshot_times=snapshot_times)
        case_outputs[name] = output
        speed = surface_arrival_speed_mm_min(output, e1_vertex, e2_vertex, radius_mm=args.roi_radius_mm)
        edge_stats = edge_speed_stats(output)
        rows.append(
            {
                'case_label': name,
                'n_vertices': mesh.n_vertices,
                'n_faces': mesh.n_faces,
                'stimulus_vertex': stimulus_vertex,
                'electrode_1_vertex': e1_vertex,
                'electrode_2_vertex': e2_vertex,
                'dt_used_s': output.dt_used,
                'arrival_speed_mm_min': speed,
                'median_edge_speed_mm_min': edge_stats['median_edge_speed_mm_min'],
                'deep_edge_speed_mm_min': edge_stats['deep_edge_speed_mm_min'],
                'shallow_edge_speed_mm_min': edge_stats['shallow_edge_speed_mm_min'],
                'min_perfusion': float(np.nanmin(output.perfusion)),
                'min_oxygen': float(np.nanmin(output.oxygen)),
                'mean_baseline_reserve': float(np.nanmean(output.baseline_reserve)),
                'mean_d_perp': float(np.nanmean(output.d_perp)),
                'mean_d_parallel': float(np.nanmean(output.d_parallel)),
                'crossed_fraction': float(np.mean(np.isfinite(output.arrival_times))),
                'vascular_feedback': bool(output.params.enable_vascular_feedback),
                'anisotropy': bool(output.params.enable_anisotropy),
            }
        )
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
        'elapsed_s': round(elapsed, 2),
        'cases': rows,
    }
    (output_root / 'surface_representative_summary.json').write_text(json.dumps(summary_payload, indent=2), encoding='utf-8')
    plot_case_fields(mesh, case_outputs, stimulus_vertex, e1_vertex, e2_vertex, output_root / 'surface_representative_fields.png')

    print(f'Surface representative run complete in {elapsed:.1f}s')
    print(f'Outputs written to {output_root}')
    for row in rows:
        print(
            f"  {row['case_label']}: speed={row['arrival_speed_mm_min']:.3f} mm/min, "
            f"minF={row['min_perfusion']:.3f}, minO={row['min_oxygen']:.3f}"
        )


if __name__ == '__main__':
    main()

