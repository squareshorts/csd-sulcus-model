from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy import sparse
from scipy.sparse import csgraph

from .surface_io import SurfaceMesh
from .surface_ops import SurfaceOperators, build_surface_operators, compute_vertex_normals, estimate_explicit_dt


@dataclass(frozen=True)
class SurfaceParams:
    final_t_end: float = 60.0
    dt: float | None = None
    auto_dt_safety: float = 0.12

    D0: float = 0.010
    depth_decay: float = 1.10
    thickness_transport_gain: float = 0.20
    anisotropy_ratio: float = 1.15
    enable_anisotropy: bool = True
    enable_vascular_feedback: bool = True
    enable_electromagnetic_cancellation: bool = True
    electromagnetic_coupling_gain: float = 0.30
    electromagnetic_parallel_fraction: float = 0.35
    electromagnetic_depth_weight: float = 0.65
    electromagnetic_tilt_weight: float = 0.35
    electromagnetic_reference_depth_quantile: float = 0.25

    k_rest: float = 3.5
    k_peak: float = 60.0
    k_threshold: float = 9.0
    k_threshold_slope: float = 1.5
    k_release_rate: float = 0.16
    k_clearance_rate: float = 0.040
    k_buffer_tau_s: float = 40.0
    k_buffer_depletion_rate: float = 0.18
    k_arrival_threshold: float = 10.0

    stim_k: float = 30.0
    stim_radius_mm: float = 1.2
    min_arrival_t: float = 0.5

    tau_F: float = 12.0
    tau_C: float = 40.0
    tau_O: float = 18.0
    a_dil: float = 0.28
    a_con0: float = 0.22
    lambda_oxygen: float = 0.14
    vascular_excitability_gain: float = 0.18

    beta_depth: float = 0.28
    beta_thickness: float = 0.12
    chi_depth: float = 0.45
    chi_vascular_risk: float = 0.40

    min_perfusion: float = 0.25
    max_perfusion: float = 1.45
    min_oxygen: float = 0.20
    max_oxygen: float = 1.25


@dataclass(frozen=True)
class SurfaceSimulationOutput:
    mesh: SurfaceMesh
    params: SurfaceParams
    operators: SurfaceOperators
    arrival_times: np.ndarray
    potassium: np.ndarray
    buffer_available: np.ndarray
    perfusion: np.ndarray
    constriction: np.ndarray
    oxygen: np.ndarray
    baseline_reserve: np.ndarray
    constriction_susceptibility: np.ndarray
    electromagnetic_cancellation: np.ndarray
    d_parallel: np.ndarray
    d_perp: np.ndarray
    dt_used: float
    snapshot_times: np.ndarray
    snapshot_potassium: np.ndarray
    snapshot_perfusion: np.ndarray
    snapshot_oxygen: np.ndarray


def _normalize_field(field: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    field = np.asarray(field, dtype=float)
    lo = float(np.nanmin(field))
    hi = float(np.nanmax(field))
    if not np.isfinite(lo) or not np.isfinite(hi) or abs(hi - lo) < eps:
        return np.zeros_like(field)
    return (field - lo) / (hi - lo)


def _activation(k: np.ndarray, threshold: np.ndarray, slope: float) -> np.ndarray:
    slope = max(float(slope), 1e-6)
    return 0.5 * (1.0 + np.tanh((k - threshold) / slope))


def _auto_dt(operators: SurfaceOperators, params: SurfaceParams) -> float:
    return estimate_explicit_dt(operators.lumped_mass, operators.stiffness, safety=params.auto_dt_safety)


def _reference_gyral_normal(
    sulcal_depth: np.ndarray,
    vertex_normals: np.ndarray,
    depth_quantile: float,
) -> np.ndarray:
    quantile = float(np.clip(depth_quantile, 0.0, 1.0))
    cutoff = float(np.quantile(sulcal_depth, quantile))
    shallow = sulcal_depth <= cutoff
    if not np.any(shallow):
        shallow = sulcal_depth <= float(np.nanmin(sulcal_depth) + 1e-12)

    reference = np.nanmean(vertex_normals[shallow], axis=0)
    norm = float(np.linalg.norm(reference))
    if not np.isfinite(norm) or norm < 1e-12:
        reference = np.array([0.0, 0.0, 1.0], dtype=float)
        norm = 1.0
    return reference / norm


def build_electromagnetic_cancellation_field(
    mesh: SurfaceMesh,
    vertex_normals: np.ndarray,
    params: SurfaceParams,
) -> np.ndarray:
    if not params.enable_electromagnetic_cancellation:
        return np.zeros(mesh.n_vertices, dtype=float)

    sulcal_depth = np.clip(np.asarray(mesh.sulcal_depth, dtype=float), 0.0, 1.0)
    reference_normal = _reference_gyral_normal(
        sulcal_depth,
        np.asarray(vertex_normals, dtype=float),
        params.electromagnetic_reference_depth_quantile,
    )
    alignment = np.abs(np.sum(vertex_normals * reference_normal[None, :], axis=1))
    alignment = np.clip(alignment, 0.0, 1.0)
    wall_tilt = np.sqrt(np.clip(1.0 - alignment**2, 0.0, 1.0))

    depth_weight = max(float(params.electromagnetic_depth_weight), 0.0)
    tilt_weight = max(float(params.electromagnetic_tilt_weight), 0.0)
    total = depth_weight + tilt_weight
    if total <= 1e-12:
        depth_weight = 1.0
        tilt_weight = 0.0
        total = 1.0

    cancellation = (depth_weight * sulcal_depth + tilt_weight * wall_tilt) / total
    return np.clip(cancellation, 0.0, 1.0)


def build_surface_fields(mesh: SurfaceMesh, params: SurfaceParams) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sulcal_depth = np.clip(np.asarray(mesh.sulcal_depth, dtype=float), 0.0, 1.0)
    inverse_thickness = _normalize_field(1.0 / np.maximum(np.asarray(mesh.thickness, dtype=float), 1e-3))
    vertex_normals = compute_vertex_normals(mesh.vertices, mesh.faces)
    electromagnetic_cancellation = build_electromagnetic_cancellation_field(mesh, vertex_normals, params)

    thickness_factor = 1.0 + params.thickness_transport_gain * inverse_thickness
    electromagnetic_gain = float(np.clip(params.electromagnetic_coupling_gain, 0.0, 0.95))
    parallel_fraction = float(np.clip(params.electromagnetic_parallel_fraction, 0.0, 1.0))
    perpendicular_scale = 1.0 - electromagnetic_gain * electromagnetic_cancellation
    parallel_scale = 1.0 - electromagnetic_gain * parallel_fraction * electromagnetic_cancellation
    perpendicular_scale = np.clip(perpendicular_scale, 0.05, 1.0)
    parallel_scale = np.clip(parallel_scale, 0.05, 1.0)

    d_perp = params.D0 * np.exp(-params.depth_decay * sulcal_depth) * thickness_factor * perpendicular_scale
    anisotropy_ratio = params.anisotropy_ratio if params.enable_anisotropy else 1.0
    d_parallel = anisotropy_ratio * params.D0 * np.exp(-params.depth_decay * sulcal_depth) * thickness_factor * parallel_scale

    baseline_reserve = 1.0 - params.beta_depth * sulcal_depth - params.beta_thickness * inverse_thickness
    baseline_reserve = np.clip(baseline_reserve, params.min_perfusion, 1.05)

    constriction_susceptibility = params.a_con0 * (
        1.0 + params.chi_depth * sulcal_depth + params.chi_vascular_risk * np.asarray(mesh.vascular_risk, dtype=float)
    )
    return d_parallel, d_perp, baseline_reserve, constriction_susceptibility, electromagnetic_cancellation


def nearest_vertex(mesh: SurfaceMesh, point: Sequence[float]) -> int:
    point_arr = np.asarray(point, dtype=float).reshape(1, 3)
    distances = np.linalg.norm(mesh.vertices - point_arr, axis=1)
    return int(np.argmin(distances))


def euclidean_roi(mesh: SurfaceMesh, seed_vertex: int, radius_mm: float) -> np.ndarray:
    center = mesh.vertices[int(seed_vertex)]
    return np.linalg.norm(mesh.vertices - center[None, :], axis=1) <= float(radius_mm)


def edge_length_graph(operators: SurfaceOperators, n_vertices: int) -> sparse.csr_matrix:
    rows = np.concatenate([operators.edge_i, operators.edge_j])
    cols = np.concatenate([operators.edge_j, operators.edge_i])
    data = np.concatenate([operators.edge_lengths, operators.edge_lengths])
    graph = sparse.coo_matrix((data, (rows, cols)), shape=(n_vertices, n_vertices)).tocsr()
    graph.sum_duplicates()
    return graph


def geodesic_roi(operators: SurfaceOperators, seed_vertex: int, radius_mm: float) -> np.ndarray:
    graph = edge_length_graph(operators, int(operators.lumped_mass.shape[0]))
    distances = csgraph.dijkstra(graph, directed=False, indices=int(seed_vertex))
    return np.asarray(distances <= float(radius_mm), dtype=bool)


def median_arrival(arrival_times: np.ndarray, mask: np.ndarray) -> float:
    values = np.asarray(arrival_times[mask], dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float('nan')
    return float(np.median(values))


def surface_arrival_speed_mm_min(
    output: SurfaceSimulationOutput,
    e1_vertex: int,
    e2_vertex: int,
    radius_mm: float = 1.0,
    use_geodesic_roi: bool = True,
) -> float:
    if use_geodesic_roi:
        roi_1 = geodesic_roi(output.operators, e1_vertex, radius_mm)
        roi_2 = geodesic_roi(output.operators, e2_vertex, radius_mm)
    else:
        roi_1 = euclidean_roi(output.mesh, e1_vertex, radius_mm)
        roi_2 = euclidean_roi(output.mesh, e2_vertex, radius_mm)

    t1 = median_arrival(output.arrival_times, roi_1)
    t2 = median_arrival(output.arrival_times, roi_2)
    if not np.isfinite(t1) or not np.isfinite(t2) or t2 <= t1:
        return float('nan')

    graph = edge_length_graph(output.operators, output.mesh.n_vertices)
    distance = float(csgraph.dijkstra(graph, directed=False, indices=int(e1_vertex))[int(e2_vertex)])
    if not np.isfinite(distance) or distance <= 0.0:
        return float('nan')
    return 60.0 * distance / (t2 - t1)


def edge_speed_stats(output: SurfaceSimulationOutput, deep_quantile: float = 0.80) -> dict[str, float]:
    arrival_i = output.arrival_times[output.operators.edge_i]
    arrival_j = output.arrival_times[output.operators.edge_j]
    dt = np.abs(arrival_j - arrival_i)
    valid = np.isfinite(dt) & (dt > 1e-6)
    edge_speed = np.full_like(dt, np.nan, dtype=float)
    edge_speed[valid] = 60.0 * output.operators.edge_lengths[valid] / dt[valid]

    edge_depth = 0.5 * (
        output.mesh.sulcal_depth[output.operators.edge_i] + output.mesh.sulcal_depth[output.operators.edge_j]
    )
    deep_cut = float(np.quantile(output.mesh.sulcal_depth, deep_quantile))
    deep_mask = valid & (edge_depth >= deep_cut)
    shallow_mask = valid & (edge_depth < deep_cut)

    def _median(values: np.ndarray, mask: np.ndarray) -> float:
        subset = values[mask]
        subset = subset[np.isfinite(subset)]
        if subset.size == 0:
            return float('nan')
        return float(np.median(subset))

    return {
        'median_edge_speed_mm_min': _median(edge_speed, valid),
        'deep_edge_speed_mm_min': _median(edge_speed, deep_mask),
        'shallow_edge_speed_mm_min': _median(edge_speed, shallow_mask),
    }


def run_surface_simulation(
    mesh: SurfaceMesh,
    params: SurfaceParams,
    *,
    stimulus_vertex: int,
    snapshot_times: Sequence[float] = (),
) -> SurfaceSimulationOutput:
    d_parallel, d_perp, baseline_reserve, constriction_susceptibility, electromagnetic_cancellation = build_surface_fields(mesh, params)
    operators = build_surface_operators(mesh, d_parallel=d_parallel, d_perp=d_perp)
    dt_used = float(params.dt) if params.dt is not None else _auto_dt(operators, params)

    steps = int(round(params.final_t_end / dt_used))
    requested_snapshots = np.asarray(tuple(snapshot_times), dtype=float)
    requested_snapshots = np.clip(requested_snapshots, 0.0, params.final_t_end)
    snapshot_indices = np.rint(requested_snapshots / dt_used).astype(int) if requested_snapshots.size else np.array([], dtype=int)
    actual_snapshot_times = snapshot_indices * dt_used

    n_vertices = mesh.n_vertices
    potassium = np.full(n_vertices, params.k_rest, dtype=float)
    buffer_available = np.ones(n_vertices, dtype=float)
    perfusion = baseline_reserve.copy()
    constriction = np.zeros(n_vertices, dtype=float)
    oxygen = np.ones(n_vertices, dtype=float)

    stimulus_mask = euclidean_roi(mesh, int(stimulus_vertex), params.stim_radius_mm)
    potassium[stimulus_mask] = np.maximum(potassium[stimulus_mask], params.stim_k)

    arrival_times = np.full(n_vertices, np.nan, dtype=float)
    uncrossed = np.ones(n_vertices, dtype=bool)

    snapshot_potassium = np.empty((snapshot_indices.size, n_vertices), dtype=float)
    snapshot_perfusion = np.empty((snapshot_indices.size, n_vertices), dtype=float)
    snapshot_oxygen = np.empty((snapshot_indices.size, n_vertices), dtype=float)
    snap_cursor = 0

    for step in range(steps + 1):
        time_s = step * dt_used
        if snap_cursor < snapshot_indices.size and step == snapshot_indices[snap_cursor]:
            snapshot_potassium[snap_cursor] = potassium
            snapshot_perfusion[snap_cursor] = perfusion
            snapshot_oxygen[snap_cursor] = oxygen
            snap_cursor += 1

        if time_s >= params.min_arrival_t:
            crossed = uncrossed & (potassium >= params.k_arrival_threshold)
            arrival_times[crossed] = time_s
            uncrossed[crossed] = False

        if step >= steps:
            continue

        vascular_reserve = np.clip(perfusion * oxygen, params.min_oxygen, params.max_perfusion)
        if params.enable_vascular_feedback:
            threshold = params.k_threshold * (1.0 - params.vascular_excitability_gain * (1.0 - np.clip(vascular_reserve, 0.0, 1.0)))
            threshold = np.clip(threshold, 0.55 * params.k_threshold, 1.15 * params.k_threshold)
            clearance_factor = buffer_available * np.clip(vascular_reserve, params.min_oxygen, params.max_perfusion)
        else:
            threshold = np.full(n_vertices, params.k_threshold, dtype=float)
            clearance_factor = buffer_available

        activation = _activation(potassium, threshold, params.k_threshold_slope)
        release = params.k_release_rate * activation * np.maximum(params.k_peak - potassium, 0.0)
        clearance = params.k_clearance_rate * clearance_factor * np.maximum(potassium - params.k_rest, 0.0)
        diffusion = -(operators.stiffness @ potassium) * operators.inv_lumped_mass

        potassium = potassium + dt_used * (release - clearance + diffusion)
        buffer_available = buffer_available + dt_used * (
            (1.0 - buffer_available) / params.k_buffer_tau_s - params.k_buffer_depletion_rate * activation * buffer_available
        )

        if params.enable_vascular_feedback:
            perfusion = perfusion + dt_used * (
                (baseline_reserve - perfusion) / params.tau_F + params.a_dil * activation - constriction_susceptibility * constriction
            )
            constriction = constriction + dt_used * ((activation - constriction) / params.tau_C)
            oxygen = oxygen + dt_used * ((perfusion - oxygen) / params.tau_O - params.lambda_oxygen * activation)
        else:
            perfusion = baseline_reserve.copy()
            constriction.fill(0.0)
            oxygen.fill(1.0)

        np.clip(potassium, params.k_rest, params.k_peak, out=potassium)
        np.clip(buffer_available, 0.05, 1.25, out=buffer_available)
        np.clip(perfusion, params.min_perfusion, params.max_perfusion, out=perfusion)
        np.clip(oxygen, params.min_oxygen, params.max_oxygen, out=oxygen)
        np.clip(constriction, 0.0, 1.5, out=constriction)

    return SurfaceSimulationOutput(
        mesh=mesh,
        params=params,
        operators=operators,
        arrival_times=arrival_times,
        potassium=potassium,
        buffer_available=buffer_available,
        perfusion=perfusion,
        constriction=constriction,
        oxygen=oxygen,
        baseline_reserve=baseline_reserve,
        constriction_susceptibility=constriction_susceptibility,
        electromagnetic_cancellation=electromagnetic_cancellation,
        d_parallel=d_parallel,
        d_perp=d_perp,
        dt_used=dt_used,
        snapshot_times=actual_snapshot_times,
        snapshot_potassium=snapshot_potassium,
        snapshot_perfusion=snapshot_perfusion,
        snapshot_oxygen=snapshot_oxygen,
    )

