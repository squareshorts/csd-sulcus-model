from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy import sparse
from scipy.sparse import csgraph

from .surface_io import SurfaceMesh
from .surface_ops import SurfaceOperators, build_surface_operators, estimate_explicit_dt


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
    vascular_excitability_gain: float = 0.04
    clearance_perfusion_gain: float = 0.20
    clearance_oxygen_gain: float = 0.08
    threshold_baseline_vulnerability_gain: float = 0.08
    threshold_constriction_gain: float | None = None
    min_clearance_factor: float = 0.35
    max_threshold_factor: float = 1.40

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


def _resolve_threshold_constriction_gain(params: SurfaceParams) -> float:
    if params.threshold_constriction_gain is None:
        return float(params.vascular_excitability_gain)
    return float(params.threshold_constriction_gain)


def _vascular_clearance_modulation(perfusion: np.ndarray, oxygen: np.ndarray, params: SurfaceParams) -> np.ndarray:
    perfusion_term = 1.0 - params.clearance_perfusion_gain * (1.0 - np.clip(perfusion, 0.0, 1.0))
    oxygen_term = 1.0 - params.clearance_oxygen_gain * (1.0 - np.clip(oxygen, 0.0, 1.0))
    return np.clip(perfusion_term * oxygen_term, params.min_clearance_factor, 1.25)


def _vascular_threshold_field(
    baseline_reserve: np.ndarray,
    constriction: np.ndarray,
    params: SurfaceParams,
) -> np.ndarray:
    baseline_vulnerability = np.clip(1.0 - np.clip(baseline_reserve, 0.0, 1.0), 0.0, 1.0)
    constriction_gain = _resolve_threshold_constriction_gain(params)
    threshold_factor = (
        1.0
        + params.threshold_baseline_vulnerability_gain * baseline_vulnerability
        + constriction_gain * np.clip(constriction, 0.0, 1.5)
    )
    return np.clip(
        params.k_threshold * threshold_factor,
        params.k_threshold,
        params.max_threshold_factor * params.k_threshold,
    )


def _auto_dt(operators: SurfaceOperators, params: SurfaceParams) -> float:
    return estimate_explicit_dt(operators.lumped_mass, operators.stiffness, safety=params.auto_dt_safety)


def build_surface_fields(mesh: SurfaceMesh, params: SurfaceParams) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sulcal_depth = np.clip(np.asarray(mesh.sulcal_depth, dtype=float), 0.0, 1.0)
    inverse_thickness = _normalize_field(1.0 / np.maximum(np.asarray(mesh.thickness, dtype=float), 1e-3))

    thickness_factor = 1.0 + params.thickness_transport_gain * inverse_thickness
    d_perp = params.D0 * np.exp(-params.depth_decay * sulcal_depth) * thickness_factor
    anisotropy_ratio = params.anisotropy_ratio if params.enable_anisotropy else 1.0
    d_parallel = anisotropy_ratio * d_perp

    baseline_reserve = 1.0 - params.beta_depth * sulcal_depth - params.beta_thickness * inverse_thickness
    baseline_reserve = np.clip(baseline_reserve, params.min_perfusion, 1.05)

    constriction_susceptibility = params.a_con0 * (
        1.0 + params.chi_depth * sulcal_depth + params.chi_vascular_risk * np.asarray(mesh.vascular_risk, dtype=float)
    )
    return d_parallel, d_perp, baseline_reserve, constriction_susceptibility


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
    d_parallel, d_perp, baseline_reserve, constriction_susceptibility = build_surface_fields(mesh, params)
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

        if params.enable_vascular_feedback:
            threshold = _vascular_threshold_field(baseline_reserve, constriction, params)
            clearance_factor = buffer_available * _vascular_clearance_modulation(perfusion, oxygen, params)
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
            constriction_target = constriction_susceptibility * activation
            perfusion = perfusion + dt_used * (
                (baseline_reserve - perfusion) / params.tau_F + params.a_dil * activation - constriction
            )
            constriction = constriction + dt_used * ((constriction_target - constriction) / params.tau_C)
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
        d_parallel=d_parallel,
        d_perp=d_perp,
        dt_used=dt_used,
        snapshot_times=actual_snapshot_times,
        snapshot_potassium=snapshot_potassium,
        snapshot_perfusion=snapshot_perfusion,
        snapshot_oxygen=snapshot_oxygen,
    )

