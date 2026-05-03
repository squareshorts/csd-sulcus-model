from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy import sparse
from scipy.sparse import csgraph, linalg as spla

from .surface_io import SurfaceMesh
from .surface_model import build_surface_fields, edge_speed_stats, median_arrival
from .surface_ops import SurfaceOperators, build_surface_operators, estimate_explicit_dt


THERMAL_VOLTAGE_MV = 26.64


@dataclass(frozen=True)
class MechanisticSurfaceParams:
    final_t_end: float = 220.0
    dt: float | None = None
    auto_dt_safety: float = 0.10
    mechanistic_dt_scale: float = 4.0

    D0: float = 0.010
    depth_decay: float = 1.10
    thickness_transport_gain: float = 0.20
    anisotropy_ratio: float = 1.15
    enable_anisotropy: bool = True
    enable_vascular_feedback: bool = False
    enable_dipole_alignment: bool = True

    ecs_volume_fraction_base: float = 0.20
    ecs_volume_fraction_min: float = 0.08
    ecs_depth_volume_fraction_loss: float = 0.28
    ecs_thickness_volume_fraction_loss: float = 0.10
    ecs_tortuosity_base: float = 1.60
    ecs_depth_tortuosity_gain: float = 0.35
    ecs_thickness_tortuosity_gain: float = 0.10
    ecs_swelling_tau: float = 12.0
    ecs_swelling_recovery_tau: float = 28.0
    ecs_swelling_volume_fraction_gain: float = 0.34
    ecs_swelling_tortuosity_gain: float = 0.28
    osmotic_swelling_gain: float = 5.0
    osmotic_swelling_half_saturation: float = 0.06
    activity_swelling_gain: float = 0.20
    swelling_target_max: float = 1.10

    potassium_diffusivity_scale: float = 1.00
    sodium_diffusivity_scale: float = 0.82
    chloride_diffusivity_scale: float = 1.08
    field_reference_mV: float = 5.0
    electrodiffusion_mobility_fraction: float = 0.50

    k_e_rest: float = 3.5
    k_i_rest: float = 140.0
    na_e_rest: float = 145.0
    na_i_rest: float = 18.0
    cl_e_rest: float = 112.0
    cl_i_rest: float = 7.0

    p_k_leak: float = 1.00
    p_na_leak: float = 0.035
    p_cl_leak: float = 0.45
    g_k_leak: float = 0.14
    g_na_leak: float = 0.035
    g_cl_leak: float = 0.06
    p_k_active: float = 1.25
    p_na_active: float = 1.55
    p_cl_active: float = 0.18
    g_k_active: float = 0.34
    g_na_active: float = 0.46
    g_cl_active: float = 0.14
    membrane_flux_scale: float = 0.038
    pump_flux_scale: float = 0.011
    pump_max_rate: float = 1.25
    pump_half_na_i: float = 14.0
    pump_half_k_e: float = 4.2

    activation_tau: float = 4.5
    activation_k_threshold: float = 8.5
    activation_k_slope: float = 1.35
    activation_voltage_threshold_mv: float = -58.0
    activation_voltage_slope_mv: float = 6.5
    arrival_voltage_threshold_mv: float = -28.0

    charge_field_gain_per_mM: float = 0.084
    dipole_field_gain: float = 1.50
    potential_screening: float = 0.65
    dipole_screening_length_mm: float = 4.0
    dipole_cutoff_mm: float = 12.0
    dipole_kernel_mode: str = "aligned"
    dipole_kernel_seed: int = 13

    stim_radius_mm: float = 1.2
    stimulus_k_e: float = 22.0
    stimulus_na_e_drop: float = 10.0
    stimulus_theta: float = 0.92
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
class MechanisticSurfaceSimulationOutput:
    mesh: SurfaceMesh
    params: MechanisticSurfaceParams
    operators: SurfaceOperators
    arrival_times: np.ndarray
    membrane_voltage_mv: np.ndarray
    activation: np.ndarray
    potassium_e: np.ndarray
    sodium_e: np.ndarray
    chloride_e: np.ndarray
    potassium_i: np.ndarray
    sodium_i: np.ndarray
    chloride_i: np.ndarray
    pump_rate: np.ndarray
    perfusion: np.ndarray
    constriction: np.ndarray
    oxygen: np.ndarray
    swelling: np.ndarray
    baseline_reserve: np.ndarray
    constriction_susceptibility: np.ndarray
    d_parallel: np.ndarray
    d_perp: np.ndarray
    ecs_volume_fraction: np.ndarray
    ecs_tortuosity: np.ndarray
    electric_potential: np.ndarray
    dt_used: float
    snapshot_times: np.ndarray
    snapshot_voltage_mv: np.ndarray
    snapshot_potassium_e: np.ndarray
    snapshot_potential: np.ndarray


def _logistic(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def _auto_dt(operators: SurfaceOperators, params: MechanisticSurfaceParams) -> float:
    base_dt = estimate_explicit_dt(operators.lumped_mass, operators.stiffness, safety=params.auto_dt_safety)
    return base_dt / max(float(params.mechanistic_dt_scale), 1.0)


def _dynamic_extracellular_fields(
    d_parallel_base: np.ndarray,
    d_perp_base: np.ndarray,
    ecs_volume_fraction_base: np.ndarray,
    ecs_tortuosity_base: np.ndarray,
    swelling: np.ndarray,
    params: MechanisticSurfaceParams,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    swelling_state = np.clip(swelling, 0.0, params.swelling_target_max)
    ecs_volume_fraction = ecs_volume_fraction_base * (
        1.0 - params.ecs_swelling_volume_fraction_gain * swelling_state
    )
    ecs_volume_fraction = np.clip(
        ecs_volume_fraction,
        params.ecs_volume_fraction_min,
        params.ecs_volume_fraction_base,
    )
    ecs_tortuosity = ecs_tortuosity_base * (
        1.0 + params.ecs_swelling_tortuosity_gain * swelling_state
    )
    ecs_tortuosity = np.clip(
        ecs_tortuosity,
        1.0,
        params.ecs_tortuosity_base * 3.0,
    )
    diffusion_scale = (
        ecs_volume_fraction / np.maximum(ecs_volume_fraction_base, 1e-6)
    ) * (ecs_tortuosity_base / np.maximum(ecs_tortuosity, 1.0)) ** 2
    return (
        d_parallel_base * diffusion_scale,
        d_perp_base * diffusion_scale,
        ecs_volume_fraction,
        ecs_tortuosity,
    )


def _nernst_potential(outside: np.ndarray, inside: np.ndarray, valence: float) -> np.ndarray:
    ratio = np.maximum(outside, 1e-6) / np.maximum(inside, 1e-6)
    return (THERMAL_VOLTAGE_MV / float(valence)) * np.log(ratio)


def _ghk_voltage(
    k_e: np.ndarray,
    na_e: np.ndarray,
    cl_e: np.ndarray,
    k_i: np.ndarray,
    na_i: np.ndarray,
    cl_i: np.ndarray,
    p_k: np.ndarray,
    p_na: np.ndarray,
    p_cl: np.ndarray,
) -> np.ndarray:
    numerator = p_k * np.maximum(k_e, 1e-6) + p_na * np.maximum(na_e, 1e-6) + p_cl * np.maximum(cl_i, 1e-6)
    denominator = p_k * np.maximum(k_i, 1e-6) + p_na * np.maximum(na_i, 1e-6) + p_cl * np.maximum(cl_e, 1e-6)
    return THERMAL_VOLTAGE_MV * np.log(np.maximum(numerator, 1e-6) / np.maximum(denominator, 1e-6))


def _pump_rate(
    na_i: np.ndarray,
    k_e: np.ndarray,
    oxygen: np.ndarray,
    params: MechanisticSurfaceParams,
) -> np.ndarray:
    na_term = np.maximum(na_i, 1e-6) ** 3 / (np.maximum(na_i, 1e-6) ** 3 + params.pump_half_na_i**3)
    k_term = np.maximum(k_e, 1e-6) ** 2 / (np.maximum(k_e, 1e-6) ** 2 + params.pump_half_k_e**2)
    oxygen_term = np.clip(oxygen, params.min_oxygen, 1.0)
    return params.pump_max_rate * na_term * k_term * oxygen_term


def _activation_target(
    k_e: np.ndarray,
    membrane_voltage_mv: np.ndarray,
    params: MechanisticSurfaceParams,
) -> np.ndarray:
    k_drive = (k_e - params.activation_k_threshold) / max(params.activation_k_slope, 1e-6)
    v_drive = (membrane_voltage_mv - params.activation_voltage_threshold_mv) / max(params.activation_voltage_slope_mv, 1e-6)
    return _logistic(k_drive + v_drive)


def _membrane_currents(
    membrane_voltage_mv: np.ndarray,
    e_k: np.ndarray,
    e_na: np.ndarray,
    e_cl: np.ndarray,
    activation: np.ndarray,
    params: MechanisticSurfaceParams,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    g_k = params.g_k_leak + params.g_k_active * activation
    g_na = params.g_na_leak + params.g_na_active * activation
    g_cl = params.g_cl_leak + params.g_cl_active * activation
    i_k = g_k * (membrane_voltage_mv - e_k)
    i_na = g_na * (membrane_voltage_mv - e_na)
    i_cl = g_cl * (membrane_voltage_mv - e_cl)
    return i_k, i_na, i_cl


def _edge_conductivity(
    operators: SurfaceOperators,
    d_parallel: np.ndarray,
    d_perp: np.ndarray,
    diffusivity_scale: float,
) -> np.ndarray:
    d_parallel_edge = 0.5 * (d_parallel[operators.edge_i] + d_parallel[operators.edge_j]) * diffusivity_scale
    d_perp_edge = 0.5 * (d_perp[operators.edge_i] + d_perp[operators.edge_j]) * diffusivity_scale
    return d_perp_edge + (d_parallel_edge - d_perp_edge) * operators.edge_alignment_sq


def _edge_flux_divergence(operators: SurfaceOperators, flux_ij: np.ndarray) -> np.ndarray:
    delta = np.zeros_like(operators.lumped_mass, dtype=float)
    np.add.at(delta, operators.edge_i, -flux_ij)
    np.add.at(delta, operators.edge_j, flux_ij)
    return delta * operators.inv_lumped_mass


def _ion_transport_rhs(
    concentration: np.ndarray,
    valence: float,
    diffusivity_scale: float,
    operators: SurfaceOperators,
    d_parallel: np.ndarray,
    d_perp: np.ndarray,
    electric_potential: np.ndarray,
    params: MechanisticSurfaceParams,
) -> np.ndarray:
    edge_conductivity = _edge_conductivity(operators, d_parallel, d_perp, diffusivity_scale)
    diffusive_flux = (
        operators.base_edge_weights
        * edge_conductivity
        * (concentration[operators.edge_i] - concentration[operators.edge_j])
    )
    rhs = _edge_flux_divergence(operators, diffusive_flux)
    concentration_edge = 0.5 * (concentration[operators.edge_i] + concentration[operators.edge_j])
    potential_drop = electric_potential[operators.edge_i] - electric_potential[operators.edge_j]
    beta_ed = params.electrodiffusion_mobility_fraction * params.field_reference_mV / THERMAL_VOLTAGE_MV
    drift_flux = (
        operators.base_edge_weights
        * beta_ed
        * float(valence)
        * edge_conductivity
        * concentration_edge
        * potential_drop
    )
    return rhs + _edge_flux_divergence(operators, drift_flux)


def _build_dipole_interaction_matrix(
    operators: SurfaceOperators,
    params: MechanisticSurfaceParams,
) -> sparse.csr_matrix:
    n_vertices = int(operators.lumped_mass.shape[0])
    if not params.enable_dipole_alignment:
        return sparse.csr_matrix((n_vertices, n_vertices), dtype=float)

    mode = str(params.dipole_kernel_mode).strip().lower()
    if mode not in {"aligned", "distance_only", "scrambled_normals"}:
        raise ValueError(
            "dipole_kernel_mode must be one of 'aligned', 'distance_only', or 'scrambled_normals'."
        )

    screening_length = max(float(params.dipole_screening_length_mm), 1e-6)
    cutoff = max(float(params.dipole_cutoff_mm), screening_length)
    vertex_normals = np.asarray(operators.vertex_normals, dtype=float)
    if mode == "scrambled_normals":
        rng = np.random.default_rng(int(params.dipole_kernel_seed))
        vertex_normals = vertex_normals[rng.permutation(n_vertices)]

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for i in range(n_vertices):
        distances = np.asarray(
            csgraph.dijkstra(operators.graph, directed=False, indices=i, limit=cutoff),
            dtype=float,
        )
        edge_js = np.where(np.isfinite(distances))[0]
        edge_js = edge_js[(edge_js != i) & (distances[edge_js] > 0.0)]
        if edge_js.size == 0:
            continue
        if mode == "distance_only":
            orientation_weight = np.ones(edge_js.size, dtype=float)
        else:
            normal_opposition = 0.5 * (
                1.0 - np.sum(vertex_normals[i][None, :] * vertex_normals[edge_js], axis=1)
            )
            orientation_weight = np.clip(normal_opposition, 0.0, 1.0)
        kernel = np.exp(-distances[edge_js] / screening_length) * orientation_weight
        keep = kernel > 1e-8
        edge_js = edge_js[keep]
        kernel = kernel[keep]
        if kernel.size == 0:
            continue
        kernel = kernel / np.sum(kernel)
        rows.extend([i] * int(edge_js.size))
        cols.extend(edge_js.tolist())
        data.extend(kernel.tolist())

    if not data:
        return sparse.csr_matrix((n_vertices, n_vertices), dtype=float)
    return sparse.coo_matrix((data, (rows, cols)), shape=(n_vertices, n_vertices)).tocsr()


def _build_potential_solver(
    mesh: SurfaceMesh,
    d_parallel_base: np.ndarray,
    d_perp_base: np.ndarray,
    params: MechanisticSurfaceParams,
) -> tuple[SurfaceOperators, callable]:
    electrical_parallel = 1.8 * d_parallel_base
    electrical_perp = 1.8 * d_perp_base
    phi_operators = build_surface_operators(mesh, d_parallel=electrical_parallel, d_perp=electrical_perp)
    system = phi_operators.stiffness + params.potential_screening * sparse.diags(phi_operators.lumped_mass)
    solver = spla.factorized(system.tocsc())
    return phi_operators, solver


def _electric_potential(
    charge_density: np.ndarray,
    membrane_current: np.ndarray,
    dipole_interaction: sparse.csr_matrix,
    phi_operators: SurfaceOperators,
    phi_solver: callable,
    params: MechanisticSurfaceParams,
) -> np.ndarray:
    dipole_drive = (
        np.asarray(dipole_interaction @ (membrane_current * phi_operators.lumped_mass), dtype=float).reshape(-1)
        if dipole_interaction.nnz > 0
        else np.zeros_like(charge_density)
    )
    rhs = phi_operators.lumped_mass * (
        params.charge_field_gain_per_mM * charge_density - params.dipole_field_gain * dipole_drive
    )
    phi = np.asarray(phi_solver(rhs), dtype=float).reshape(-1)
    phi -= float(np.average(phi, weights=phi_operators.lumped_mass))
    return phi


def _vascular_clearance_modulation(
    perfusion: np.ndarray,
    oxygen: np.ndarray,
    params: MechanisticSurfaceParams,
) -> np.ndarray:
    perfusion_term = 1.0 - params.clearance_perfusion_gain * (1.0 - np.clip(perfusion, 0.0, 1.0))
    oxygen_term = 1.0 - params.clearance_oxygen_gain * (1.0 - np.clip(oxygen, 0.0, 1.0))
    return np.clip(perfusion_term * oxygen_term, params.min_clearance_factor, 1.25)


def _resolve_threshold_constriction_gain(params: MechanisticSurfaceParams) -> float:
    if params.threshold_constriction_gain is None:
        return float(params.vascular_excitability_gain)
    return float(params.threshold_constriction_gain)


def _vascular_threshold_field(
    baseline_reserve: np.ndarray,
    constriction: np.ndarray,
    params: MechanisticSurfaceParams,
) -> np.ndarray:
    baseline_vulnerability = np.clip(1.0 - np.clip(baseline_reserve, 0.0, 1.0), 0.0, 1.0)
    constriction_gain = _resolve_threshold_constriction_gain(params)
    threshold_factor = (
        1.0
        + params.threshold_baseline_vulnerability_gain * baseline_vulnerability
        + constriction_gain * np.clip(constriction, 0.0, 1.5)
    )
    return np.clip(
        params.activation_k_threshold * threshold_factor,
        params.activation_k_threshold,
        params.max_threshold_factor * params.activation_k_threshold,
    )


def _snapshot_indices(snapshot_times: Sequence[float], dt: float, final_t_end: float) -> tuple[np.ndarray, np.ndarray]:
    requested = np.asarray(tuple(snapshot_times), dtype=float)
    if requested.size == 0:
        return np.array([], dtype=int), np.array([], dtype=float)
    requested = np.clip(requested, 0.0, final_t_end)
    indices = np.rint(requested / dt).astype(int)
    return indices, indices * dt


def _resting_reference(
    n_vertices: int,
    params: MechanisticSurfaceParams,
) -> tuple[float, float, float, float, float, float, float, float]:
    p_k = np.full(n_vertices, params.p_k_leak, dtype=float)
    p_na = np.full(n_vertices, params.p_na_leak, dtype=float)
    p_cl = np.full(n_vertices, params.p_cl_leak, dtype=float)
    v_rest = _ghk_voltage(
        np.full(n_vertices, params.k_e_rest, dtype=float),
        np.full(n_vertices, params.na_e_rest, dtype=float),
        np.full(n_vertices, params.cl_e_rest, dtype=float),
        np.full(n_vertices, params.k_i_rest, dtype=float),
        np.full(n_vertices, params.na_i_rest, dtype=float),
        np.full(n_vertices, params.cl_i_rest, dtype=float),
        p_k,
        p_na,
        p_cl,
    )[0]
    e_k_rest = _nernst_potential(np.array([params.k_e_rest]), np.array([params.k_i_rest]), 1.0)[0]
    e_na_rest = _nernst_potential(np.array([params.na_e_rest]), np.array([params.na_i_rest]), 1.0)[0]
    e_cl_rest = _nernst_potential(np.array([params.cl_e_rest]), np.array([params.cl_i_rest]), -1.0)[0]
    i_k_rest, i_na_rest, i_cl_rest = _membrane_currents(
        np.full(n_vertices, v_rest, dtype=float),
        np.full(n_vertices, e_k_rest, dtype=float),
        np.full(n_vertices, e_na_rest, dtype=float),
        np.full(n_vertices, e_cl_rest, dtype=float),
        np.zeros(n_vertices, dtype=float),
        params,
    )
    pump_rest = _pump_rate(
        np.full(n_vertices, params.na_i_rest, dtype=float),
        np.full(n_vertices, params.k_e_rest, dtype=float),
        np.ones(n_vertices, dtype=float),
        params,
    )[0]
    return (
        float(v_rest),
        float(i_k_rest[0]),
        float(i_na_rest[0]),
        float(i_cl_rest[0]),
        float(pump_rest),
        float(e_k_rest),
        float(e_na_rest),
        float(e_cl_rest),
    )


def run_mechanistic_surface_simulation(
    mesh: SurfaceMesh,
    params: MechanisticSurfaceParams,
    *,
    stimulus_vertex: int,
    snapshot_times: Sequence[float] = (),
) -> MechanisticSurfaceSimulationOutput:
    (
        d_parallel_base,
        d_perp_base,
        baseline_reserve,
        constriction_susceptibility,
        ecs_volume_fraction_base,
        ecs_tortuosity_base,
    ) = build_surface_fields(mesh, params)
    operators = build_surface_operators(mesh, d_parallel=d_parallel_base, d_perp=d_perp_base)
    dt_used = float(params.dt) if params.dt is not None else _auto_dt(operators, params)
    phi_operators, phi_solver = _build_potential_solver(mesh, d_parallel_base, d_perp_base, params)
    dipole_interaction = _build_dipole_interaction_matrix(operators, params)

    n_vertices = mesh.n_vertices
    steps = int(round(params.final_t_end / dt_used))
    snap_indices, actual_snapshot_times = _snapshot_indices(snapshot_times, dt_used, params.final_t_end)
    snapshot_voltage_mv = np.empty((snap_indices.size, n_vertices), dtype=float)
    snapshot_potassium_e = np.empty((snap_indices.size, n_vertices), dtype=float)
    snapshot_potential = np.empty((snap_indices.size, n_vertices), dtype=float)
    snap_cursor = 0
    (
        v_rest,
        i_k_rest,
        i_na_rest,
        i_cl_rest,
        pump_rest,
        _e_k_rest,
        _e_na_rest,
        _e_cl_rest,
    ) = _resting_reference(n_vertices, params)

    potassium_e = np.full(n_vertices, params.k_e_rest, dtype=float)
    sodium_e = np.full(n_vertices, params.na_e_rest, dtype=float)
    chloride_e = np.full(n_vertices, params.cl_e_rest, dtype=float)
    potassium_i = np.full(n_vertices, params.k_i_rest, dtype=float)
    sodium_i = np.full(n_vertices, params.na_i_rest, dtype=float)
    chloride_i = np.full(n_vertices, params.cl_i_rest, dtype=float)
    activation = np.zeros(n_vertices, dtype=float)
    membrane_voltage_mv = np.full(n_vertices, v_rest, dtype=float)
    perfusion = baseline_reserve.copy()
    constriction = np.zeros(n_vertices, dtype=float)
    oxygen = np.ones(n_vertices, dtype=float)
    swelling = np.zeros(n_vertices, dtype=float)
    pump_rate = np.full(n_vertices, pump_rest, dtype=float)
    electric_potential = np.zeros(n_vertices, dtype=float)
    d_parallel = d_parallel_base.copy()
    d_perp = d_perp_base.copy()
    ecs_volume_fraction = ecs_volume_fraction_base.copy()
    ecs_tortuosity = ecs_tortuosity_base.copy()

    stimulus_mask = np.linalg.norm(mesh.vertices - mesh.vertices[int(stimulus_vertex)][None, :], axis=1) <= params.stim_radius_mm
    potassium_e[stimulus_mask] = np.maximum(potassium_e[stimulus_mask], params.stimulus_k_e)
    sodium_e[stimulus_mask] = np.maximum(sodium_e[stimulus_mask] - params.stimulus_na_e_drop, 5.0)
    activation[stimulus_mask] = np.maximum(activation[stimulus_mask], params.stimulus_theta)

    intracellular_rest_osm = params.k_i_rest + params.na_i_rest + params.cl_i_rest
    arrival_times = np.full(n_vertices, np.nan, dtype=float)
    uncrossed = np.ones(n_vertices, dtype=bool)

    for step in range(steps + 1):
        time_s = step * dt_used
        if snap_cursor < snap_indices.size and step == snap_indices[snap_cursor]:
            snapshot_voltage_mv[snap_cursor] = membrane_voltage_mv
            snapshot_potassium_e[snap_cursor] = potassium_e
            snapshot_potential[snap_cursor] = electric_potential
            snap_cursor += 1

        if time_s >= params.min_arrival_t:
            crossed = uncrossed & (membrane_voltage_mv >= params.arrival_voltage_threshold_mv)
            arrival_times[crossed] = time_s
            uncrossed[crossed] = False

        if step >= steps:
            continue

        if params.enable_vascular_feedback:
            k_threshold = _vascular_threshold_field(baseline_reserve, constriction, params)
            pump_modulation = _vascular_clearance_modulation(perfusion, oxygen, params)
        else:
            k_threshold = np.full(n_vertices, params.activation_k_threshold, dtype=float)
            pump_modulation = np.ones(n_vertices, dtype=float)

        p_k = params.p_k_leak + params.p_k_active * activation
        p_na = params.p_na_leak + params.p_na_active * activation
        p_cl = params.p_cl_leak + params.p_cl_active * activation
        membrane_voltage_mv = _ghk_voltage(
            potassium_e,
            sodium_e,
            chloride_e,
            potassium_i,
            sodium_i,
            chloride_i,
            p_k,
            p_na,
            p_cl,
        )
        membrane_voltage_mv = np.clip(membrane_voltage_mv, -95.0, 35.0)

        e_k = _nernst_potential(potassium_e, potassium_i, 1.0)
        e_na = _nernst_potential(sodium_e, sodium_i, 1.0)
        e_cl = _nernst_potential(chloride_e, chloride_i, -1.0)
        i_k, i_na, i_cl = _membrane_currents(membrane_voltage_mv, e_k, e_na, e_cl, activation, params)
        pump_rate = _pump_rate(sodium_i, potassium_e, oxygen * pump_modulation, params)
        delta_i_k = i_k - i_k_rest
        delta_i_na = i_na - i_na_rest
        delta_i_cl = i_cl - i_cl_rest
        delta_pump = pump_rate - pump_rest
        net_membrane_current = delta_i_k + delta_i_na + delta_i_cl + delta_pump

        d_parallel, d_perp, ecs_volume_fraction, ecs_tortuosity = _dynamic_extracellular_fields(
            d_parallel_base,
            d_perp_base,
            ecs_volume_fraction_base,
            ecs_tortuosity_base,
            swelling,
            params,
        )

        charge_density = (
            (potassium_e - params.k_e_rest)
            + (sodium_e - params.na_e_rest)
            - (chloride_e - params.cl_e_rest)
        )
        electric_potential = _electric_potential(
            charge_density,
            net_membrane_current,
            dipole_interaction,
            phi_operators,
            phi_solver,
            params,
        )

        k_transport = _ion_transport_rhs(
            potassium_e,
            1.0,
            params.potassium_diffusivity_scale,
            operators,
            d_parallel,
            d_perp,
            electric_potential,
            params,
        )
        na_transport = _ion_transport_rhs(
            sodium_e,
            1.0,
            params.sodium_diffusivity_scale,
            operators,
            d_parallel,
            d_perp,
            electric_potential,
            params,
        )
        cl_transport = _ion_transport_rhs(
            chloride_e,
            -1.0,
            params.chloride_diffusivity_scale,
            operators,
            d_parallel,
            d_perp,
            electric_potential,
            params,
        )

        alpha = np.maximum(ecs_volume_fraction, 1e-6)
        beta = np.maximum(1.0 - ecs_volume_fraction, 1e-6)
        k_membrane = params.membrane_flux_scale * delta_i_k - 2.0 * params.pump_flux_scale * delta_pump
        na_membrane = params.membrane_flux_scale * delta_i_na + 3.0 * params.pump_flux_scale * delta_pump
        cl_membrane = -params.membrane_flux_scale * delta_i_cl

        potassium_e = potassium_e + dt_used * (k_transport + k_membrane / alpha)
        sodium_e = sodium_e + dt_used * (na_transport + na_membrane / alpha)
        chloride_e = chloride_e + dt_used * (cl_transport + cl_membrane / alpha)

        potassium_i = potassium_i - dt_used * (k_membrane / beta)
        sodium_i = sodium_i - dt_used * (na_membrane / beta)
        chloride_i = chloride_i - dt_used * (cl_membrane / beta)

        theta_target = _activation_target(potassium_e, membrane_voltage_mv, params)
        theta_target = np.where(potassium_e >= k_threshold, np.maximum(theta_target, 0.55), theta_target)
        activation = activation + dt_used * ((theta_target - activation) / max(params.activation_tau, 1e-6))

        osmotic_drive = np.maximum((potassium_i + sodium_i + chloride_i - intracellular_rest_osm) / intracellular_rest_osm, 0.0)
        osmotic_term = (
            params.osmotic_swelling_gain
            * osmotic_drive
            / (osmotic_drive + max(params.osmotic_swelling_half_saturation, 1e-6))
        )
        activity_term = params.activity_swelling_gain * np.clip(activation, 0.0, 1.5)
        swelling_drive = osmotic_term + activity_term
        swelling_target = params.swelling_target_max * swelling_drive / (1.0 + swelling_drive)
        swelling_tau = np.where(
            swelling_target >= swelling,
            params.ecs_swelling_tau,
            params.ecs_swelling_recovery_tau,
        )
        swelling = swelling + dt_used * ((swelling_target - swelling) / np.maximum(swelling_tau, 1e-6))

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

        np.clip(potassium_e, 1.0, 80.0, out=potassium_e)
        np.clip(sodium_e, 20.0, 170.0, out=sodium_e)
        np.clip(chloride_e, 20.0, 170.0, out=chloride_e)
        np.clip(potassium_i, 20.0, 160.0, out=potassium_i)
        np.clip(sodium_i, 2.0, 80.0, out=sodium_i)
        np.clip(chloride_i, 2.0, 80.0, out=chloride_i)
        np.clip(activation, 0.0, 1.5, out=activation)
        np.clip(swelling, 0.0, params.swelling_target_max, out=swelling)
        np.clip(perfusion, params.min_perfusion, params.max_perfusion, out=perfusion)
        np.clip(oxygen, params.min_oxygen, params.max_oxygen, out=oxygen)
        np.clip(constriction, 0.0, 1.5, out=constriction)

    return MechanisticSurfaceSimulationOutput(
        mesh=mesh,
        params=params,
        operators=operators,
        arrival_times=arrival_times,
        membrane_voltage_mv=membrane_voltage_mv,
        activation=activation,
        potassium_e=potassium_e,
        sodium_e=sodium_e,
        chloride_e=chloride_e,
        potassium_i=potassium_i,
        sodium_i=sodium_i,
        chloride_i=chloride_i,
        pump_rate=pump_rate,
        perfusion=perfusion,
        constriction=constriction,
        oxygen=oxygen,
        swelling=swelling,
        baseline_reserve=baseline_reserve,
        constriction_susceptibility=constriction_susceptibility,
        d_parallel=d_parallel,
        d_perp=d_perp,
        ecs_volume_fraction=ecs_volume_fraction,
        ecs_tortuosity=ecs_tortuosity,
        electric_potential=electric_potential,
        dt_used=dt_used,
        snapshot_times=actual_snapshot_times,
        snapshot_voltage_mv=snapshot_voltage_mv,
        snapshot_potassium_e=snapshot_potassium_e,
        snapshot_potential=snapshot_potential,
    )


def mechanistic_surface_arrival_speed_mm_min(
    output: MechanisticSurfaceSimulationOutput,
    e1_vertex: int,
    e2_vertex: int,
    radius_mm: float = 1.0,
) -> float:
    graph_rows = np.concatenate([output.operators.edge_i, output.operators.edge_j])
    graph_cols = np.concatenate([output.operators.edge_j, output.operators.edge_i])
    graph_data = np.concatenate([output.operators.edge_lengths, output.operators.edge_lengths])
    graph = sparse.coo_matrix(
        (graph_data, (graph_rows, graph_cols)),
        shape=(output.mesh.n_vertices, output.mesh.n_vertices),
    ).tocsr()
    roi_1 = np.asarray(csgraph.dijkstra(graph, directed=False, indices=int(e1_vertex)) <= float(radius_mm), dtype=bool)
    roi_2 = np.asarray(csgraph.dijkstra(graph, directed=False, indices=int(e2_vertex)) <= float(radius_mm), dtype=bool)
    t1 = median_arrival(output.arrival_times, roi_1)
    t2 = median_arrival(output.arrival_times, roi_2)
    if not np.isfinite(t1) or not np.isfinite(t2) or t2 <= t1:
        return float("nan")
    distance = float(csgraph.dijkstra(graph, directed=False, indices=int(e1_vertex))[int(e2_vertex)])
    if not np.isfinite(distance) or distance <= 0.0:
        return float("nan")
    return 60.0 * distance / (t2 - t1)


def mechanistic_edge_speed_stats(output: MechanisticSurfaceSimulationOutput, deep_quantile: float = 0.80) -> dict[str, float]:
    return edge_speed_stats(output, deep_quantile=deep_quantile)
