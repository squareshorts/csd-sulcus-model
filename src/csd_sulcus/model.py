from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.ndimage import gaussian_filter


@dataclass(frozen=True)
class Params:
    nx: int = 200
    ny: int = 140
    dx: float = 0.10
    dt: float = 0.002
    final_t_end: float = 400.0

    a: float = 0.75
    b: float = 0.02
    eps: float = 0.02
    D0: float = 0.01

    sulcus_center_x: float = 0.50
    sulcus_width_mm: float = 4.0
    sulcus_curved: bool = True
    sulcus_curve_amp: float = 0.10
    sulcus_curve_cycles: float = 1.0
    sulcus_curve_phase: float = 0.0

    g_gyrus: float = 1.00
    g_sulcus_min: float = 0.75
    g_smooth_mm: float = 1.2
    g_profile: str = "flat"
    g_profile_sigma_mm: float = 0.0

    diffusion_mode: str = "scalar"
    tensor_tangent_attenuation_ratio: float = 0.40
    tensor_constraint_mode: str = "manual"
    microstructure_target_tangent_normal_ratio: float = 1.15
    microstructure_ratio_min: float = 1.05
    microstructure_ratio_max: float = 1.30

    kinetics_model: str = "barkley"
    k_rest: float = 3.5
    k_peak: float = 60.0
    k_threshold: float = 9.0
    k_threshold_slope: float = 1.5
    k_release_rate: float = 0.16
    k_clearance_rate: float = 0.040
    k_buffer_tau_s: float = 40.0
    k_buffer_depletion_rate: float = 0.18
    stim_k: float = 30.0
    k_arrival_threshold: float = 10.0

    stim_radius_mm: float = 1.2
    stim_u: float = 0.95
    stim_location: tuple[float, float] = (0.15, 0.50)

    u_threshold: float = 0.50
    min_arrival_t: float = 0.5
    target_gyrus_mm_min: float = 3.0
    seed: int = 0


@dataclass
class SimulationOutput:
    arr: np.ndarray
    g_field: np.ndarray
    sulc_mask: np.ndarray
    phi: np.ndarray
    params: Params
    snapshot_times: np.ndarray
    snapshot_fields: np.ndarray
    g_tangent: np.ndarray | None = None
    g_normal: np.ndarray | None = None


def sulcus_centerline_x_norm(y_norm: np.ndarray, p: Params) -> np.ndarray:
    if not p.sulcus_curved:
        return np.full_like(y_norm, p.sulcus_center_x)
    return p.sulcus_center_x + p.sulcus_curve_amp * np.sin(
        2.0 * np.pi * p.sulcus_curve_cycles * y_norm + p.sulcus_curve_phase
    )


def sulcus_centerline_mm(p: Params) -> tuple[np.ndarray, np.ndarray]:
    x_mm = np.arange(p.nx) * p.dx
    y_mm = np.arange(p.ny) * p.dx
    y_norm = y_mm / (y_mm.max() + 1e-12)
    x_c_norm = sulcus_centerline_x_norm(y_norm, p)
    x_c_mm = x_c_norm * x_mm.max()
    return x_c_mm, y_mm


def build_sulcus_fields(p: Params) -> tuple[np.ndarray, np.ndarray]:
    nx, ny = p.nx, p.ny
    x_mm = np.arange(nx) * p.dx

    if p.sulcus_width_mm <= 0:
        phi = np.full((nx, ny), np.inf)
        mask = np.zeros((nx, ny), dtype=bool)
        return phi, mask

    x_c_mm, _ = sulcus_centerline_mm(p)
    phi = x_mm[:, None] - x_c_mm[None, :]
    mask = np.abs(phi) <= (0.5 * p.sulcus_width_mm)
    return phi, mask


def build_orientation_field(p: Params) -> tuple[np.ndarray, np.ndarray]:
    x_c_mm, y_mm = sulcus_centerline_mm(p)
    if y_mm.size > 2:
        dx_dy = np.gradient(x_c_mm, y_mm, edge_order=2)
    else:
        dx_dy = np.gradient(x_c_mm, y_mm)
    denom = np.sqrt(1.0 + dx_dy**2)
    tx_line = dx_dy / denom
    ty_line = 1.0 / denom
    tx = np.repeat(tx_line[None, :], p.nx, axis=0)
    ty = np.repeat(ty_line[None, :], p.nx, axis=0)
    return tx, ty


def build_coupling_weight(p: Params, dipole_on: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    phi, sulc_mask = build_sulcus_fields(p)
    weight = np.zeros((p.nx, p.ny), dtype=float)

    if not dipole_on:
        return weight, sulc_mask, phi

    profile = p.g_profile.lower()
    if profile == "flat":
        weight = sulc_mask.astype(float)
        if p.g_smooth_mm > 0:
            sigma = p.g_smooth_mm / p.dx
            weight = gaussian_filter(weight, sigma=sigma)
            weight[sulc_mask] = 1.0
            np.clip(weight, 0.0, 1.0, out=weight)
    elif profile == "gaussian":
        sigma_mm = p.g_profile_sigma_mm if p.g_profile_sigma_mm > 0 else max(0.5 * p.sulcus_width_mm, p.dx)
        weight = np.exp(-0.5 * (phi / sigma_mm) ** 2)
    else:
        raise ValueError(f"Unsupported g_profile: {p.g_profile}")

    return weight, sulc_mask, phi


def build_electromagnetic_cancellation_weight(
    p: Params,
    dipole_on: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mesoscale attenuation weight for opposed-bank dipole cancellation inside the sulcus."""
    return build_coupling_weight(p, dipole_on)


def resolve_tensor_tangent_attenuation_ratio(p: Params) -> float:
    mode = p.tensor_constraint_mode.lower()
    if mode in {"manual", "none"}:
        return float(np.clip(p.tensor_tangent_attenuation_ratio, 0.0, 1.0))

    if mode not in {"cortical_microstructure", "cortical_gray_matter"}:
        raise ValueError(f"Unsupported tensor_constraint_mode: {p.tensor_constraint_mode}")

    target_ratio = float(
        np.clip(
            p.microstructure_target_tangent_normal_ratio,
            p.microstructure_ratio_min,
            p.microstructure_ratio_max,
        )
    )
    if np.isclose(p.g_gyrus, p.g_sulcus_min):
        return 1.0

    g_tangent_min = np.clip(target_ratio * p.g_sulcus_min, p.g_sulcus_min, p.g_gyrus)
    eta = (p.g_gyrus - g_tangent_min) / (p.g_gyrus - p.g_sulcus_min)
    return float(np.clip(eta, 0.0, 1.0))


def build_diffusion_fields(
    p: Params,
    dipole_on: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    weight, sulc_mask, phi = build_electromagnetic_cancellation_weight(p, dipole_on)

    g_normal = p.g_gyrus - (p.g_gyrus - p.g_sulcus_min) * weight
    diffusion_mode = p.diffusion_mode.lower()

    if diffusion_mode == "scalar":
        g_tangent = g_normal.copy()
        dxx = p.D0 * g_normal
        dxy = np.zeros_like(dxx)
        dyy = p.D0 * g_normal
        g_effective = g_normal.copy()
    elif diffusion_mode == "tensor":
        tx, ty = build_orientation_field(p)
        nx = ty
        ny = -tx

        tangent_ratio = resolve_tensor_tangent_attenuation_ratio(p)
        g_tangent = p.g_gyrus - tangent_ratio * (p.g_gyrus - p.g_sulcus_min) * weight

        dxx = p.D0 * (g_tangent * tx * tx + g_normal * nx * nx)
        dxy = p.D0 * (g_tangent * tx * ty + g_normal * nx * ny)
        dyy = p.D0 * (g_tangent * ty * ty + g_normal * ny * ny)
        g_effective = np.sqrt(g_tangent * g_normal)
    else:
        raise ValueError(f"Unsupported diffusion_mode: {p.diffusion_mode}")

    return g_effective, sulc_mask, phi, dxx, dxy, dyy, g_tangent, g_normal


def build_g_field(p: Params, dipole_on: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    g_field, sulc_mask, phi, _, _, _, _, _ = build_diffusion_fields(p, dipole_on)
    return g_field, sulc_mask, phi


def divergence_tensor_flux(u: np.ndarray, dxx: np.ndarray, dxy: np.ndarray, dyy: np.ndarray, dx: float) -> np.ndarray:
    nx, ny = u.shape
    u_pad = np.pad(u, ((1, 1), (1, 1)), mode="edge")

    ux_center = (u_pad[2:, 1:-1] - u_pad[:-2, 1:-1]) / (2.0 * dx)
    uy_center = (u_pad[1:-1, 2:] - u_pad[1:-1, :-2]) / (2.0 * dx)

    ux_face_x = (u[1:, :] - u[:-1, :]) / dx
    uy_face_x = 0.5 * (uy_center[1:, :] + uy_center[:-1, :])
    dxx_face_x = 0.5 * (dxx[1:, :] + dxx[:-1, :])
    dxy_face_x = 0.5 * (dxy[1:, :] + dxy[:-1, :])
    flux_x = dxx_face_x * ux_face_x + dxy_face_x * uy_face_x

    uy_face_y = (u[:, 1:] - u[:, :-1]) / dx
    ux_face_y = 0.5 * (ux_center[:, 1:] + ux_center[:, :-1])
    dyy_face_y = 0.5 * (dyy[:, 1:] + dyy[:, :-1])
    dxy_face_y = 0.5 * (dxy[:, 1:] + dxy[:, :-1])
    flux_y = dxy_face_y * ux_face_y + dyy_face_y * uy_face_y

    flux_x_pad = np.zeros((nx + 1, ny), dtype=float)
    flux_x_pad[1:nx, :] = flux_x
    flux_y_pad = np.zeros((nx, ny + 1), dtype=float)
    flux_y_pad[:, 1:ny] = flux_y

    div_x = (flux_x_pad[1:, :] - flux_x_pad[:-1, :]) / dx
    div_y = (flux_y_pad[:, 1:] - flux_y_pad[:, :-1]) / dx
    return div_x + div_y


def barkley_step(
    u: np.ndarray,
    v: np.ndarray,
    dxx: np.ndarray,
    dxy: np.ndarray,
    dyy: np.ndarray,
    p: Params,
) -> tuple[np.ndarray, np.ndarray]:
    f = (1.0 / p.eps) * u * (1.0 - u) * (u - (v + p.b) / p.a)
    g = u - v
    diff = divergence_tensor_flux(u, dxx, dxy, dyy, p.dx)

    u_new = u + p.dt * (f + diff)
    v_new = v + p.dt * g

    np.clip(u_new, -0.1, 1.2, out=u_new)
    np.clip(v_new, -0.2, 1.5, out=v_new)
    return u_new, v_new


def potassium_buffer_step(
    k: np.ndarray,
    buffer_available: np.ndarray,
    dxx: np.ndarray,
    dxy: np.ndarray,
    dyy: np.ndarray,
    p: Params,
) -> tuple[np.ndarray, np.ndarray]:
    activation = 0.5 * (1.0 + np.tanh((k - p.k_threshold) / p.k_threshold_slope))
    release = p.k_release_rate * activation * np.maximum(p.k_peak - k, 0.0)
    clearance = p.k_clearance_rate * buffer_available * np.maximum(k - p.k_rest, 0.0)
    diff = divergence_tensor_flux(k, dxx, dxy, dyy, p.dx)

    k_new = k + p.dt * (release - clearance + diff)
    buffer_new = buffer_available + p.dt * (
        (1.0 - buffer_available) / p.k_buffer_tau_s - p.k_buffer_depletion_rate * activation * buffer_available
    )

    np.clip(k_new, p.k_rest, p.k_peak, out=k_new)
    np.clip(buffer_new, 0.05, 1.25, out=buffer_new)
    return k_new, buffer_new


def initialise_state(p: Params) -> tuple[np.ndarray, np.ndarray]:
    kinetics_model = p.kinetics_model.lower()
    if kinetics_model == "barkley":
        return np.zeros((p.nx, p.ny), dtype=float), np.zeros((p.nx, p.ny), dtype=float)
    if kinetics_model == "potassium_buffer":
        u = np.full((p.nx, p.ny), p.k_rest, dtype=float)
        v = np.ones((p.nx, p.ny), dtype=float)
        return u, v
    raise ValueError(f"Unsupported kinetics_model: {p.kinetics_model}")


def apply_stimulus(u: np.ndarray, p: Params) -> np.ndarray:
    cx = int(round(p.stim_location[0] * (p.nx - 1)))
    cy = int(round(p.stim_location[1] * (p.ny - 1)))
    rr = int(round(p.stim_radius_mm / p.dx))
    X, Y = np.ogrid[: p.nx, : p.ny]
    mask = (X - cx) ** 2 + (Y - cy) ** 2 <= rr**2

    u2 = u.copy()
    stim_value = p.stim_u if p.kinetics_model.lower() == "barkley" else p.stim_k
    u2[mask] = np.maximum(u2[mask], stim_value)
    return u2


def is_arrival_crossed(u: np.ndarray, p: Params) -> np.ndarray:
    kinetics_model = p.kinetics_model.lower()
    if kinetics_model == "barkley":
        return u >= p.u_threshold
    if kinetics_model == "potassium_buffer":
        return u >= p.k_arrival_threshold
    raise ValueError(f"Unsupported kinetics_model: {p.kinetics_model}")


def _snapshot_indices(snapshot_times: Sequence[float], dt: float, steps: int) -> tuple[np.ndarray, np.ndarray]:
    if not snapshot_times:
        return np.array([], dtype=int), np.array([], dtype=float)

    requested = np.asarray(snapshot_times, dtype=float)
    requested = np.clip(requested, 0.0, steps * dt)
    indices = np.rint(requested / dt).astype(int)
    actual_times = indices * dt
    return indices, actual_times


def run_simulation(
    p: Params,
    dipole_on: bool,
    snapshot_times: Sequence[float] = (),
) -> SimulationOutput:
    g_field, sulc_mask, phi, dxx, dxy, dyy, g_tangent, g_normal = build_diffusion_fields(p, dipole_on)

    u, v = initialise_state(p)
    u = apply_stimulus(u, p)

    steps = int(round(p.final_t_end / p.dt))
    snap_indices, actual_snapshot_times = _snapshot_indices(snapshot_times, p.dt, steps)
    snapshot_fields = np.empty((len(snap_indices), p.nx, p.ny), dtype=float)
    snap_cursor = 0

    arr = np.full((p.nx, p.ny), np.nan, dtype=float)
    uncrossed = np.ones((p.nx, p.ny), dtype=bool)
    kinetics_model = p.kinetics_model.lower()

    for k in range(steps + 1):
        t = k * p.dt

        if snap_cursor < len(snap_indices) and k == snap_indices[snap_cursor]:
            snapshot_fields[snap_cursor] = u
            snap_cursor += 1

        if t >= p.min_arrival_t:
            crossed = uncrossed & is_arrival_crossed(u, p)
            arr[crossed] = t
            uncrossed[crossed] = False

        if k < steps:
            if kinetics_model == "barkley":
                u, v = barkley_step(u, v, dxx, dxy, dyy, p)
            elif kinetics_model == "potassium_buffer":
                u, v = potassium_buffer_step(u, v, dxx, dxy, dyy, p)
            else:
                raise ValueError(f"Unsupported kinetics_model: {p.kinetics_model}")

    return SimulationOutput(
        arr=arr,
        g_field=g_field,
        sulc_mask=sulc_mask,
        phi=phi,
        params=p,
        snapshot_times=actual_snapshot_times,
        snapshot_fields=snapshot_fields,
        g_tangent=g_tangent,
        g_normal=g_normal,
    )

