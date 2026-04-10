from __future__ import annotations

import dataclasses as dc
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from .model import Params, SimulationOutput, run_simulation


@dataclass(frozen=True)
class ElectrodeMeasurement:
    scaled_speed_mm_min: float
    model_speed_mm_min: float
    distance_mm: float
    t1: float
    t2: float


@dataclass(frozen=True)
class ConditionComparison:
    control: SimulationOutput
    dipole: SimulationOutput
    fixed_scale: float
    control_measurement: ElectrodeMeasurement
    dipole_measurement: ElectrodeMeasurement
    delay_sulcus_s: float
    delay_upstream_s: float
    delay_downstream_s: float
    peak_delay_s: float


@dataclass(frozen=True)
class ReferenceBandSummary:
    label: str
    lower_mm_min: float
    upper_mm_min: float
    n_total: int
    n_within: int
    observed_min_mm_min: float
    observed_max_mm_min: float
    observed_mean_mm_min: float


def roi_mask(center_norm: tuple[float, float], radius_mm: float, p: Params) -> np.ndarray:
    cx = int(round(center_norm[0] * (p.nx - 1)))
    cy = int(round(center_norm[1] * (p.ny - 1)))
    rr = int(round(radius_mm / p.dx))
    X, Y = np.ogrid[: p.nx, : p.ny]
    return (X - cx) ** 2 + (Y - cy) ** 2 <= rr**2


def median_arrival(arr: np.ndarray, mask: np.ndarray) -> float:
    vals = arr[mask]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan")
    return float(np.median(vals))


def electrode_arrival(arr: np.ndarray, p: Params, center: tuple[float, float], radius_mm: float) -> float:
    return median_arrival(arr, roi_mask(center, radius_mm, p))


def electrode_speed_mm_min(
    arr: np.ndarray,
    p: Params,
    e1: tuple[float, float],
    e2: tuple[float, float],
    radius_mm: float = 1.0,
    fixed_scale: float | None = None,
) -> ElectrodeMeasurement:
    t1 = electrode_arrival(arr, p, e1, radius_mm)
    t2 = electrode_arrival(arr, p, e2, radius_mm)

    x1 = e1[0] * (p.nx - 1) * p.dx
    x2 = e2[0] * (p.nx - 1) * p.dx
    y1 = e1[1] * (p.ny - 1) * p.dx
    y2 = e2[1] * (p.ny - 1) * p.dx
    d_mm = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    dt = t2 - t1
    v_model = (d_mm / dt) * 60.0 if np.isfinite(dt) and dt > 0 else float("nan")

    if fixed_scale is None or not np.isfinite(v_model):
        scaled = float("nan")
    else:
        scaled = v_model * fixed_scale

    return ElectrodeMeasurement(
        scaled_speed_mm_min=float(scaled),
        model_speed_mm_min=float(v_model),
        distance_mm=float(d_mm),
        t1=float(t1),
        t2=float(t2),
    )


def fixed_scale_from_control(
    control: SimulationOutput,
    p: Params,
    e1: tuple[float, float],
    e2: tuple[float, float],
    radius_mm: float,
) -> float:
    measurement = electrode_speed_mm_min(control.arr, p, e1, e2, radius_mm, fixed_scale=None)
    if not np.isfinite(measurement.model_speed_mm_min):
        raise ValueError("Unable to calibrate the control condition because the model speed is not finite.")
    return p.target_gyrus_mm_min / measurement.model_speed_mm_min


def compartment_masks(
    phi: np.ndarray,
    sulc_mask: np.ndarray,
    sulcus_width_mm: float,
    band_mm: float = 2.0,
) -> dict[str, np.ndarray]:
    half_width = 0.5 * sulcus_width_mm
    upstream_mask = (phi < -half_width) & (phi >= -(half_width + band_mm))
    downstream_mask = (phi > half_width) & (phi <= (half_width + band_mm))
    return {
        "sulcus": sulc_mask,
        "upstream": upstream_mask,
        "downstream": downstream_mask,
    }


def _median_field_value(field: np.ndarray, mask: np.ndarray) -> float:
    vals = field[mask]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan")
    return float(np.median(vals))


def compartment_field_medians(
    field: np.ndarray,
    phi: np.ndarray,
    sulc_mask: np.ndarray,
    sulcus_width_mm: float,
    band_mm: float = 2.0,
) -> dict[str, float]:
    masks = compartment_masks(phi, sulc_mask, sulcus_width_mm, band_mm=band_mm)
    return {name: _median_field_value(field, mask) for name, mask in masks.items()}


def arrival_speed_map_mm_min(
    arrival_times_s: np.ndarray,
    p: Params,
    max_speed_mm_min: float | None = 15.0,
) -> np.ndarray:
    grad_x = np.full_like(arrival_times_s, np.nan, dtype=float)
    grad_y = np.full_like(arrival_times_s, np.nan, dtype=float)

    center_x = arrival_times_s[1:-1, :]
    left = arrival_times_s[:-2, :]
    right = arrival_times_s[2:, :]
    valid_x = np.isfinite(center_x) & np.isfinite(left) & np.isfinite(right)
    grad_x[1:-1, :][valid_x] = (right[valid_x] - left[valid_x]) / (2.0 * p.dx)

    center_y = arrival_times_s[:, 1:-1]
    down = arrival_times_s[:, :-2]
    up = arrival_times_s[:, 2:]
    valid_y = np.isfinite(center_y) & np.isfinite(down) & np.isfinite(up)
    grad_y[:, 1:-1][valid_y] = (up[valid_y] - down[valid_y]) / (2.0 * p.dx)

    grad_norm = np.sqrt(grad_x**2 + grad_y**2)
    speed = np.full_like(arrival_times_s, np.nan, dtype=float)
    valid = np.isfinite(grad_norm) & (grad_norm > 0)
    speed[valid] = 60.0 / grad_norm[valid]

    if max_speed_mm_min is not None:
        speed[speed > max_speed_mm_min] = np.nan
    return speed


def compare_against_reference_control(
    control: SimulationOutput,
    control_params: Params,
    case_params: Params,
    e1: tuple[float, float],
    e2: tuple[float, float],
    radius_mm: float = 1.0,
    snapshot_times: Sequence[float] = (),
) -> ConditionComparison:
    dipole = run_simulation(case_params, dipole_on=True, snapshot_times=snapshot_times)

    fixed_scale = fixed_scale_from_control(control, control_params, e1, e2, radius_mm)
    control_measurement = electrode_speed_mm_min(control.arr, control_params, e1, e2, radius_mm, fixed_scale=fixed_scale)
    dipole_measurement = electrode_speed_mm_min(dipole.arr, case_params, e1, e2, radius_mm, fixed_scale=fixed_scale)

    control_arr_s = control.arr / fixed_scale
    dipole_arr_s = dipole.arr / fixed_scale
    delay_stats = compartment_field_medians(
        dipole_arr_s - control_arr_s,
        dipole.phi,
        dipole.sulc_mask,
        case_params.sulcus_width_mm,
    )
    peak_delay_s = float(np.nanmax(dipole_arr_s - control_arr_s)) if np.isfinite(dipole_arr_s - control_arr_s).any() else float("nan")

    return ConditionComparison(
        control=control,
        dipole=dipole,
        fixed_scale=float(fixed_scale),
        control_measurement=control_measurement,
        dipole_measurement=dipole_measurement,
        delay_sulcus_s=delay_stats["sulcus"],
        delay_upstream_s=delay_stats["upstream"],
        delay_downstream_s=delay_stats["downstream"],
        peak_delay_s=peak_delay_s,
    )


def compare_control_vs_dipole(
    p: Params,
    e1: tuple[float, float],
    e2: tuple[float, float],
    radius_mm: float = 1.0,
    snapshot_times: Sequence[float] = (),
) -> ConditionComparison:
    control = run_simulation(p, dipole_on=False, snapshot_times=snapshot_times)
    return compare_against_reference_control(
        control,
        p,
        p,
        e1,
        e2,
        radius_mm=radius_mm,
        snapshot_times=snapshot_times,
    )


def sweep_g_sulcus(
    p: Params,
    e1: tuple[float, float],
    e2: tuple[float, float],
    radius_mm: float,
    fixed_scale: float,
    gmins: Iterable[float],
    control_speed_mm_min: float,
) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for gm in gmins:
        p_g = dc.replace(p, g_sulcus_min=float(gm))
        out = run_simulation(p_g, dipole_on=(gm < 1.0))
        measurement = electrode_speed_mm_min(out.arr, p_g, e1, e2, radius_mm, fixed_scale=fixed_scale)
        delta = measurement.scaled_speed_mm_min - control_speed_mm_min
        rows.append(
            {
                "g_sulcus_min": float(gm),
                "speed_mm_min": float(measurement.scaled_speed_mm_min),
                "delta_speed_mm_min": float(delta),
                "delta_speed_percent": float(100.0 * delta / control_speed_mm_min),
                "theory_speed_mm_min": float(control_speed_mm_min * np.sqrt(gm)),
            }
        )
    return rows


def multi_seed_robustness(
    p: Params,
    e1: tuple[float, float],
    e2: tuple[float, float],
    radius_mm: float,
    fixed_scale: float,
    num_seeds: int = 5,
) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for seed in range(num_seeds):
        p_seed = dc.replace(p, seed=seed)
        control = run_simulation(p_seed, dipole_on=False)
        dipole = run_simulation(p_seed, dipole_on=True)

        control_measurement = electrode_speed_mm_min(
            control.arr,
            p_seed,
            e1,
            e2,
            radius_mm,
            fixed_scale=fixed_scale,
        )
        dipole_measurement = electrode_speed_mm_min(
            dipole.arr,
            p_seed,
            e1,
            e2,
            radius_mm,
            fixed_scale=fixed_scale,
        )
        rows.append(
            {
                "seed": float(seed),
                "control_speed_mm_min": float(control_measurement.scaled_speed_mm_min),
                "dipole_speed_mm_min": float(dipole_measurement.scaled_speed_mm_min),
                "delta_speed_mm_min": float(
                    dipole_measurement.scaled_speed_mm_min - control_measurement.scaled_speed_mm_min
                ),
            }
        )
    return rows


def summarize_reference_band(
    values: Sequence[float],
    *,
    label: str,
    lower_mm_min: float,
    upper_mm_min: float,
) -> ReferenceBandSummary:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError(f"No finite values provided for reference band summary '{label}'.")

    within = (finite >= lower_mm_min) & (finite <= upper_mm_min)
    return ReferenceBandSummary(
        label=label,
        lower_mm_min=float(lower_mm_min),
        upper_mm_min=float(upper_mm_min),
        n_total=int(finite.size),
        n_within=int(np.sum(within)),
        observed_min_mm_min=float(np.min(finite)),
        observed_max_mm_min=float(np.max(finite)),
        observed_mean_mm_min=float(np.mean(finite)),
    )
