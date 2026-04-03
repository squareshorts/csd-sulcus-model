import dataclasses as dc

import numpy as np

from csd_sulcus.analysis import compare_control_vs_dipole, multi_seed_robustness, sweep_g_sulcus
from csd_sulcus.model import Params, build_diffusion_fields, build_g_field


E1 = (0.43, 0.50)
E2 = (0.62, 0.50)
R_MM = 1.0


def quick_params() -> Params:
    return dc.replace(
        Params(),
        nx=90,
        ny=64,
        final_t_end=40.0,
        sulcus_width_mm=3.0,
        g_smooth_mm=0.6,
    )


def test_control_calibrates_to_target_speed() -> None:
    comparison = compare_control_vs_dipole(quick_params(), E1, E2, radius_mm=R_MM, snapshot_times=(8.0, 12.0))
    assert np.isclose(
        comparison.control_measurement.scaled_speed_mm_min,
        comparison.control.params.target_gyrus_mm_min,
        atol=1e-6,
    )


def test_velocity_sweep_is_monotonic() -> None:
    comparison = compare_control_vs_dipole(quick_params(), E1, E2, radius_mm=R_MM)
    rows = sweep_g_sulcus(
        quick_params(),
        E1,
        E2,
        radius_mm=R_MM,
        fixed_scale=comparison.fixed_scale,
        gmins=[1.0, 0.9, 0.8, 0.75],
        control_speed_mm_min=comparison.control_measurement.scaled_speed_mm_min,
    )
    speeds = [row["speed_mm_min"] for row in rows]
    assert speeds == sorted(speeds, reverse=True)


def test_multi_seed_output_is_deterministic() -> None:
    comparison = compare_control_vs_dipole(quick_params(), E1, E2, radius_mm=R_MM)
    rows = multi_seed_robustness(
        quick_params(),
        E1,
        E2,
        radius_mm=R_MM,
        fixed_scale=comparison.fixed_scale,
        num_seeds=3,
    )
    deltas = [row["delta_speed_mm_min"] for row in rows]
    assert np.allclose(deltas, deltas[0])


def test_gaussian_profile_is_bounded() -> None:
    p = dc.replace(quick_params(), g_profile="gaussian", g_sulcus_min=0.75)
    g_field, _, _ = build_g_field(p, dipole_on=True)
    assert np.nanmin(g_field) >= p.g_sulcus_min - 1e-9
    assert np.nanmax(g_field) <= p.g_gyrus + 1e-9


def test_tensor_diffusion_is_positive_definite() -> None:
    p = dc.replace(quick_params(), diffusion_mode="tensor", g_profile="flat", g_sulcus_min=0.75)
    _, _, _, dxx, dxy, dyy, _, _ = build_diffusion_fields(p, dipole_on=True)
    determinant = dxx * dyy - dxy**2
    assert np.nanmin(determinant) > 0
