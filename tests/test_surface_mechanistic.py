import dataclasses as dc
import sys
from pathlib import Path

import numpy as np

from csd_sulcus.surface_io import generate_folded_strip_mesh
from csd_sulcus.surface_mechanistic import (
    MechanisticSurfaceParams,
    mechanistic_surface_arrival_speed_mm_min,
    run_mechanistic_surface_simulation,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))

from run_surface_representative import choose_auto_vertices


def _run_case(mesh, params: MechanisticSurfaceParams):
    stimulus, e1, e2 = choose_auto_vertices(mesh)
    output = run_mechanistic_surface_simulation(mesh, params, stimulus_vertex=stimulus)
    speed = mechanistic_surface_arrival_speed_mm_min(output, e1, e2, radius_mm=1.0)
    return output, speed


def test_mechanistic_surface_model_runs_with_finite_wave_metrics() -> None:
    mesh = generate_folded_strip_mesh(
        nx=32,
        ny=16,
        length_mm=18.0,
        width_mm=8.0,
        fold_depth_mm=2.4,
        fold_sigma_mm=1.5,
    )
    params = MechanisticSurfaceParams(final_t_end=180.0)
    output, speed = _run_case(mesh, params)

    assert np.isfinite(speed)
    assert np.isfinite(output.arrival_times).any()
    assert np.nanmax(output.potassium_e) > params.k_e_rest + 5.0
    assert np.nanmin(output.sodium_e) < params.na_e_rest - 10.0
    assert np.nanmax(output.swelling) > 0.2
    assert np.nanmax(np.abs(output.electric_potential)) > 1.0
    assert np.nanmin(output.membrane_voltage_mv) < -40.0
    assert np.nanmax(output.membrane_voltage_mv) > -35.0


def test_mechanistic_dipole_slows_folded_surface_but_not_flat_control() -> None:
    folded_mesh = generate_folded_strip_mesh(
        nx=30,
        ny=16,
        length_mm=18.0,
        width_mm=8.0,
        fold_depth_mm=2.4,
        fold_sigma_mm=1.5,
    )
    flat_mesh = generate_folded_strip_mesh(
        nx=30,
        ny=16,
        length_mm=18.0,
        width_mm=8.0,
        fold_depth_mm=0.0,
        fold_sigma_mm=1.5,
    )
    base = MechanisticSurfaceParams(final_t_end=180.0, enable_vascular_feedback=False)

    _, folded_baseline_speed = _run_case(folded_mesh, dc.replace(base, enable_dipole_alignment=False))
    _, folded_dipole_speed = _run_case(folded_mesh, dc.replace(base, enable_dipole_alignment=True))
    _, flat_baseline_speed = _run_case(flat_mesh, dc.replace(base, enable_dipole_alignment=False))
    _, flat_dipole_speed = _run_case(flat_mesh, dc.replace(base, enable_dipole_alignment=True))

    assert np.isfinite(folded_baseline_speed)
    assert np.isfinite(folded_dipole_speed)
    assert folded_dipole_speed < folded_baseline_speed - 0.05
    assert np.isclose(flat_dipole_speed, flat_baseline_speed, atol=1e-8)


def test_mechanistic_vascular_extension_changes_coupled_state_variables() -> None:
    mesh = generate_folded_strip_mesh(
        nx=30,
        ny=16,
        length_mm=18.0,
        width_mm=8.0,
        fold_depth_mm=2.4,
        fold_sigma_mm=1.5,
    )
    base = MechanisticSurfaceParams(final_t_end=180.0, enable_dipole_alignment=True)
    uncoupled, _ = _run_case(mesh, dc.replace(base, enable_vascular_feedback=False))
    coupled, _ = _run_case(mesh, dc.replace(base, enable_vascular_feedback=True))

    shared = np.isfinite(uncoupled.arrival_times) & np.isfinite(coupled.arrival_times)
    assert np.any(shared)
    assert np.nanmax(coupled.constriction) > 0.0
    assert np.nanmin(coupled.oxygen) < np.nanmin(uncoupled.oxygen)
    assert np.nanmin(coupled.perfusion) < np.nanmin(uncoupled.perfusion)
    assert np.nanmedian(np.abs(coupled.arrival_times[shared] - uncoupled.arrival_times[shared])) > 0.5


def test_mechanistic_null_models_weaken_orientation_specific_slowing() -> None:
    mesh = generate_folded_strip_mesh(
        nx=30,
        ny=16,
        length_mm=18.0,
        width_mm=8.0,
        fold_depth_mm=2.4,
        fold_sigma_mm=1.5,
    )
    base = MechanisticSurfaceParams(final_t_end=180.0, enable_vascular_feedback=False)

    _, baseline_speed = _run_case(mesh, dc.replace(base, enable_dipole_alignment=False))
    _, aligned_speed = _run_case(mesh, dc.replace(base, enable_dipole_alignment=True, dipole_kernel_mode='aligned'))
    _, distance_only_speed = _run_case(
        mesh,
        dc.replace(base, enable_dipole_alignment=True, dipole_kernel_mode='distance_only'),
    )
    _, scrambled_speed = _run_case(
        mesh,
        dc.replace(base, enable_dipole_alignment=True, dipole_kernel_mode='scrambled_normals'),
    )

    aligned_slowdown = baseline_speed - aligned_speed
    distance_only_slowdown = baseline_speed - distance_only_speed
    scrambled_slowdown = baseline_speed - scrambled_speed

    assert aligned_slowdown > 0.05
    assert aligned_slowdown > distance_only_slowdown + 0.03
    assert aligned_slowdown > scrambled_slowdown + 0.03


def test_mechanistic_swelling_uses_soft_cap_instead_of_hard_saturation() -> None:
    mesh = generate_folded_strip_mesh(
        nx=32,
        ny=16,
        length_mm=18.0,
        width_mm=8.0,
        fold_depth_mm=2.4,
        fold_sigma_mm=1.5,
    )
    params = MechanisticSurfaceParams(final_t_end=180.0, enable_vascular_feedback=False)
    output, _ = _run_case(mesh, params)

    assert np.nanmax(output.swelling) > 0.2
    assert np.nanmax(output.swelling) < 0.95 * output.params.swelling_target_max
