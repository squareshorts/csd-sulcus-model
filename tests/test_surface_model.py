import dataclasses as dc

import numpy as np

from csd_sulcus.surface_io import generate_folded_strip_mesh
from csd_sulcus.surface_model import (
    SurfaceParams,
    build_surface_fields,
    edge_speed_stats,
    run_surface_simulation,
    surface_arrival_speed_mm_min,
)
from csd_sulcus.surface_ops import build_surface_operators, estimate_explicit_dt


def synthetic_vertices(nx: int, ny: int) -> tuple[int, int, int]:
    center = ny // 2
    stim = 2 * ny + center
    e1 = 3 * ny + center
    e2 = 5 * ny + center
    return stim, e1, e2


def test_generate_folded_strip_mesh_has_expected_fields() -> None:
    mesh = generate_folded_strip_mesh(nx=18, ny=10)
    assert mesh.vertices.shape == (180, 3)
    assert mesh.faces.shape[1] == 3
    assert mesh.sulcal_depth.shape == (180,)
    assert np.nanmax(mesh.sulcal_depth) <= 1.0
    assert np.nanmin(mesh.sulcal_depth) >= 0.0
    assert mesh.preferred_axis is not None


def test_surface_operator_builds_positive_mass_and_stable_dt() -> None:
    mesh = generate_folded_strip_mesh(nx=20, ny=12)
    params = SurfaceParams()
    d_perp = np.full(mesh.n_vertices, params.D0)
    d_parallel = 1.1 * d_perp
    operators = build_surface_operators(mesh, d_parallel=d_parallel, d_perp=d_perp)

    assert np.all(operators.lumped_mass > 0.0)
    assert np.all(np.isfinite(operators.vertex_normals))
    assert np.all(np.isfinite(operators.tangent_directions))
    assert np.allclose(np.asarray(operators.stiffness.sum(axis=1)).ravel(), 0.0, atol=1e-8)
    assert estimate_explicit_dt(operators.lumped_mass, operators.stiffness) > 0.0


def test_surface_vascular_model_runs_and_produces_finite_speed() -> None:
    mesh = generate_folded_strip_mesh(nx=22, ny=12)
    stim, e1, e2 = synthetic_vertices(22, 12)
    params = SurfaceParams(final_t_end=180.0, auto_dt_safety=0.10, tau_F=8.0, tau_O=10.0)
    output = run_surface_simulation(mesh, params, stimulus_vertex=stim, snapshot_times=(45.0, 90.0, 135.0))

    speed = surface_arrival_speed_mm_min(output, e1, e2, radius_mm=1.0)
    stats = edge_speed_stats(output)

    assert np.isfinite(output.arrival_times).any()
    assert np.nanmin(output.perfusion) < 1.0
    assert np.nanmin(output.oxygen) < 1.0
    assert np.nanmax(output.swelling) > 0.0
    assert np.isfinite(speed)
    assert np.isfinite(stats['median_edge_speed_mm_min'])
    assert output.snapshot_potassium.shape[0] == 3
    assert output.ecs_volume_fraction.shape == (mesh.n_vertices,)
    assert output.ecs_tortuosity.shape == (mesh.n_vertices,)
    assert output.electric_potential.shape == (mesh.n_vertices,)


def test_extracellular_diffusion_is_reduced_in_deep_fold() -> None:
    mesh = generate_folded_strip_mesh(nx=24, ny=18)
    params = SurfaceParams(enable_anisotropy=True)
    d_parallel, d_perp, _, _, ecs_volume_fraction, ecs_tortuosity = build_surface_fields(mesh, params)

    shallow = mesh.sulcal_depth <= np.quantile(mesh.sulcal_depth, 0.25)
    deep = mesh.sulcal_depth >= np.quantile(mesh.sulcal_depth, 0.80)

    assert np.median(ecs_volume_fraction[deep]) < np.median(ecs_volume_fraction[shallow])
    assert np.median(ecs_tortuosity[deep]) > np.median(ecs_tortuosity[shallow])
    assert np.median(d_perp[deep]) < np.median(d_perp[shallow])
    assert np.median(d_parallel[deep]) < np.median(d_parallel[shallow])


def test_quasistatic_dipole_field_is_sign_changing_and_nontrivial() -> None:
    mesh = generate_folded_strip_mesh(nx=22, ny=12)
    base = SurfaceParams(final_t_end=180.0, auto_dt_safety=0.10, enable_vascular_feedback=False)
    stim, _, _ = synthetic_vertices(22, 12)
    output = run_surface_simulation(
        mesh,
        dc.replace(base, enable_electromagnetic_dipole=True),
        stimulus_vertex=stim,
    )
    assert np.nanmin(output.electric_potential) < 0.0
    assert np.nanmax(output.electric_potential) > 0.0
    assert np.nanmax(np.abs(output.electric_potential)) > 0.05


def test_electromagnetic_dipole_changes_arrival_dynamics() -> None:
    mesh = generate_folded_strip_mesh(nx=22, ny=12)
    base = SurfaceParams(final_t_end=180.0, auto_dt_safety=0.10, enable_vascular_feedback=False)
    stim, _, _ = synthetic_vertices(22, 12)
    no_em = run_surface_simulation(
        mesh,
        dc.replace(base, enable_electromagnetic_dipole=False),
        stimulus_vertex=stim,
    )
    with_em = run_surface_simulation(
        mesh,
        dc.replace(base, enable_electromagnetic_dipole=True, electrodiffusion_mobility_gain=0.18),
        stimulus_vertex=stim,
    )
    shared = np.isfinite(no_em.arrival_times) & np.isfinite(with_em.arrival_times)
    assert np.any(shared)
    assert np.median(np.abs(with_em.arrival_times[shared] - no_em.arrival_times[shared])) > 0.25


def test_vascular_feedback_changes_coupled_dynamics() -> None:
    mesh = generate_folded_strip_mesh(nx=22, ny=12)
    stim, e1, e2 = synthetic_vertices(22, 12)
    base = SurfaceParams(final_t_end=180.0, auto_dt_safety=0.10, enable_electromagnetic_dipole=True)
    uncoupled = run_surface_simulation(mesh, dc.replace(base, enable_vascular_feedback=False), stimulus_vertex=stim)
    coupled = run_surface_simulation(mesh, dc.replace(base, enable_vascular_feedback=True), stimulus_vertex=stim)

    uncoupled_speed = surface_arrival_speed_mm_min(uncoupled, e1, e2, radius_mm=1.0)
    coupled_speed = surface_arrival_speed_mm_min(coupled, e1, e2, radius_mm=1.0)

    assert np.nanmax(coupled.constriction) > 0.0
    assert np.nanmin(coupled.oxygen) < np.nanmin(uncoupled.oxygen)
    assert np.isfinite(uncoupled_speed)
    assert np.isfinite(coupled_speed)
    assert coupled_speed != uncoupled_speed
