import sys
from pathlib import Path

from csd_sulcus.surface_io import generate_folded_strip_mesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))

from run_surface_representative import CASE_DEFINITIONS, CASE_ORDER, choose_auto_vertices, projection_2d


def test_surface_case_definitions_cover_four_families() -> None:
    assert CASE_ORDER == [
        'surface_scalar_transport',
        'surface_tensor_transport',
        'surface_scalar_vascular',
        'surface_tensor_vascular',
    ]
    assert set(CASE_DEFINITIONS) == set(CASE_ORDER)
    assert not CASE_DEFINITIONS['surface_scalar_transport'].enable_anisotropy
    assert not CASE_DEFINITIONS['surface_scalar_transport'].enable_vascular_feedback
    assert CASE_DEFINITIONS['surface_tensor_transport'].enable_anisotropy
    assert not CASE_DEFINITIONS['surface_tensor_transport'].enable_vascular_feedback
    assert not CASE_DEFINITIONS['surface_scalar_vascular'].enable_anisotropy
    assert CASE_DEFINITIONS['surface_scalar_vascular'].enable_vascular_feedback
    assert CASE_DEFINITIONS['surface_tensor_vascular'].enable_anisotropy
    assert CASE_DEFINITIONS['surface_tensor_vascular'].enable_vascular_feedback


def test_choose_auto_vertices_places_e2_across_the_fold() -> None:
    mesh = generate_folded_strip_mesh(nx=40, ny=18, length_mm=18.0, width_mm=8.0, fold_depth_mm=2.0, fold_sigma_mm=1.2)
    stim, e1, e2 = choose_auto_vertices(mesh)
    projected = projection_2d(mesh)

    assert projected[stim, 1] < 0.0
    assert projected[e1, 1] < 0.0
    assert projected[e2, 1] > 0.0
    assert mesh.sulcal_depth[e1] > mesh.sulcal_depth[stim]
    assert mesh.sulcal_depth[e2] > mesh.sulcal_depth[stim]
