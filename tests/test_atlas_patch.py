import numpy as np

from csd_sulcus.atlas_patch import extract_patch_pair_from_mesh
from csd_sulcus.surface_io import generate_folded_strip_mesh


def test_extract_patch_pair_from_mesh_separates_sulcal_and_flatter_regions() -> None:
    mesh = generate_folded_strip_mesh(
        nx=44,
        ny=22,
        length_mm=20.0,
        width_mm=8.0,
        fold_depth_mm=2.6,
        fold_sigma_mm=1.2,
    )

    patch_pair = extract_patch_pair_from_mesh(mesh, patch_radius_mm=4.5, min_separation_mm=8.0)

    assert patch_pair.sulcal_patch.center_global_idx != patch_pair.flat_patch.center_global_idx
    assert patch_pair.sulcal_patch.mesh.n_vertices > 20
    assert patch_pair.flat_patch.mesh.n_vertices > 20
    assert np.mean(patch_pair.sulcal_patch.mesh.sulcal_depth) > np.mean(patch_pair.flat_patch.mesh.sulcal_depth) + 0.10
    assert patch_pair.sulcal_roi_mask.shape == (mesh.n_vertices,)
    assert patch_pair.flat_roi_mask.shape == (mesh.n_vertices,)
    assert np.any(patch_pair.sulcal_roi_mask & ~patch_pair.flat_roi_mask)
    assert np.any(patch_pair.flat_roi_mask & ~patch_pair.sulcal_roi_mask)
