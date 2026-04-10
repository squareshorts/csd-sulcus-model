from pathlib import Path
import shutil

import numpy as np

from csd_sulcus.surface_io import generate_folded_strip_mesh, load_surface_mesh
from csd_sulcus.surface_ops import compute_vertex_normals
from csd_sulcus.surface_prep import (
    derive_vascular_risk,
    prepare_surface_bundle,
    rectify_sulcal_depth,
    write_surface_bundle,
)


def test_rectify_sulcal_depth_respects_sign_mode() -> None:
    raw = np.array([-3.0, -1.0, 0.0, 2.0])
    neg = rectify_sulcal_depth(raw, sign_mode='negative-is-deep')
    pos = rectify_sulcal_depth(raw, sign_mode='positive-is-deep')

    assert np.isclose(neg[0], 1.0)
    assert np.isclose(neg[-1], 0.0)
    assert np.isclose(pos[0], 0.0)
    assert np.isclose(pos[-1], 1.0)


def test_prepare_surface_bundle_derives_unit_tangent_axis() -> None:
    mesh = generate_folded_strip_mesh(nx=18, ny=10)
    raw_sulc = -mesh.sulcal_depth
    bundle = prepare_surface_bundle(mesh.vertices, mesh.faces, raw_sulc, mesh.thickness)

    assert bundle.sulcal_depth.shape == (mesh.n_vertices,)
    assert bundle.preferred_axis.shape == (mesh.n_vertices, 3)
    assert np.allclose(np.linalg.norm(bundle.preferred_axis, axis=1), 1.0, atol=1e-6)

    normals = compute_vertex_normals(bundle.vertices, bundle.faces)
    tangency = np.sum(bundle.preferred_axis * normals, axis=1)
    assert np.allclose(tangency, 0.0, atol=1e-5)


def test_prepare_surface_bundle_roundtrips_to_npz() -> None:
    mesh = generate_folded_strip_mesh(nx=16, ny=8)
    raw_sulc = -mesh.sulcal_depth
    vascular_risk = derive_vascular_risk(mesh.sulcal_depth, mesh.thickness)
    bundle = prepare_surface_bundle(
        mesh.vertices,
        mesh.faces,
        raw_sulc,
        mesh.thickness,
        vascular_risk=vascular_risk,
    )

    temp_dir = Path('.tmp_surface_prep_roundtrip')
    temp_dir.mkdir(exist_ok=True)
    try:
        output_path = temp_dir / 'surface_bundle.npz'
        write_surface_bundle(output_path, bundle)
        loaded = load_surface_mesh(output_path)

        assert loaded.vertices.shape == mesh.vertices.shape
        assert loaded.faces.shape == mesh.faces.shape
        assert np.allclose(loaded.sulcal_depth, bundle.sulcal_depth)
        assert np.allclose(loaded.thickness, bundle.thickness)
        assert np.allclose(loaded.vascular_risk, bundle.vascular_risk)
        assert np.allclose(loaded.preferred_axis, bundle.preferred_axis)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
