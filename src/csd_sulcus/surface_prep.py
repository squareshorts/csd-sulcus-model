from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from .surface_io import SurfaceMesh, _load_gifti_mesh, _load_npz_mesh, _load_obj_mesh, _load_scalar_field
from .surface_ops import compute_vertex_normals, face_areas_and_normals


SulcSignMode = Literal['negative-is-deep', 'positive-is-deep', 'absolute']


@dataclass(frozen=True)
class PreparedSurfaceBundle:
    vertices: np.ndarray
    faces: np.ndarray
    sulcal_depth: np.ndarray
    thickness: np.ndarray
    vascular_risk: np.ndarray
    preferred_axis: np.ndarray


def normalize_unit_interval(field: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    arr = np.asarray(field, dtype=float).reshape(-1)
    lo = float(np.nanmin(arr))
    hi = float(np.nanmax(arr))
    if not np.isfinite(lo) or not np.isfinite(hi) or abs(hi - lo) < eps:
        return np.zeros_like(arr, dtype=float)
    return (arr - lo) / (hi - lo)


def rectify_sulcal_depth(raw_sulc: np.ndarray, sign_mode: SulcSignMode = 'negative-is-deep') -> np.ndarray:
    raw = np.asarray(raw_sulc, dtype=float).reshape(-1)
    if sign_mode == 'negative-is-deep':
        depth = np.maximum(-raw, 0.0)
    elif sign_mode == 'positive-is-deep':
        depth = np.maximum(raw, 0.0)
    elif sign_mode == 'absolute':
        depth = np.abs(raw)
    else:
        raise ValueError(f'Unsupported sulcal sign mode: {sign_mode}')
    return normalize_unit_interval(depth)


def derive_midthickness(white_vertices: np.ndarray, pial_vertices: np.ndarray) -> np.ndarray:
    white = np.asarray(white_vertices, dtype=float)
    pial = np.asarray(pial_vertices, dtype=float)
    if white.shape != pial.shape:
        raise ValueError(f'White and pial surfaces must have the same vertex shape, got {white.shape} and {pial.shape}.')
    return 0.5 * (white + pial)


def _normalize_vectors(vectors: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return vectors / norms


def face_scalar_gradients(vertices: np.ndarray, faces: np.ndarray, scalar_field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scalar = np.asarray(scalar_field, dtype=float).reshape(-1)
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    edge01 = v1 - v0
    edge02 = v2 - v0
    face_areas, face_normals = face_areas_and_normals(vertices, faces)

    basis_1 = _normalize_vectors(edge01)
    basis_2 = np.cross(face_normals, basis_1)
    basis_2 = _normalize_vectors(basis_2)

    x1 = np.linalg.norm(edge01, axis=1)
    x2 = np.sum(edge02 * basis_1, axis=1)
    y2 = np.sum(edge02 * basis_2, axis=1)
    x1 = np.maximum(x1, 1e-12)
    y2 = np.where(np.abs(y2) < 1e-12, np.sign(y2) * 1e-12 + (y2 == 0.0) * 1e-12, y2)

    df1 = scalar[faces[:, 1]] - scalar[faces[:, 0]]
    df2 = scalar[faces[:, 2]] - scalar[faces[:, 0]]

    grad_1 = df1 / x1
    grad_2 = (df2 - x2 * grad_1) / y2
    gradients = grad_1[:, None] * basis_1 + grad_2[:, None] * basis_2
    return gradients, face_areas


def vertex_scalar_gradients(vertices: np.ndarray, faces: np.ndarray, scalar_field: np.ndarray) -> np.ndarray:
    face_gradients, face_areas = face_scalar_gradients(vertices, faces, scalar_field)
    vertex_gradients = np.zeros_like(vertices, dtype=float)
    weights = face_areas[:, None]
    np.add.at(vertex_gradients, faces[:, 0], face_gradients * weights)
    np.add.at(vertex_gradients, faces[:, 1], face_gradients * weights)
    np.add.at(vertex_gradients, faces[:, 2], face_gradients * weights)

    vertex_weights = np.zeros(vertices.shape[0], dtype=float)
    np.add.at(vertex_weights, faces[:, 0], face_areas)
    np.add.at(vertex_weights, faces[:, 1], face_areas)
    np.add.at(vertex_weights, faces[:, 2], face_areas)
    vertex_weights = np.maximum(vertex_weights, 1e-12)
    return vertex_gradients / vertex_weights[:, None]


def derive_preferred_axis(vertices: np.ndarray, faces: np.ndarray, sulcal_depth: np.ndarray, fallback_axis: tuple[float, float, float] = (1.0, 0.0, 0.0)) -> np.ndarray:
    normals = compute_vertex_normals(vertices, faces)
    depth_grad = vertex_scalar_gradients(vertices, faces, sulcal_depth)
    bank_direction = depth_grad - np.sum(depth_grad * normals, axis=1, keepdims=True) * normals
    bank_norm = np.linalg.norm(bank_direction, axis=1)

    preferred_axis = np.cross(normals, bank_direction)
    preferred_norm = np.linalg.norm(preferred_axis, axis=1)

    fallback = np.repeat(np.asarray(fallback_axis, dtype=float)[None, :], vertices.shape[0], axis=0)
    fallback = fallback - np.sum(fallback * normals, axis=1, keepdims=True) * normals

    bad = (bank_norm < 1e-10) | (preferred_norm < 1e-10)
    if np.any(bad):
        preferred_axis[bad] = fallback[bad]

    alt_bad = np.linalg.norm(preferred_axis, axis=1) < 1e-10
    if np.any(alt_bad):
        alt = np.repeat(np.array([0.0, 1.0, 0.0], dtype=float)[None, :], int(np.sum(alt_bad)), axis=0)
        alt = alt - np.sum(alt * normals[alt_bad], axis=1, keepdims=True) * normals[alt_bad]
        preferred_axis[alt_bad] = alt
    return _normalize_vectors(preferred_axis)


def derive_vascular_risk(sulcal_depth: np.ndarray, thickness: np.ndarray, depth_weight: float = 0.7) -> np.ndarray:
    sulcal_depth = normalize_unit_interval(sulcal_depth)
    inverse_thickness = normalize_unit_interval(1.0 / np.maximum(np.asarray(thickness, dtype=float).reshape(-1), 1e-3))
    depth_weight = float(np.clip(depth_weight, 0.0, 1.0))
    return np.clip(depth_weight * sulcal_depth + (1.0 - depth_weight) * inverse_thickness, 0.0, 1.0)


def read_surface_geometry(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    surface_path = Path(path)
    suffix = surface_path.suffix.lower()
    if suffix == '.npz':
        vertices, faces, _ = _load_npz_mesh(surface_path)
        return vertices, faces
    if suffix == '.obj':
        return _load_obj_mesh(surface_path)
    if suffix == '.gii' or surface_path.name.endswith('.surf.gii'):
        return _load_gifti_mesh(surface_path)

    try:
        import nibabel as nib
    except ImportError as exc:  # pragma: no cover - optional dependency for local use
        raise ImportError(
            'Reading FreeSurfer surface files requires nibabel. Install the surface extras or use OBJ/NPZ/GIFTI.'
        ) from exc

    try:
        vertices, faces = nib.freesurfer.read_geometry(str(surface_path))
    except Exception as exc:  # pragma: no cover - depends on local files
        raise ValueError(f'Unsupported surface geometry file: {surface_path}') from exc
    return np.asarray(vertices, dtype=float), np.asarray(faces, dtype=int)


def read_surface_scalar(path: str | Path, n_vertices: int) -> np.ndarray:
    field_path = Path(path)
    suffix = field_path.suffix.lower()
    if suffix in {'.npy', '.npz', '.gii'} or field_path.name.endswith('.shape.gii'):
        return _load_scalar_field(field_path, n_vertices)

    try:
        import nibabel as nib
    except ImportError as exc:  # pragma: no cover - optional dependency for local use
        raise ImportError(
            'Reading FreeSurfer scalar files requires nibabel. Install the surface extras or export the field as NPY.'
        ) from exc

    try:
        scalar = nib.freesurfer.read_morph_data(str(field_path))
    except Exception as exc:  # pragma: no cover - depends on local files
        raise ValueError(f'Unsupported scalar field file: {field_path}') from exc
    scalar = np.asarray(scalar, dtype=float).reshape(-1)
    if scalar.shape[0] != n_vertices:
        raise ValueError(f'Field {field_path} has length {scalar.shape[0]}, expected {n_vertices}.')
    return scalar


def prepare_surface_bundle(
    vertices: np.ndarray,
    faces: np.ndarray,
    raw_sulc: np.ndarray,
    thickness: np.ndarray,
    *,
    sulc_sign_mode: SulcSignMode = 'negative-is-deep',
    vascular_risk: np.ndarray | None = None,
    preferred_axis: np.ndarray | None = None,
) -> PreparedSurfaceBundle:
    vertices = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces, dtype=int)
    sulcal_depth = rectify_sulcal_depth(raw_sulc, sign_mode=sulc_sign_mode)
    thickness = np.asarray(thickness, dtype=float).reshape(-1)
    if thickness.shape[0] != vertices.shape[0]:
        raise ValueError(f'Thickness field must have length {vertices.shape[0]}, got {thickness.shape[0]}.')

    if vascular_risk is None:
        vascular_risk = derive_vascular_risk(sulcal_depth, thickness)
    else:
        vascular_risk = normalize_unit_interval(vascular_risk)

    if preferred_axis is None:
        preferred_axis = derive_preferred_axis(vertices, faces, sulcal_depth)
    else:
        preferred_axis = np.asarray(preferred_axis, dtype=float)
        if preferred_axis.ndim == 1 and preferred_axis.size == 3:
            preferred_axis = np.repeat(preferred_axis[None, :], vertices.shape[0], axis=0)
        if preferred_axis.shape != (vertices.shape[0], 3):
            raise ValueError(f'preferred_axis must have shape ({vertices.shape[0]}, 3) or (3,).')
        preferred_axis = _normalize_vectors(preferred_axis)

    return PreparedSurfaceBundle(
        vertices=vertices,
        faces=faces,
        sulcal_depth=sulcal_depth,
        thickness=thickness,
        vascular_risk=np.asarray(vascular_risk, dtype=float).reshape(-1),
        preferred_axis=preferred_axis,
    )


def write_surface_bundle(path: str | Path, bundle: PreparedSurfaceBundle) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        vertices=bundle.vertices,
        faces=bundle.faces,
        sulcal_depth=bundle.sulcal_depth,
        thickness=bundle.thickness,
        vascular_risk=bundle.vascular_risk,
        preferred_axis=bundle.preferred_axis,
    )


def load_bundle_as_mesh(path: str | Path) -> SurfaceMesh:
    bundle_path = Path(path)
    with np.load(bundle_path, allow_pickle=False) as data:
        return SurfaceMesh(
            vertices=np.asarray(data['vertices'], dtype=float),
            faces=np.asarray(data['faces'], dtype=int),
            sulcal_depth=np.asarray(data['sulcal_depth'], dtype=float),
            thickness=np.asarray(data['thickness'], dtype=float),
            vascular_risk=np.asarray(data['vascular_risk'], dtype=float),
            preferred_axis=np.asarray(data['preferred_axis'], dtype=float),
            metadata={'source': str(bundle_path)},
        )
