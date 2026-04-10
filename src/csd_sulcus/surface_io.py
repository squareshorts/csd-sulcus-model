from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import warnings

import numpy as np


@dataclass
class SurfaceMesh:
    vertices: np.ndarray
    faces: np.ndarray
    sulcal_depth: np.ndarray
    thickness: np.ndarray
    vascular_risk: np.ndarray
    preferred_axis: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.vertices = np.asarray(self.vertices, dtype=float)
        self.faces = np.asarray(self.faces, dtype=int)
        self.sulcal_depth = _as_vertex_field(self.sulcal_depth, self.n_vertices, 'sulcal_depth')
        self.thickness = _as_vertex_field(self.thickness, self.n_vertices, 'thickness')
        self.vascular_risk = _as_vertex_field(self.vascular_risk, self.n_vertices, 'vascular_risk')

        if self.vertices.ndim != 2 or self.vertices.shape[1] != 3:
            raise ValueError('Surface vertices must have shape (n_vertices, 3).')
        if self.faces.ndim != 2 or self.faces.shape[1] != 3:
            raise ValueError('Surface faces must have shape (n_faces, 3).')
        if np.any(self.faces < 0) or np.any(self.faces >= self.vertices.shape[0]):
            raise ValueError('Surface faces reference invalid vertex indices.')

        if self.preferred_axis is not None:
            preferred_axis = np.asarray(self.preferred_axis, dtype=float)
            if preferred_axis.ndim == 1 and preferred_axis.size == 3:
                preferred_axis = np.repeat(preferred_axis[None, :], self.n_vertices, axis=0)
            if preferred_axis.shape != (self.n_vertices, 3):
                raise ValueError('preferred_axis must have shape (n_vertices, 3) or (3,).')
            self.preferred_axis = preferred_axis

    @property
    def n_vertices(self) -> int:
        return int(self.vertices.shape[0])

    @property
    def n_faces(self) -> int:
        return int(self.faces.shape[0])


def _as_vertex_field(values: np.ndarray | list[float], n_vertices: int, label: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.shape[0] != n_vertices:
        raise ValueError(f'{label} must have length {n_vertices}, got {arr.shape[0]}.')
    return arr


def _normalize_field(field: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    field = np.asarray(field, dtype=float)
    lo = float(np.nanmin(field))
    hi = float(np.nanmax(field))
    if not np.isfinite(lo) or not np.isfinite(hi) or abs(hi - lo) < eps:
        return np.zeros_like(field, dtype=float)
    return (field - lo) / (hi - lo)


def _load_obj_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    for line in path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        if line.startswith('v '):
            _, xs, ys, zs = line.split()[:4]
            vertices.append([float(xs), float(ys), float(zs)])
            continue
        if line.startswith('f '):
            tokens = line.split()[1:]
            indices = [int(token.split('/')[0]) - 1 for token in tokens]
            if len(indices) < 3:
                continue
            for j in range(1, len(indices) - 1):
                faces.append([indices[0], indices[j], indices[j + 1]])
    if not vertices or not faces:
        raise ValueError(f'OBJ mesh at {path} did not contain vertices and faces.')
    return np.asarray(vertices, dtype=float), np.asarray(faces, dtype=int)


def _load_npz_mesh(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as data:
        if 'vertices' not in data or 'faces' not in data:
            raise ValueError(f"NPZ mesh at {path} must contain 'vertices' and 'faces'.")
        fields = {key: np.asarray(data[key]) for key in data.files if key not in {'vertices', 'faces'}}
        return np.asarray(data['vertices'], dtype=float), np.asarray(data['faces'], dtype=int), fields


def _load_gifti_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    try:
        import nibabel as nib
    except ImportError as exc:  # pragma: no cover - depends on optional local install
        raise ImportError(
            'Reading GIFTI surfaces requires nibabel. Install the optional surface extras '
            'or provide an OBJ/NPZ mesh instead.'
        ) from exc

    image = nib.load(str(path))
    pointset = None
    triangles = None
    for darray in image.darrays:
        intent = getattr(darray, 'intent', '')
        if intent == 'NIFTI_INTENT_POINTSET':
            pointset = np.asarray(darray.data, dtype=float)
        elif intent == 'NIFTI_INTENT_TRIANGLE':
            triangles = np.asarray(darray.data, dtype=int)
    if pointset is None or triangles is None:
        raise ValueError(f'GIFTI file {path} must contain POINTSET and TRIANGLE arrays.')
    return pointset, triangles


def _load_scalar_field(path: Path, n_vertices: int) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == '.npy':
        field = np.load(path)
    elif suffix == '.npz':
        with np.load(path, allow_pickle=False) as data:
            if len(data.files) != 1:
                raise ValueError(f'Field file {path} must contain exactly one array.')
            field = np.asarray(data[data.files[0]])
    elif suffix in {'.gii', '.shape.gii'} or path.name.endswith('.shape.gii'):
        try:
            import nibabel as nib
        except ImportError as exc:  # pragma: no cover - depends on optional local install
            raise ImportError(
                'Reading GIFTI scalar fields requires nibabel. Install the optional surface extras '
                'or export the field as NPY.'
            ) from exc
        image = nib.load(str(path))
        if not image.darrays:
            raise ValueError(f'GIFTI field file {path} does not contain any data arrays.')
        field = np.asarray(image.darrays[0].data)
    else:
        raise ValueError(f'Unsupported field file format for {path}.')
    field = np.asarray(field, dtype=float).reshape(-1)
    if field.shape[0] != n_vertices:
        raise ValueError(f'Field {path} has length {field.shape[0]}, expected {n_vertices}.')
    return field


def _load_vector_field(path: Path, n_vertices: int) -> np.ndarray:
    vector = np.asarray(np.load(path), dtype=float)
    if vector.ndim == 1 and vector.size == 3:
        vector = np.repeat(vector[None, :], n_vertices, axis=0)
    if vector.shape != (n_vertices, 3):
        raise ValueError(f'Vector field {path} must have shape ({n_vertices}, 3) or (3,).')
    return vector


def generate_folded_strip_mesh(
    nx: int = 64,
    ny: int = 28,
    length_mm: float = 22.0,
    width_mm: float = 10.0,
    fold_depth_mm: float = 2.4,
    fold_sigma_mm: float = 1.5,
    thickness_mm: float = 2.6,
) -> SurfaceMesh:
    x = np.linspace(0.0, float(length_mm), int(nx))
    y = np.linspace(-0.5 * float(width_mm), 0.5 * float(width_mm), int(ny))
    xx, yy = np.meshgrid(x, y, indexing='ij')

    zz = -float(fold_depth_mm) * np.exp(-0.5 * (yy / float(fold_sigma_mm)) ** 2)
    vertices = np.column_stack([xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)])

    faces: list[list[int]] = []
    for i in range(int(nx) - 1):
        for j in range(int(ny) - 1):
            v00 = i * int(ny) + j
            v01 = v00 + 1
            v10 = (i + 1) * int(ny) + j
            v11 = v10 + 1
            faces.append([v00, v10, v11])
            faces.append([v00, v11, v01])

    sulcal_depth = _normalize_field(-zz.reshape(-1))
    thickness = np.full(vertices.shape[0], float(thickness_mm), dtype=float) - 0.35 * sulcal_depth
    vascular_risk = np.clip(0.15 + 0.85 * sulcal_depth, 0.0, 1.0)
    preferred_axis = np.tile(np.array([1.0, 0.0, 0.0], dtype=float), (vertices.shape[0], 1))

    metadata = {
        'source': 'synthetic_folded_strip',
        'length_mm': float(length_mm),
        'width_mm': float(width_mm),
        'fold_depth_mm': float(fold_depth_mm),
    }
    return SurfaceMesh(
        vertices=vertices,
        faces=np.asarray(faces, dtype=int),
        sulcal_depth=sulcal_depth,
        thickness=thickness,
        vascular_risk=vascular_risk,
        preferred_axis=preferred_axis,
        metadata=metadata,
    )


def load_surface_mesh(
    mesh_path: str | Path,
    *,
    sulcal_depth_path: str | Path | None = None,
    thickness_path: str | Path | None = None,
    vascular_risk_path: str | Path | None = None,
    preferred_axis_path: str | Path | None = None,
) -> SurfaceMesh:
    path = Path(mesh_path)
    suffix = path.suffix.lower()
    embedded_fields: dict[str, np.ndarray] = {}

    if suffix == '.npz':
        vertices, faces, embedded_fields = _load_npz_mesh(path)
    elif suffix == '.obj':
        vertices, faces = _load_obj_mesh(path)
    elif suffix == '.gii' or path.name.endswith('.surf.gii'):
        vertices, faces = _load_gifti_mesh(path)
    else:
        raise ValueError(f'Unsupported mesh format for {path}. Use NPZ, OBJ, or GIFTI.')

    n_vertices = int(vertices.shape[0])

    sulcal_depth = (
        _load_scalar_field(Path(sulcal_depth_path), n_vertices)
        if sulcal_depth_path is not None
        else embedded_fields.get('sulcal_depth')
    )
    thickness = (
        _load_scalar_field(Path(thickness_path), n_vertices)
        if thickness_path is not None
        else embedded_fields.get('thickness')
    )
    vascular_risk = (
        _load_scalar_field(Path(vascular_risk_path), n_vertices)
        if vascular_risk_path is not None
        else embedded_fields.get('vascular_risk')
    )
    preferred_axis = (
        _load_vector_field(Path(preferred_axis_path), n_vertices)
        if preferred_axis_path is not None
        else embedded_fields.get('preferred_axis')
    )

    if sulcal_depth is None:
        warnings.warn(
            'No sulcal depth field was supplied; using a geometry-only proxy derived from z-position. '
            'Provide a sulcal-depth map from FreeSurfer/HCP for anatomical runs.',
            stacklevel=2,
        )
        sulcal_depth = _normalize_field(-vertices[:, 2])
    else:
        sulcal_depth = _normalize_field(sulcal_depth)

    if thickness is None:
        thickness = np.full(n_vertices, 2.5, dtype=float)
    if vascular_risk is None:
        vascular_risk = np.clip(sulcal_depth.copy(), 0.0, 1.0)

    metadata = {
        'source': str(path),
        'sulcal_depth_proxy': bool(sulcal_depth_path is None and 'sulcal_depth' not in embedded_fields),
    }
    return SurfaceMesh(
        vertices=vertices,
        faces=faces,
        sulcal_depth=sulcal_depth,
        thickness=thickness,
        vascular_risk=vascular_risk,
        preferred_axis=preferred_axis,
        metadata=metadata,
    )
