from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import warnings

import numpy as np
from scipy.sparse import csgraph

from .surface_io import SurfaceMesh
from .surface_ops import build_surface_operators
from .surface_prep import derive_midthickness, prepare_surface_bundle, read_surface_geometry, read_surface_scalar


@dataclass(frozen=True)
class AtlasPatch:
    label: str
    mesh: SurfaceMesh
    global_vertex_indices: np.ndarray
    center_global_idx: int
    stimulus_vertex: int
    electrode_1_vertex: int
    electrode_2_vertex: int


@dataclass(frozen=True)
class AtlasPatchPair:
    atlas_mesh: SurfaceMesh
    sulcal_patch: AtlasPatch
    flat_patch: AtlasPatch
    sulcal_roi_mask: np.ndarray
    flat_roi_mask: np.ndarray
    atlas_source_dir: Path | None


@dataclass(frozen=True)
class AtlasMultiPatchPanel:
    atlas_mesh: SurfaceMesh
    sulcal_patches: list[AtlasPatch]
    flat_patches: list[AtlasPatch]
    sulcal_roi_masks: list[np.ndarray]
    flat_roi_masks: list[np.ndarray]
    atlas_source_dir: Path | None

def load_fsaverage10k_left_mesh(cache_dir: str | Path) -> tuple[SurfaceMesh, Path]:
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                'ignore',
                message=r'.*doesn\'t match a supported version!.*',
                category=Warning,
            )
            from nilearn import datasets
    except ImportError as exc:  # pragma: no cover - optional local dependency
        raise ImportError(
            'Atlas patch extraction requires nilearn to fetch the public fsaverage atlas. '
            'Install the optional surface extras or nilearn explicitly.'
        ) from exc

    fsaverage = datasets.fetch_surf_fsaverage(mesh='fsaverage6', data_dir=str(cache_dir))
    assets = {
        'white': Path(fsaverage.white_left),
        'pial': Path(fsaverage.pial_left),
        'sulc': Path(fsaverage.sulc_left),
    }
    white_vertices, faces = read_surface_geometry(assets['white'])
    pial_vertices, pial_faces = read_surface_geometry(assets['pial'])
    if not np.array_equal(faces, pial_faces):
        raise ValueError('TemplateFlow fsaverage10k white and pial meshes do not share topology.')

    vertices = derive_midthickness(white_vertices, pial_vertices)
    thickness = np.linalg.norm(np.asarray(pial_vertices, dtype=float) - np.asarray(white_vertices, dtype=float), axis=1)
    raw_sulc = read_surface_scalar(assets['sulc'], vertices.shape[0])
    bundle = prepare_surface_bundle(
        vertices,
        faces,
        raw_sulc,
        thickness,
        sulc_sign_mode='negative-is-deep',
    )
    mesh = SurfaceMesh(
        vertices=bundle.vertices,
        faces=bundle.faces,
        sulcal_depth=bundle.sulcal_depth,
        thickness=bundle.thickness,
        vascular_risk=bundle.vascular_risk,
        preferred_axis=bundle.preferred_axis,
        metadata={
            'source': 'nilearn_fsaverage6_left',
            'white_path': str(assets['white']),
            'pial_path': str(assets['pial']),
            'sulc_path': str(assets['sulc']),
        },
    )
    return mesh, Path(assets['white']).parent


def _estimate_patch_axis(coords: np.ndarray) -> np.ndarray:
    centered = np.asarray(coords, dtype=float) - np.mean(coords, axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    return np.asarray(vh[0], dtype=float)


def _reconstruct_path(predecessors: np.ndarray, start_vertex: int, end_vertex: int) -> np.ndarray:
    path = [int(end_vertex)]
    cursor = int(end_vertex)
    while cursor != int(start_vertex):
        cursor = int(predecessors[cursor])
        if cursor < 0:
            return np.asarray([int(start_vertex), int(end_vertex)], dtype=int)
        path.append(cursor)
    path.reverse()
    return np.asarray(path, dtype=int)


def _choose_axis_vertices(mesh: SurfaceMesh) -> tuple[int, int, int]:
    coords = mesh.vertices
    centered = np.asarray(coords, dtype=float) - np.mean(coords, axis=0, keepdims=True)
    axis = _estimate_patch_axis(coords)
    projection = centered @ axis
    proj_min = float(np.min(projection))
    proj_max = float(np.max(projection))
    span = max(proj_max - proj_min, 1e-6)
    stimulus_target = proj_min + 0.10 * span
    stimulus = int(np.argmin(np.abs(projection - stimulus_target)))

    operators = build_surface_operators(mesh, d_parallel=1.0, d_perp=1.0)
    distances, predecessors = csgraph.dijkstra(
        operators.graph,
        directed=False,
        indices=stimulus,
        return_predecessors=True,
    )
    distances = np.asarray(distances, dtype=float)
    forward_candidates = np.where(
        np.isfinite(distances)
        & (np.arange(mesh.n_vertices) != stimulus)
        & (projection > projection[stimulus])
    )[0]
    candidate_pool = forward_candidates if forward_candidates.size > 0 else np.where(np.isfinite(distances))[0]
    candidate_pool = candidate_pool[candidate_pool != stimulus]
    if candidate_pool.size == 0:
        return stimulus, stimulus, stimulus

    electrode_2 = int(candidate_pool[np.argmax(distances[candidate_pool])])
    path = _reconstruct_path(np.asarray(predecessors, dtype=int), stimulus, electrode_2)
    if path.size >= 3:
        electrode_1 = int(path[path.size // 2])
        if electrode_1 in {stimulus, electrode_2}:
            electrode_1 = int(path[max(1, path.size // 2 - 1)])
    else:
        target = proj_min + 0.45 * span
        order = np.argsort(np.abs(projection - target))
        electrode_1 = next(int(idx) for idx in order if int(idx) not in {stimulus, electrode_2})
    return int(stimulus), int(electrode_1), int(electrode_2)


def _extract_submesh(mesh: SurfaceMesh, roi_mask: np.ndarray, label: str, center_global_idx: int) -> AtlasPatch:
    kept = np.where(np.asarray(roi_mask, dtype=bool))[0]
    old_to_new = -np.ones(mesh.n_vertices, dtype=int)
    old_to_new[kept] = np.arange(kept.size, dtype=int)

    face_mask = np.asarray(roi_mask[mesh.faces].all(axis=1), dtype=bool)
    sub_faces_old = mesh.faces[face_mask]
    sub_faces = old_to_new[sub_faces_old]
    preferred_axis = None if mesh.preferred_axis is None else mesh.preferred_axis[kept]
    submesh = SurfaceMesh(
        vertices=mesh.vertices[kept],
        faces=sub_faces,
        sulcal_depth=mesh.sulcal_depth[kept],
        thickness=mesh.thickness[kept],
        vascular_risk=mesh.vascular_risk[kept],
        preferred_axis=preferred_axis,
        metadata={
            **mesh.metadata,
            'patch_label': label,
            'patch_vertices': int(kept.size),
            'center_global_idx': int(center_global_idx),
        },
    )
    stimulus, electrode_1, electrode_2 = _choose_axis_vertices(submesh)
    return AtlasPatch(
        label=label,
        mesh=submesh,
        global_vertex_indices=kept,
        center_global_idx=int(center_global_idx),
        stimulus_vertex=int(stimulus),
        electrode_1_vertex=int(electrode_1),
        electrode_2_vertex=int(electrode_2),
    )


def choose_patch_centers(
    mesh: SurfaceMesh,
    *,
    min_separation_mm: float = 30.0,
    flat_quantile: float = 0.25,
) -> tuple[int, int, np.ndarray]:
    operators = build_surface_operators(mesh, d_parallel=1.0, d_perp=1.0)
    sulcal_center = int(np.argmax(np.asarray(mesh.sulcal_depth, dtype=float)))
    dist_from_sulcus = np.asarray(csgraph.dijkstra(operators.graph, directed=False, indices=sulcal_center), dtype=float)

    depth = np.asarray(mesh.sulcal_depth, dtype=float)
    thickness = np.asarray(mesh.thickness, dtype=float)
    flat_candidates = np.where(
        (dist_from_sulcus >= float(min_separation_mm))
        & (depth <= float(np.quantile(depth, flat_quantile)))
    )[0]
    if flat_candidates.size == 0:
        raise RuntimeError('No sufficiently flat atlas patch candidates found; reduce separation or inspect the sulc map.')

    thickness_scale = max(float(np.nanstd(thickness)), 1e-6)
    score = depth[flat_candidates] + 0.20 * np.abs(thickness[flat_candidates] - thickness[sulcal_center]) / thickness_scale
    flat_center = int(flat_candidates[np.argmin(score)])
    return sulcal_center, flat_center, operators.graph


def choose_multi_patch_centers(
    mesh: SurfaceMesh,
    *,
    n_sulcal: int = 5,
    n_flat: int = 5,
    min_separation_mm: float = 30.0,
    flat_quantile: float = 0.25,
) -> tuple[list[int], list[int], np.ndarray]:
    operators = build_surface_operators(mesh, d_parallel=1.0, d_perp=1.0)
    graph = operators.graph
    depth = np.asarray(mesh.sulcal_depth, dtype=float)
    thickness = np.asarray(mesh.thickness, dtype=float)

    sulcal_centers = []
    available_mask = np.ones(mesh.n_vertices, dtype=bool)

    for _ in range(n_sulcal):
        masked_depth = np.where(available_mask, depth, -np.inf)
        if not np.any(available_mask) or np.all(masked_depth == -np.inf):
            break
        center = int(np.argmax(masked_depth))
        sulcal_centers.append(center)

        distances = np.asarray(csgraph.dijkstra(graph, directed=False, indices=center), dtype=float)
        available_mask &= (distances >= min_separation_mm)

    flat_centers = []
    flat_threshold = float(np.quantile(depth, flat_quantile))
    thickness_scale = max(float(np.nanstd(thickness)), 1e-6)

    available_mask = np.ones(mesh.n_vertices, dtype=bool)
    for center in sulcal_centers:
        dist = np.asarray(csgraph.dijkstra(graph, directed=False, indices=center), dtype=float)
        available_mask &= (dist >= min_separation_mm)

    for _ in range(n_flat):
        valid = available_mask & (depth <= flat_threshold)
        if not np.any(valid):
            break

        score = np.where(valid, depth + 0.20 * np.abs(thickness - float(np.mean(thickness))) / thickness_scale, np.inf)
        center = int(np.argmin(score))
        flat_centers.append(center)

        dist = np.asarray(csgraph.dijkstra(graph, directed=False, indices=center), dtype=float)
        available_mask &= (dist >= min_separation_mm)

    return sulcal_centers, flat_centers, graph


def grow_patch(graph, center_idx: int, radius_mm: float) -> tuple[np.ndarray, np.ndarray]:
    distances = np.asarray(csgraph.dijkstra(graph, directed=False, indices=int(center_idx)), dtype=float)
    return np.asarray(distances <= float(radius_mm), dtype=bool), distances


def extract_patch_pair_from_mesh(
    atlas_mesh: SurfaceMesh,
    *,
    patch_radius_mm: float = 12.0,
    min_separation_mm: float = 30.0,
    atlas_source_dir: str | Path | None = None,
) -> AtlasPatchPair:
    sulcal_center, flat_center, graph = choose_patch_centers(atlas_mesh, min_separation_mm=min_separation_mm)
    sulcal_roi_mask, _ = grow_patch(graph, sulcal_center, patch_radius_mm)
    flat_roi_mask, _ = grow_patch(graph, flat_center, patch_radius_mm)
    sulcal_patch = _extract_submesh(atlas_mesh, sulcal_roi_mask, 'atlas_sulcal_patch', sulcal_center)
    flat_patch = _extract_submesh(atlas_mesh, flat_roi_mask, 'atlas_flat_patch', flat_center)
    return AtlasPatchPair(
        atlas_mesh=atlas_mesh,
        sulcal_patch=sulcal_patch,
        flat_patch=flat_patch,
        sulcal_roi_mask=sulcal_roi_mask,
        flat_roi_mask=flat_roi_mask,
        atlas_source_dir=None if atlas_source_dir is None else Path(atlas_source_dir),
    )


def extract_multi_patch_panel_from_mesh(
    atlas_mesh: SurfaceMesh,
    *,
    n_sulcal: int = 5,
    n_flat: int = 5,
    patch_radius_mm: float = 12.0,
    min_separation_mm: float = 30.0,
    atlas_source_dir: str | Path | None = None,
) -> AtlasMultiPatchPanel:
    sulcal_centers, flat_centers, graph = choose_multi_patch_centers(
        atlas_mesh, n_sulcal=n_sulcal, n_flat=n_flat, min_separation_mm=min_separation_mm
    )

    sulcal_patches = []
    sulcal_roi_masks = []
    for i, center in enumerate(sulcal_centers):
        roi_mask, _ = grow_patch(graph, center, patch_radius_mm)
        patch = _extract_submesh(atlas_mesh, roi_mask, f'atlas_sulcal_patch_{i}', center)
        sulcal_patches.append(patch)
        sulcal_roi_masks.append(roi_mask)

    flat_patches = []
    flat_roi_masks = []
    for i, center in enumerate(flat_centers):
        roi_mask, _ = grow_patch(graph, center, patch_radius_mm)
        patch = _extract_submesh(atlas_mesh, roi_mask, f'atlas_flat_patch_{i}', center)
        flat_patches.append(patch)
        flat_roi_masks.append(roi_mask)

    return AtlasMultiPatchPanel(
        atlas_mesh=atlas_mesh,
        sulcal_patches=sulcal_patches,
        flat_patches=flat_patches,
        sulcal_roi_masks=sulcal_roi_masks,
        flat_roi_masks=flat_roi_masks,
        atlas_source_dir=None if atlas_source_dir is None else Path(atlas_source_dir),
    )


def prepare_atlas_patch_pair(
    cache_dir: str | Path,
    *,
    patch_radius_mm: float = 12.0,
    min_separation_mm: float = 30.0,
) -> AtlasPatchPair:
    atlas_mesh, atlas_source_dir = load_fsaverage10k_left_mesh(cache_dir)
    return extract_patch_pair_from_mesh(
        atlas_mesh,
        patch_radius_mm=patch_radius_mm,
        min_separation_mm=min_separation_mm,
        atlas_source_dir=atlas_source_dir,
    )


def prepare_atlas_multi_patch_panel(
    cache_dir: str | Path,
    *,
    n_sulcal: int = 5,
    n_flat: int = 5,
    patch_radius_mm: float = 12.0,
    min_separation_mm: float = 30.0,
) -> AtlasMultiPatchPanel:
    atlas_mesh, atlas_source_dir = load_fsaverage10k_left_mesh(cache_dir)
    return extract_multi_patch_panel_from_mesh(
        atlas_mesh,
        n_sulcal=n_sulcal,
        n_flat=n_flat,
        patch_radius_mm=patch_radius_mm,
        min_separation_mm=min_separation_mm,
        atlas_source_dir=atlas_source_dir,
    )
