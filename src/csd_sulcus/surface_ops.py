from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from .surface_io import SurfaceMesh


@dataclass(frozen=True)
class SurfaceOperators:
    stiffness: sparse.csr_matrix
    lumped_mass: np.ndarray
    inv_lumped_mass: np.ndarray
    graph: sparse.csr_matrix
    vertex_normals: np.ndarray
    tangent_directions: np.ndarray
    bank_directions: np.ndarray
    edge_i: np.ndarray
    edge_j: np.ndarray
    edge_lengths: np.ndarray
    base_edge_weights: np.ndarray
    edge_alignment_sq: np.ndarray


def _normalize_vectors(vectors: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return vectors / norms


def face_areas_and_normals(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    double_area = np.linalg.norm(cross, axis=1)
    normals = np.zeros_like(cross)
    valid = double_area > 0.0
    normals[valid] = cross[valid] / double_area[valid, None]
    return 0.5 * double_area, normals


def compute_vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    _, face_normals = face_areas_and_normals(vertices, faces)
    accum = np.zeros_like(vertices, dtype=float)
    np.add.at(accum, faces[:, 0], face_normals)
    np.add.at(accum, faces[:, 1], face_normals)
    np.add.at(accum, faces[:, 2], face_normals)
    return _normalize_vectors(accum)


def compute_lumped_mass(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    face_areas, _ = face_areas_and_normals(vertices, faces)
    mass = np.zeros(vertices.shape[0], dtype=float)
    face_contrib = face_areas / 3.0
    np.add.at(mass, faces[:, 0], face_contrib)
    np.add.at(mass, faces[:, 1], face_contrib)
    np.add.at(mass, faces[:, 2], face_contrib)
    return np.maximum(mass, 1e-12)


def build_tangent_frames(
    mesh: SurfaceMesh,
    vertex_normals: np.ndarray,
    fallback_axis: tuple[float, float, float] = (1.0, 0.0, 0.0),
) -> tuple[np.ndarray, np.ndarray]:
    if mesh.preferred_axis is not None:
        axis_field = np.asarray(mesh.preferred_axis, dtype=float)
    else:
        axis_field = np.repeat(np.asarray(fallback_axis, dtype=float)[None, :], mesh.n_vertices, axis=0)

    tangent = axis_field - np.sum(axis_field * vertex_normals, axis=1, keepdims=True) * vertex_normals
    tangent_norm = np.linalg.norm(tangent, axis=1)
    bad = tangent_norm < 1e-10
    if np.any(bad):
        alt_axis = np.repeat(np.array([0.0, 1.0, 0.0], dtype=float)[None, :], int(np.sum(bad)), axis=0)
        alt_tangent = np.cross(vertex_normals[bad], alt_axis)
        alt_norm = np.linalg.norm(alt_tangent, axis=1)
        alt_bad = alt_norm < 1e-10
        if np.any(alt_bad):
            alt_tangent[alt_bad] = np.cross(
                vertex_normals[bad][alt_bad],
                np.repeat(np.array([0.0, 0.0, 1.0], dtype=float)[None, :], int(np.sum(alt_bad)), axis=0),
            )
        tangent[bad] = alt_tangent
    tangent = _normalize_vectors(tangent)
    bank = _normalize_vectors(np.cross(vertex_normals, tangent))
    return tangent, bank


def cotangent_edge_data(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    e01 = v1 - v0
    e02 = v2 - v0
    e10 = v0 - v1
    e12 = v2 - v1
    e20 = v0 - v2
    e21 = v1 - v2

    cross0 = np.linalg.norm(np.cross(e01, e02), axis=1)
    cross1 = np.linalg.norm(np.cross(e12, e10), axis=1)
    cross2 = np.linalg.norm(np.cross(e20, e21), axis=1)
    cross0 = np.maximum(cross0, 1e-12)
    cross1 = np.maximum(cross1, 1e-12)
    cross2 = np.maximum(cross2, 1e-12)

    cot0 = np.sum(e01 * e02, axis=1) / cross0
    cot1 = np.sum(e12 * e10, axis=1) / cross1
    cot2 = np.sum(e20 * e21, axis=1) / cross2

    i = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]])
    j = np.concatenate([faces[:, 2], faces[:, 0], faces[:, 1]])
    weights = 0.5 * np.concatenate([cot0, cot1, cot2])

    upper_i = np.minimum(i, j)
    upper_j = np.maximum(i, j)
    matrix = sparse.coo_matrix((weights, (upper_i, upper_j)), shape=(vertices.shape[0], vertices.shape[0])).tocsr()
    matrix.sum_duplicates()
    coo = matrix.tocoo()
    edge_i = np.asarray(coo.row, dtype=int)
    edge_j = np.asarray(coo.col, dtype=int)
    edge_weights = np.asarray(coo.data, dtype=float)
    edge_lengths = np.linalg.norm(vertices[edge_j] - vertices[edge_i], axis=1)
    return edge_i, edge_j, edge_weights, edge_lengths


def build_weighted_stiffness(
    n_vertices: int,
    edge_i: np.ndarray,
    edge_j: np.ndarray,
    edge_weights: np.ndarray,
) -> sparse.csr_matrix:
    diagonal = np.zeros(n_vertices, dtype=float)
    np.add.at(diagonal, edge_i, edge_weights)
    np.add.at(diagonal, edge_j, edge_weights)

    rows = np.concatenate([edge_i, edge_j, np.arange(n_vertices)])
    cols = np.concatenate([edge_j, edge_i, np.arange(n_vertices)])
    data = np.concatenate([-edge_weights, -edge_weights, diagonal])
    matrix = sparse.coo_matrix((data, (rows, cols)), shape=(n_vertices, n_vertices)).tocsr()
    matrix.sum_duplicates()
    return matrix


def estimate_explicit_dt(lumped_mass: np.ndarray, stiffness: sparse.csr_matrix, safety: float = 0.15) -> float:
    diagonal = stiffness.diagonal()
    valid = diagonal > 1e-12
    if not np.any(valid):
        return 0.01
    return float(safety * np.min(lumped_mass[valid] / diagonal[valid]))


def build_surface_operators(
    mesh: SurfaceMesh,
    d_parallel: float | np.ndarray,
    d_perp: float | np.ndarray,
    fallback_axis: tuple[float, float, float] = (1.0, 0.0, 0.0),
) -> SurfaceOperators:
    n_vertices = mesh.n_vertices
    d_parallel_vertex = np.broadcast_to(np.asarray(d_parallel, dtype=float), (n_vertices,)).astype(float)
    d_perp_vertex = np.broadcast_to(np.asarray(d_perp, dtype=float), (n_vertices,)).astype(float)

    vertex_normals = compute_vertex_normals(mesh.vertices, mesh.faces)
    tangent, bank = build_tangent_frames(mesh, vertex_normals, fallback_axis=fallback_axis)
    lumped_mass = compute_lumped_mass(mesh.vertices, mesh.faces)
    inv_lumped_mass = 1.0 / lumped_mass

    edge_i, edge_j, base_edge_weights, edge_lengths = cotangent_edge_data(mesh.vertices, mesh.faces)
    edge_vectors = mesh.vertices[edge_j] - mesh.vertices[edge_i]
    edge_unit = _normalize_vectors(edge_vectors)

    edge_tangent = tangent[edge_i] + tangent[edge_j]
    edge_tangent_norm = np.linalg.norm(edge_tangent, axis=1)
    bad = edge_tangent_norm < 1e-10
    if np.any(bad):
        edge_tangent[bad] = tangent[edge_i[bad]]
    edge_tangent = _normalize_vectors(edge_tangent)

    align_sq = np.sum(edge_unit * edge_tangent, axis=1) ** 2
    d_parallel_edge = 0.5 * (d_parallel_vertex[edge_i] + d_parallel_vertex[edge_j])
    d_perp_edge = 0.5 * (d_perp_vertex[edge_i] + d_perp_vertex[edge_j])
    edge_conductivity = d_perp_edge + (d_parallel_edge - d_perp_edge) * align_sq

    weighted_edge = base_edge_weights * edge_conductivity
    stiffness = build_weighted_stiffness(n_vertices, edge_i, edge_j, weighted_edge)
    graph_rows = np.concatenate([edge_i, edge_j])
    graph_cols = np.concatenate([edge_j, edge_i])
    graph_data = np.concatenate([edge_lengths, edge_lengths])
    graph = sparse.coo_matrix((graph_data, (graph_rows, graph_cols)), shape=(n_vertices, n_vertices)).tocsr()
    graph.sum_duplicates()

    return SurfaceOperators(
        stiffness=stiffness,
        lumped_mass=lumped_mass,
        inv_lumped_mass=inv_lumped_mass,
        graph=graph,
        vertex_normals=vertex_normals,
        tangent_directions=tangent,
        bank_directions=bank,
        edge_i=edge_i,
        edge_j=edge_j,
        edge_lengths=edge_lengths,
        base_edge_weights=base_edge_weights,
        edge_alignment_sq=align_sq,
    )

