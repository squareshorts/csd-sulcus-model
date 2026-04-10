from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from csd_sulcus.surface_prep import (
    prepare_surface_bundle,
    read_surface_geometry,
    read_surface_scalar,
    derive_midthickness,
    write_surface_bundle,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Prepare a ready-to-run cortical surface NPZ bundle from FreeSurfer or HCP-style inputs.'
    )
    geometry = parser.add_mutually_exclusive_group(required=True)
    geometry.add_argument('--mesh', type=Path, default=None, help='Direct midthickness mesh (OBJ, NPZ, GIFTI, or FreeSurfer geometry).')
    geometry.add_argument('--white', type=Path, default=None, help='White-matter surface for deriving midthickness.')
    parser.add_argument('--pial', type=Path, default=None, help='Pial surface for deriving midthickness when --white is used.')
    parser.add_argument('--sulc', type=Path, required=True, help='Sulcal-depth / sulc scalar field (GIFTI, NPY/NPZ, or FreeSurfer morph).')
    parser.add_argument('--thickness', type=Path, required=True, help='Thickness scalar field (GIFTI, NPY/NPZ, or FreeSurfer morph).')
    parser.add_argument('--vascular-risk', type=Path, default=None, help='Optional vascular-risk scalar field. If omitted it is derived from depth and thickness.')
    parser.add_argument('--preferred-axis', type=Path, default=None, help='Optional preferred tangential axis (.npy with shape (n_vertices,3) or (3,)).')
    parser.add_argument('--sulc-sign', choices=['negative-is-deep', 'positive-is-deep', 'absolute'], default='negative-is-deep')
    parser.add_argument('--output', type=Path, required=True, help='Output NPZ bundle path.')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.white is not None and args.pial is None:
        raise SystemExit('--pial is required when --white is provided.')

    if args.mesh is not None:
        vertices, faces = read_surface_geometry(args.mesh)
        mesh_source = str(args.mesh)
    else:
        white_vertices, faces = read_surface_geometry(args.white)
        pial_vertices, pial_faces = read_surface_geometry(args.pial)
        if pial_faces.shape != faces.shape or not (pial_faces == faces).all():
            raise SystemExit('White and pial surfaces must share the same face topology to derive midthickness.')
        vertices = derive_midthickness(white_vertices, pial_vertices)
        mesh_source = f'midthickness({args.white.name},{args.pial.name})'

    n_vertices = int(vertices.shape[0])
    raw_sulc = read_surface_scalar(args.sulc, n_vertices)
    thickness = read_surface_scalar(args.thickness, n_vertices)
    vascular_risk = read_surface_scalar(args.vascular_risk, n_vertices) if args.vascular_risk is not None else None

    preferred_axis = None
    if args.preferred_axis is not None:
        import numpy as np
        preferred_axis = np.asarray(np.load(args.preferred_axis), dtype=float)

    bundle = prepare_surface_bundle(
        vertices,
        faces,
        raw_sulc,
        thickness,
        sulc_sign_mode=args.sulc_sign,
        vascular_risk=vascular_risk,
        preferred_axis=preferred_axis,
    )
    write_surface_bundle(args.output, bundle)

    print(f'Prepared surface bundle: {args.output}')
    print(f'  Mesh source: {mesh_source}')
    print(f'  Vertices: {bundle.vertices.shape[0]}')
    print(f'  Faces: {bundle.faces.shape[0]}')
    print(f'  Sulcal depth range: {bundle.sulcal_depth.min():.3f} to {bundle.sulcal_depth.max():.3f}')
    print(f'  Thickness range: {bundle.thickness.min():.3f} to {bundle.thickness.max():.3f}')
    print(f'  Vascular risk range: {bundle.vascular_risk.min():.3f} to {bundle.vascular_risk.max():.3f}')


if __name__ == '__main__':
    main()
