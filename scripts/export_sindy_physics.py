import os
import sys
from pathlib import Path
import dataclasses as dc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from csd_sulcus.surface_io import generate_folded_strip_mesh
from csd_sulcus.surface_mechanistic import (
    MechanisticSurfaceParams,
    run_mechanistic_surface_simulation,
)
from csd_sulcus.sindy_export import SindyExporter

def main():
    out_dir = ROOT / 'outputs' / 'sindy_physics_export'
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Generate geometries
    print("Generating meshes...")
    folded_mesh = generate_folded_strip_mesh(
        nx=100,
        ny=40,
        length_mm=40.0,
        width_mm=10.0,
        fold_depth_mm=8.0,
        fold_sigma_mm=1.5,
    )
    # Generate a flat mesh
    flat_mesh = generate_folded_strip_mesh(
        nx=100,
        ny=40,
        length_mm=40.0,
        width_mm=10.0,
        fold_depth_mm=0.0, # flat
        fold_sigma_mm=1.5,
    )
    
    # Stimulus vertex (e.g. at x=2.0 mm)
    def find_stim_vertex(mesh, x_target=2.0):
        import numpy as np
        dists = np.abs(mesh.vertices[:, 0] - x_target)
        return int(np.argmin(dists))
        
    stim_folded = find_stim_vertex(folded_mesh)
    stim_flat = find_stim_vertex(flat_mesh)
    
    base_params = MechanisticSurfaceParams(
        final_t_end=150.0,
        enable_dipole_alignment=False,
        enable_vascular_feedback=False,
    )
    
    conditions = [
        {
            'name': 'flat_no_dipole',
            'mesh': flat_mesh,
            'stim': stim_flat,
            'params': dc.replace(base_params, enable_dipole_alignment=False)
        },
        {
            'name': 'folded_no_dipole',
            'mesh': folded_mesh,
            'stim': stim_folded,
            'params': dc.replace(base_params, enable_dipole_alignment=False)
        },
        {
            'name': 'folded_dipole_aligned',
            'mesh': folded_mesh,
            'stim': stim_folded,
            'params': dc.replace(base_params, enable_dipole_alignment=True, dipole_kernel_mode='aligned')
        },
        {
            'name': 'folded_distance_only_null',
            'mesh': folded_mesh,
            'stim': stim_folded,
            'params': dc.replace(base_params, enable_dipole_alignment=True, dipole_kernel_mode='distance_only')
        },
        {
            'name': 'folded_scrambled_normal_null',
            'mesh': folded_mesh,
            'stim': stim_folded,
            'params': dc.replace(base_params, enable_dipole_alignment=True, dipole_kernel_mode='scrambled_normals')
        }
    ]
    
    for cond in conditions:
        print(f"Running condition: {cond['name']}")
        exporter = SindyExporter(
            condition_name=cond['name'],
            params=cond['params'],
            dt=0.02, # Approx auto dt, exporter just needs it for snapshot timing
            n_vertices=cond['mesh'].n_vertices
        )
        exporter.set_geometry(cond['mesh'])
        
        # Override dt for the simulation to be safe
        cond['params'] = dc.replace(cond['params'], dt=0.02)
        exporter.dt = 0.02
        
        run_mechanistic_surface_simulation(
            cond['mesh'],
            cond['params'],
            stimulus_vertex=cond['stim'],
            sindy_exporter=exporter
        )
        
        exporter.save_to_disk(str(out_dir))
        
if __name__ == '__main__':
    main()
