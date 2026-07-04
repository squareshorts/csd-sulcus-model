import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import asdict

class SindyExporter:
    def __init__(self, condition_name: str, params, dt: float, n_vertices: int):
        self.condition_name = condition_name
        self.params = params
        self.dt = dt
        self.n_vertices = n_vertices
        
        self.regression_rows = []
        self.block_summary_rows = []
        self.snapshots = {
            'time': [],
            'K_e': [],
            'V_m': [],
            'theta': [],
            'alpha': [],
            'phi': [],
            'Na_e': [],
            'Cl_e': [],
            'lambda_tortuosity': []
        }
        self.geometry_data = {}
        
    def set_geometry(self, mesh):
        self.geometry_data = {
            'vertices': mesh.vertices.copy(),
            'faces': mesh.faces.copy(),
        }
        if hasattr(mesh, 'normals'):
            self.geometry_data['normals'] = mesh.normals.copy()
        if hasattr(mesh, 'fold_depth'):
            self.geometry_data['fold_depth'] = mesh.fold_depth.copy()
        if hasattr(mesh, 'curvature'):
            self.geometry_data['curvature'] = mesh.curvature.copy()

    def capture_step(self, step: int, time_s: float, state: dict, rhs: dict, mesh):
        # 1. Update block summary
        self.block_summary_rows.append({
            'time': time_s,
            'mean_abs_reaction_rhs': float(np.mean(np.abs(rhs['reaction_rhs_K_e']))),
            'mean_abs_diffusion_rhs': float(np.mean(np.abs(rhs['diffusion_rhs_K_e']))),
            'mean_abs_electrodiffusion_rhs': float(np.mean(np.abs(rhs['electrodiffusion_rhs_K_e']))),
            'mean_abs_ecs_rhs': 0.0, # not separable cleanly in this form
            'mean_abs_geometry_rhs': 0.0, # inherently part of lap and phi
            'mean_abs_total_rhs': float(np.mean(np.abs(rhs['total_rhs_K_e']))),
            'number_wavefront_points': int(np.sum(state['wavefront_mask']))
        })
        
        # 2. Save snapshots (only save ~100 snapshots, so every N steps)
        # We don't know total steps easily, assume saving every 1.0s or so. 
        # CSD is usually ~100-200s, so every 1.0s is 100-200 snapshots.
        if abs(time_s - round(time_s)) < self.dt * 0.6:  # approximately integer seconds
            self.snapshots['time'].append(time_s)
            self.snapshots['K_e'].append(state['K_e'].copy())
            self.snapshots['V_m'].append(state['V_m'].copy())
            self.snapshots['theta'].append(state['theta'].copy())
            self.snapshots['alpha'].append(state['alpha'].copy())
            self.snapshots['phi'].append(state['phi'].copy())
            self.snapshots['Na_e'].append(state['Na_e'].copy())
            self.snapshots['Cl_e'].append(state['Cl_e'].copy())
            self.snapshots['lambda_tortuosity'].append(state['lambda_tortuosity'].copy())

        # 3. Stratified sampling for regression
        # Subsample to keep memory and file size reasonable.
        # Say, all wavefront points, and 2% of other points.
        is_wf = state['wavefront_mask']
        is_stim = state['stimulus_mask']
        
        # Randomly select ~2% of non-wavefront points
        rand_mask = np.random.rand(self.n_vertices) < 0.02
        
        # Combine
        sample_mask = is_wf | rand_mask
        sample_indices = np.where(sample_mask)[0]
        
        x = mesh.vertices[:, 0]
        y = mesh.vertices[:, 1]
        z = mesh.vertices[:, 2]

        for i in sample_indices:
            self.regression_rows.append({
                'condition': self.condition_name,
                'time': time_s,
                'point_id': i,
                'x': x[i],
                'y': y[i],
                'z': z[i],
                'K_e': state['K_e'][i],
                'V_m': state['V_m'][i],
                'theta': state['theta'][i],
                'alpha': state['alpha'][i],
                'phi': state['phi'][i],
                'Na_e': state['Na_e'][i],
                'Cl_e': state['Cl_e'][i],
                'lambda': state['lambda_tortuosity'][i],
                'lap_K_e': rhs['lap_K_e'][i],
                'div_K_grad_phi': rhs['div_K_grad_phi'][i],
                'reaction_rhs_K_e': rhs['reaction_rhs_K_e'][i],
                'diffusion_rhs_K_e': rhs['diffusion_rhs_K_e'][i],
                'electrodiffusion_rhs_K_e': rhs['electrodiffusion_rhs_K_e'][i],
                'ecs_rhs_K_e': 0.0,
                'geometry_rhs_K_e': 0.0,
                'total_rhs_K_e': rhs['total_rhs_K_e'][i],
                'dK_e_dt_exact': rhs['total_rhs_K_e'][i],
                'wavefront_flag': bool(is_wf[i]),
                'stimulus_region_flag': bool(is_stim[i]),
                'boundary_flag': False, # Not computed trivially here
                'quiescent_flag': not bool(is_wf[i])
            })
            
    def save_to_disk(self, out_dir: str):
        path = Path(out_dir) / self.condition_name
        path.mkdir(parents=True, exist_ok=True)
        
        # 1. metadata.json
        meta = {
            'condition_name': self.condition_name,
            'simulator_version': 'export-hook',
            'dt': self.dt,
            'duration': self.block_summary_rows[-1]['time'] if self.block_summary_rows else 0,
            'number_of_time_steps': len(self.block_summary_rows),
            'number_of_spatial_points': self.n_vertices,
            'mesh_type': 'SurfaceMesh',
            'parameter_values': asdict(self.params),
            'dipole_alignment_enabled': self.params.enable_dipole_alignment,
            'dipole_kernel_mode': self.params.dipole_kernel_mode,
            'units': 'mm, min, mV, mM'
        }
        with open(path / 'metadata.json', 'w') as f:
            json.dump(meta, f, indent=2)
            
        # 2. geometry.npz
        if self.geometry_data:
            np.savez_compressed(path / 'geometry.npz', **self.geometry_data)
            
        # 3. state_snapshots.npz
        snap_arrays = {k: np.array(v) for k, v in self.snapshots.items()}
        np.savez_compressed(path / 'state_snapshots.npz', **snap_arrays)
        
        # 4. regression_samples.csv
        df_reg = pd.DataFrame(self.regression_rows)
        # subsample if > 200_000 rows to keep size manageable
        if len(df_reg) > 200_000:
            df_reg = df_reg.sample(n=200_000, random_state=42)
        df_reg.to_csv(path / 'regression_samples.csv', index=False)
        
        # 5. block_rhs_summary.csv
        df_block = pd.DataFrame(self.block_summary_rows)
        df_block.to_csv(path / 'block_rhs_summary.csv', index=False)
        
        # 6. virtual_electrode_traces.csv (empty mockup for now since mechanistic script does it externally)
        pd.DataFrame().to_csv(path / 'virtual_electrode_traces.csv', index=False)
        
        print(f"Exported {self.condition_name} to {path} (Regression rows: {len(df_reg)})")
