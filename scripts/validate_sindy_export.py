import os
import glob
import json
import pandas as pd
import numpy as np
from pathlib import Path

def validate():
    export_dir = Path(os.path.dirname(__file__)).parents[0] / 'outputs' / 'sindy_physics_export'
    if not export_dir.exists():
        print("No export dir found!")
        return

    conditions = [d for d in export_dir.iterdir() if d.is_dir()]
    
    inventory_rows = []
    report_lines = ["# SINDy Export Validation Report\n"]
    
    for cond in conditions:
        name = cond.name
        
        meta_path = cond / 'metadata.json'
        geom_path = cond / 'geometry.npz'
        snap_path = cond / 'state_snapshots.npz'
        reg_path = cond / 'regression_samples.csv'
        block_path = cond / 'block_rhs_summary.csv'
        
        has_meta = meta_path.exists()
        has_geom = geom_path.exists()
        has_snap = snap_path.exists()
        has_reg = reg_path.exists()
        has_block = block_path.exists()
        
        n_reg = 0
        n_snap = 0
        alignment_passed = False
        wavefront_present = False
        k_min = k_max = v_min = v_max = phi_min = phi_max = np.nan
        
        if has_snap:
            try:
                snaps = np.load(snap_path)
                if 'time' in snaps:
                    n_snap = len(snaps['time'])
            except:
                pass
                
        missing_cols = []
        if has_reg:
            df = pd.read_csv(reg_path)
            n_reg = len(df)
            cols = set(df.columns)
            
            # check min max
            if 'K_e' in cols: k_min, k_max = df['K_e'].min(), df['K_e'].max()
            if 'V_m' in cols: v_min, v_max = df['V_m'].min(), df['V_m'].max()
            if 'phi' in cols: phi_min, phi_max = df['phi'].min(), df['phi'].max()
            
            if 'wavefront_flag' in cols:
                wavefront_present = df['wavefront_flag'].any()
                
            # Alignment check: reaction + diffusion + electrodiffusion == total
            if {'reaction_rhs_K_e', 'diffusion_rhs_K_e', 'electrodiffusion_rhs_K_e', 'total_rhs_K_e'}.issubset(cols):
                computed_total = df['reaction_rhs_K_e'] + df['diffusion_rhs_K_e'] + df['electrodiffusion_rhs_K_e']
                max_diff = np.max(np.abs(computed_total - df['total_rhs_K_e']))
                alignment_passed = bool(max_diff < 1e-6)
                
            req_cols = [
                'condition', 'time', 'point_id', 'x', 'y', 'K_e', 'V_m', 'theta', 'alpha', 'phi',
                'lap_K_e', 'div_K_grad_phi', 'reaction_rhs_K_e', 'diffusion_rhs_K_e', 'electrodiffusion_rhs_K_e',
                'ecs_rhs_K_e', 'geometry_rhs_K_e', 'total_rhs_K_e', 'dK_e_dt_exact', 
                'wavefront_flag', 'stimulus_region_flag', 'boundary_flag', 'quiescent_flag'
            ]
            for c in req_cols:
                if c not in cols:
                    missing_cols.append(c)
                    
        inventory_rows.append({
            'condition': name,
            'has_metadata': has_meta,
            'has_geometry': has_geom,
            'has_snapshots': has_snap,
            'has_regression': has_reg,
            'has_block_summary': has_block,
            'n_snapshots': n_snap,
            'n_regression_rows': n_reg,
            'alignment_passed': alignment_passed,
            'wavefront_present': wavefront_present,
            'missing_cols': "|".join(missing_cols),
            'k_min': k_min, 'k_max': k_max,
            'v_min': v_min, 'v_max': v_max,
            'phi_min': phi_min, 'phi_max': phi_max
        })
        
        report_lines.append(f"## {name}")
        report_lines.append(f"- **Snapshots**: {n_snap}")
        report_lines.append(f"- **Regression rows**: {n_reg}")
        report_lines.append(f"- **Wavefront samples present**: {'Yes' if wavefront_present else 'No'}")
        report_lines.append(f"- **Time alignment passed**: {'Yes' if alignment_passed else 'No'}")
        report_lines.append(f"- **Missing columns**: {', '.join(missing_cols) if missing_cols else 'None'}")
        report_lines.append(f"- **Ranges**: K_e [{k_min:.2f}, {k_max:.2f}], V_m [{v_min:.2f}, {v_max:.2f}], phi [{phi_min:.2f}, {phi_max:.2f}]\n")
        
    df_inv = pd.DataFrame(inventory_rows)
    df_inv.to_csv(export_dir / 'export_inventory.csv', index=False)
    
    with open(export_dir / 'export_validation_report.md', 'w') as f:
        f.write("\n".join(report_lines))
        
    print("Validation complete.")
    print(df_inv)

if __name__ == '__main__':
    validate()
