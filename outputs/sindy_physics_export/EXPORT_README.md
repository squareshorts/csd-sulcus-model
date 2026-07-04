# SINDy Physics Export Provenance

- **Simulator Repo Path**: `C:/work/CSD`
- **Active Branch**: `export/sindy-physics-fields`
- **Latest Commit Hash**: `f2deab1`
- **Export Tag**: `sindy-export-v1`
- **Export Path**: `C:/work/CSD/outputs/sindy_physics_export`

## Exported Conditions
- `flat_no_dipole`
- `folded_no_dipole`
- `folded_dipole_aligned`
- `folded_distance_only_null`
- `folded_scrambled_normal_null`

## Exported Variables
Dense spatiotemporal fields saved in `regression_samples.csv` and geometric vertex coordinates in `geometry.npz`:
- `K_e`, `V_m`, `theta`, `alpha`, `phi`
- Exact spatial derivatives: `lap_K_e`, `div_K_grad_phi`
- Exact grouped RHS physics blocks: `reaction_rhs_K_e`, `diffusion_rhs_K_e`, `electrodiffusion_rhs_K_e`
- Total temporal derivative: `total_rhs_K_e` (representing $dK_e/dt$)

## Exact RHS Alignment Rule
All state variables and RHS/block terms correspond to the exactly same time $t$. The export captures snapshots of the active fluxes mathematically resolved immediately prior to the Forward Euler explicit update inside `run_mechanistic_surface_simulation`.

## Validation Result
The exporter strictly guarantees and has mathematically verified that the sum of the physics blocks matches the total RHS perfectly, avoiding any $O(dt)$ misalignment:
$$ \text{Reaction} + \text{Diffusion} + \text{Electrodiffusion} = \text{Total RHS} $$

## Additional Artifacts
- **Inventory File**: `export_inventory.csv`
- **Validation Report**: `export_validation_report.md`

> [!WARNING]
> Do not commit the large exported array data (`regression_samples.csv`, `geometry.npz`) to Git! These are heavy output artifacts intended strictly for external downstream processing in the `sindy-csd-geometry` repository.
