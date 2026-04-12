# CSD Sulcus Model

This repository contains a reproducible cortical spreading depolarization (CSD) modeling workflow for studying sulcal slowing in folded cortex. The project supports isotropic scalar coupling reductions, orientation-aware tensor transport, a surface-based reduced biophysical SD model in which extracellular potassium diffusion is the central transport process and a quasi-static dipole field contributes an electrodiffusive correction, and a stronger mechanistic surface electrodiffusion model with explicit extracellular and intracellular K/Na/Cl states, membrane leak and activated currents, Na/K pump dynamics, dynamic extracellular-space swelling, and an optional vascular extension. The repository also includes a physiology-constrained tensor extension tied to conservative cortical anisotropy priors for checking whether the scalar-versus-tensor ordering survives beyond Barkley kinetics.

## Repository Contents

- `scripts/run_supplementary_analyses.py`: supplementary analyses such as eta sensitivity and grid convergence
- `scripts/run_biophysical_grid.py`: manuscript-scale biophysical K-buffer sensitivity study
- `scripts/run_biophysical_validation.py`: microstructure-constrained biophysical validation sweep
- `scripts/run_physiology_anchor.py`: upgraded physiological speed/delay anchoring logic
- `scripts/run_surface_representative.py`: cortical-surface representative scaffold comparing diffusion-only, anisotropic diffusion, dipole electrodiffusion, and vascularly coupled electrodiffusion families
- `scripts/run_surface_mechanistic_study.py`: stronger mechanistic surface study comparing the same multi-ion electrodiffusion model with and without sulcal dipole alignment, plus a folded-geometry statistics sweep, an atlas-based sulcal-versus-flatter patch sanity check, and a propagation figure
- `scripts/prepare_surface_bundle.py`: prepares a ready-to-run cortical-surface NPZ bundle from FreeSurfer or HCP-style files
- `src/csd_sulcus/`: reusable model, analysis, and plotting code
- `tests/test_study.py`: smoke tests for determinism, monotonic slowing, tensor positivity, and physiology-extension ordering
- `manuscript/reframed_submission.tex`: current submission manuscript
- `manuscript/figures/`: manuscript-local figure assets for Overleaf or journal upload
- `outputs/`: generated numerical summaries, CSV tables, and figure exports

## Installation

The project requires Python 3.10 or newer.

```bash
python -m pip install -e .
```

For test dependencies:

```bash
python -m pip install -e .[dev]
```

For cortical-surface GIFTI support and the atlas patch sanity check:

```bash
python -m pip install -e .[surface]
```

## Main Workflows

Quick baseline run:

```bash
python scripts/run_study.py --quick
```

Manuscript-scale baseline run:

```bash
python scripts/run_study.py
```

Quick extended scalar-versus-tensor study:

```bash
python scripts/run_extended_study.py --quick
```

Full manuscript-scale extended study:

```bash
python scripts/run_extended_study.py --output-root outputs/extended_full
```

Supplementary analyses:

```bash
python scripts/run_supplementary_analyses.py
```

Quick biophysical validation sweep:
```bash
python scripts/run_biophysical_validation.py --quick
```

Full microstructure-constrained biophysical validation:
```bash
python scripts/run_biophysical_validation.py
```

Full biophysical grid sweep:
```bash
python scripts/run_biophysical_grid.py
```

Upgraded physiological anchoring:
```bash
python scripts/run_physiology_anchor.py
```

Quick surface representative scaffold:
```bash
python scripts/run_surface_representative.py --quick
```

Prepare a surface bundle from FreeSurfer or HCP-style inputs:
```bash
python scripts/prepare_surface_bundle.py --mesh path\to\midthickness.surf.gii --sulc path\to\sulc.shape.gii --thickness path\to\thickness.shape.gii --output data\lh_surface_bundle.npz
```

If you only have white and pial surfaces:
```bash
python scripts/prepare_surface_bundle.py --white path\to\lh.white --pial path\to\lh.pial --sulc path\to\lh.sulc --thickness path\to\lh.thickness --output data\lh_surface_bundle.npz
```

Run the surface representative model on the prepared bundle:
```bash
python scripts/run_surface_representative.py --mesh data\lh_surface_bundle.npz
```

Run the stronger mechanistic surface study:
```bash
python scripts/run_surface_mechanistic_study.py
```

## Preparing Sulcal Data

The surface solver needs three per-vertex anatomical inputs and one mesh topology:

- A midthickness cortical mesh. HCP-style `*.midthickness.surf.gii` works directly. If you only have FreeSurfer `white` and `pial`, the prep script averages them into a midthickness surface.
- A sulcal-depth or `sulc` map. FreeSurfer and HCP-style `sulc` files are supported; by default the prep script assumes the FreeSurfer convention `negative-is-deep`.
- A cortical thickness map. FreeSurfer `thickness` or HCP-style `*.thickness.shape.gii` works.
- An optional vascular-risk field. If you do not provide one, the script derives it from normalized sulcal depth and inverse thickness.

The prep script also derives a tangential preferred-axis field from the sulcal-depth gradient so the anisotropic transport term has an anatomically grounded surface direction field. In the surface solver, sulcal depth and thickness also parameterize extracellular-space volume fraction and tortuosity, so the diffusion operator reflects reduced ECS transport in deeper tissue before the quasi-static dipole field is applied as an electrodiffusive correction.

## Key Outputs

- `outputs/extended_full/extended_summary.md`: full manuscript-scale headline results
- `outputs/extended_full/extended_results.csv`: full scalar/tensor sensitivity grid
- `outputs/supplementary/grid_convergence.csv`: grid-convergence study
- `outputs/supplementary/eta_sensitivity.csv`: tensor tangential-attenuation sensitivity
- `outputs/physiology_extension_quick/physiology_summary.md`: quick extension headline results
- `outputs/physiology_extension_quick/physiology_representative.csv`: representative Barkley and potassium-buffer comparisons
- `outputs/surface_representative*/surface_representative_summary.csv`: representative reduced surface-model summary table
- `outputs/surface_mechanistic_study/mechanistic_study_summary.json`: stronger mechanistic surface-study summary and paired statistics
- `outputs/surface_mechanistic_study/mechanistic_geometry_sweep.csv`: folded-geometry sweep for the mechanistic surface model
- `outputs/surface_mechanistic_study/mechanistic_atlas_patch_check.csv`: atlas-based sulcal-versus-flatter patch sanity-check table
- `outputs/surface_mechanistic_study/mechanistic_wave_propagation.png`: representative propagation snapshots and cross-sulcal kymographs
- `data/*_surface_bundle.npz`: prepared cortical-surface bundle containing vertices, faces, sulcal depth, thickness, vascular risk, and preferred-axis fields
- `manuscript/reframed_submission.tex`: current submission draft with local figure paths

## Testing

```bash
python -m pytest
```

The tests are reduced-domain smoke checks intended to validate code behavior quickly rather than reproduce manuscript-scale outputs.

## Citation And Archiving

- `CITATION.cff` provides GitHub's citation metadata.
- `.zenodo.json` provides Zenodo-specific software metadata for GitHub release archiving.
- `RELEASE.md` contains the practical checklist for the first public push and the first Zenodo-backed GitHub release.

## Release Note

This repository is prepared for public release under the MIT License and includes the metadata needed for GitHub citation support and Zenodo release archiving.
