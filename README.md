# CSD Sulcus Model

This repository contains a reproducible cortical spreading depolarization (CSD) modeling workflow for studying sulcal slowing in folded cortex. The project supports isotropic scalar coupling reductions, orientation-aware tensor transport, a physiology-constrained tensor extension tied to conservative cortical anisotropy priors, and a reduced biophysical potassium-buffer SD model for checking whether the scalar-versus-tensor ordering survives beyond Barkley kinetics.

## Repository Contents

- `scripts/run_supplementary_analyses.py`: supplementary analyses such as eta sensitivity and grid convergence
- `scripts/run_biophysical_grid.py`: manuscript-scale biophysical K-buffer sensitivity study
- `scripts/run_biophysical_validation.py`: microstructure-constrained biophysical validation sweep
- `scripts/run_physiology_anchor.py`: upgraded physiological speed/delay anchoring logic
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

## Key Outputs

- `outputs/extended_full/extended_summary.md`: full manuscript-scale headline results
- `outputs/extended_full/extended_results.csv`: full scalar/tensor sensitivity grid
- `outputs/supplementary/grid_convergence.csv`: grid-convergence study
- `outputs/supplementary/eta_sensitivity.csv`: tensor tangential-attenuation sensitivity
- `outputs/physiology_extension_quick/physiology_summary.md`: quick extension headline results
- `outputs/physiology_extension_quick/physiology_representative.csv`: representative Barkley and potassium-buffer comparisons
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
