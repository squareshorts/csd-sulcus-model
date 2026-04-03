# CSD Sulcus Model

This repository contains a reproducible cortical spreading depolarization (CSD) modeling workflow for studying sulcal slowing in folded cortex. The project now supports both isotropic scalar coupling reductions and orientation-aware tensor transport, together with manuscript-ready figures, supplementary analyses, and release metadata for GitHub and Zenodo archiving.

## Repository Contents

- `src/csd_sulcus/`: reusable model, analysis, and plotting code
- `scripts/run_study.py`: baseline study runner
- `scripts/run_extended_study.py`: manuscript-scale scalar-versus-tensor sensitivity study
- `scripts/run_supplementary_analyses.py`: supplementary analyses such as eta sensitivity and grid convergence
- `tests/test_study.py`: smoke tests for determinism, monotonic slowing, and tensor positivity
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

## Key Outputs

- `outputs/extended_full/extended_summary.md`: full manuscript-scale headline results
- `outputs/extended_full/extended_results.csv`: full scalar/tensor sensitivity grid
- `outputs/supplementary/grid_convergence.csv`: grid-convergence study
- `outputs/supplementary/eta_sensitivity.csv`: tensor tangential-attenuation sensitivity
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
