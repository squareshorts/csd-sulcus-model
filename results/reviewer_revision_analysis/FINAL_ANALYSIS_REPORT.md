# Final computational revision analysis report

## 1. Executive factual summary

The authoritative current representative run was reproducibly identified and rerun. Cross-fold speed was 2.609169983155 mm/min without the dipole kernel and 2.536871858740 mm/min with aligned dipole coupling (slowdown 0.072298124415 mm/min; 2.770924% relative to no dipole). Center-vertex E1-E2 delays were 127.973060795749 and 131.420987853502 s; the increase was 3.447927057753 s. The between-condition downstream E2 arrival shift was 3.447927057753 s. Maximum absolute Ve was 15.433485430845 and 16.384820032185 mV.

## 2. Repository and environment

Branch `export/sindy-physics-fields`, commit `4996a1065f65c7b9dc7f758561a2fbeadc3ca34d`, Python `3.11.15 (main, Jun  2 2026, 22:29:49) [MSC v.1944 64 bit (AMD64)]` on `Windows-10-10.0.26200-SP0`. Full package, CPU, memory, GPU, initial status, and pre-analysis dirty-tree state are in `00_environment.txt`.

## 3. Representative-run provenance

The current implementation and declared bounded swelling specification identify the post-`7d054b8` run as authoritative. The mesh, physical stimulus/readout locations, threshold, duration, and time step were verified in code. The pipeline uses no randomness for aligned/no-dipole runs; seed 20260714 is fixed for all potentially random operations.

## 4. Authoritative representative metrics

Speed uses 60 times the 5.566144315619 mm center-to-center geodesic distance divided by the 1 mm ROI median-arrival difference. The original saved delay is instead the E2 center-vertex arrival minus E1 center-vertex arrival. ROI delays were 127.998045774428 and 131.645852661617 s. Both definitions are preserved in the canonical CSV/JSON.

## 5. Competing Figure 3 / Table 2 values

The 2.558/2.476 mm/min, 130.5/134.7 s, and 18.99/19.50 mV set was generated before the swelling law was changed to the current bounded saturating target with a separate recovery branch. Its output-side Table S2 was never regenerated and is stale. The 2.609/2.537 set comes from the current representative CSV/JSON and current manuscript-side Table S2. Neither set is a null-control result. A pre-existing uncommitted edit to `scripts/run_fig2_rep_quantitative.py` hard-codes the stale labels while still importing the current simulation configuration, producing a mixed working-tree Figure 2 artifact. No production file was changed here.

## 6. Geometry-profile outputs

All 12 implemented Gaussian folds plus a computed flat control are provided with equal physical scaling. Because the 28-point cross-section has no y=0 vertex, sampled maximum depths are slightly below analytic requested depths; exact sampled coordinates and fitted sigma/FWHM checks are tabulated.

## 7. Difference-map findings

Signed Ke and arrival-time differences were computed as dipole minus no dipole. Display times were fixed at 25%, 50%, and 75% of the no-dipole E1-E2 traversal interval. Paired absolute maps share limits; signed maps are zero-centered. Arrays, coordinate conventions, and selected times are saved separately.

## 8. Shoulder/gyral-effect findings

Required classification: **no shoulder-specific acceleration**. The minimum representative path Delta T was -9.794111642314 s. Region definitions depend only on analytic Gaussian landmarks, with a predeclared shoulder-to-flatter boundary at |y|=2 sigma. Across the 12 deterministic geometries, classifications were {'no material difference': 26, 'slowing': 34, 'acceleration': 24}.

## 9. Numerical robustness of localized acceleration

The two shoulders were checked on the 64x28 reference mesh, a 72x32 neighboring mesh, and with the reference time step halved. Local speed was evaluated with 5-, 7-, and 9-vertex Savitzky-Golay derivatives and unsmoothed adjacent segments. Detailed outcomes are in `04_shoulder_analysis/numerical_check.csv`; the classification above requires agreement across these definitions and resolutions.

## 10. Exact model-equation audit

All 37 requested implementation items are transcribed with expressions, defaults, units/status, code locations, manuscript coverage, and discrepancies. Several quantities are phenomenological implementation scales rather than dimensionally derived biophysical conversions. In particular, the code has no Faraday/current-to-molar derivation or membrane area-to-volume factor; it uses fixed factors 0.038 and 0.011.

## 11. Initial and boundary conditions

Initial ion states, GHK voltage, activation, focal perturbation, clipping, and implicit natural no-flux boundaries are documented in `05_model_audit/initial_and_boundary_conditions.md`. The model is a two-compartment surface reduction with no through-thickness coordinate.

## 12. Skull/surrounding-conductor feasibility

The current solver has no CSF, meninges, skull, scalp, external conductivity, or boundary-layer option. Internal kappa/sigma/gain parameters cannot defensibly be relabeled as skull conductivity. The skull question therefore cannot be answered without a three-dimensional volume-conductor or derived boundary-layer extension; no arbitrary sensitivity was run.

## 13. Files generated

- `results/reviewer_revision_analysis/00_environment.txt`
- `results/reviewer_revision_analysis/00_provenance_map.csv`
- `results/reviewer_revision_analysis/00_repository_inventory.md`
- `results/reviewer_revision_analysis/01_canonical_representative/canonical_config.json`
- `results/reviewer_revision_analysis/01_canonical_representative/canonical_metrics.csv`
- `results/reviewer_revision_analysis/01_canonical_representative/canonical_metrics.json`
- `results/reviewer_revision_analysis/01_canonical_representative/dipole_aligned_timeseries.npz`
- `results/reviewer_revision_analysis/01_canonical_representative/mesh_and_readouts.npz`
- `results/reviewer_revision_analysis/01_canonical_representative/no_dipole_timeseries.npz`
- `results/reviewer_revision_analysis/01_canonical_representative/provenance_decision.md`
- `results/reviewer_revision_analysis/01_canonical_representative/run_log.txt`
- `results/reviewer_revision_analysis/02_geometry_profiles/geometry_equation_and_implementation.md`
- `results/reviewer_revision_analysis/02_geometry_profiles/profile_coordinates.csv`
- `results/reviewer_revision_analysis/02_geometry_profiles/realized_geometry_parameters.csv`
- `results/reviewer_revision_analysis/02_geometry_profiles/representative_surface_readouts.pdf`
- `results/reviewer_revision_analysis/02_geometry_profiles/representative_surface_readouts.png`
- `results/reviewer_revision_analysis/02_geometry_profiles/representative_surface_readouts.svg`
- `results/reviewer_revision_analysis/02_geometry_profiles/synthetic_fold_profiles.pdf`
- `results/reviewer_revision_analysis/02_geometry_profiles/synthetic_fold_profiles.png`
- `results/reviewer_revision_analysis/02_geometry_profiles/synthetic_fold_profiles.svg`
- `results/reviewer_revision_analysis/03_difference_visualization/arrival_time_difference_map.pdf`
- `results/reviewer_revision_analysis/03_difference_visualization/arrival_time_difference_map.png`
- `results/reviewer_revision_analysis/03_difference_visualization/arrival_time_difference_map.svg`
- `results/reviewer_revision_analysis/03_difference_visualization/matched_maps.pdf`
- `results/reviewer_revision_analysis/03_difference_visualization/matched_maps.png`
- `results/reviewer_revision_analysis/03_difference_visualization/matched_maps.svg`
- `results/reviewer_revision_analysis/03_difference_visualization/paired_kymographs.pdf`
- `results/reviewer_revision_analysis/03_difference_visualization/paired_kymographs.png`
- `results/reviewer_revision_analysis/03_difference_visualization/paired_kymographs.svg`
- `results/reviewer_revision_analysis/03_difference_visualization/path_arrival_and_deltaT.pdf`
- `results/reviewer_revision_analysis/03_difference_visualization/path_arrival_and_deltaT.png`
- `results/reviewer_revision_analysis/03_difference_visualization/path_arrival_and_deltaT.svg`
- `results/reviewer_revision_analysis/03_difference_visualization/plotted_arrays.npz`
- `results/reviewer_revision_analysis/03_difference_visualization/plotting_metadata.json`
- `results/reviewer_revision_analysis/03_difference_visualization/visualization_methods.md`
- `results/reviewer_revision_analysis/04_shoulder_analysis/all_geometries_region_metrics.csv`
- `results/reviewer_revision_analysis/04_shoulder_analysis/curvature_and_deltaT.pdf`
- `results/reviewer_revision_analysis/04_shoulder_analysis/curvature_and_deltaT.png`
- `results/reviewer_revision_analysis/04_shoulder_analysis/curvature_and_deltaT.svg`
- `results/reviewer_revision_analysis/04_shoulder_analysis/numerical_check.csv`
- `results/reviewer_revision_analysis/04_shoulder_analysis/regional_effect_summary.pdf`
- `results/reviewer_revision_analysis/04_shoulder_analysis/regional_effect_summary.png`
- `results/reviewer_revision_analysis/04_shoulder_analysis/regional_effect_summary.svg`
- `results/reviewer_revision_analysis/04_shoulder_analysis/representative_pathwise_metrics.csv`
- `results/reviewer_revision_analysis/04_shoulder_analysis/shoulder_analysis_report.md`
- `results/reviewer_revision_analysis/05_model_audit/code_manuscript_discrepancies.md`
- `results/reviewer_revision_analysis/05_model_audit/initial_and_boundary_conditions.md`
- `results/reviewer_revision_analysis/05_model_audit/model_equation_audit.csv`
- `results/reviewer_revision_analysis/05_model_audit/model_equation_audit.md`
- `results/reviewer_revision_analysis/05_model_audit/parameter_audit.csv`
- `results/reviewer_revision_analysis/06_skull_feasibility.md`
- `results/reviewer_revision_analysis/07_consistency_checks.csv`
- `results/reviewer_revision_analysis/07_reproduction_commands.txt`
- `results/reviewer_revision_analysis/07_test_verification.txt`
- `results/reviewer_revision_analysis/FINAL_ANALYSIS_REPORT.md`
- `results/reviewer_revision_analysis/manifest_sha256.txt`

## 14. Reproduction

Run `C:\work\CSD\.venv\Scripts\python.exe scripts/reviewer_revision_analysis/run_all_revision_analyses.py` from `C:\work\CSD`. Exact commands are in `07_reproduction_commands.txt`. All 27 automated consistency checks passed.

## 15. Incomplete analyses or unresolved ambiguities

- A local scalar surface-normal opposition index is not implemented; opposition exists only pairwise inside the nonlocal kernel. It was not invented.
- The manuscript symbol J_s^m has no explicit sign convention mapping to the code's `*_membrane` variables; the apparent extracellular sign mismatch is therefore reported as unresolved.
- Skull effects are outside the current solver.
- No atlas rerun was needed for the reviewer-requested direct synthetic shoulder test; existing atlas provenance was inventoried only.

## 16. Git safety confirmation

No manuscript, supplement, bibliography, response-letter, cover-letter, or production figure was edited by this analysis. New writes are confined to `scripts/reviewer_revision_analysis/` and `results/reviewer_revision_analysis/`. The repository already contained tracked and untracked changes before this work; they were preserved. No unexpected new tracked modification was detected.

Final `git diff --name-only` (all entries pre-existing):

```text
manuscript/figures/fig2_rep_quantitative.pdf
manuscript/figures/fig2_rep_quantitative.png
scripts/run_fig2_rep_quantitative.py
src/csd_sulcus_model.egg-info/PKG-INFO
src/csd_sulcus_model.egg-info/SOURCES.txt
src/csd_sulcus_model.egg-info/requires.txt
warning: in the working copy of 'scripts/run_fig2_rep_quantitative.py', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'src/csd_sulcus_model.egg-info/SOURCES.txt', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'src/csd_sulcus_model.egg-info/dependency_links.txt', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'src/csd_sulcus_model.egg-info/requires.txt', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'src/csd_sulcus_model.egg-info/top_level.txt', LF will be replaced by CRLF the next time Git touches it
```

Final `git status --short`:

```text
M manuscript/figures/fig2_rep_quantitative.pdf
 M manuscript/figures/fig2_rep_quantitative.png
 M scripts/run_fig2_rep_quantitative.py
 M src/csd_sulcus_model.egg-info/PKG-INFO
 M src/csd_sulcus_model.egg-info/SOURCES.txt
 M src/csd_sulcus_model.egg-info/dependency_links.txt
 M src/csd_sulcus_model.egg-info/requires.txt
 M src/csd_sulcus_model.egg-info/top_level.txt
?? outputs/sindy_physics_export/flat_no_dipole/
?? outputs/sindy_physics_export/folded_dipole_aligned/
?? outputs/sindy_physics_export/folded_distance_only_null/
?? outputs/sindy_physics_export/folded_no_dipole/
?? outputs/sindy_physics_export/folded_scrambled_normal_null/
?? results/
?? scripts/reviewer_revision_analysis/
?? uv.lock
```
