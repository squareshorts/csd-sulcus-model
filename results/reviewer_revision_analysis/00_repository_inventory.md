# Repository inventory and provenance reconnaissance

- Branch: `export/sindy-physics-fields`
- Commit: `4996a1065f65c7b9dc7f758561a2fbeadc3ca34d`
- The working tree was already dirty before this analysis. Those pre-existing files were treated as user-owned and were not modified.

## Relevant files

| Category | Path(s) | Role |
|---|---|---|
| Representative folded simulation | `scripts/run_surface_mechanistic_study.py` | _representative_rows; REPRESENTATIVE_CASES |
| Representative solver | `src/csd_sulcus/surface_mechanistic.py` | MechanisticSurfaceParams; run_mechanistic_surface_simulation |
| Synthetic mesh and Gaussian fold | `src/csd_sulcus/surface_io.py` | generate_folded_strip_mesh |
| Readout selection | `scripts/run_surface_representative.py` | choose_auto_vertices |
| Speed/readout functions | `src/csd_sulcus/surface_mechanistic.py` | mechanistic_surface_arrival_speed_mm_min |
| Current representative CSV | `outputs/surface_mechanistic_study/mechanistic_representative_summary.csv` | 2.609/2.537 current saved outputs |
| Current mechanistic summary | `outputs/surface_mechanistic_study/mechanistic_study_summary.json` | representative, sweep, null, atlas, convergence summaries |
| Stale generated Table S2 | `outputs/surface_mechanistic_study/table_s2_exact_representative_run.tex` | 2.558/2.476 pre-swelling-revision outputs |
| Current manuscript Table S2 source | `manuscript/table_s2_exact_representative_run.tex` | 2.609/2.537 current outputs; read only |
| Figure 2 generator | `scripts/run_fig2_rep_quantitative.py` | working-tree hard-coded display values; read only |
| Figure 2 production assets | `manuscript/figures/fig2_rep_quantitative.pdf; manuscript/figures/fig2_rep_quantitative.png` | pre-existing working-tree modifications; read only |
| Figure 3 propagation source | `scripts/run_surface_mechanistic_study.py` | _save_propagation_figure |
| Figure 3 archived data/figure | `outputs/surface_mechanistic_study/mechanistic_wave_propagation.png` | generated from current mechanistic pipeline |
| Synthetic geometry sweep | `outputs/surface_mechanistic_study/mechanistic_geometry_sweep.csv` | 52x24 meshes, 210 s |
| Atlas patches | `src/csd_sulcus/atlas_patch.py; outputs/surface_mechanistic_study/mechanistic_atlas_patch_check.csv` | multi-patch pipeline and outputs |
| Null kernels | `outputs/surface_mechanistic_study/mechanistic_null_models.csv` | aligned, distance-only, scrambled-normal |
| Convergence | `outputs/surface_mechanistic_study/mechanistic_convergence.csv` | neighboring mesh and time-step checks |
| Model equations/parameters | `src/csd_sulcus/surface_mechanistic.py; src/csd_sulcus/surface_model.py; src/csd_sulcus/surface_ops.py` | implemented model and discretization |
| Manuscript declarations | `manuscript/reframed_submission.tex` | primary specification and reported values; read only |
| SINDy/null raw snapshots | `outputs/sindy_physics_export/*` | untracked pre-existing physics exports; not used as representative authority |

## Provenance resolution

The 2.558/2.476 mm/min result is an archived generated table from the implementation before commit `7d054b8` changed the swelling law from an unbounded/clipped 1.5 state to a bounded saturating target with separate recovery. The representative CSV and JSON were rerun after that model change and contain 2.609/2.537 mm/min. The archived output-side Table S2 was not regenerated and is stale.

The current tracked Figure 2 generator at HEAD uses 2.609/2.537, but the working tree contains a pre-existing uncommitted edit that replaces those labels with the stale 2.558/2.476 values. The current mechanistic propagation figure and manuscript-side Table S2 use the post-revision values. Neither result comes from the distance-only or scrambled-normal null pipeline; the null pipeline uses the same vertices but distinct kernel modes and is stored separately.

Both representative result sets used the 64x28 (1,792 vertex, 3,402 face) mesh; stimulus vertex 471; E1 vertex 636; E2 vertex 624; 1.2 mm initial perturbation; -28 mV arrival threshold; 220 s duration; and automatic 0.049969957 s step. The material difference is the swelling implementation and its defaults, not geometry, electrodes, seed, stimulus, threshold, duration, or time step.

## Readout-definition finding

The saved `cross_fold_delay_s` is E2 vertex arrival minus E1 vertex arrival. The saved speed instead uses the same E1/E2 centers but median arrival over separate 1 mm geodesic regions, divided into the center-to-center geodesic distance. The two time differences are close but not definitionally identical. New canonical outputs preserve both.
