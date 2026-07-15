# Canonical representative provenance decision

The authoritative run is the current post-`7d054b8` implementation on the declared 64x28 synthetic folded strip. It exactly verifies the declared 22x10 mm domain, requested depth 2.4 mm and sigma 1.5 mm, 1,792 vertices, 3,402 faces, 1.2 mm initial-only stimulus, -28 mV arrival threshold, 220 s duration, automatically selected vertices 471/636/624, and 0.049969957359 s time step.

This choice is based on implementation chronology and the manuscript's declared bounded saturating swelling specification, not effect size. The competing 2.558/2.476 result was generated before the swelling implementation was changed and survives only in an output-side Table S2 that was not regenerated. The current representative CSV/JSON, manuscript-side Table S2, and tracked Figure 2 source all identify the post-revision run. The working-tree Figure 2 labels were manually changed back to the stale values without changing the underlying trace simulation, creating a mixed artifact.

The null-control pipeline is not the source of either representative pair. Its distance-only and scrambled-normal kernel modes are separate rows and outputs.

## Readout definitions

- Original saved `cross_fold_delay_s`: E2 center-vertex arrival minus E1 center-vertex arrival.
- Speed denominator: E2 median arrival minus E1 median arrival over 1 mm geodesic ROIs centered on the same vertices.
- Speed formula: `60 * 5.566144315619 mm / ROI delay (s)`.
- Between-condition downstream shift: dipole E2 center arrival minus no-dipole E2 center arrival.

Both center-vertex and ROI-median delays are reported in the canonical metrics; they are not averaged or substituted.
