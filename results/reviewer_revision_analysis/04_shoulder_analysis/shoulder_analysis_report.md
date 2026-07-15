# Shoulder/gyral-effect analysis

**Required classification: no shoulder-specific acceleration.**

Regions were fixed from the implemented Gaussian geometry before examining paired differences. Inflection points are y=+/-sigma. Fundus/bank boundaries are y=+/-0.5 sigma. Shoulder-to-flatter boundaries are y=+/-2 sigma, where the Gaussian profile amplitude is exp(-2) of requested depth.

Arrival arrays are shown unsmoothed. Local speed uses a quadratic Savitzky-Golay derivative with nominal 7-vertex window; 5- and 9-vertex windows and unsmoothed adjacent-segment speeds are included as sensitivity definitions.

The representative minimum Delta T over the full path was -9.794112 s (positive means later with dipole).
Across the 12 deterministic folded geometries, region classifications were: {'no material difference': 26, 'slowing': 34, 'acceleration': 24}.

Delta T was not positive at every path vertex. Regions containing values more negative than one reference time step were: first sulcal bank, fundus, and post-fold flatter region. Neither representative shoulder region met that criterion.
Mean Delta T was 1.442883 s where analytic signed curvature was positive and -0.352288 s where it was negative. The effect therefore did not follow a simple curvature-sign reversal.
Delay accumulated across the fundus, opposite bank, and second shoulder. It then decreased in the outer post-fold flatter region. Arrival time versus fixed cross-sectional path distance was non-monotonic there, so a directional local speed is not numerically defined and the apparent outer-region acceleration is not shoulder evidence.

No population-level biological inference is made. The sweep summary describes deterministic paired simulations only.

The implementation has no precomputed per-vertex surface-normal opposition index. Opposition is pairwise inside the nonlocal kernel, so inventing a local path field would change the model; the path CSV records this as unavailable.

## Representative regional summaries

| Region | Mean Delta T (s) | Minimum Delta T (s) | Mean local speed difference (mm/min) |
|---|---:|---:|---:|
| pre-fold flatter region | 0.008328 | 0.000000 | -0.018443 |
| first shoulder/curvature-transition region | 0.012492 | 0.000000 | -0.004846 |
| first sulcal bank | -0.049970 | -0.099940 | 0.043797 |
| fundus | 0.949429 | -2.048768 | -0.286699 |
| opposite bank | 3.922642 | 2.448528 | -0.260827 |
| second shoulder/curvature-transition region | 4.572251 | 2.098738 | -64.654226 |
| post-fold flatter region | -4.239118 | -9.794112 | nan |

The numerical check repeats both shoulders on a neighboring 72x32 mesh and with the reference time step halved. The reference and half-step shoulders show no negative Delta T; the neighboring mesh has a one-step negative fluctuation at the first shoulder with mutually inconsistent local-speed definitions. This does not support shoulder-specific acceleration.
