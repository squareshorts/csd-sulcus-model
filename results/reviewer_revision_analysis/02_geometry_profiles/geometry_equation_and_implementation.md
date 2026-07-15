# Geometry equation and implementation

The exact code in `src/csd_sulcus/surface_io.py:160-205` evaluates

`z(y) = -d exp[-0.5 (y/sigma)^2]`

on a tensor grid with `x=linspace(0,22,nx)` and `y=linspace(-5,5,ny)`. The representative uses `nx=64`, `ny=28`; the family profiles in `profile_coordinates.csv` therefore use the exact 28 y samples of the representative mesh. The flat control is computed by setting `d=0`, not inserted as a reported result.

Because `ny=28` is even, y=0 is not a mesh vertex. The sampled maximum depth is consequently slightly smaller than the requested analytic depth; `realized_geometry_parameters.csv` reports this sampling difference. A fit of the exact sampled coordinates recovers sigma, and both analytic and sampled-interpolated full width at half maximum are reported.

All profile panels use equal physical axis scaling in millimeters. The representative depth 2.4 mm, sigma 1.5 mm panel is highlighted without vertical exaggeration.
