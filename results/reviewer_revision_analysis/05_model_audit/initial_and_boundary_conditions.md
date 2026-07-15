# Initial and boundary conditions

## Initial state

Uniform concentrations are Ke=3.5, Ki=140.0, Nae=145.0, Nai=18.0, Cle=112.0, and Cli=7.0 mM. Vm is the GHK value from those concentrations and leak permeabilities. Theta, swelling, constriction, and phi_hat start at zero; oxygen starts at one; perfusion starts at the geometry-dependent baseline reserve.

The focal perturbation is applied once at t=0 to vertices within 1.2 mm **three-dimensional Euclidean** distance of the frozen stimulus vertex: Ke is raised to at least 22 mM, Nae is reduced by 10 mM with a 5 mM floor, and theta is raised to at least 0.92. There is no sustained source.

## Boundaries

Ion transport and potential use mesh-edge cotangent operators with no exterior-edge source or flux. On the open rectangular strip this is an implicit natural zero-normal-flux boundary. The screened potential solve has no explicit Dirichlet boundary; after solving, the mass-weighted mean is subtracted. There is no absorbing layer, surrounding conductor, or continuation beyond the finite mesh.

## Reduced compartments

All concentrations are scalar nodal surface fields. The code has no thickness coordinate, membrane area-to-volume ratio, or through-thickness gradients. Intracellular and extracellular compartments are therefore locally well mixed in the unresolved direction; alpha and 1-alpha rescale effective compartment source terms.
