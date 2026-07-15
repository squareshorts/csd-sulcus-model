# Code-manuscript discrepancies and omissions

These findings report the current implementation; they do not propose replacement prose.

- **1 beta_ed**: Code uses the numerical thermal voltage directly rather than explicit F, R, T constants.
- **2 phi_hat and Ve**: Mass-weighted mean-zero gauge/fallback is not stated.
- **3 gamma_rho**: Phenomenological gain; no derivation or physical charge conversion in code.
- **4 gamma_d**: Units are not defined by code.
- **5 sigma_phi**: It is not a calibrated electrical conductivity.
- **6 kappa**: No physical screening-length derivation.
- **7 complete Na/K pump**: Manuscript omits explicit factors, exponents, clipping, and constants.
- **8 f_Na**: Hill exponent 3 not documented.
- **9 f_K**: Hill exponent 2 not documented.
- **10 f_O2**: Exact clipping is undocumented.
- **11 Hill exponents**: Missing from manuscript.
- **12 half-activation constants**: None.
- **13 all transmembrane fluxes J_s^m**: Manuscript equation uses -J/alpha extracellular but code adds these J/alpha; J sign convention is not explicitly defined.
- **14 current-to-molar conversions**: No Faraday constant, membrane capacitance, area, or derivation is implemented.
- **15 membrane area / surface-to-volume**: A physical surface-to-volume ratio is absent.
- **16 pump stoichiometry**: Signs depend on undocumented J convention.
- **17 GHK voltage**: Exact equation and 1e-6 floors are omitted.
- **18 Nernst potentials**: Exact formula/floor omitted.
- **19 leak conductances**: Units and values omitted.
- **20 active permeabilities/conductances**: Exact values and distinction between GHK P and current g omitted.
- **21 theta ODE**: Exact update order is after concentration update but uses pre-update Vm.
- **22 theta sigmoid/slopes/bounds**: 0.55 target floor, logistic clipping [-60,60], and theta cap 1.5 omitted.
- **23 swelling target s_infinity**: Exact saturation law and constants omitted.
- **24 alpha(s)**: Exact gain and geometric alpha0 construction not fully documented.
- **25 lambda(s)**: Exact multiplier omitted.
- **26 swelling recovery**: Branch condition omitted.
- **27 initial concentrations**: Initial concentrations absent from manuscript table.
- **28 initial voltage/activation**: Exact initial voltage/state not reported.
- **29 exact focal perturbation**: Stimulus region is Euclidean 3D, not geodesic; manuscript does not state this.
- **30 ionic boundary conditions**: Boundary condition omitted.
- **31 potential boundary conditions**: Boundary/gauge omitted.
- **32 finite mesh boundaries**: No absorbing layer or infinite-domain approximation.
- **33 concentration field interpretation**: Not explicitly identified as cortical-column averages or thickness-integrated fields.
- **34 unresolved-thickness mixing**: Well-mixed-through-thickness assumption is implicit, not stated.
- **35 V0,m_ed,ell_d,d_c,gamma terms**: Code normalization means gamma_d multiplies a row-normalized average, not an unnormalized integral.
- **36 caps/floors/clipping/fallbacks**: Extensive safeguards and explicit-Euler arrival quantization omitted.
- **37 null kernels**: Kernel >1e-8 pruning and exact seed omitted.

The most consequential reproducibility omissions are the phenomenological current-to-concentration factors, absence of a membrane area/volume conversion, exact pump factors and Hill exponents, exact theta and swelling laws, all clipping rules, implicit boundary conditions, and unresolved-thickness mixing assumption. The manuscript's extracellular J sign cannot be reconciled unambiguously because J is not explicitly assigned the code's sign convention.
