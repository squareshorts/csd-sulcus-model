# Submission Positioning

## Core claim

Sulcal geometry slows CSD across transport and kinetics families, but isotropic coupling reductions overestimate the magnitude of that slowing relative to orientation-aware tensor transport.

## Why this is stronger

- It is a robustness paper rather than a single-mechanism paper.
- It tests whether the sulcal effect survives both a transport change and a kinetics change.
- It turns isotropic attenuation into an upper-bound model rather than the only evidentiary base.
- It gives reviewers a sharper computational claim: the sign of sulcal slowing is robust, but its amplitude depends on how directional structure is represented.

## Full manuscript-scale scalar-versus-tensor message

- Representative scalar-flat case: 2.582 mm/min
- Representative tensor-flat case: 2.689 mm/min
- Representative scalar-Gaussian case: 2.618 mm/min
- Representative tensor-Gaussian case: 2.718 mm/min
- Mean tensor minus scalar gain across the full grid: +0.062 mm/min for flat profiles and +0.057 mm/min for Gaussian profiles
- Strongest slowing in the full grid: scalar-flat, width 5.5 mm, g = 0.75, with delta speed = -0.448 mm/min

## Physiology-extension message

Quick representative runs show the same ordering after both extensions:

- Barkley scalar: 2.562 mm/min
- Barkley microstructure-constrained tensor: 2.652 mm/min
- Potassium-buffer scalar: 2.570 mm/min
- Potassium-buffer microstructure-constrained tensor: 2.652 mm/min
- Positive tensor-minus-scalar gain across the full microstructure sweep in both kinetics families

## Recommended framing

Suggested contribution statement:
We show that sulcal delays are robust to substantial changes in model structure. They persist when tensor transport is constrained to modest cortex-like anisotropy and when Barkley kinetics are replaced with a reduced potassium-buffer SD model, but their amplitude is systematically larger in matched isotropic scalar models.

## What is already in place

- The full manuscript-scale scalar-versus-tensor grid has been run.
- The physiology-constrained tensor extension is implemented in `scripts/run_physiology_extension.py`.
- A reduced potassium-buffer SD model reproduces the scalar-versus-tensor ordering in quick representative runs.
- The main heatmaps, local-speed triptych, and physiology-extension figures all support the manuscript claim visually.

## Highest-value next upgrade

- Run the physiology extension at manuscript scale and fold those results into the main manuscript.
- Add a short Methods/Results subsection using the literature-backed text in `manuscript/physiology_extension_notes.md`.
