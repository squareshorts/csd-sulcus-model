# Submission Positioning

## Core claim

Localized sulcal slowing of CSD is robust across model families, but isotropic coupling reductions overestimate its magnitude relative to orientation-aware tensor transport.

## Why this is stronger

- It is a robustness paper, not a single-mechanism paper.
- It directly addresses anisotropy instead of listing it as a limitation.
- It turns a minimal isotropic model into one member of a comparison set rather than the sole evidentiary base.
- It gives a sharper computational claim that reviewers can test and disagree with at the model-structure level.

## Full manuscript-scale message

- Representative scalar-flat case: 2.582 mm/min
- Representative tensor-flat case: 2.689 mm/min
- Representative scalar-Gaussian case: 2.618 mm/min
- Representative tensor-Gaussian case: 2.718 mm/min
- Mean tensor minus scalar gain across the full grid: +0.062 mm/min for flat profiles and +0.057 mm/min for Gaussian profiles
- Strongest slowing in the full grid: scalar-flat, width 5.5 mm, g = 0.75, with delta speed = -0.448 mm/min

## Recommended framing

Suggested title:
Localized sulcal slowing of cortical spreading depolarization persists in orientation-aware tensor coupling models, while isotropic reductions overestimate its magnitude

Suggested contribution statement:
We show that sulcal delays are robust to substantial changes in model structure, but that their amplitude depends strongly on whether coupling loss is treated isotropically or with direction-aware tensor transport.

## What is already submission-ready

- The full manuscript-scale scalar-versus-tensor grid has been run.
- The representative table now has full-resolution numbers.
- The main heatmaps and the local-speed triptych support the manuscript claim visually.

## Highest-value next upgrade after submission

- Add a targeted sweep over `tensor_tangent_attenuation_ratio` to show how anisotropy strength changes the moderation effect.
- Polish journal-specific formatting and, if desired, add a targeted tensor-ratio sensitivity appendix.


