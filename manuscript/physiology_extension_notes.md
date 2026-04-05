# Physiology Extension Notes

## Purpose

This note records the manuscript-ready rationale for two additions to the modeling pipeline:

1. a physiology-constrained tensor formulation, in which the tangential-to-normal coupling ratio is restricted to a conservative cortex-like band rather than chosen only as a free parameter; and
2. a reduced biophysical spreading-depolarization model based on extracellular potassium and buffer availability, used to test whether the scalar-versus-tensor ordering survives beyond Barkley kinetics.

## Literature grounding for the tensor constraint

The current tensor constraint is intentionally conservative. It does **not** claim to fit a unique biological measurement of cortical anisotropy. Instead, it encodes the weaker directional structure expected in cortical gray matter relative to white matter.

Primary studies motivating this choice are:

- McNab et al. reported that diffusion anisotropy in adult cortical gray matter is low but nonzero, with mean fractional anisotropy around 0.20 in cortex versus about 0.80 in white matter, and showed that diffusion orientation can be analyzed relative to the folded cortical surface rather than only in scanner coordinates \citep{McNab2013}.
- Truong et al. showed that cortical gray matter diffusion anisotropy has a depth dependence and is predominantly radial in vivo, again supporting the idea that cortical transport is weakly anisotropic rather than isotropic \citep{Truong2014}.
- Leuze et al. demonstrated a coherent tangential diffusion component and layer-specific intracortical connectivity in human cortex, which motivates preserving easier along-sulcus transport in a tensor formulation instead of collapsing the fold into a purely isotropic attenuation field \citep{Leuze2014}.

Taken together, these studies support the use of a **modest** tangential-to-normal coupling ratio rather than a white-matter-like anisotropy. In the current implementation, the physiology-constrained tensor mode clamps the target tangential/normal ratio to 1.05--1.30, with 1.15 as the default. This should be described as a conservative mesoscale prior informed by cortical diffusion studies, not as a direct fit of CSD transport coefficients to dMRI eigenvalues.

## Literature grounding for the reduced biophysical SD model

The potassium-buffer model is meant as a bridge between minimal excitable-media dynamics and fuller ion-dynamics models. It retains an explicit propagating extracellular potassium field and a dynamic buffer-availability variable, but remains much cheaper to simulate than electrodiffusive SD models.

This reduced formulation is informed by prior biophysical SD work showing that extracellular potassium accumulation, clearance, and tissue ion dynamics are central to spreading depolarization propagation \citep{Tuckwell1978,Wei2014,Zandt2015}. The current model should therefore be described as a reduced biophysical SD model, not as a full ionic or electrodiffusive account.

## Quick extension results currently in the repo

Quick representative runs are stored under `outputs/physiology_extension_quick/` and show:

- Barkley scalar: 2.562 mm/min
- Barkley microstructure-constrained tensor: 2.652 mm/min
- Potassium-buffer scalar: 2.570 mm/min
- Potassium-buffer microstructure-constrained tensor: 2.652 mm/min
- Barkley tensor-minus-scalar gain: +0.090 mm/min
- Potassium-buffer tensor-minus-scalar gain: +0.081 mm/min

Across the microstructure sweep, tensor-minus-scalar gain remains positive from target tangential/normal ratios of 1.05 to 1.30 in both kinetics families.

## Paste-ready manuscript text

### Methods paragraph: physiology-constrained tensor

```tex
To reduce the arbitrariness of the tensor parameterization, we introduced a physiology-constrained tensor mode in which the tangential-to-normal coupling ratio was restricted to a conservative cortex-like range rather than chosen only as a free attenuation ratio. This choice was motivated by diffusion-MRI studies showing that adult cortical gray matter is weakly but measurably anisotropic, with orientation defined relative to cortical geometry, depth-dependent radial structure, and coherent tangential intracortical components \citep{McNab2013,Truong2014,Leuze2014}. Because these measurements do not provide a direct estimate of CSD transport coefficients, we did not fit the tensor directly to dMRI eigenvalues. Instead, we used a modest tangential/normal coupling ratio band of 1.05--1.30, with 1.15 as the default, to encode weak directional preference at the mesoscale while avoiding white-matter-like anisotropy.
```

### Methods paragraph: reduced biophysical SD model

```tex
To test whether the scalar-versus-tensor ordering depended on Barkley kinetics, we added a reduced biophysical spreading-depolarization model in which the propagating variable is extracellular potassium and the recovery variable is buffer availability. Potassium release was activated sigmoidally above threshold, clearance depended on the available buffer fraction, and the same spatial transport operator used in the Barkley model was applied to the potassium field. This reduced formulation was intended as a computationally tractable bridge toward fuller ion-dynamics models of spreading depolarization \citep{Tuckwell1978,Wei2014,Zandt2015}, rather than as a complete electrodiffusive description.
```

### Results paragraph: extension ordering

```tex
The scalar-versus-tensor ordering also survived two additional model extensions. First, when the tensor field was constrained to a conservative cortex-like tangential/normal coupling band (1.05--1.30, default 1.15), the representative Barkley case remained faster in the tensor formulation (2.652 mm/min) than in the matched scalar formulation (2.562 mm/min). Second, the same ordering appeared in a reduced biophysical potassium-buffer SD model, in which the representative scalar and microstructure-constrained tensor cases propagated at 2.570 and 2.652 mm/min, respectively. Thus, the sign of the tensor-minus-scalar difference was preserved both when tensor anisotropy was tied to cortical microstructure constraints and when the kinetics were moved beyond the Barkley model.
```

### Discussion paragraph: interpretation

```tex
These extensions tighten the interpretation of the tensor result. The moderation of sulcal slowing is not limited to a hand-tuned attenuation ratio in Barkley kinetics. It persists when the tensor is restricted to a conservative anisotropy band informed by cortical diffusion studies and when the excitable-medium kinetics are replaced with a reduced biophysical potassium-buffer model. This does not make the present model fully biophysical, but it does show that the scalar-versus-tensor ordering is robust to two substantively different forms of model refinement.
```

## Recommended next run before manuscript use

Before these paragraphs are moved into the main paper, the highest-value computational step is:

```powershell
python scripts/run_physiology_extension.py
```

That full run will replace the quick extension numbers with manuscript-scale values while keeping the same literature framing.
