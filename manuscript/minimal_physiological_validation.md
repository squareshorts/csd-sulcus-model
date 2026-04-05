# Minimal Physiological Validation Recommendation

## Recommendation

Add the accepted-candidate speed-range anchor first.

Why this should come first:
- It is the smallest external validation layer already supported by the current codebase.
- It upgrades the paper from a purely internal scalar-versus-tensor comparison to a physiologically anchored mechanistic comparison.
- It preserves the present claim: tensor transport moderates sulcal slowing but does not reverse it.
- It avoids overclaiming full ionic realism.

## Exact Analyses

1. Reuse the accepted candidates from `outputs/biophysical_validation/biophysical_validation.json`.
2. Report the accepted scalar, tensor, and control speed ranges.
3. Check those accepted speeds against two published CSD bands:
   - experimental cortex: 2.0--5.0 mm/min
   - human malignant stroke: 1.7--9.2 mm/min
4. Report whether the tensor-minus-scalar ordering remains positive across all accepted candidates.

## Exact Outputs

- Figure: `outputs/physiology_anchor/figures/fig_physiology_anchor.png`
- Table: `outputs/physiology_anchor/physiology_anchor_table.tex`
- Summary: `outputs/physiology_anchor/physiology_anchor_summary.md`

Observed values from the current accepted candidate set:
- accepted candidates: 63
- scalar speed range: 2.606 to 2.750 mm/min
- tensor speed range: 2.683 to 2.792 mm/min
- mean tensor-minus-scalar gain: 0.071 mm/min
- experimental-range coverage: 63/63 scalar, 63/63 tensor, 63/63 control
- human-range coverage: 63/63 scalar, 63/63 tensor, 63/63 control

## Manuscript Revisions

Sections revised in `manuscript/reframed_submission.tex`:
- Abstract
- Introduction
- Methods: `Reduced biophysical spreading-depolarization model and validation sweep`
- Results: `Microstructure-constrained biophysical validation`
- Discussion
- Code availability

New manuscript assets referenced:
- `../outputs/physiology_anchor/physiology_anchor_table.tex`
- `../outputs/physiology_anchor/figures/fig_physiology_anchor.png`

## Claim Scope

This validation supports the following narrow claim:
- the accepted physiology-constrained scalar and tensor cases occupy published CSD speed regimes, and the tensor-faster-than-scalar ordering is preserved across that accepted set.

This validation does not justify the following stronger claims:
- that the model is a full ionic or electrodiffusive SD model
- that the transport tensor is uniquely identified from physiology
- that matching published speed ranges validates the mechanism by itself

## Practical Note

Do not base the paper's physiological anchor on `outputs/physiology_extension/` alone. That output set contains potassium-buffer representative runs that do not preserve the scalar-versus-tensor ordering, whereas the accepted-candidate validation set in `outputs/biophysical_validation/` does preserve the intended constrained claim.
