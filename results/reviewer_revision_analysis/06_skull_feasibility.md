# Skull / surrounding-conductor feasibility

1. **Represented layers:** The repository's mechanistic surface solver represents neither CSF, meninges, skull, scalp, nor any surrounding volume-conductor layer. It evolves two overlapping ionic surface compartments on the cortical mesh only.
2. **External conductivity boundary:** The screened surface potential equation cannot assign an exterior conductivity or interface condition. Its finite-strip boundary is an implicit natural no-flux boundary.
3. **Existing physical skull parameter:** None. `potential_screening` (manuscript kappa), the 1.8 electrical surface-operator multiplier (sigma_phi), `field_reference_mV`, and dipole gains are internal phenomenological surface-equation parameters, not skull conductivity.
4. **Interpretability of changing an internal parameter:** Relabeling or varying one of those terms as skull conductivity would be arbitrary because the implementation contains no derivation connecting it to a layered volume conductor or conductivity contrast.
5. **Answerability:** The reviewer's skull question is not answerable with the current model. A three-dimensional volume-conductor or defensibly derived boundary-layer extension with CSF/skull interfaces would be required.

No sensitivity simulation was run because the required external-conductivity/boundary-layer option does not exist. This is a feasibility limit, not a null result about skull effects.
