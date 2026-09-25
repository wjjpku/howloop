# Shared-prefix semantic counterfactual

Prospective refinement of the task pairing, not a re-labeling of results from the arbitrary-graph pilot.

Why: across unrelated graphs, a multi-call pattern transplant may replay an incoherent sequence of reads on base content. The final-edge hypothesis then has no guarantee of describing the transplanted intermediate computation. Test a more controlled pairing, retaining the negative arbitrary-graph results.

Construction: source directed cycle c0→...→c9→c0. Base cycle c0→...→c7→c9→c8→c0. Source starts c0 and asks for eight transfers; base asks from a different starting node. Both graphs are ten-node single rings. The source-start path is identical through seven transfers. Source answer c8, base answer, and base counterfactual c9 must all differ. The final-edge and eight-hop-from-source-start counterfactuals now agree by construction. This structural criterion is fixed without running the model.

Discovery: 8 new graph pairs, seed 2026092501, excluding arbitrary-graph discovery and reserved confirmation graphs. New confirmation: 64 pairs, seed 2026092601, also excluding all preceding graphs. Token slots remain strictly aligned.

Both runs use full J. Transfer source attention patterns, head outputs, or values into the base run; self-pattern and self-output controls. Initial scopes: all layers (positive/global comparison), late layers32–47, and local layers33–34, across calls2–4. No universal per-call hop assumption. All answers and pair acceptance are fixed before model execution; no model-success exclusion. Report crossed specificity for predefined base counterfactual versus source answer, including other answers and unchanged base answers.

A positive group result does not identify a single head. Any localization is a new discovery step followed by held-out confirmation, full-name generation and control patterns.

## Prospective localization after the eight-pair group result
Eight-pair late-region signature: pattern→counterfactual7/8; output→source8/8. Local33–34 pattern→counterfactual7/8; output→counterfactual5/8 and source2/8. Search whether a smaller component reproduces useful aspects without assuming all head outputs encode final content.

Use only the same shared-path discovery8. Scan all16 heads in layers33,34,35,39,43,47. Patch one head at one shared layer jointly across calls2–4 (not one call), with both self controls and source pattern/output patches. Site[-1,layer,head] denotes joint calls2–4. Retain every outcome, including route-like output signatures and failures. No head selection on reserved64 confirmation. No global-optimality claim outside scanned layers.
