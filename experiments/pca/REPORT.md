# One-hop versus two-hop hidden-state PCA

N10 Graph A; exact backbone/map hashes in manifest.json. No paper edits.
512 training-excluded reserved graphs, all 10 starts, both independent map fits; no success/collision filtering. Training/test split is by graph (256/256). PCA axes fit on training graphs only, separately per stage and representation; plots contain held-out graphs. No feature scaling before PCA. Probe uses standardized logistic regression; reported accuracy classifies steering condition, not task answer correctness.

Panels: (a) shared h6 before J; (b) J_one(h6) versus J_two(h6); (c) F(J_one(h6)) versus F(J_two(h6)). Axes are independently fitted per panel; do not read spatial motion across panels. Gray a is one shared state cloud, not two classes that were separated before intervention.

Token-mean PCA cleanly separates conditions (100% test classification in PC1/2 for both fits, before/after F). Answer-token PCA overlaps (about 55% before F and 50% after F), while the full 256-dimensional answer state is over 99.9% linearly classifiable. Across-map-fit transfer is 98.75–99.79% for the full answer state. This supports a robust distinction between steered state distributions, not a proven one-dimensional causal control variable. A common offset introduced by different affine maps may dominate token-mean PCA. Causal control-direction claims would require interventions on the discovered direction.

All scores: summary.json. Activations: states.npz. Full extraction/analysis scripts included.
