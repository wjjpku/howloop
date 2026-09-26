# Dense affine matched rerun

User authorized replacement of low-rank J by dense affine J for all mechanism/control experiments. Existing Ouro letter-walk J is already dense and is not retrained. No backbone training; no paper replacement before comparison.

Five frozen N10 D8L6 backbones A–E (seeds 6,3,4,5,7), two controller seeds, two target maps each. Dense row-vector map hW+b, 256x256 W and 256 bias (65,792 parameters), identity initialization. Existing D+AB+b rank48 maps initialize to the same identity function. Optimization parameterization necessarily differs. Preserve original training sampler, exclusions, seed, 8,000 updates, batch128, AdamW lr1e-4, weight_decay0, clipping1, validation every400, earliest best validation checkpoint. Compare training token-stream hashes with archived matching fit; fail on mismatch.

Same confirmation graphs: complete 168-condition matrix (all directed raw/one/two transfers, Q/K/V/pattern/output/MLP/block inputs, layers1/2/both), two-step compositions including omission controls, and original corruption/restoration/cross-graph/mediation battery. Run the battery for all four heads; primary comparison uses original locked discovery head, not a head selected on dense confirmation. Distinct-label scoring masks and eligibility denominators retained; report changed eligibility and paired intersections when comparing cross-graph/restoration effects. Fresh-corrupted donor implementation with n10_selected_mechanism_20260924/datasets.json is reused.

Storage: /data/wujiaju/dense_affine_mechanism_20260925; logs /data/wujiaju/logs/dense_affine_*.log. Artifacts kept separately from originals. GPU0: A,C; GPU1: B,D; GPU3: E. Each GPU exclusively empty at preflight. Expected output under 1 GB. No automatic reminder created.
