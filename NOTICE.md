# Source and attribution record

This package contains the anonymous submission source, experiment code, numerical inputs, and validation scripts. Model weights are external except for the small dense-controller bundles under `experiments/dense_routing/local/`.

The Ouro-2.6B model configuration, tokenizer, and implementation under `vendor/remote/models/Ouro-2.6B` retain upstream attribution and their original notices. Conference template files under `paper/` retain their original notices.

Machine-specific path roots in text files were replaced with neutral placeholders for this export. The source code and archived result records may therefore have byte hashes different from the original run manifests. `provenance/SHA256SUMS.json` checks the exact contents of this export. Original model-weight hashes in `provenance/checkpoints.json` are unchanged.

The graph-mechanism and eight-node graph runtime directories share the N10 code snapshot. This export stores that shared snapshot once and keeps the experiment-specific files at their respective paths. `scripts/prepare_runtime.py` reconstructs each runtime directory from those files.
