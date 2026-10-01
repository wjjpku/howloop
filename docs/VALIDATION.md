# Validation record for arXiv:2609.39892v1

The pinned [public preprint](https://arxiv.org/abs/2609.39892v1) is the 36-page `paper/arxiv_v1.pdf`. Its `paper/main.tex` source has SHA-256 `a899f438566e5365e752875aceb2e4cb48898e0477cb96365ea3a7c0a75ce562` and references 21 unique figure assets. `provenance/figures.json` has one row for every asset. The TeX source and figure PDFs were checked against the arXiv v1 source package; Figure 8 is the one figure asset changed from the earlier submission snapshot.

`python reproduce.py` generated all 21 referenced assets and passed the seed-6 two-step and original eight-step saved-prediction checks. It also audited arXiv v1's independent Figure 8 confirmation: 512 additional graphs, 3,175 selected examples, five backbone seeds, two controller fits each, and 3,000 graph-bootstrap draws. The new graph set is disjoint from the earlier composition graph set. These are saved-result and plot checks, not a new GPU inference run.

The matched-supervision audit checked five paired runs and 60 map fits. The three-backbone mechanism aggregate, other saved-result families, and Parity phase audit also passed. Appendix H saved-result audits recomputed the Qwen four-branch table from 5,120 generated-answer records and the KG table and curve from 32,768 paired-query records. The exact file set is checked by `scripts/verify_integrity.py`.

Large checkpoints are excluded. Full independent training and every GPU intervention therefore remain outside this CPU-only validation. Locally generated figure PDFs need not have byte-identical encodings to the PDFs in the arXiv source package.
