# Validation record

The pinned source is `paper/main.tex` SHA-256 `26e0a17d82c698374cfb1b308826bf978a32b4003937ede03ef379d05d71413c`. It locally compiles to a 34-page PDF and references 21 unique figure assets. `provenance/figures.json` has one row for each asset.

The CPU reproduction command generated all 21 assets. The seed-6 two-step audit and eight-step per-prefix audit matched the saved predictions. The matched-supervision audit checked all five paired runs and 60 map fits; the three-seed mechanism aggregate was recomputed. Other saved-result and Parity phase audits passed. `scripts/verify_integrity.py` validates the final packaged hashes. These checks verify archived numerical evidence and regenerated plots, not new GPU training.

The original large checkpoints are excluded. A full independent re-execution of training and every GPU intervention is therefore outside this ZIP-only verification. The regenerated Figure 4 schematic is a simplified rendering; the paper's supplied figure remains in `paper/figures/`.

Appendix H saved-result audits recomputed the Qwen four-branch table from 5,120 generated-answer records and the KG table from 32,768 paired-query records. The KG length-curve PDF was regenerated from the same query records. Full original-weight GPU evaluation and training were not rerun.
