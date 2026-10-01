# Reviewer quick start

This repository or its ZIP download corresponds to [arXiv:2609.39892v1](https://arxiv.org/abs/2609.39892v1), the 36-page manuscript in `paper/arxiv_v1.pdf`. The source is `paper/main.tex`, SHA-256 `a899f438566e5365e752875aceb2e4cb48898e0477cb96365ea3a7c0a75ce562`; it references 21 figure assets, including the KG figure in Appendix H.

With Python 3.12, run these commands from the repository root or extracted ZIP directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-plot.txt
python scripts/verify_integrity.py
python reproduce.py
python scripts/audit_results.py
python scripts/audit_parity_phase.py
python experiments/revision/audit_paper_matched.py
python experiments/revision/mechanism_aggregate.py
```

`verify_integrity.py` checks file hashes and confirms that every TeX figure reference has a provenance entry. `reproduce.py` writes the 21 paper assets to `outputs/figures/` and audits the two-step, eight-step, KG, and Qwen saved results. The KG audit checks 1,024 paired queries at each of lengths 1–32 and reproduces the four Appendix H table rows. The Qwen audit checks all four continuation branches on the same 1,280 prompts and reproduces the table's complete-answer counts. The Qwen result is a table and has no figure.

arXiv v1's Figure 8 uses an additional independent graph set. `reproduce.py` also checks its 3,175 retained examples, five backbones, two controller fits each, and saved per-call predictions against `experiments/revision/arxiv_v1_composition/summary.json`.

`docs/EXPERIMENTS.md` connects figures, tables, protocols, and archived inputs. `provenance/additional_controls.json` records the Qwen/KG checkpoint hashes and source adaptations. The large original model checkpoints are excluded from this ZIP, so the CPU commands verify saved predictions rather than rerun training or every GPU forward pass. The regenerated Figure 4 schematic is simplified; the numerical panels use archived aggregate data.
