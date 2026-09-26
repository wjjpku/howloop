# Reviewer quick start

This ZIP corresponds to the 34-page manuscript in `paper/submission.pdf`. The source is `paper/main.tex`, SHA-256 `26e0a17d82c698374cfb1b308826bf978a32b4003937ede03ef379d05d71413c`; it references 21 figure assets, including the new KG figure in Appendix H.

With Python 3.12, run these commands from the extracted `PaperExperiment` directory:

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

`docs/EXPERIMENTS.md` connects figures, tables, protocols, and archived inputs. `provenance/additional_controls.json` records the Qwen/KG checkpoint hashes and source adaptations. The large original model checkpoints are excluded from this ZIP, so the CPU commands verify saved predictions rather than rerun training or every GPU forward pass. The regenerated Figure 4 schematic is simplified; the numerical panels use archived aggregate data.
