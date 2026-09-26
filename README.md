# PaperExperiment — anonymous reviewer reproduction archive

This archive is pinned to the revised ICLR 2027 `paper/main.tex` (SHA-256 `26e0a17d82c698374cfb1b308826bf978a32b4003937ede03ef379d05d71413c`). The accompanying `paper/submission.pdf` is 34 pages, and the source references 21 figure assets. This version includes the Qwen3-8B and synthetic KG experiments in Appendix H.

From the extracted `PaperExperiment` directory, use Python 3.12:

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

`reproduce.py` generates all 21 manuscript figure assets under `outputs/figures/` and checks the saved Qwen and KG predictions. The Qwen check recomputes the four suffix branches from 5,120 prompt-level records. The KG check recomputes all 32 length points from 32,768 paired queries and the four table rows in Appendix H. Figure 1 is supplied conceptual artwork; the 20 other assets are generated from archived numerical inputs. Generated PDF encodings need not match the paper's embedded PDFs byte for byte.

`REVIEWER_GUIDE.md` is the short reviewer walkthrough. `docs/EXPERIMENTS.md` maps every paper result to evidence and code; `provenance/figures.json` maps each figure. Large checkpoints are not included. The CPU checks validate archived predictions and aggregation, not an independent GPU retraining. Original-weight requirements and limits are in `docs/RUNNING.md` and `docs/REPRODUCIBILITY_LIMITS.md`.
