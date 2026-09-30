# howloop

Reproduction archive for **“Shared Weights, Selected Computations: How Looped Transformers Route What Each Loop Does.”** It contains the manuscript snapshot, experiment and plotting code, archived numerical evidence, and scripts that check the reported results.

The included [manuscript PDF](paper/submission.pdf) has 34 pages. Its [source](paper/main.tex) references 21 figure assets. The archive covers the graph, Ouro, Parity, long-range, matched-supervision, Qwen3-8B, and synthetic knowledge-graph experiments in that snapshot. [The experiment index](docs/EXPERIMENTS.md) maps each reported result to its inputs and code.

## Quick start

Use Python 3.12 on Linux or macOS. From a clone:

```bash
git clone https://github.com/wjjpku/howloop.git
cd howloop
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-plot.txt
python scripts/verify_integrity.py
python reproduce.py
```

If using a downloaded ZIP, start at `python3 -m venv .venv` inside its extracted directory. `verify_integrity.py` checks the file manifest and the manuscript-to-figure mapping. `reproduce.py` writes the manuscript figure assets to `outputs/figures/` and checks the archived Qwen and KG predictions. Figure 1 is supplied conceptual artwork; the other 20 assets are generated from archived numerical inputs. PDF files generated locally may differ byte-for-byte from those embedded in the manuscript.

Run the remaining saved-result checks with:

```bash
python scripts/audit_results.py
python scripts/audit_parity_phase.py
python experiments/revision/audit_paper_matched.py
python experiments/revision/mechanism_aggregate.py
```

The Qwen audit recalculates four suffix branches from 5,120 prompt-level records. The KG audit recalculates 32 length points from 32,768 paired queries and the four Appendix H table rows. [REVIEWER_GUIDE.md](REVIEWER_GUIDE.md) gives the short walkthrough; [docs/VALIDATION.md](docs/VALIDATION.md) records what was checked for this version.

## What this archive can verify

| Material included | What you can check |
| --- | --- |
| Manuscript source, figure inputs, and plotting scripts | Regenerate all 21 figure assets referenced by the manuscript. |
| Saved predictions and result records | Recalculate reported counts, aggregates, Qwen and KG tables, and matched-run checks. |
| Training and evaluation source snapshots | Inspect protocols and prepare original-weight runs using [docs/RUNNING.md](docs/RUNNING.md). |

Large model checkpoints are **not included**. The CPU commands above verify archived predictions and their aggregation; they do not independently repeat training or every GPU forward pass. Checkpoint identities and hashes are recorded under [`provenance/`](provenance/). [Reproducibility limits](docs/REPRODUCIBILITY_LIMITS.md) distinguish these checks from full original-weight execution.

## Citation and license

Cite the paper and this repository at **https://github.com/wjjpku/howloop**. GitHub's “Cite this repository” menu reads [CITATION.cff](CITATION.cff). Repository-authored code and documentation use the [MIT license](LICENSE); the manuscript and third-party components retain their own terms, as described in [NOTICE.md](NOTICE.md).

This archive is pinned to [`paper/main.tex`](paper/main.tex) with SHA-256 `26e0a17d82c698374cfb1b308826bf978a32b4003937ede03ef379d05d71413c`. The complete file manifest is [`provenance/SHA256SUMS.json`](provenance/SHA256SUMS.json).
