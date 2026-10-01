# Experiment-to-paper index

Scope is fixed by [arXiv:2609.39892v1](https://arxiv.org/abs/2609.39892v1), `paper/main.tex`, and `provenance/figures.json`. The 21 TeX figure references are all mapped there. Seed and cohort identities are kept separate below.

| Paper result | Archived evidence and code |
|---|---|
| Fig. 2 and trajectory appendix | `experiments/revision/trajectories/`, `experiments/selected_mechanism/heatmaps/`, `plots/revision_figures.py`; D8L8 seeds 0–11 and D8L6 seeds 3–7 |
| Fig. 3 target selection and two-step composition | `experiments/n10/`, `experiments/composition/A_fit{1,2}.npz`, `plots/fig3_submission.py`; seed 6, 4,110 target examples and 3,200 composition examples per fit |
| Fig. 8 eight-step composition | `experiments/revision/arxiv_v1_composition/` contains the independent 512-graph confirmation, ten saved backbone/map fits, 34 fixed sequences, `summary.json`, and `analyze.py`; seed 6 supplies the plotted curve on 3,175 retained examples. The earlier 512-graph composition set remains under `experiments/revision/long_composition/` and is checked by `scripts/audit_revision.py`. |
| Fig. 4 Graph mechanism | `experiments/graph_mechanism/` for seed 6, `experiments/selected_mechanism/` for seeds 10 and 13, `experiments/revision/mechanism_aggregate.py`, `experiments/target_exchange/`; do not combine model seeds as controller replicates |
| Fig. 4 Ouro mechanism | `experiments/ouro_256_steering/`, `experiments/ouro_256_mechanism/`; 256-pair records and analysis scripts |
| Fig. 9 and matched-supervision appendix | `experiments/revision/matched/`, `experiments/revision/{matched_train,matched_maps,matched_campaign,audit_paper_matched}.py`; five final-only/stepwise pairs, 60 map fits, 4,137 common test examples |
| Figs. 5–6 | `experiments/graph_continuation/`, `experiments/long_range/`, `experiments/parity_phase/`, `plots/fig5.py`, `plots/fig6.py` |
| Fig. 7 | `experiments/ouro_antonym/`, `plots/fig7_submission.py` |
| Appendix H, Qwen table | `experiments/qwen/results/predictions.jsonl` and `panel.json`, `experiments/qwen/audit_saved.py`, training and four-branch evaluation code in `experiments/qwen/code/`; five tasks, 1,280 paired prompts per branch |
| Appendix H, KG table and Fig. H | `experiments/kg/results/key_cells_1024.jsonl`, `experiments/kg/audit_saved.py`, `plots/figH_kg.py`, training and evaluation code in `experiments/kg/code/` and `core/`; lengths 1–32, 1,024 paired queries per length |

The `experiments/ouro_letter/` directory holds the related letter-walk mechanism evidence referenced in the manuscript. Original checkpoint identities and limitations are documented under `provenance/` and `docs/REPRODUCIBILITY_LIMITS.md`. Old-version dense-routing, head-subset, and native-layer control packages are excluded from this version.
