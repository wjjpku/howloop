# One Set of Weights, Many Algorithms — Manuscript v66

This is the current manuscript source for **One Set of Weights, Many Algorithms: How Looped Transformers Route Computation Across Loops**.

Upload this folder's contents. Set `main.tex` as the main document and use **pdfLaTeX**. Overleaf runs BibTeX automatically.

- `main.tex`: the full manuscript and appendices, with comments at section and figure boundaries.
- `references.bib`: the only bibliography database.
- `figures/`: all required vector figures, in a single directory. Filenames follow the manuscript figure numbers.
- `iclr2027_conference.sty` and `.bst`: conference formatting and bibliography style.
- `natbib.sty` and `fancyhdr.sty`: dependencies supplied with the source package.

## Figure files

| Figure | Files | Purpose |
|---|---|---|
| 1 | `fig01_overview.pdf` | Main overview |
| 2 | `fig02a`–`fig02d` | Four intermediate-readout panels |
| 3 | `fig03a`, `fig03b` | Target accuracy and PCA |
| 4 | `fig04_attention_mechanism.pdf` | Attention schematic and intervention results |
| 5 | `fig05a`, `fig05b`, `fig05_legend` | Parity and graph continuation, shared legend |
| 6 | `fig06_parity_timing.pdf` | Readout timing |
| 7 | `fig07_supervision_capacity.pdf` | Supervision and controller comparison |
| 8 | `fig08_graph_seeds.pdf` | Additional graph backbones |
| 9 | `fig09a`–`fig09c` | Additional parity seeds |

Panel files remain separate where LaTeX assembles them and maintains subfigure references. No historical figures, plotting scripts, or build outputs are included. Edit captions in `main.tex`; preserve existing labels so cross-references keep working.

## What to edit

- Prose, headings, captions, tables, cross-references, and layout: `main.tex`.
- Citation metadata and new references: `references.bib`.
- Figures: replace the relevant PDF only when changing that figure.
- Keep the four `.sty` / `.bst` files unchanged unless the conference template must change.
- Update this README or `.gitignore` only when project instructions or structure change.
