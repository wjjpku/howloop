# Source and attribution record

The repository's original experiment code, plotting code, archived numerical inputs, and documentation are released under the root MIT license. The [arXiv v1 manuscript](https://arxiv.org/abs/2609.39892v1), its figure PDFs, and the README overview image derived from Figure 1 are distributed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Third-party files retain their respective rights and notices; the root MIT license does not replace those terms. Large model weights are not distributed here.

`vendor/remote/models/Ouro-2.6B/` contains model configuration, tokenizer files, and implementation from [ByteDance/Ouro-2.6B](https://huggingface.co/ByteDance/Ouro-2.6B), which identifies its license as Apache-2.0. Its license text is included at `vendor/remote/models/Ouro-2.6B/LICENSE`. Copyright and license headers in individual upstream files remain in place. LaTeX template and package files under `paper/` retain their embedded notices.

Machine-specific path roots in the current files were replaced with neutral placeholders for this export. The source code and archived result records may therefore have byte hashes different from the original run manifests. `provenance/SHA256SUMS.json` checks the exact contents of this export. Original model-weight hashes are recorded in `provenance/checkpoints.json` and `provenance/additional_controls.json`.

The graph-mechanism and eight-node graph runtime directories share the N10 code snapshot. This export stores that shared snapshot once and keeps the experiment-specific files at their respective paths. `scripts/prepare_runtime.py` reconstructs each runtime directory from those files. `docs/REPRODUCIBILITY_LIMITS.md` distinguishes saved-result audits from fresh GPU execution.
