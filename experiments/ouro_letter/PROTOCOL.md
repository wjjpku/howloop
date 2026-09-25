# Ouro semantic-patching discovery, 2026-09-23

Frozen step-200 Ouro-2.6B and L4 step-500 affine J; four calls, J before calls 2–4. No training. Objective: test route/content separation, localized recovery, and routing mediation, including negative outcomes.

## Stage 1: semantic-patching discovery
- Fixed ten-node directed cycles, depth 8, template 0. Source/base share rule source ordering and token boundaries. Every semantic slot must have identical token positions and length. Token identity may change. Random seed alone does not imply graph disjointness; archive graph edge identities and reject repeated source/base graphs across splits.
- Primary alternatives fixed before patching: base answer, source answer, and base successor of the source's seventh-hop holder. All three must differ. This is a hypothesis about the last edge read, not an assumption that each recurrent call is one hop.
- Secondary diagnostic: base eight-hop answer from source start. Kept distinct from primary; do not select whichever target fits results.
- All source/base examples retained regardless of model accuracy. Report an additional clean-both-correct stratum separately.
- Source-pattern patch computes alpha_source V_base; source-output patch imports z_source before o_proj. Same call/layer/query positions. Subsequent computation reruns. Layer scan at head 5 is exploratory, not a claimed localized head.
- Coarse scan: zero-based layers 8,16,24,32,33,34,35,40,44,47; calls 2,3,4; initial pilot patches head 5 at all query positions at exactly one site (same tensor workload as the archived H5 test); expand head candidates after measuring this pilot. Refine heads and token positions only after this scan.
- Eight discovery pairs, seed 2026092301. Confirmation pairs use independent seed 2026092401 and exclude all discovery graphs. Confirmatory settings will be frozen in a separate file before evaluation.
- Report first-token full-vocabulary argmax and probability/margins for three distinct names. First tokens must be distinct. Full-name generation needed for final confirmation.
- Identity output patch must reproduce logits exactly. Pattern reconstruction checked separately against native SDPA (numerical precision difference documented); self-pattern control paired with source patch.
- Before positive interpretation require crossed specificity beyond self-patch, adjacent-head and unrelated-source controls. Failure is evidence, not grounds to silently redefine counterfactual.

## Follow-ups
Candidate head corrupt–restore using alternative-start queries, then same-input J/raw pattern rescue and damage. Strong damage alone will not be called sufficient mediation. No existing H5 site is assumed optimal.

## Resource/provenance
All outputs under /data/wujiaju/ouro_semantic_20260923; logs under /data/wujiaju/logs. Explicit GPU pinning, no modifications of other processes, no new training/checkpoints. Archive SHA256 identities, exact prompts, graph pairs, settings, code hash, raw per-condition outputs, peak memory and process state.

## Exploration expansion (specified before expanded-head results)
After the initial H5 implementation/pilot, scan all 16 heads in shared layers 33 and 34, separately at calls 2,3,4, using the same eight discovery pairs. This region contains the prior causal H5 and L33.H12/H14 candidates. No claim of exhaustive global head search. All sites receive both self controls and source pattern/output interventions. Candidate selection uses crossed specificity, not only accuracy damage. The 64 confirmation pairs are generated but will remain unevaluated until site and contrasts are frozen. If no crossed candidate appears, report the negative terminal-read test and examine other routing hypotheses explicitly rather than relabeling the counterfactual.
