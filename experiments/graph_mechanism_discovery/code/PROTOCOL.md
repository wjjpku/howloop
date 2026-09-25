# N10 graph migration, 2026-09-23

This is a fresh N10 campaign. Never relabel archived N8 outcomes.

## Registered cohorts
- D8L6 local control: seeds 6, 3, 4, 5, 7; final-only CE, uniform requested depth 1–8, six shared-block calls.
- D8L8 descriptive trajectories: seeds 0, 1, 6, 8; the same uniform-depth final-only task, eight calls.
- D8L8 continuation: seeds 100–111; fixed depth 8 final-only CE, eight calls.
- All main backbones: two blocks, width 256, four heads, MLP width 1024, 20,000 updates, batch 512, AdamW lr 3e-4, weight decay .3, warmup 500, gradient clip 1, bf16 autocast. Earliest best D8 selection accuracy selects the checkpoint; final checkpoint is also retained. This selection rule and explicit holdouts are redesign changes, not a pure node-count causal comparison.
- Initialization uses seed + 1009*L for the uniform-depth cohorts and the registered seed for the fixed-depth cohort, preserving the respective source conventions.

## Data and exclusion
Before training, create 128 selection graphs, 32 discovery graphs, 512 confirmation permutations, 512 ten-node cycles, and four smoke graphs, all unique and mutually disjoint. All single-output-swap donors of discovery/confirmation/smoke graphs are also excluded from all backbone and J training. Total held-out graph union: 25,720. Train sampling rejects the graph's int64 base-10 permutation code. Manifest hashes identify exact locks; seeds alone do not establish disjointness.

## Local controller and mechanism
Five D8L6 backbones each get two fits (seeds 1,2) for each target f9/f10. J = D+AB+b, rank48, all tokens, identity initialization via random A/zero B. Freeze F and decoder; optimize pure post-F CE for 8000 updates, batch128, AdamW lr1e-4 wd0, clip1, validation every400 on the selection lock; earliest best accuracy. Seed is set before J initialization. One-hop and two-hop fits are matched except for the target. Test each saved map before/after F and with graph-cluster shuffle.

All 4 layer-2 heads are compared on discovery only: maximize eligible J_one pattern-to-base-counterfactual accuracy averaged across both fits, ties by smaller head. Freeze the head before confirmation. Retain empty eligible cohorts explicitly. Confirmation uses all ten currents per graph; report both unconditional target behavior and clean-readout-conditioned semantic patch outcomes. Same-input pattern rescue/damage has no success filter. Single-hop and two-hop results are never pooled. All head, pattern, output and value patches keep physical position 34 as the answer token (sequence length35); attention scale sqrt64 stays8, depth stays8.

## Continuation
Twelve separate fixed-depth D8L8 backbones, two rank48 pure on-policy CE maps each. Fit all calls1–16 using one shared map before each F, 8000 updates, batch128, lr1e-4, selection by mean successor CE. Evaluate through call128 on the full confirmation set with collision-excluded scores and ordinary scores, separately report the topology-defined ring subset. Fit1 also runs D-only, no-AB, identity-D, AB-only, no-bias, mean-D, shuffled-D, spectrum-matched random delta, executor-off and graph-cluster shuffle; raw and full for both fits. The graph ring period is10, so an endpoint-holding baseline must be recomputed per window, not replaced with a universal10%.

## Figure and paper
Section3 should contain two D8L8 and two D8L6 illustrative trajectory panels. Retain all registered model outcomes; panel choice is descriptive, not confirmation. Show node-category match across calls0–16 and ten node categories. F-call index is not semantic hop count. No new numbered paper version; compile an unnumbered preview only after the corresponding figures and numerical statements have actual N10 support.

## Additional appendix migration
Restricted-controller classes, matched final-only/per-loop supervision, regularization sensitivity, and the separately trained single-backbone long-horizon case are separate pipelines. Their historical numbers must not silently enter the N10 paper. The main backbone queue does not imply these pipelines have run. Ouro, parity, and KG are separate tasks and remain unchanged.


## Supplementary execution details
Restricted controllers: 4 classes x 3 fits x 5 backbones, 4608 parameters each, 800 updates, lr1e-3; 2048 unique train graphs outside all locks, the128-graph selection lock, first256 confirmation graphs, all10 currents. Direct saved-checkpoint replay must agree exactly.

Sensitivity: Backbone A gets two regularized fits per target, source algorithm unchanged: dense residual regression initialization,400 CE+10 normalized state-MSE+pre-call CE updates, then residual SVD48. Training/calibration graphs exclude all locks. This uses the pure-CE-discovered locked head and the same512 confirmation graphs. Random training-domain diagnostics are labelled separately.

Supervision: new paired N10 shuffled-edge fixed-D8 models, seed3 with identical initial weights and minibatch stream,20k updates each, final-only vs per-call f^t target CE. Use final checkpoints. For each backbone, fit one answer-token dense map shared at eight post-F boundaries,5000 updates, for target depths9/10/12/15 x seeds0/1/2. The fixed8 source objective/optimizer is retained; splits are redesigned using shared N10 locks. Requested target depth is not the number of F calls.

Separate h64 case: a redesigned curriculum on the registered trajectory L8seed0, initialized at identity, rank48 all-token J, native h8 prefix. Fixed continuation horizons1/2/4/8/16/32/64,1000 updates per stage,batch32,AdamW lr1e-4,mean successor CE over all covered continuation calls. Evaluate after native h8 for128 continuation calls, with always-J, J-through16, raw schedules. This is explicitly NOT budget- or initialization-matched to the old phase-aligned affine h64 fit. The curriculum covers64 continuation calls; only later calls qualify as beyond the fitted horizon. No accuracy-based stopping or selection of a successful backbone.
