# Ouro semantic mechanism experiment: interim results

## Completed: H5 discovery pilot
Same frozen step-200 backbone and step-500 L4 map, checkpoint SHA verified. Eight token-aligned source/base pairs, each a ten-person directed ring, depth eight. Base and source answers and last-edge base counterfactual all distinct. All sixteen unpatched runs answer correctly.

At head 5 of ten sampled layers and calls 2,3,4: 240 paired interventions per patch type. Pattern patches produced base/source/counterfactual answers in 234/1/0 cases (5 other); output patches in 235/0/0 (5 other). These 240 interventions reuse eight pairs; they are NOT 240 independent graphs. All 240 self-output patches reproduce logits exactly; all 240 self-pattern reconstructions retain the base answer. Peak GPU memory 10.155 GiB, elapsed evaluation 209.38 seconds on a shared GPU.

Interpretation: this pilot gives no crossed semantic signature for the prespecified terminal-edge hypothesis. It does not establish that H5 has no role: the tested intervention acts at one call, while prior ablations cover multiple calls. It also does not test all heads or disprove nonterminal intermediate routing.

## In progress: all heads in layers 33/34
Same eight discovery pairs; all 16 heads, calls 2/3/4; same four patch arms. This region contains prior H5 and L33.H12/H14 candidates. See STATUS.json for live handle. A separate 64-pair confirmation set excludes discovery graphs and has not been evaluated. No paper claims changed.

## Completed: all-head discovery in layers 33/34
Eight pairs × 32 heads × 3 calls = 768 interventions per patch type (only eight independent pairs). Neither source-pattern nor source-output patches produced the respective predefined counterfactual/source target at any site. All self-output restorations are exactly equal in logits; all self-pattern controls retain the correct base answer. This strengthens the negative result for single-site terminal-edge transfer in the selected region, not for all possible distributed mechanisms.

## Same-input routing mediation
One-case calibration passes identity checks with peak10.155GiB. For this case, late-layer pattern rescue succeeds (correct-answer probability0.955), late-value rescue fails (probability0.0724). Full-network value rescue also succeeds and must be reported; the mechanism is not exclusively pattern-mediated everywhere. Eight-case discovery is running; no confirmatory claim yet.

## Prospectively prepared paired-graph refinement
SHARED_PATH_PROTOCOL.md specifies a graph pairing whose source-start seven-edge prefix is shared, with different eighth recipients. This makes the terminal-edge and full-source-start base counterfactual identical. It is a separate experiment; it does not rewrite the arbitrary-graph pilot's targets or erase its negative result.

## Completed: eight-case mediation discovery
Native0/8; fullJ8/8. Late layers32–47 (all heads, calls2–4): pattern rescue8/8, output rescue8/8, value rescue3/8, reverse pattern damage0/8. Early pattern rescue1/8, middle0/8, local33–34 region0/8. Full-network pattern rescue8/8, value rescue4/8. All self-output logits exact; all self-pattern conditions remain0/8 (max logit discrepancy0.125 from reconstruction). These are exploratory eight-case results only.

## Complete-generation implementation check
One discovery case: first generated token matches direct forward in all8 generation arms, no truncations. Native answer David vs target Iris; late-pattern rescue generates Iris; unrelated-pattern control Henry and wrong-call control Grace. Measured peak10.4875GiB, elapsed61.58sec.

## Shared-prefix semantic pilot (one case only)
Both clean runs correct. Across late layers32–47 and calls2–4, pattern patch produces base counterfactual(probability0.980), output patch source answer(probability0.982). Local33–34 pattern AND output patches both produce counterfactual, so locality does not yet reproduce the route/content separation. The remaining7 discovery cases are not evaluated yet.

## Running
Locked64 mediation confirmation, protocol/code/data hashes in remote manifest. See STATUS.json. Do not treat the n=1 shared-prefix pilot or n=8 mediation screen as independent confirmation.
