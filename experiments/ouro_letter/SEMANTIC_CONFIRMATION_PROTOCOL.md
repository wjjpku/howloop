# Locked shared-prefix semantic confirmation

Locked after 8 discovery pairs: late-pattern counterfactual7/8; late-output source8/8; all clean base/source runs correct. This is independent from the64-instance mediation cohort.

Data: shared_path_confirmation.json, seed2026092601,64 pairs, excluding all discovery and other reserved graphs. Ten-node directed cycles with a common source-start path for seven steps and different eighth recipients. Base query starts elsewhere. All base/source/counterfactual answers and their first token IDs differ. No model-success exclusion.

Primary scope fixed: all heads, layers32–47, calls2–4, all prompt positions. Full J in all source and receiving runs. Pattern transfers alpha_source V_base; output transfers z_source; V-only transfers alpha_base V_source. Main contrasts: pattern-versus-output counterfactual rate and output-versus-pattern source-answer rate. Conditions also include base self-output/self-pattern, early and middle patterns (equal-size16-layer controls), local33–34 patterns and outputs, cyclically misaligned source-call patterns. Baselines: original base, source, and natural counterfactual query (base graph asked from source start) to verify the counterfactual is computable by the unchanged model. Source-dependent, oracle activation patching is not a deployable controller.

Primary readout first-answer-token full-vocabulary argmax; full-name greedy generation for base/source/counterfactual baselines, primary late pattern/output/value, local pattern/output, and wrong-call pattern. Prompt-prefill-only patches, J remains active during generation; max_new_tokens16, truncations recorded, generated first token must match direct forward. No intervention-site search on these64 pairs.

Report paired differences with10000 paired bootstrap draws, full four-way outcomes (base/source/counterfactual/other), all-self numerical deviations, and no exclusions. One checkpoint/map and one controlled graph construction: does not imply arbitrary-graph semantic patchability or a single-head circuit. Negative arbitrary-graph discovery is retained.
