# Query corruption and head-output restoration in Ouro

Prospective local-usefulness test. Fixed site: shared layer34, H5 (zero-based), selected in the prior independent Ouro study, not selected on the new confirmation cohort. All16 single heads are measured for comparison, plus restoration of all heads. Calls2–4 jointly, prompt prefill only; J retained throughout.

Use base graph/query from the shared-path dataset. The corruption source is the natural counterfactual run: same graph, different starting person, same eight-hop request. Graph statements and token positions are identical, only query start differs. Base and alternative-start answers are distinct by construction.

At layer34, replace all query vectors Q with the alternative-start run's Q (post rotary processing at the attention backend, same call), retaining receiving K,V and recomputing attention with the normal SDPA backend. Restore the clean base z=alphaV at H5, each of the15 other heads, or all16 heads before o_proj. No substitution of residual states or other layers. All-head restoration must recover original logits exactly under this intervention schedule.

Report unfiltered accuracy and the prespecified conditional cohort: originally correct base runs made incorrect by query corruption. Recovery is the fraction of this broken cohort returned to the base answer. Report how many examples enter this cohort; no conditioning on restoration success. Best other head is a conservative comparison, not a selected confirmatory head. H5 superiority is not guaranteed and will not be assumed if absent.

Discovery uses shared-path discovery8. Confirmation will use a separate64-pair set seed2026092701, excluding all preceding graph identities, with no threshold/site changes. Complete generation for base, corruption, H5 restore, neighboring H4 restore, and all-head restore. First generated token must match the forward prediction; generation uses receiver J with no generated-token patch. This is conditional repair, not an autonomous error-correction controller.
