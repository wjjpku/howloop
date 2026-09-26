# Multi-head attention-pattern substitution
Frozen N10 seeds3/5/7, existing dense J_one. No training. Test 256 subsets of 8 heads (2 layers, 4 heads each) within the seventh F call. Only post-softmax attention patterns are patched from the full J_one run. Recipient values are recomputed naturally; they are never directly patched. First-layer patches can indirectly change later values and queries.

Four families: answer query only vs all query positions; untouched residual vs prior fixed residual attenuation (B L2 alpha0, D L1 alpha.05, E L2 alpha0). These parameters are inherited, not retuned. Thus 1024 conditions including empty subsets. Full-J and native baselines included. Primary outcome one-hop f(u), u=f^8(s); common inclusion requires u,f(u),f²(u) distinct, with no correctness filtering.

Fresh discovery64 and confirmation256 graphs sampled from the locked donor pool after excluding prior named cohorts; 10 starts per graph. Discovery fit1 only. Before confirmation, select best subset at each size1..8 for each family, ties by mask. Retain all empty, individual-head, full-layer, and full8-head controls plus a subset-deletion control for each selected headset. Both original J fits evaluated on confirmation. Wrong-current donor patterns permute current nodes within the SAME graph, keeping recipient residual scaling unchanged.

Report all selected conditions, including failed combinations, same-layer vs cross-layer pairs, layer-wide vs selective substitution, and bootstrap graph-level uncertainty. Selection hashes frozen before confirmation. No claim of independent training replication from the two J fits; no claim that output recovery alone establishes a complete algorithm. No paper edits.

Before-confirmation amendment: since every unattenuated subset scored zero on discovery, validate ALL 256 subsets in both position scopes and both J fits on new graphs, rather than only arbitrary tie-selected subsets. Selection file updated before confirmation launch.
