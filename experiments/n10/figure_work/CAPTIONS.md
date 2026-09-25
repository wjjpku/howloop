# Section 3 figure preview

## Trajectories
**Readout trajectories on N10 cycles.** Each panel reports the fraction of decoded predictions matching each node category $f^k(s)$ after 0–16 calls to the frozen recurrent block $F$, evaluated on 512 held-out ten-node cycles with all ten starting nodes. The dashed line marks the trained recurrence budget. These are descriptive examples, with no filtering by prediction success. Intermediate readouts do not by themselves establish the number of graph operations performed internally.

The selected panels are D8L8 seeds 10 and 5, and D8L6 backbones A and E. These examples were selected for visual clarity and contrasting readout paths, from twelve D8L8 and five D8L6 seeds; they are not an unbiased summary of trajectory frequencies. D8L8 seed 5 reads out the starting node after the first F call, then predominantly advances by two nodes per call until reaching the task endpoint. D8L8 seed 10 predominantly advances by one node per call through call 8, then reads out f^9(s) after the training horizon. All panels use the complete locked cycle evaluation set, without selecting individual examples.

## Post-F target control
**One- and two-hop control from the same frozen state.** For N10 D8L6 backbone A, the three rows show decoded outputs from $F(h_6)$, $F(J_{one}(h_6))$, and $F(J_{two}(h_6))$. Relative to the task endpoint $u=f^8(s)$, predictions are categorized as stay ($u$), one hop ($f(u)$), two hops ($f^2(u)$), or others. We retain the same 4,110 examples where these three labels are pairwise distinct, without filtering by prediction success. Each controlled row averages two independently trained controllers; the no-J baseline is counted once. The figure shows outputs after F only.

These controller fits are not independent backbone replications. Per-fit counts are retained in control_counts.csv, and all five backbones are shown in the supplementary control figure.
