# Registered 12-backbone horizon decomposition

This is a post-hoc decomposition of the already locked and completed
`paper2027.graph.g4.disjoint_aggregate.v1` evaluation. It does not rerun or
select models.

- Population: every completed registered backbone (12/12), both controller
  replicas per backbone, and all 512 final-lock graph permutations.
- Primary metric: strict successor accuracy, excluding cases where successor
  and endpoint coincide.
- Fixed intervals: calls 9--16 (training-covered continuation) and calls
  17--128 (strictly beyond the controller's training horizon).
- Uncertainty for the two fixed AUCs: nested bootstrap over backbone,
  controller replica, and graph permutation, in that order, with 10,000
  draws. The per-call curve is descriptive and uses a bootstrap over the 12
  independent backbone means after averaging registered replicas and graphs.
- Controls shown on the curve: raw backbone, executor-off, and batch-shuffled
  boundary maps. Controls have one registered controller replica per backbone.

The per-call curve is descriptive. The two fixed horizon AUCs are the planned
summary statistics for revising the paper's finite-continuation claim.
