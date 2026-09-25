from __future__ import annotations

from dataclasses import dataclass, replace

import torch


UNKNOWN = 0
ZERO = 1
ONE = 2

LEAF = 0
AND = 1
OR = 2
XOR = 3
XNOR = 4
MAX_BALANCED_DAG_ATTEMPTS = 2_048
TARGET_CANDIDATES_PER_ROUND = 1_024
MAX_CANDIDATES_PER_ROW = 64


@dataclass(frozen=True)
class BooleanDAGConfig:
    node_count: int = 24
    leaf_count: int = 6
    train_max_depth: int = 4
    eval_max_depth: int = 8
    balanced_output_xor: bool = False
    balanced_logic: bool = False

    def __post_init__(self) -> None:
        if self.leaf_count < 3 or self.node_count <= self.leaf_count:
            raise ValueError("the two-rail scaffold requires at least three leaves and one gate")
        if self.balanced_output_xor and self.leaf_count < 4:
            raise ValueError("balanced output XOR requires three data leaves and one mask leaf")
        required_gates = 2 * self.eval_max_depth - 1
        if self.gate_count < required_gates:
            raise ValueError(
                f"depth {self.eval_max_depth} requires at least {required_gates} gates"
            )
        if not 1 <= self.train_max_depth <= self.eval_max_depth:
            raise ValueError("train_max_depth must be in [1, eval_max_depth]")

    @property
    def gate_count(self) -> int:
        return self.node_count - self.leaf_count

    @property
    def invalid_parent_id(self) -> int:
        return self.node_count


@dataclass(frozen=True)
class BooleanDAGBatch:
    self_ids: torch.Tensor
    kinds: torch.Tensor
    parent_ids: torch.Tensor
    initial_states: torch.Tensor
    root_mask: torch.Tensor
    levels: torch.Tensor
    values: torch.Tensor
    root_values: torch.Tensor
    depths: torch.Tensor
    has_reused_root_ancestor: torch.Tensor

    @property
    def batch_size(self) -> int:
        return self.self_ids.shape[0]

    @property
    def node_count(self) -> int:
        return self.self_ids.shape[1]

    def to(self, device: torch.device | str) -> BooleanDAGBatch:
        return replace(
            self,
            **{
                name: value.to(device)
                for name, value in self.__dict__.items()
                if isinstance(value, torch.Tensor)
            },
        )


def _random_scores(
    shape: tuple[int, ...],
    *,
    device: torch.device,
    generator: torch.Generator | None,
) -> torch.Tensor:
    return torch.rand(shape, device=device, generator=generator)


def _evaluate_canonical(
    *,
    kinds: torch.Tensor,
    parent_indices: torch.Tensor,
    levels: torch.Tensor,
    leaf_values: torch.Tensor,
    max_depth: int,
) -> torch.Tensor:
    batch_size, node_count = kinds.shape
    leaf_count = leaf_values.shape[1]
    values = torch.zeros(batch_size, node_count, dtype=torch.long, device=kinds.device)
    values[:, :leaf_count] = leaf_values
    for level in range(1, max_depth + 1):
        left = values.gather(1, parent_indices[:, :, 0].clamp_max(node_count - 1))
        right = values.gather(1, parent_indices[:, :, 1].clamp_max(node_count - 1))
        xor = left ^ right
        computed = torch.where(
            kinds.eq(AND),
            left & right,
            torch.where(
                kinds.eq(OR),
                left | right,
                torch.where(kinds.eq(XOR), xor, 1 - xor),
            ),
        )
        update = levels.eq(level)
        values = torch.where(update, computed, values)
    return values


def _make_canonical_candidates(
    cfg: BooleanDAGConfig,
    depths: torch.Tensor,
    root_kinds: torch.Tensor,
    *,
    generator: torch.Generator | None,
) -> dict[str, torch.Tensor]:
    device = depths.device
    batch_size = len(depths)
    node_count = cfg.node_count
    leaf_count = cfg.leaf_count
    gate_count = cfg.gate_count

    kinds = torch.zeros(batch_size, node_count, dtype=torch.long, device=device)
    minimum_kind = XOR if cfg.balanced_logic else AND
    maximum_kind = XNOR + 1 if cfg.balanced_logic else XOR + 1
    kinds[:, leaf_count:] = torch.randint(
        minimum_kind,
        maximum_kind,
        (batch_size, gate_count),
        device=device,
        generator=generator,
    )

    gate_levels = (
        _random_scores((batch_size, gate_count), device=device, generator=generator)
        * depths[:, None]
    ).long() + 1
    for level in range(1, cfg.eval_max_depth):
        rows = depths.gt(level)
        first_gate = 2 * (level - 1)
        gate_levels[rows, first_gate] = level
        gate_levels[rows, first_gate + 1] = level
    root_gate = torch.where(depths.eq(1), torch.zeros_like(depths), 2 * (depths - 1))
    gate_levels.scatter_(1, root_gate[:, None], depths[:, None])
    kinds[:, leaf_count:].scatter_(1, root_gate[:, None], root_kinds[:, None])

    levels = torch.zeros(batch_size, node_count, dtype=torch.long, device=device)
    levels[:, leaf_count:] = gate_levels
    current_level = gate_levels[:, :, None]
    candidate_levels = levels[:, None, :]
    exact_previous = candidate_levels.eq(current_level - 1)
    any_previous = candidate_levels.lt(current_level)
    if cfg.balanced_output_xor:
        data_node = torch.arange(node_count, device=device).ne(leaf_count - 1)
        exact_previous &= data_node.view(1, 1, -1)
        any_previous &= data_node.view(1, 1, -1)

    first_scores = _random_scores(
        (batch_size, gate_count, node_count), device=device, generator=generator
    ).masked_fill(~exact_previous, -1.0)
    parent_first = first_scores.argmax(dim=-1)
    second_mask = any_previous & torch.arange(node_count, device=device).view(1, 1, -1).ne(
        parent_first[:, :, None]
    )
    second_scores = _random_scores(
        (batch_size, gate_count, node_count), device=device, generator=generator
    ).masked_fill(~second_mask, -1.0)
    parent_second = second_scores.argmax(dim=-1)
    gate_parents = torch.stack((parent_first, parent_second), dim=-1)

    # Two rails per level force a reused subexpression in every depth>1 root ancestry.
    for level in range(1, cfg.eval_max_depth):
        rows = depths.gt(level)
        first_gate = 2 * (level - 1)
        if level == 1:
            gate_parents[rows, first_gate] = torch.tensor([0, 1], device=device)
            gate_parents[rows, first_gate + 1] = torch.tensor([0, 2], device=device)
        else:
            previous_first = leaf_count + 2 * (level - 2)
            previous_second = previous_first + 1
            forced = torch.tensor([previous_first, previous_second], device=device)
            gate_parents[rows, first_gate] = forced
            gate_parents[rows, first_gate + 1] = forced
    for depth in range(2, cfg.eval_max_depth + 1):
        rows = depths.eq(depth)
        root_gate_index = 2 * (depth - 1)
        previous_first = leaf_count + 2 * (depth - 2)
        forced = torch.tensor([previous_first, previous_first + 1], device=device)
        gate_parents[rows, root_gate_index] = forced
    if cfg.balanced_output_xor:
        mask_leaf = leaf_count - 1
        for depth in range(1, cfg.eval_max_depth + 1):
            rows = depths.eq(depth)
            root_gate_index = 2 * (depth - 1)
            raw_parent = 0 if depth == 1 else leaf_count + 2 * (depth - 2)
            gate_parents[rows, root_gate_index] = torch.tensor(
                [raw_parent, mask_leaf], device=device
            )
            kinds[rows, leaf_count + root_gate_index] = XOR

    parent_indices = torch.full(
        (batch_size, node_count, 2),
        cfg.invalid_parent_id,
        dtype=torch.long,
        device=device,
    )
    parent_indices[:, leaf_count:] = gate_parents
    leaf_values = torch.randint(
        0,
        2,
        (batch_size, leaf_count),
        device=device,
        generator=generator,
    )
    values = _evaluate_canonical(
        kinds=kinds,
        parent_indices=parent_indices,
        levels=levels,
        leaf_values=leaf_values,
        max_depth=int(depths.max()),
    )
    root_indices = leaf_count + root_gate
    root_values = values.gather(1, root_indices[:, None]).squeeze(1)
    return {
        "kinds": kinds,
        "parent_indices": parent_indices,
        "levels": levels,
        "values": values,
        "root_indices": root_indices,
        "root_values": root_values,
    }


def make_boolean_dag_batch(
    cfg: BooleanDAGConfig,
    batch_size: int,
    device: torch.device,
    *,
    depths: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> BooleanDAGBatch:
    if batch_size < 2 or batch_size % 2:
        raise ValueError("batch_size must be a positive even number for exact root balance")
    if depths is None:
        depths = torch.arange(batch_size, device=device) % cfg.train_max_depth + 1
        depths = depths[torch.randperm(batch_size, device=device, generator=generator)]
    else:
        depths = depths.to(device=device, dtype=torch.long)
        if depths.shape != (batch_size,):
            raise ValueError("depths must have shape [batch_size]")
    if depths.min() < 1 or depths.max() > cfg.eval_max_depth:
        raise ValueError("requested depths are outside the configured evaluation range")

    schedule = torch.arange(batch_size, device=device)
    desired_root_values = schedule % 2
    desired_root_kinds = AND + (schedule // 2) % 3
    joint_permutation = torch.randperm(batch_size, device=device, generator=generator)
    desired_root_values = desired_root_values[joint_permutation]
    desired_root_kinds = desired_root_kinds[joint_permutation]
    if cfg.balanced_output_xor:
        canonical = _make_canonical_candidates(
            cfg,
            depths,
            torch.full_like(desired_root_kinds, XOR),
            generator=generator,
        )
    else:
        canonical: dict[str, torch.Tensor] = {}
        pending = torch.arange(batch_size, device=device)
        attempts = 0
        while pending.numel():
            candidates_per_row = min(
                MAX_CANDIDATES_PER_ROW,
                max(1, TARGET_CANDIDATES_PER_ROUND // len(pending)),
            )
            attempts += candidates_per_row
            if attempts > MAX_BALANCED_DAG_ATTEMPTS:
                raise RuntimeError("could not generate balanced nonconstant Boolean DAGs")
            repeated_pending = pending.repeat_interleave(candidates_per_row)
            candidates = _make_canonical_candidates(
                cfg,
                depths[repeated_pending],
                desired_root_kinds[repeated_pending],
                generator=generator,
            )
            accepted = candidates["root_values"].eq(
                desired_root_values[repeated_pending]
            ).view(len(pending), candidates_per_row)
            has_accepted = accepted.any(dim=1)
            first_accepted = accepted.long().argmax(dim=1)
            selected = (
                torch.arange(len(pending), device=device) * candidates_per_row
                + first_accepted
            )
            accepted_rows = pending[has_accepted]
            if accepted_rows.numel():
                for name, values in candidates.items():
                    if name not in canonical:
                        canonical[name] = torch.empty(
                            (batch_size, *values.shape[1:]),
                            dtype=values.dtype,
                            device=device,
                        )
                    canonical[name][accepted_rows] = values[selected[has_accepted]]
            pending = pending[~has_accepted]

    label_map = _random_scores(
        (batch_size, cfg.node_count), device=device, generator=generator
    ).argsort(dim=1)
    slot_nodes = _random_scores(
        (batch_size, cfg.node_count), device=device, generator=generator
    ).argsort(dim=1)
    self_ids = label_map.gather(1, slot_nodes)

    parent_indices = canonical["parent_indices"]
    valid_parent = parent_indices.lt(cfg.node_count)
    parent_labels = label_map.gather(1, parent_indices.clamp_max(cfg.node_count - 1).flatten(1)).view_as(
        parent_indices
    )
    parent_labels = torch.where(
        valid_parent,
        parent_labels,
        torch.full_like(parent_labels, cfg.invalid_parent_id),
    )

    def gather_slots(values: torch.Tensor) -> torch.Tensor:
        if values.ndim == 2:
            return values.gather(1, slot_nodes)
        return values.gather(1, slot_nodes[:, :, None].expand(-1, -1, values.shape[-1]))

    kinds = gather_slots(canonical["kinds"])
    values = gather_slots(canonical["values"])
    levels = gather_slots(canonical["levels"])
    parent_ids = gather_slots(parent_labels)
    root_canonical = torch.zeros(
        batch_size, cfg.node_count, dtype=torch.bool, device=device
    )
    root_canonical.scatter_(1, canonical["root_indices"][:, None], True)
    root_mask = gather_slots(root_canonical)
    initial_states = torch.where(
        kinds.eq(LEAF),
        values + 1,
        torch.full_like(values, UNKNOWN),
    )
    batch = BooleanDAGBatch(
        self_ids=self_ids,
        kinds=kinds,
        parent_ids=parent_ids,
        initial_states=initial_states,
        root_mask=root_mask,
        levels=levels,
        values=values,
        root_values=canonical["root_values"],
        depths=depths,
        has_reused_root_ancestor=(
            depths.gt(2) if cfg.balanced_output_xor else depths.gt(1)
        ),
    )
    return batch


def _index_batch(batch: BooleanDAGBatch, indices: torch.Tensor) -> BooleanDAGBatch:
    return replace(
        batch,
        **{
            name: value[indices]
            for name, value in batch.__dict__.items()
            if isinstance(value, torch.Tensor)
        },
    )


def _concatenate_batches(batches: list[BooleanDAGBatch]) -> BooleanDAGBatch:
    first = batches[0]
    return replace(
        first,
        **{
            name: torch.cat([getattr(batch, name) for batch in batches], dim=0)
            for name, value in first.__dict__.items()
            if isinstance(value, torch.Tensor)
        },
    )


def make_topology_matched_boolean_dag_batch(
    cfg: BooleanDAGConfig,
    batch_size: int,
    device: torch.device,
    *,
    root_depth: int,
    generator: torch.Generator | None = None,
) -> BooleanDAGBatch:
    """Reroot depth-8 graphs while balancing the queried gate kind and value."""
    if batch_size < 6 or batch_size % 6:
        raise ValueError("topology-matched batch_size must be a positive multiple of 6")
    if not 1 <= root_depth <= cfg.eval_max_depth:
        raise ValueError("root_depth must be within the configured evaluation range")

    selected_batches: list[BooleanDAGBatch] = []
    selected_slots: list[torch.Tensor] = []
    examples_per_cell = batch_size // 6
    master_depth = torch.full(
        (max(60, examples_per_cell * 6),),
        cfg.eval_max_depth,
        device=device,
        dtype=torch.long,
    )
    for desired_kind in (AND, OR, XOR):
        for desired_value in (0, 1):
            remaining = examples_per_cell
            attempts = 0
            while remaining:
                attempts += 1
                if attempts > 128:
                    raise RuntimeError("could not generate balanced topology-matched queries")
                candidate = make_boolean_dag_batch(
                    cfg,
                    len(master_depth),
                    device,
                    depths=master_depth,
                    generator=generator,
                )
                eligible = (
                    candidate.levels.eq(root_depth)
                    & candidate.kinds.eq(desired_kind)
                    & candidate.values.eq(desired_value)
                )
                if root_depth == cfg.eval_max_depth:
                    eligible &= candidate.root_mask
                elif root_depth > 1:
                    same_parent_pair = candidate.parent_ids[:, :, None, :].eq(
                        candidate.parent_ids[:, None, :, :]
                    ).all(dim=-1)
                    same_level = candidate.levels[:, :, None].eq(candidate.levels[:, None, :])
                    distinct_node = ~torch.eye(
                        cfg.node_count,
                        device=device,
                        dtype=torch.bool,
                    )[None]
                    duplicated_parent_pair = (
                        same_parent_pair & same_level & distinct_node
                    ).any(dim=-1)
                    eligible &= duplicated_parent_pair
                accepted = eligible.any(dim=1).nonzero(as_tuple=False).squeeze(1)
                if not accepted.numel():
                    continue
                accepted = accepted[:remaining]
                selected_batches.append(_index_batch(candidate, accepted))
                selected_slots.append(eligible[accepted].long().argmax(dim=1))
                remaining -= len(accepted)

    batch = _concatenate_batches(selected_batches)
    slots = torch.cat(selected_slots)
    root_mask = torch.zeros_like(batch.root_mask)
    root_mask.scatter_(1, slots[:, None], True)
    depths = torch.full_like(batch.depths, root_depth)
    root_values = batch.values.gather(1, slots[:, None]).squeeze(1)
    has_reused_root_ancestor = torch.full_like(
        batch.has_reused_root_ancestor,
        root_depth > 1,
    )
    batch = replace(
        batch,
        root_mask=root_mask,
        root_values=root_values,
        depths=depths,
        has_reused_root_ancestor=has_reused_root_ancestor,
    )
    permutation = torch.randperm(batch_size, device=device, generator=generator)
    return _index_batch(batch, permutation)


def wavefront_targets(batch: BooleanDAGBatch, *, readouts: int) -> torch.Tensor:
    if readouts < 1:
        raise ValueError("readouts must be >= 1")
    resolved = batch.levels[:, None, :] <= torch.arange(
        1,
        readouts + 1,
        device=batch.levels.device,
    ).view(1, readouts, 1)
    values = (batch.values + 1)[:, None, :].expand(-1, readouts, -1)
    return torch.where(resolved, values, torch.full_like(values, UNKNOWN))


def permute_batch_slots(
    batch: BooleanDAGBatch,
    permutation: torch.Tensor | None = None,
) -> tuple[BooleanDAGBatch, torch.Tensor]:
    if permutation is None:
        permutation = torch.rand(
            batch.batch_size,
            batch.node_count,
            device=batch.self_ids.device,
        ).argsort(dim=1)
    if permutation.shape != batch.self_ids.shape:
        raise ValueError("permutation must have shape [batch, node_count]")
    inverse = torch.empty_like(permutation)
    inverse.scatter_(
        1,
        permutation,
        torch.arange(batch.node_count, device=permutation.device).expand(batch.batch_size, -1),
    )

    def gather(values: torch.Tensor) -> torch.Tensor:
        if values.ndim == 2:
            return values.gather(1, permutation)
        return values.gather(1, permutation[:, :, None].expand(-1, -1, values.shape[-1]))

    permuted = replace(
        batch,
        self_ids=gather(batch.self_ids),
        kinds=gather(batch.kinds),
        parent_ids=gather(batch.parent_ids),
        initial_states=gather(batch.initial_states),
        root_mask=gather(batch.root_mask),
        levels=gather(batch.levels),
        values=gather(batch.values),
    )
    return permuted, inverse


def assert_boolean_oracle(batch: BooleanDAGBatch) -> None:
    batch_size, node_count = batch.self_ids.shape
    expected_ids = torch.arange(node_count, device=batch.self_ids.device).expand(batch_size, -1)
    assert torch.equal(batch.self_ids.sort(dim=1).values, expected_ids)
    assert torch.all(batch.root_mask.sum(dim=1).eq(1))
    leaf_counts = batch.kinds.eq(LEAF).sum(dim=1)
    assert torch.all(leaf_counts.eq(leaf_counts[0]))

    id_to_slot = torch.empty_like(batch.self_ids)
    id_to_slot.scatter_(
        1,
        batch.self_ids,
        torch.arange(node_count, device=batch.self_ids.device).expand(batch_size, -1),
    )
    gate_mask = batch.kinds.ne(LEAF)
    gate_parents = batch.parent_ids[gate_mask]
    assert torch.all(gate_parents.lt(node_count))
    parent_slots = id_to_slot.gather(
        1,
        batch.parent_ids.clamp_max(node_count - 1).flatten(1),
    ).view_as(batch.parent_ids)
    parent_levels = batch.levels.gather(1, parent_slots.flatten(1)).view_as(parent_slots)
    parent_values = batch.values.gather(1, parent_slots.flatten(1)).view_as(parent_slots)
    assert torch.all(parent_levels[gate_mask].max(dim=-1).values + 1 == batch.levels[gate_mask])

    left = parent_values[:, :, 0]
    right = parent_values[:, :, 1]
    xor = left ^ right
    computed = torch.where(
        batch.kinds.eq(AND),
        left & right,
        torch.where(
            batch.kinds.eq(OR),
            left | right,
            torch.where(batch.kinds.eq(XOR), xor, 1 - xor),
        ),
    )
    assert torch.equal(computed[gate_mask], batch.values[gate_mask])
    assert torch.equal(batch.initial_states[gate_mask], torch.zeros_like(batch.initial_states[gate_mask]))
    assert torch.equal(
        batch.initial_states[~gate_mask],
        batch.values[~gate_mask] + 1,
    )
    assert torch.equal(batch.levels[batch.root_mask], batch.depths)
    assert torch.equal(batch.values[batch.root_mask], batch.root_values)
    assert torch.all(batch.has_reused_root_ancestor[batch.depths.gt(2)])
