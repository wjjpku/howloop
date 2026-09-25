from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path
from typing import Literal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from reasoning_loop.boolean_dag_data import (
    AND,
    LEAF,
    OR,
    XOR,
    BooleanDAGBatch,
    BooleanDAGConfig,
    make_boolean_dag_batch,
    make_topology_matched_boolean_dag_batch,
)
from reasoning_loop.boolean_dag_model import (
    BidirectionalSelfAttention,
    BooleanDAGBlock,
    BooleanDAGModelConfig,
    build_boolean_dag_model,
)
from reasoning_loop.graph_path_loop import pick_device, set_seed


BranchName = Literal["attention", "mlp"]


def make_leaf_counterfactual(
    batch: BooleanDAGBatch,
    *,
    generator: torch.Generator | None = None,
) -> BooleanDAGBatch:
    values = batch.values.clone()
    leaf_mask = batch.kinds.eq(LEAF)
    random_values = torch.randint(
        0,
        2,
        values.shape,
        device=values.device,
        generator=generator,
    )
    values = torch.where(leaf_mask, random_values, values)
    id_to_slot = torch.empty_like(batch.self_ids)
    id_to_slot.scatter_(
        1,
        batch.self_ids,
        torch.arange(batch.node_count, device=values.device).expand(batch.batch_size, -1),
    )
    parent_slots = id_to_slot.gather(
        1,
        batch.parent_ids.clamp_max(batch.node_count - 1).flatten(1),
    ).view_as(batch.parent_ids)
    for level in range(1, int(batch.levels.max()) + 1):
        parent_values = values.gather(1, parent_slots.flatten(1)).view_as(parent_slots)
        left, right = parent_values.unbind(dim=-1)
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
        values = torch.where(batch.levels.eq(level), computed, values)
    initial_states = torch.where(leaf_mask, values + 1, torch.zeros_like(values))
    root_values = values[batch.root_mask]
    return replace(
        batch,
        values=values,
        initial_states=initial_states,
        root_values=root_values,
    )


def root_parent_slots(batch: BooleanDAGBatch) -> torch.Tensor:
    root_slots = batch.root_mask.long().argmax(dim=1)
    parent_ids = batch.parent_ids.gather(
        1,
        root_slots[:, None, None].expand(-1, 1, 2),
    ).squeeze(1)
    id_to_slot = torch.empty_like(batch.self_ids)
    id_to_slot.scatter_(
        1,
        batch.self_ids,
        torch.arange(batch.node_count, device=batch.self_ids.device).expand(batch.batch_size, -1),
    )
    return id_to_slot.gather(1, parent_ids)


def replace_selected_slots(
    receiver: torch.Tensor,
    donor: torch.Tensor,
    slots: torch.Tensor,
) -> torch.Tensor:
    if receiver.shape != donor.shape or receiver.ndim != 3:
        raise ValueError("receiver and donor must have identical [batch, node, hidden] shapes")
    if slots.ndim != 2 or slots.shape[0] != receiver.shape[0]:
        raise ValueError("slots must have shape [batch, selected_node]")
    patched = receiver.clone()
    patched.scatter_(
        1,
        slots[:, :, None].expand(-1, -1, receiver.shape[-1]),
        donor.gather(1, slots[:, :, None].expand(-1, -1, receiver.shape[-1])),
    )
    return patched


def _replace_one_slot(
    values: torch.Tensor,
    slots: torch.Tensor,
    replacements: torch.Tensor,
) -> torch.Tensor:
    if slots.shape != (values.shape[0],) or replacements.shape != (
        values.shape[0],
        values.shape[-1],
    ):
        raise ValueError("target slot or replacement shape is incompatible with branch output")
    patched = values.clone()
    patched[torch.arange(values.shape[0], device=values.device), slots] = replacements
    return patched


def apply_block_instrumented(
    block: BooleanDAGBlock,
    x: torch.Tensor,
    *,
    target_slots: torch.Tensor,
    patch_branch: BranchName | None = None,
    patch_values: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if (patch_branch is None) != (patch_values is None):
        raise ValueError("patch_branch and patch_values must be provided together")
    rows = torch.arange(x.shape[0], device=x.device)
    attention = block.attn(block.ln_1(x))
    clean_attention = attention[rows, target_slots].clone()
    if patch_branch == "attention":
        attention = _replace_one_slot(attention, target_slots, patch_values)
    after_attention = x + attention
    mlp = block.mlp(block.ln_2(after_attention))
    clean_mlp = mlp[rows, target_slots].clone()
    if patch_branch == "mlp":
        mlp = _replace_one_slot(mlp, target_slots, patch_values)
    return after_attention + mlp, {
        "attention": clean_attention,
        "mlp": clean_mlp,
    }


def attention_head_contributions(
    attention: BidirectionalSelfAttention,
    x: torch.Tensor,
) -> torch.Tensor:
    batch_size, node_count, d_model = x.shape
    qkv = attention.qkv(x).view(
        batch_size,
        node_count,
        3,
        attention.n_heads,
        attention.d_head,
    )
    query, key, value = qkv.unbind(dim=2)
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)
    attended = F.scaled_dot_product_attention(
        query,
        key,
        value,
        dropout_p=attention.dropout if attention.training else 0.0,
        is_causal=False,
    )
    output_weights = attention.out_proj.weight.view(
        d_model,
        attention.n_heads,
        attention.d_head,
    )
    return torch.einsum("bhnd,ohd->bhno", attended, output_weights)


def root_attention_mass(
    block: BooleanDAGBlock,
    x: torch.Tensor,
    *,
    root_slots: torch.Tensor,
    parent_slots: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    normalized = block.ln_1(x)
    batch_size, node_count, _ = normalized.shape
    qkv = block.attn.qkv(normalized).view(
        batch_size,
        node_count,
        3,
        block.attn.n_heads,
        block.attn.d_head,
    )
    query, key, _value = qkv.unbind(dim=2)
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    rows = torch.arange(batch_size, device=x.device)
    root_query = query[rows, :, root_slots]
    weights = torch.einsum("bhd,bhnd->bhn", root_query, key).mul(
        block.attn.d_head**-0.5
    ).softmax(dim=-1)
    parent_mass = weights.gather(
        2,
        parent_slots[:, None, :].expand(-1, block.attn.n_heads, -1),
    ).sum(dim=-1)
    root_mass = weights.gather(
        2,
        root_slots[:, None, None].expand(-1, block.attn.n_heads, 1),
    ).squeeze(-1)
    return parent_mass, root_mass


def apply_block_with_head_patch(
    block: BooleanDAGBlock,
    x: torch.Tensor,
    *,
    target_slots: torch.Tensor,
    head_mask: torch.Tensor | None = None,
    patch_values: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if (head_mask is None) != (patch_values is None):
        raise ValueError("head_mask and patch_values must be provided together")
    contributions = attention_head_contributions(block.attn, block.ln_1(x))
    rows = torch.arange(x.shape[0], device=x.device)
    target_contributions = contributions[rows, :, target_slots, :].clone()
    if head_mask is not None:
        if head_mask.shape != (block.attn.n_heads,) or patch_values.shape != target_contributions.shape:
            raise ValueError("head patch shapes are incompatible with the attention module")
        replacements = target_contributions.clone()
        replacements[:, head_mask] = patch_values[:, head_mask]
        contributions = contributions.clone()
        contributions[rows, :, target_slots, :] = replacements
    after_attention = x + contributions.sum(dim=1)
    return after_attention + block.mlp(block.ln_2(after_attention)), target_contributions


def normalized_recovery(
    clean: torch.Tensor,
    corrupt: torch.Tensor,
    patched: torch.Tensor,
) -> torch.Tensor:
    denominator = clean - corrupt
    return torch.where(
        denominator.abs() > 1e-6,
        (patched - corrupt) / denominator,
        torch.full_like(denominator, float("nan")),
    )


def _root_slots(batch: BooleanDAGBatch) -> torch.Tensor:
    return batch.root_mask.long().argmax(dim=1)


def _root_logits(model: torch.nn.Module, state: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
    rows = torch.arange(state.shape[0], device=state.device)
    return model._readout(state)[0][rows, slots]


def _select_nonparent_slots(
    batch: BooleanDAGBatch,
    parent_slots: torch.Tensor,
    root_slots: torch.Tensor,
) -> torch.Tensor:
    rows = torch.arange(batch.batch_size, device=batch.self_ids.device)
    excluded = torch.zeros_like(batch.root_mask)
    excluded[rows, root_slots] = True
    excluded.scatter_(1, parent_slots, True)
    eligible = batch.kinds.ne(LEAF) & ~excluded
    preferred_level = batch.depths[:, None] - 1
    scores = torch.rand(
        batch.batch_size,
        batch.node_count,
        device=batch.self_ids.device,
    ) + 2.0 * batch.levels.eq(preferred_level)
    scores.masked_fill_(~eligible, -1.0)
    return scores.topk(2, dim=1).indices


def _head_masks(n_heads: int, device: torch.device) -> list[torch.Tensor]:
    masks = [
        torch.tensor([(bits >> head) & 1 for head in range(n_heads)], device=device).bool()
        for bits in range(1 << n_heads)
    ]
    return sorted(masks, key=lambda mask: (int(mask.sum()), tuple(mask.tolist())))


def _mask_label(mask: torch.Tensor) -> str:
    selected = mask.nonzero(as_tuple=False).squeeze(1).tolist()
    return "+".join(str(head) for head in selected) if selected else "none"


def _safe_accuracy(correct: float, count: int) -> float:
    return correct / count if count else float("nan")


@torch.no_grad()
def run_counterfactual_circuit(
    *,
    checkpoint: Path,
    out_dir: Path,
    depths: list[int],
    batches: int,
    batch_size: int,
    device: torch.device,
    topology_control: bool = True,
    standard_continuation: str = "repeat_last",
) -> dict[str, object]:
    checkpoint_data = torch.load(checkpoint, map_location=device)
    architecture = checkpoint_data["architecture"]
    if standard_continuation not in {"repeat_last", "cycle"}:
        raise ValueError("standard_continuation must be repeat_last or cycle")
    if architecture != "standard" and standard_continuation != "repeat_last":
        raise ValueError("standard_continuation only applies to standard checkpoints")
    data_cfg = BooleanDAGConfig(**checkpoint_data["data_config"])
    model_cfg = BooleanDAGModelConfig(**checkpoint_data["model_config"])
    if not depths or min(depths) < 2 or max(depths) > data_cfg.eval_max_depth:
        raise ValueError("causal depths must lie within 2 through the configured maximum")
    model = build_boolean_dag_model(
        architecture=architecture,
        data_cfg=data_cfg,
        model_cfg=model_cfg,
    ).to(device)
    model.load_state_dict(checkpoint_data["model"])
    model.eval()
    set_seed(20260715)

    def effective_block(step_index: int) -> BooleanDAGBlock:
        if architecture == "looped":
            return model.block
        if architecture == "periodic2":
            return model.blocks[step_index % len(model.blocks)]
        if standard_continuation == "cycle":
            return model.blocks[step_index % len(model.blocks)]
        return model.blocks[min(step_index, len(model.blocks) - 1)]

    masks = _head_masks(model_cfg.n_heads, device)
    depth_rows: list[dict[str, object]] = []
    subset_rows: list[dict[str, object]] = []
    attention_rows: list[dict[str, object]] = []
    for depth in depths:
        totals = {
            "examples": 0,
            "valid": 0,
            "parent_pair_readout_correct": 0,
            "parent_swap_donor": 0.0,
            "nonparent_swap_base": 0.0,
            "clean_valid": 0,
            "zero_attention_base": 0.0,
            "zero_mlp_base": 0.0,
            "shuffled_attention_base": 0.0,
            "shuffled_mlp_base": 0.0,
            "rollback_valid": 0,
            "parent_rollback_base": 0.0,
            "nonparent_rollback_base": 0.0,
        }
        subset_totals = {
            _mask_label(mask): {
                "count": 0,
                "sufficiency": 0.0,
                "necessity": 0.0,
                "shuffled": 0.0,
                "recovery_sum": 0.0,
                "recovery_count": 0,
            }
            for mask in masks
        }
        attention_parent_sum = torch.zeros(model_cfg.n_heads, device=device)
        attention_root_sum = torch.zeros(model_cfg.n_heads, device=device)
        attention_count = 0
        for _ in range(batches):
            if topology_control:
                base = make_topology_matched_boolean_dag_batch(
                    data_cfg,
                    batch_size,
                    device,
                    root_depth=depth,
                )
            else:
                base = make_boolean_dag_batch(
                    data_cfg,
                    batch_size,
                    device,
                    depths=torch.full((batch_size,), depth, device=device),
                )
            donor = make_leaf_counterfactual(base)
            root_slots = _root_slots(base)
            parent_slots = root_parent_slots(base)
            nonparent_slots = _select_nonparent_slots(base, parent_slots, root_slots)
            rows = torch.arange(batch_size, device=device)

            base_history = [model.encode(base)]
            donor_history = [model.encode(donor)]
            for step_index in range(depth - 1):
                history_block = effective_block(step_index)
                base_history.append(history_block(base_history[-1]))
                donor_history.append(history_block(donor_history[-1]))
            base_pre = base_history[-1]
            donor_pre = donor_history[-1]
            target_block = effective_block(depth - 1)
            parent_mass, root_mass = root_attention_mass(
                target_block,
                base_pre,
                root_slots=root_slots,
                parent_slots=parent_slots,
            )
            attention_parent_sum += parent_mass.sum(dim=0)
            attention_root_sum += root_mass.sum(dim=0)
            attention_count += batch_size
            base_output, base_branches = apply_block_instrumented(
                target_block,
                base_pre,
                target_slots=root_slots,
            )
            _, base_heads = apply_block_with_head_patch(
                target_block,
                base_pre,
                target_slots=root_slots,
            )
            donor_output = target_block(donor_pre)
            parent_swapped_pre = replace_selected_slots(base_pre, donor_pre, parent_slots)
            parent_swapped_output, swapped_heads = apply_block_with_head_patch(
                target_block,
                parent_swapped_pre,
                target_slots=root_slots,
            )
            nonparent_swapped_output = target_block(
                replace_selected_slots(base_pre, donor_pre, nonparent_slots)
            )

            base_targets = base.root_values + 1
            donor_targets = donor.root_values + 1
            base_parent_targets = base.values.gather(1, parent_slots) + 1
            donor_parent_targets = donor.values.gather(1, parent_slots) + 1
            base_parent_predictions = model._readout(base_pre)[0].gather(
                1,
                parent_slots[:, :, None].expand(-1, -1, 3),
            ).argmax(dim=-1)
            donor_parent_predictions = model._readout(donor_pre)[0].gather(
                1,
                parent_slots[:, :, None].expand(-1, -1, 3),
            ).argmax(dim=-1)
            parent_pair_readout_correct = (
                base_parent_predictions.eq(base_parent_targets).all(dim=1)
                & donor_parent_predictions.eq(donor_parent_targets).all(dim=1)
            )
            base_logits = _root_logits(model, base_output, root_slots)
            donor_logits = _root_logits(model, donor_output, root_slots)
            swapped_logits = _root_logits(model, parent_swapped_output, root_slots)
            nonparent_logits = _root_logits(model, nonparent_swapped_output, root_slots)
            clean_valid = base_logits.argmax(dim=1).eq(base_targets)
            zero_attention_logits = _root_logits(
                model,
                apply_block_instrumented(
                    target_block,
                    base_pre,
                    target_slots=root_slots,
                    patch_branch="attention",
                    patch_values=torch.zeros_like(base_branches["attention"]),
                )[0],
                root_slots,
            )
            zero_mlp_logits = _root_logits(
                model,
                apply_block_instrumented(
                    target_block,
                    base_pre,
                    target_slots=root_slots,
                    patch_branch="mlp",
                    patch_values=torch.zeros_like(base_branches["mlp"]),
                )[0],
                root_slots,
            )
            shuffled_attention_logits = _root_logits(
                model,
                apply_block_instrumented(
                    target_block,
                    base_pre,
                    target_slots=root_slots,
                    patch_branch="attention",
                    patch_values=base_branches["attention"].roll(1, dims=0),
                )[0],
                root_slots,
            )
            shuffled_mlp_logits = _root_logits(
                model,
                apply_block_instrumented(
                    target_block,
                    base_pre,
                    target_slots=root_slots,
                    patch_branch="mlp",
                    patch_values=base_branches["mlp"].roll(1, dims=0),
                )[0],
                root_slots,
            )
            clean_count = int(clean_valid.sum())
            totals["clean_valid"] += clean_count
            for name, logits in (
                ("zero_attention_base", zero_attention_logits),
                ("zero_mlp_base", zero_mlp_logits),
                ("shuffled_attention_base", shuffled_attention_logits),
                ("shuffled_mlp_base", shuffled_mlp_logits),
            ):
                totals[name] += float(
                    logits.argmax(dim=1)[clean_valid].eq(base_targets[clean_valid]).sum()
                )
            valid = (
                base_targets.ne(donor_targets)
                & base_parent_targets.ne(donor_parent_targets).any(dim=1)
                & base_parent_predictions.eq(base_parent_targets).all(dim=1)
                & donor_parent_predictions.eq(donor_parent_targets).all(dim=1)
                & base_logits.argmax(dim=1).eq(base_targets)
                & donor_logits.argmax(dim=1).eq(donor_targets)
            )
            valid_count = int(valid.sum())
            totals["examples"] += batch_size
            totals["parent_pair_readout_correct"] += int(parent_pair_readout_correct.sum())
            totals["valid"] += valid_count
            totals["parent_swap_donor"] += float(
                swapped_logits.argmax(dim=1)[valid].eq(donor_targets[valid]).sum()
            )
            totals["nonparent_swap_base"] += float(
                nonparent_logits.argmax(dim=1)[valid].eq(base_targets[valid]).sum()
            )

            early_pre = base_history[-2]
            early_parent_predictions = model._readout(early_pre)[0].gather(
                1,
                parent_slots[:, :, None].expand(-1, -1, 3),
            ).argmax(dim=-1)
            rollback_valid = (
                early_parent_predictions.eq(0).all(dim=1)
                & base_parent_predictions.eq(base_parent_targets).all(dim=1)
                & base_logits.argmax(dim=1).eq(base_targets)
            )
            rollback_count = int(rollback_valid.sum())
            parent_rollback_logits = _root_logits(
                model,
                target_block(replace_selected_slots(base_pre, early_pre, parent_slots)),
                root_slots,
            )
            nonparent_rollback_logits = _root_logits(
                model,
                target_block(replace_selected_slots(base_pre, early_pre, nonparent_slots)),
                root_slots,
            )
            totals["rollback_valid"] += rollback_count
            totals["parent_rollback_base"] += float(
                parent_rollback_logits.argmax(dim=1)[rollback_valid]
                .eq(base_targets[rollback_valid])
                .sum()
            )
            totals["nonparent_rollback_base"] += float(
                nonparent_rollback_logits.argmax(dim=1)[rollback_valid]
                .eq(base_targets[rollback_valid])
                .sum()
            )

            base_margin = base_logits[rows, donor_targets] - base_logits[rows, base_targets]
            swapped_margin = swapped_logits[rows, donor_targets] - swapped_logits[rows, base_targets]
            for mask in masks:
                label = _mask_label(mask)
                sufficient_output, _ = apply_block_with_head_patch(
                    target_block,
                    base_pre,
                    target_slots=root_slots,
                    head_mask=mask,
                    patch_values=swapped_heads,
                )
                necessary_output, _ = apply_block_with_head_patch(
                    target_block,
                    parent_swapped_pre,
                    target_slots=root_slots,
                    head_mask=mask,
                    patch_values=base_heads,
                )
                shuffled_output, _ = apply_block_with_head_patch(
                    target_block,
                    base_pre,
                    target_slots=root_slots,
                    head_mask=mask,
                    patch_values=swapped_heads.roll(1, dims=0),
                )
                sufficient_logits = _root_logits(model, sufficient_output, root_slots)
                necessary_logits = _root_logits(model, necessary_output, root_slots)
                shuffled_logits = _root_logits(model, shuffled_output, root_slots)
                stats = subset_totals[label]
                stats["count"] += valid_count
                stats["sufficiency"] += float(
                    sufficient_logits.argmax(dim=1)[valid].eq(donor_targets[valid]).sum()
                )
                stats["necessity"] += float(
                    necessary_logits.argmax(dim=1)[valid].eq(base_targets[valid]).sum()
                )
                stats["shuffled"] += float(
                    shuffled_logits.argmax(dim=1)[valid].eq(donor_targets[valid]).sum()
                )
                sufficient_margin = (
                    sufficient_logits[rows, donor_targets]
                    - sufficient_logits[rows, base_targets]
                )
                recovery = normalized_recovery(swapped_margin, base_margin, sufficient_margin)
                finite = valid & recovery.isfinite()
                stats["recovery_sum"] += float(recovery[finite].sum())
                stats["recovery_count"] += int(finite.sum())

        valid_count = int(totals["valid"])
        clean_count = int(totals["clean_valid"])
        rollback_count = int(totals["rollback_valid"])
        depth_rows.append(
            {
                "depth": depth,
                "examples": totals["examples"],
                "counterfactual_valid_examples": valid_count,
                "counterfactual_valid_rate": valid_count / int(totals["examples"]),
                "parent_pair_readout_correct_rate": int(
                    totals["parent_pair_readout_correct"]
                )
                / int(totals["examples"]),
                "parent_swap_follows_donor_accuracy": _safe_accuracy(
                    totals["parent_swap_donor"], valid_count
                ),
                "nonparent_swap_keeps_base_accuracy": _safe_accuracy(
                    totals["nonparent_swap_base"], valid_count
                ),
                "zero_attention_keeps_base_accuracy": _safe_accuracy(
                    totals["zero_attention_base"], clean_count
                ),
                "zero_mlp_keeps_base_accuracy": _safe_accuracy(
                    totals["zero_mlp_base"], clean_count
                ),
                "shuffled_attention_keeps_base_accuracy": _safe_accuracy(
                    totals["shuffled_attention_base"], clean_count
                ),
                "shuffled_mlp_keeps_base_accuracy": _safe_accuracy(
                    totals["shuffled_mlp_base"], clean_count
                ),
                "rollback_valid_examples": rollback_count,
                "parent_rollback_keeps_base_accuracy": _safe_accuracy(
                    totals["parent_rollback_base"], rollback_count
                ),
                "nonparent_rollback_keeps_base_accuracy": _safe_accuracy(
                    totals["nonparent_rollback_base"], rollback_count
                ),
            }
        )
        for mask in masks:
            label = _mask_label(mask)
            stats = subset_totals[label]
            count = int(stats["count"])
            subset_rows.append(
                {
                    "depth": depth,
                    "head_subset": label,
                    "subset_size": int(mask.sum()),
                    "valid_examples": count,
                    "sufficiency_donor_accuracy": _safe_accuracy(stats["sufficiency"], count),
                    "necessity_base_accuracy": _safe_accuracy(stats["necessity"], count),
                    "shuffled_donor_accuracy": _safe_accuracy(stats["shuffled"], count),
                    "normalized_margin_recovery": _safe_accuracy(
                        stats["recovery_sum"], int(stats["recovery_count"])
                    ),
                }
            )
        for head in range(model_cfg.n_heads):
            attention_rows.append(
                {
                    "depth": depth,
                    "head": head,
                    "examples": attention_count,
                    "parent_attention_mass": float(
                        attention_parent_sum[head] / attention_count
                    ),
                    "root_attention_mass": float(attention_root_sum[head] / attention_count),
                }
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    for filename, rows in (
        ("depth_metrics.csv", depth_rows),
        ("head_subset_metrics.csv", subset_rows),
        ("head_attention_mass.csv", attention_rows),
    ):
        with (out_dir / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    qualifying = [
        row
        for row in subset_rows
        if np.isfinite(row["sufficiency_donor_accuracy"])
        and row["sufficiency_donor_accuracy"] >= 0.95
        and row["necessity_base_accuracy"] >= 0.95
    ]
    minimal_by_depth: dict[str, str | None] = {}
    for depth in depths:
        candidates = [row for row in qualifying if row["depth"] == depth]
        candidates.sort(key=lambda row: (row["subset_size"], row["head_subset"]))
        minimal_by_depth[str(depth)] = candidates[0]["head_subset"] if candidates else None

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.patch.set_facecolor("white")
    x = np.arange(len(depths))
    axes[0, 0].plot(x, [row["parent_swap_follows_donor_accuracy"] for row in depth_rows], "o-", label="parent swap follows donor")
    axes[0, 0].plot(x, [row["nonparent_swap_keeps_base_accuracy"] for row in depth_rows], "o-", label="non-parent swap keeps base")
    axes[0, 0].plot(x, [row["parent_rollback_keeps_base_accuracy"] for row in depth_rows], "o--", label="parent rollback keeps answer")
    axes[0, 0].plot(x, [row["nonparent_rollback_keeps_base_accuracy"] for row in depth_rows], "o--", label="non-parent rollback keeps answer")
    axes[0, 0].set(xticks=x, xticklabels=depths, xlabel="queried root depth", ylabel="accuracy", ylim=(-0.03, 1.03), title="Causal state interventions")
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].plot(x, [row["zero_attention_keeps_base_accuracy"] for row in depth_rows], "o-", label="zero attention")
    axes[0, 1].plot(x, [row["zero_mlp_keeps_base_accuracy"] for row in depth_rows], "o-", label="zero MLP")
    axes[0, 1].plot(x, [row["shuffled_attention_keeps_base_accuracy"] for row in depth_rows], "o--", label="shuffled attention")
    axes[0, 1].plot(x, [row["shuffled_mlp_keeps_base_accuracy"] for row in depth_rows], "o--", label="shuffled MLP")
    axes[0, 1].set(xticks=x, xticklabels=depths, xlabel="queried root depth", ylabel="base-target accuracy", ylim=(-0.03, 1.03), title="Branch necessity controls")
    axes[0, 1].legend(fontsize=8)
    labels = [_mask_label(mask) for mask in masks]
    heat = np.array(
        [
            [
                next(row["sufficiency_donor_accuracy"] for row in subset_rows if row["depth"] == depth and row["head_subset"] == label)
                for label in labels
            ]
            for depth in depths
        ]
    )
    image = axes[1, 0].imshow(heat, aspect="auto", vmin=0, vmax=1, cmap="viridis")
    axes[1, 0].set(xticks=np.arange(len(labels)), xticklabels=labels, yticks=np.arange(len(depths)), yticklabels=depths, xlabel="patched head subset", ylabel="queried root depth", title="Counterfactual head-subset sufficiency")
    axes[1, 0].tick_params(axis="x", rotation=90, labelsize=7)
    fig.colorbar(image, ax=axes[1, 0], label="donor-target accuracy")
    for head in range(model_cfg.n_heads):
        axes[1, 1].plot(
            x,
            [
                next(
                    row["parent_attention_mass"]
                    for row in attention_rows
                    if row["depth"] == depth and row["head"] == head
                )
                for depth in depths
            ],
            "o-",
            label=f"head {head}",
        )
    axes[1, 1].axhline(2 / data_cfg.node_count, color="black", linestyle=":", linewidth=1, label="uniform baseline")
    axes[1, 1].set(xticks=x, xticklabels=depths, xlabel="queried root depth", ylabel="attention mass on two parents", ylim=(-0.03, 1.03), title="Functional interpretation of causal heads")
    axes[1, 1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "counterfactual_circuit.png", dpi=180, facecolor="white")
    plt.close(fig)

    behavior_spec = {
        "model": str(checkpoint),
        "architecture": (
            "one shared block recurrently applied"
            if architecture == "looped"
            else "two trained blocks repeated with period two"
            if architecture == "periodic2"
            else "independent trained blocks, then cycle from the first block beyond trained depth"
            if standard_continuation == "cycle"
            else "independent trained blocks, then repeat the last block beyond trained depth"
        ),
        "behavior": "compute a queried Boolean gate from the current states of its two parents",
        "clean_distribution": "balanced Boolean DAG queries",
        "counterfactual": "same graph and node IDs with resampled leaf values",
        "intervention": "interchange only the two queried root-parent states at loop t-1",
        "negative_control": "interchange two non-parent gate states or shuffle donor head activations",
        "scalar_metrics": [
            "counterfactual donor-target accuracy",
            "base-target accuracy",
            "normalized donor-vs-base logit-margin recovery",
        ],
        "component_granularity": "effective step x attention head",
        "topology_control": topology_control,
    }
    (out_dir / "behavior_spec.json").write_text(
        json.dumps(behavior_spec, indent=2), encoding="utf-8"
    )
    summary: dict[str, object] = {
        "task_version": checkpoint_data.get("task_version", "legacy_unversioned"),
        "checkpoint": str(checkpoint),
        "checkpoint_step": checkpoint_data.get("step"),
        "seed": checkpoint_data.get("seed"),
        "architecture": architecture,
        "standard_continuation": (
            standard_continuation if architecture == "standard" else None
        ),
        "repeated_last_block_from_depth": (
            model_cfg.steps + 1
            if architecture == "standard" and standard_continuation == "repeat_last"
            else None
        ),
        "depths": depths,
        "batches": batches,
        "batch_size": batch_size,
        "topology_control": topology_control,
        "head_subset_count": len(masks),
        "minimal_95pct_sufficient_and_necessary_subset_by_depth": minimal_by_depth,
        "depth_metrics": depth_rows,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "claim_ledger.md").write_text(
        "# Claim ledger\n\n"
        "| Status | Claim | Evidence boundary |\n"
        "|---|---|---|\n"
        "| Tested | Parent-state interchange causally changes the queried root output | Same-graph counterfactual leaves; see `depth_metrics.csv` |\n"
        "| Tested | A sparse effective-step head subset is sufficient and necessary under this intervention family | Exhaustive head subsets; see `head_subset_metrics.csv` |\n"
        "| Tested | Both the root attention update and MLP update are causally necessary | Zero and shuffled branch-output controls in `depth_metrics.csv` |\n"
        "| Control | Non-parent state swaps and shuffled head donors should not reproduce the targeted effect | Matched-size negative controls |\n"
        "| Open | The MLP implements the Boolean truth table | Requires gate/value-factorized interventions or weight analysis |\n"
        "| Open | This circuit is unique | Alternative low-overlap circuits were not searched beyond exhaustive head subsets |\n"
        "| Open | The result is stable across training seeds | Run the identical analysis on seed 1 and seed 2 |\n",
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Causal circuit analysis for Boolean DAG models.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--depths", type=int, nargs="+", default=[2, 3, 4, 5, 6])
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=510)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--topology-control", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--standard-continuation",
        choices=["repeat_last", "cycle"],
        default="repeat_last",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_counterfactual_circuit(
        checkpoint=args.checkpoint,
        out_dir=args.out_dir,
        depths=args.depths,
        batches=args.batches,
        batch_size=args.batch_size,
        device=pick_device(args.device),
        topology_control=args.topology_control,
        standard_continuation=args.standard_continuation,
    )


if __name__ == "__main__":
    main()
