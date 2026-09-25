from __future__ import annotations

import argparse
import csv
import itertools
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from reasoning_loop.graph_path_component_circuit import instrumented_forward_all
from reasoning_loop.graph_path_loop import (
    GraphPathConfig,
    LoopedGraphPathTransformer,
    TransformerBlock,
    pick_device,
    set_seed,
)
from reasoning_loop.graph_path_temporal_intervention import (
    apply_shared_stack,
    cache_raw_states,
    logits_from_raw_state,
    replace_answer_state,
    schedule_batch_metrics,
)


Branch = Literal["attn", "mlp"]
ParameterBranch = tuple[int, Branch]


@dataclass(frozen=True)
class EffectiveBranchSite:
    loop_index: int
    block_index: int
    branch: Branch

    @property
    def label(self) -> str:
        return f"L{self.loop_index + 1}.B{self.block_index + 1}.{self.branch}"


def fixed_depth_batch(
    cfg: GraphPathConfig,
    batch_size: int,
    device: torch.device,
    *,
    path_positions: int,
    successors: torch.Tensor | None = None,
    start: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build an explicit-depth batch whose query always asks for cfg.max_depth."""
    if batch_size < 1 or path_positions < 1:
        raise ValueError("batch_size and path_positions must be positive")
    if successors is None:
        successors = torch.rand(batch_size, cfg.node_count, device=device).argsort(dim=-1)
    if successors.shape != (batch_size, cfg.node_count):
        raise ValueError("successors has the wrong shape")
    if start is None:
        start = torch.randint(
            0,
            cfg.node_count,
            (batch_size,),
            dtype=torch.long,
            device=device,
        )
    if start.shape != (batch_size,):
        raise ValueError("start has the wrong shape")

    source = torch.arange(cfg.node_count, device=device)[None, :].expand(batch_size, -1)
    edge_triplets = torch.empty(
        batch_size,
        cfg.node_count,
        3,
        dtype=torch.long,
        device=device,
    )
    edge_triplets[:, :, 0] = cfg.edge_token
    edge_triplets[:, :, 1] = source
    edge_triplets[:, :, 2] = successors

    targets = torch.empty(
        batch_size,
        path_positions,
        dtype=torch.long,
        device=device,
    )
    current = start
    for position in range(path_positions):
        current = successors.gather(1, current[:, None]).squeeze(1)
        targets[:, position] = current

    tokens = torch.empty(batch_size, cfg.seq_len, dtype=torch.long, device=device)
    tokens[:, 0] = cfg.bos_token
    tokens[:, 1 : 1 + 3 * cfg.node_count] = edge_triplets.reshape(batch_size, -1)
    tokens[:, -4] = cfg.query_token
    tokens[:, -3] = start
    tokens[:, -2] = cfg.depth_token_base + cfg.max_depth - 1
    tokens[:, -1] = cfg.answer_token
    return tokens, targets, successors, start


def paired_fixed_depth_batch(
    cfg: GraphPathConfig,
    batch_size: int,
    device: torch.device,
    *,
    path_positions: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Create clean/corrupt queries on the same graph with different starts."""
    successors = torch.rand(batch_size, cfg.node_count, device=device).argsort(dim=-1)
    start_a = torch.randint(0, cfg.node_count, (batch_size,), device=device)
    offset = torch.randint(1, cfg.node_count, (batch_size,), device=device)
    start_b = (start_a + offset) % cfg.node_count
    tokens_a, targets_a, _, _ = fixed_depth_batch(
        cfg,
        batch_size,
        device,
        path_positions=path_positions,
        successors=successors,
        start=start_a,
    )
    tokens_b, targets_b, _, _ = fixed_depth_batch(
        cfg,
        batch_size,
        device,
        path_positions=path_positions,
        successors=successors,
        start=start_b,
    )
    return tokens_a, targets_a, tokens_b, targets_b, successors, start_a


def load_checkpoint(
    checkpoint: Path,
    device: torch.device,
) -> tuple[LoopedGraphPathTransformer, GraphPathConfig, dict[str, Any]]:
    payload = torch.load(checkpoint, map_location=device)
    cfg = GraphPathConfig(**payload["config"])
    model = LoopedGraphPathTransformer(cfg).to(device)
    model.load_state_dict(payload["model"])
    if payload.get("norm_transition_alpha") is not None:
        model.set_norm_transition(float(payload["norm_transition_alpha"]))
    model.eval()
    if model.block_style != "legacy":
        raise ValueError("depth-circuit analysis currently supports legacy blocks")
    return model, cfg, payload


def checkpoint_loss_mode(checkpoint: Path) -> str:
    metadata_path = checkpoint.parent / "metadata.json"
    if not metadata_path.exists():
        return "unknown"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    args = metadata.get("args", {})
    trajectory_weight = float(args.get("trajectory_aux_weight", 0.0))
    generic_aux_weight = float(args.get("aux_loss", 0.0))
    if trajectory_weight <= 0.0 and generic_aux_weight <= 0.0:
        return "final_only"
    if trajectory_weight > 0.0:
        jump = int(args.get("trajectory_aux_jump", 0))
        active = bool(args.get("trajectory_aux_active_only", False))
        scope = "active" if active else "full"
        return f"trajectory_jump{jump}_{scope}_weight{trajectory_weight:g}"
    return f"generic_aux_weight{generic_aux_weight:g}"


def _initial_state(
    model: LoopedGraphPathTransformer,
    tokens: torch.Tensor,
) -> torch.Tensor:
    return model.token_embed(tokens) + model.pos_embed.unsqueeze(0)


def _apply_branch(
    block: TransformerBlock,
    x: torch.Tensor,
    branch: Branch,
) -> torch.Tensor:
    if branch == "attn":
        update = block.attn(block.ln_1(x))
        if block.inner_norm_style == "ouro_sandwich_rms":
            update = block.attn_out_norm(update)
        return update
    update = block.mlp(block.ln_2(x))
    if block.inner_norm_style == "ouro_sandwich_rms":
        update = block.mlp_out_norm(update)
    return update


@torch.no_grad()
def rollout_branches(
    model: LoopedGraphPathTransformer,
    tokens: torch.Tensor,
    *,
    max_loops: int,
    ablated_sites: set[EffectiveBranchSite] | None = None,
    kept_parameter_branches: set[ParameterBranch] | None = None,
    patch_site: EffectiveBranchSite | None = None,
    patch_answer: torch.Tensor | None = None,
    cache_branches: bool = False,
) -> dict[str, Any]:
    if (patch_site is None) != (patch_answer is None):
        raise ValueError("patch_site and patch_answer must be supplied together")
    ablated = ablated_sites or set()
    x = _initial_state(model, tokens)
    logits: list[torch.Tensor] = []
    cache: dict[EffectiveBranchSite, torch.Tensor] = {}
    for loop_index in range(max_loops):
        for block_index in model.active_block_indices(loop_index):
            block = model.blocks[block_index]
            if not isinstance(block, TransformerBlock):
                raise TypeError("legacy TransformerBlock required")
            for branch in ("attn", "mlp"):
                site = EffectiveBranchSite(loop_index, block_index, branch)
                update = _apply_branch(block, x, branch)
                if cache_branches:
                    cache[site] = update[:, -1, :].clone()
                if site in ablated or (
                    kept_parameter_branches is not None
                    and (block_index, branch) not in kept_parameter_branches
                ):
                    update = torch.zeros_like(update)
                if patch_site == site:
                    if patch_answer.shape != update[:, -1, :].shape:
                        raise ValueError("patch_answer has the wrong shape")
                    update = update.clone()
                    update[:, -1, :] = patch_answer
                x = x + update
        if model.outer_norm is not None:
            x = model.outer_norm(x)
        logits.append(logits_from_raw_state(model, x))
    return {
        "logits_by_loop": torch.stack(logits, dim=1),
        "branch_cache": cache,
        "final_raw_state": x,
    }


def target_margin(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    target_value = logits.gather(-1, target[..., None]).squeeze(-1)
    masked = logits.clone()
    masked.scatter_(-1, target[..., None], float("-inf"))
    return target_value - masked.max(dim=-1).values


def logit_difference(
    logits: torch.Tensor,
    target_a: torch.Tensor,
    target_b: torch.Tensor,
) -> torch.Tensor:
    a = logits.gather(1, target_a[:, None]).squeeze(1)
    b = logits.gather(1, target_b[:, None]).squeeze(1)
    return a - b


def normalized_recovery(
    clean: torch.Tensor,
    corrupt: torch.Tensor,
    patched: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    denominator = clean - corrupt
    score = (patched - corrupt) / denominator
    return torch.where(
        denominator.abs() > eps,
        score,
        torch.full_like(score, float("nan")),
    )


def _all_targets(start: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return torch.cat((start[:, None], targets), dim=1)


def _mean_valid(values: torch.Tensor, valid: torch.Tensor) -> float:
    selected = values[valid & torch.isfinite(values)]
    return float(selected.mean().item()) if selected.numel() else float("nan")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _heatmap(
    values: np.ndarray,
    *,
    x_labels: list[str],
    y_labels: list[str],
    title: str,
    color_label: str,
    path: Path,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    width = max(6.0, 0.65 * len(x_labels) + 2.5)
    height = max(3.8, 0.5 * len(y_labels) + 2.0)
    plt.figure(figsize=(width, height))
    image = plt.imshow(values, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    plt.colorbar(image, label=color_label)
    plt.xticks(range(len(x_labels)), x_labels, rotation=45, ha="right")
    plt.yticks(range(len(y_labels)), y_labels)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def _parameter_branches(model: LoopedGraphPathTransformer) -> list[ParameterBranch]:
    return [
        (block_index, branch)
        for block_index in range(len(model.blocks))
        for branch in ("attn", "mlp")
    ]


def _branch_label(branch: ParameterBranch) -> str:
    return f"B{branch[0] + 1}.{branch[1]}"


@torch.no_grad()
def analyze_checkpoint(
    *,
    name: str,
    checkpoint: Path,
    out_dir: Path,
    device: torch.device,
    batch_size: int,
    batches: int,
    overloops: int,
    seed: int,
) -> dict[str, Any]:
    set_seed(seed)
    model, cfg, payload = load_checkpoint(checkpoint, device)
    run_dir = out_dir / name
    run_dir.mkdir(parents=True, exist_ok=True)
    trained_loops = cfg.max_loops
    analysis_loops = max(trained_loops, overloops)
    path_positions = max(cfg.max_depth, analysis_loops + 1)
    endpoint_position = cfg.max_depth
    eval_batches = [
        fixed_depth_batch(
            cfg,
            batch_size,
            device,
            path_positions=path_positions,
        )
        for _ in range(batches)
    ]

    baseline_acc_sum = torch.zeros(analysis_loops, path_positions + 1, device=device)
    baseline_margin_sum = torch.zeros_like(baseline_acc_sum)
    non_alias_correct = torch.zeros_like(baseline_acc_sum)
    non_alias_count = torch.zeros_like(baseline_acc_sum)
    baseline_entropy_sum = torch.zeros(analysis_loops, device=device)
    for tokens, targets, _, start in eval_batches:
        logits = model.forward_all(tokens, max_loops=analysis_loops)["logits_by_loop"]
        all_targets = _all_targets(start, targets)
        endpoint_target = all_targets[:, endpoint_position]
        prediction = logits.argmax(-1)
        for position in range(path_positions + 1):
            target = all_targets[:, position]
            correct = prediction.eq(target[:, None])
            baseline_acc_sum[:, position] += correct.float().mean(0)
            baseline_margin_sum[:, position] += target_margin(logits, target[:, None].expand(-1, analysis_loops)).mean(0)
            alias_filter = (
                torch.ones_like(target, dtype=torch.bool)
                if position == endpoint_position
                else target.ne(endpoint_target)
            )
            non_alias_correct[:, position] += (
                correct & alias_filter[:, None]
            ).float().sum(0)
            non_alias_count[:, position] += alias_filter.float().sum()
        probs = logits.softmax(-1)
        baseline_entropy_sum += (-(probs * probs.clamp_min(1e-9).log()).sum(-1)).mean(0)
    baseline_acc = baseline_acc_sum / batches
    baseline_margin = baseline_margin_sum / batches
    baseline_non_alias_acc = non_alias_correct / non_alias_count.clamp_min(1)
    baseline_entropy = baseline_entropy_sum / batches
    baseline_rows: list[dict[str, Any]] = []
    for loop_index in range(analysis_loops):
        for position in range(path_positions + 1):
            baseline_rows.append(
                {
                    "loop": loop_index + 1,
                    "path_position": position,
                    "accuracy": float(baseline_acc[loop_index, position]),
                    "non_alias_accuracy_excluding_endpoint_collision": float(
                        baseline_non_alias_acc[loop_index, position]
                    ),
                    "non_alias_count": int(non_alias_count[loop_index, position]),
                    "margin": float(baseline_margin[loop_index, position]),
                }
            )
    _write_csv(run_dir / "baseline_path_rows.csv", baseline_rows)
    _heatmap(
        baseline_acc.detach().cpu().numpy().T,
        x_labels=[str(index + 1) for index in range(analysis_loops)],
        y_labels=[f"f^{position}" for position in range(path_positions + 1)],
        title=f"{name}: max-depth query trajectory",
        color_label="accuracy",
        path=run_dir / "baseline_path_accuracy.png",
        vmin=0.0,
        vmax=1.0,
    )
    unrestricted_best_position_by_loop = baseline_acc.argmax(dim=1)
    best_position_by_loop = baseline_non_alias_acc[:, : endpoint_position + 1].argmax(dim=1)
    best_position_accuracy_by_loop = baseline_non_alias_acc[
        torch.arange(analysis_loops, device=device),
        best_position_by_loop,
    ]
    resolved_position_by_loop = best_position_accuracy_by_loop.ge(0.80)
    endpoint_target_index = endpoint_position - 1
    base_final_acc = float(baseline_acc[trained_loops - 1, endpoint_position])
    base_final_margin = float(baseline_margin[trained_loops - 1, endpoint_position])

    effective_branch_rows: list[dict[str, Any]] = []
    parameter_branches = _parameter_branches(model)
    for loop_index in range(trained_loops):
        for block_index, branch in parameter_branches:
            site = EffectiveBranchSite(loop_index, block_index, branch)
            acc_total = 0.0
            margin_total = 0.0
            for tokens, targets, _, _ in eval_batches:
                logits = rollout_branches(
                    model,
                    tokens,
                    max_loops=trained_loops,
                    ablated_sites={site},
                )["logits_by_loop"][:, -1, :]
                target = targets[:, endpoint_target_index]
                acc_total += float(logits.argmax(-1).eq(target).float().mean())
                margin_total += float(target_margin(logits, target).mean())
            acc = acc_total / batches
            margin = margin_total / batches
            effective_branch_rows.append(
                {
                    "site": site.label,
                    "loop": loop_index + 1,
                    "block": block_index + 1,
                    "branch": branch,
                    "ablated_accuracy": acc,
                    "accuracy_drop": base_final_acc - acc,
                    "ablated_margin": margin,
                    "margin_drop": base_final_margin - margin,
                }
            )
    _write_csv(run_dir / "effective_branch_ablation_rows.csv", effective_branch_rows)

    subset_rows: list[dict[str, Any]] = []
    subset_lookup: dict[frozenset[ParameterBranch], dict[str, Any]] = {}
    for size in range(len(parameter_branches) + 1):
        for subset_tuple in itertools.combinations(parameter_branches, size):
            subset = frozenset(subset_tuple)
            acc_total = 0.0
            margin_total = 0.0
            for tokens, targets, _, _ in eval_batches:
                logits = rollout_branches(
                    model,
                    tokens,
                    max_loops=trained_loops,
                    kept_parameter_branches=set(subset),
                )["logits_by_loop"][:, -1, :]
                target = targets[:, endpoint_target_index]
                acc_total += float(logits.argmax(-1).eq(target).float().mean())
                margin_total += float(target_margin(logits, target).mean())
            row = {
                "kept_branches": "+".join(_branch_label(item) for item in subset_tuple) or "none",
                "size": size,
                "accuracy": acc_total / batches,
                "margin": margin_total / batches,
            }
            subset_rows.append(row)
            subset_lookup[subset] = row
    empty_margin = float(subset_lookup[frozenset()]["margin"])
    denominator = base_final_margin - empty_margin
    for row in subset_rows:
        row["margin_recovery"] = (
            (float(row["margin"]) - empty_margin) / denominator
            if abs(denominator) > 1e-8
            else float("nan")
        )
    candidates = [
        (subset, row)
        for subset, row in subset_lookup.items()
        if float(row["accuracy"]) >= base_final_acc - 0.02
        and float(row["margin_recovery"]) >= 0.90
    ]
    if candidates:
        chosen_set, chosen_row = min(
            candidates,
            key=lambda item: (
                len(item[0]),
                -float(item[1]["accuracy"]),
                -float(item[1]["margin_recovery"]),
            ),
        )
    else:
        chosen_set = frozenset(parameter_branches)
        chosen_row = subset_lookup[chosen_set]
    complement_set = frozenset(parameter_branches) - chosen_set
    complement_row = subset_lookup[complement_set]
    for row in subset_rows:
        row["selected_circuit"] = row is chosen_row
        row["selected_complement"] = row is complement_row
    _write_csv(run_dir / "tied_branch_circuit_rows.csv", subset_rows)

    paired_batches = [
        paired_fixed_depth_batch(
            cfg,
            batch_size,
            device,
            path_positions=path_positions,
        )
        for _ in range(batches)
    ]
    patch_accumulator: dict[str, dict[str, float]] = {}
    for loop_index in range(trained_loops):
        for block_index, branch in parameter_branches:
            site = EffectiveBranchSite(loop_index, block_index, branch)
            patch_accumulator[site.label] = {
                "patch_in_sum": 0.0,
                "patch_out_effect_sum": 0.0,
                "shuffle_sum": 0.0,
                "valid_count": 0.0,
            }
    for tokens_a, targets_a, tokens_b, targets_b, _, _ in paired_batches:
        clean_a = rollout_branches(
            model,
            tokens_a,
            max_loops=trained_loops,
            cache_branches=True,
        )
        clean_b = rollout_branches(
            model,
            tokens_b,
            max_loops=trained_loops,
            cache_branches=True,
        )
        logits_a = clean_a["logits_by_loop"][:, -1, :]
        logits_b = clean_b["logits_by_loop"][:, -1, :]
        target_a = targets_a[:, endpoint_target_index]
        target_b = targets_b[:, endpoint_target_index]
        candidate = target_a.ne(target_b)
        valid = (
            candidate
            & logits_a.argmax(-1).eq(target_a)
            & logits_b.argmax(-1).eq(target_b)
        )
        clean_diff = logit_difference(logits_a, target_a, target_b)
        corrupt_diff = logit_difference(logits_b, target_a, target_b)
        for site, clean_activation in clean_a["branch_cache"].items():
            corrupt_activation = clean_b["branch_cache"][site]
            patch_in_logits = rollout_branches(
                model,
                tokens_b,
                max_loops=trained_loops,
                patch_site=site,
                patch_answer=clean_activation,
            )["logits_by_loop"][:, -1, :]
            patch_out_logits = rollout_branches(
                model,
                tokens_a,
                max_loops=trained_loops,
                patch_site=site,
                patch_answer=corrupt_activation,
            )["logits_by_loop"][:, -1, :]
            shuffle_logits = rollout_branches(
                model,
                tokens_b,
                max_loops=trained_loops,
                patch_site=site,
                patch_answer=clean_activation.roll(1, dims=0),
            )["logits_by_loop"][:, -1, :]
            patch_in = normalized_recovery(
                clean_diff,
                corrupt_diff,
                logit_difference(patch_in_logits, target_a, target_b),
            )
            patch_out = 1.0 - normalized_recovery(
                clean_diff,
                corrupt_diff,
                logit_difference(patch_out_logits, target_a, target_b),
            )
            shuffle = normalized_recovery(
                clean_diff,
                corrupt_diff,
                logit_difference(shuffle_logits, target_a, target_b),
            )
            bucket = patch_accumulator[site.label]
            finite_valid = valid & torch.isfinite(patch_in) & torch.isfinite(patch_out)
            count = float(finite_valid.sum())
            bucket["valid_count"] += count
            if count:
                bucket["patch_in_sum"] += float(patch_in[finite_valid].sum())
                bucket["patch_out_effect_sum"] += float(patch_out[finite_valid].sum())
                bucket["shuffle_sum"] += float(shuffle[finite_valid].sum())
    patch_rows: list[dict[str, Any]] = []
    for loop_index in range(trained_loops):
        for block_index, branch in parameter_branches:
            site = EffectiveBranchSite(loop_index, block_index, branch)
            bucket = patch_accumulator[site.label]
            count = bucket["valid_count"]
            patch_rows.append(
                {
                    "site": site.label,
                    "loop": loop_index + 1,
                    "block": block_index + 1,
                    "branch": branch,
                    "valid_count": int(count),
                    "patch_in_recovery": bucket["patch_in_sum"] / count if count else float("nan"),
                    "patch_out_effect": bucket["patch_out_effect_sum"] / count if count else float("nan"),
                    "shuffle_recovery": bucket["shuffle_sum"] / count if count else float("nan"),
                }
            )
    _write_csv(run_dir / "effective_branch_patching_rows.csv", patch_rows)

    effective_head_rows: list[dict[str, Any]] = []
    for loop_index in range(trained_loops):
        for block_index in range(cfg.n_layers):
            for head_index in range(cfg.n_heads):
                acc_total = 0.0
                margin_total = 0.0
                for tokens, targets, _, _ in eval_batches:
                    logits = instrumented_forward_all(
                        model,
                        tokens,
                        max_loops=trained_loops,
                        ablated_heads={(loop_index, block_index, head_index)},
                        return_attention=False,
                    )["logits_by_loop"][:, -1, :]
                    target = targets[:, endpoint_target_index]
                    acc_total += float(logits.argmax(-1).eq(target).float().mean())
                    margin_total += float(target_margin(logits, target).mean())
                acc = acc_total / batches
                margin = margin_total / batches
                effective_head_rows.append(
                    {
                        "site": f"L{loop_index + 1}.B{block_index + 1}.H{head_index}",
                        "loop": loop_index + 1,
                        "block": block_index + 1,
                        "head": head_index,
                        "ablated_accuracy": acc,
                        "accuracy_drop": base_final_acc - acc,
                        "ablated_margin": margin,
                        "margin_drop": base_final_margin - margin,
                    }
                )
    _write_csv(run_dir / "effective_head_ablation_rows.csv", effective_head_rows)

    tied_head_rows: list[dict[str, Any]] = []
    for block_index in range(cfg.n_layers):
        for head_index in range(cfg.n_heads):
            ablations = {
                (loop_index, block_index, head_index)
                for loop_index in range(trained_loops)
            }
            acc_total = 0.0
            margin_total = 0.0
            for tokens, targets, _, _ in eval_batches:
                logits = instrumented_forward_all(
                    model,
                    tokens,
                    max_loops=trained_loops,
                    ablated_heads=ablations,
                    return_attention=False,
                )["logits_by_loop"][:, -1, :]
                target = targets[:, endpoint_target_index]
                acc_total += float(logits.argmax(-1).eq(target).float().mean())
                margin_total += float(target_margin(logits, target).mean())
            acc = acc_total / batches
            margin = margin_total / batches
            tied_head_rows.append(
                {
                    "head": f"B{block_index + 1}.H{head_index}",
                    "block": block_index + 1,
                    "head_index": head_index,
                    "ablated_accuracy": acc,
                    "accuracy_drop": base_final_acc - acc,
                    "ablated_margin": margin,
                    "margin_drop": base_final_margin - margin,
                }
            )
    _write_csv(run_dir / "tied_head_ablation_rows.csv", tied_head_rows)

    parameter_heads = [
        (block_index, head_index)
        for block_index in range(cfg.n_layers)
        for head_index in range(cfg.n_heads)
    ]
    head_subset_cache: dict[frozenset[tuple[int, int]], dict[str, float]] = {}

    def evaluate_kept_heads(
        kept_heads: frozenset[tuple[int, int]],
    ) -> dict[str, float]:
        if kept_heads in head_subset_cache:
            return head_subset_cache[kept_heads]
        ablations = {
            (loop_index, block_index, head_index)
            for loop_index in range(trained_loops)
            for block_index, head_index in parameter_heads
            if (block_index, head_index) not in kept_heads
        }
        acc_total = 0.0
        margin_total = 0.0
        for tokens, targets, _, _ in eval_batches:
            logits = instrumented_forward_all(
                model,
                tokens,
                max_loops=trained_loops,
                ablated_heads=ablations,
                return_attention=False,
            )["logits_by_loop"][:, -1, :]
            target = targets[:, endpoint_target_index]
            acc_total += float(logits.argmax(-1).eq(target).float().mean())
            margin_total += float(target_margin(logits, target).mean())
        result = {
            "accuracy": acc_total / batches,
            "margin": margin_total / batches,
        }
        head_subset_cache[kept_heads] = result
        return result

    all_heads = frozenset(parameter_heads)
    no_heads = frozenset()
    all_head_metrics = evaluate_kept_heads(all_heads)
    no_head_metrics = evaluate_kept_heads(no_heads)
    head_margin_denominator = (
        all_head_metrics["margin"] - no_head_metrics["margin"]
    )

    def head_margin_recovery(metrics: dict[str, float]) -> float:
        if abs(head_margin_denominator) <= 1e-8:
            return float("nan")
        return (
            metrics["margin"] - no_head_metrics["margin"]
        ) / head_margin_denominator

    selected_heads = all_heads
    while selected_heads:
        removal_candidates = []
        for head in selected_heads:
            candidate = selected_heads - {head}
            metrics = evaluate_kept_heads(candidate)
            recovery = head_margin_recovery(metrics)
            if (
                metrics["accuracy"] >= base_final_acc - 0.02
                and recovery >= 0.90
            ):
                removal_candidates.append((candidate, metrics, recovery))
        if not removal_candidates:
            break
        selected_heads, _, _ = max(
            removal_candidates,
            key=lambda item: (
                item[1]["accuracy"],
                item[2],
            ),
        )
    selected_head_metrics = evaluate_kept_heads(selected_heads)
    complement_heads = all_heads - selected_heads
    complement_head_metrics = evaluate_kept_heads(complement_heads)
    selected_head_minimality = []
    for head in selected_heads:
        metrics = evaluate_kept_heads(selected_heads - {head})
        selected_head_minimality.append(
            {
                "removed": f"B{head[0] + 1}.H{head[1]}",
                "accuracy": metrics["accuracy"],
                "margin_recovery": head_margin_recovery(metrics),
            }
        )
    head_control_rows: list[dict[str, Any]] = []
    for subset_tuple in itertools.combinations(
        parameter_heads,
        len(selected_heads),
    ):
        subset = frozenset(subset_tuple)
        metrics = evaluate_kept_heads(subset)
        head_control_rows.append(
            {
                "kept_heads": "+".join(
                    f"B{block + 1}.H{head}"
                    for block, head in subset_tuple
                )
                or "none",
                "size": len(subset),
                "accuracy": metrics["accuracy"],
                "margin": metrics["margin"],
                "margin_recovery": head_margin_recovery(metrics),
                "selected_circuit": subset == selected_heads,
                "selected_complement": subset == complement_heads,
            }
        )
    if len(complement_heads) != len(selected_heads):
        metrics = complement_head_metrics
        head_control_rows.append(
            {
                "kept_heads": "+".join(
                    f"B{block + 1}.H{head}"
                    for block, head in sorted(complement_heads)
                )
                or "none",
                "size": len(complement_heads),
                "accuracy": metrics["accuracy"],
                "margin": metrics["margin"],
                "margin_recovery": head_margin_recovery(metrics),
                "selected_circuit": False,
                "selected_complement": True,
            }
        )
    _write_csv(run_dir / "tied_head_circuit_rows.csv", head_control_rows)

    schedule_sum: torch.Tensor | None = None
    schedule_names: list[str] = []
    cumulative_updates: torch.Tensor | None = None
    for tokens, targets, _, start in eval_batches:
        metrics = schedule_batch_metrics(
            model=model,
            tokens=tokens,
            targets_by_position=targets,
            start=start,
            base_loops=trained_loops,
            max_path_position=path_positions,
        )
        accuracy = metrics["accuracy"]
        schedule_sum = accuracy if schedule_sum is None else schedule_sum + accuracy
        schedule_names = list(metrics["condition_names"])
        cumulative_updates = metrics["cumulative_updates"]
    schedule_accuracy = schedule_sum / batches
    schedule_rows: list[dict[str, Any]] = []
    assert cumulative_updates is not None
    for condition_index, condition in enumerate(schedule_names):
        final_curve = schedule_accuracy[condition_index, -1]
        schedule_rows.append(
            {
                "condition": condition,
                "updates_at_final_slot": int(cumulative_updates[condition_index, -1]),
                "endpoint_accuracy": float(final_curve[endpoint_position]),
                "best_path_position": int(final_curve.argmax()),
                "best_path_accuracy": float(final_curve.max()),
            }
        )
    _write_csv(run_dir / "skip_repeat_rows.csv", schedule_rows)

    transplant_acc_sum = torch.zeros(
        trained_loops,
        trained_loops,
        2,
        device=device,
    )
    transplant_count = torch.zeros(
        trained_loops,
        trained_loops,
        2,
        device=device,
    )
    for _ in range(batches):
        donor_tokens, donor_targets, _, _ = fixed_depth_batch(
            cfg,
            batch_size,
            device,
            path_positions=path_positions,
        )
        receiver_tokens, _, receiver_successors, _ = fixed_depth_batch(
            cfg,
            batch_size,
            device,
            path_positions=path_positions,
        )
        donor_states = cache_raw_states(model, donor_tokens, max_loop=trained_loops)
        receiver_states = cache_raw_states(model, receiver_tokens, max_loop=trained_loops)
        for donor_index in range(trained_loops):
            if not bool(resolved_position_by_loop[donor_index]):
                continue
            position = int(best_position_by_loop[donor_index])
            donor_current = (
                donor_tokens[:, -3]
                if position == 0
                else donor_targets[:, position - 1]
            )
            for receiver_index in range(trained_loops):
                patched = replace_answer_state(
                    receiver_states[receiver_index],
                    donor_states[donor_index],
                )
                logits0 = logits_from_raw_state(model, patched)
                transplant_acc_sum[donor_index, receiver_index, 0] += (
                    logits0.argmax(-1).eq(donor_current).float().mean()
                )
                transplant_count[donor_index, receiver_index, 0] += 1
                next_target = receiver_successors.gather(
                    1,
                    donor_current[:, None],
                ).squeeze(1)
                updated = apply_shared_stack(
                    model,
                    patched,
                    loop_index=receiver_index + 1,
                )
                logits1 = logits_from_raw_state(model, updated)
                transplant_acc_sum[donor_index, receiver_index, 1] += (
                    logits1.argmax(-1).eq(next_target).float().mean()
                )
                transplant_count[donor_index, receiver_index, 1] += 1
    transplant_acc = transplant_acc_sum / transplant_count.clamp_min(1)
    transplant_acc = torch.where(
        transplant_count > 0,
        transplant_acc,
        torch.full_like(transplant_acc, float("nan")),
    )
    transplant_rows: list[dict[str, Any]] = []
    for donor_index in range(trained_loops):
        for receiver_index in range(trained_loops):
            transplant_rows.append(
                {
                    "donor_loop": donor_index + 1,
                    "donor_best_path_position": int(best_position_by_loop[donor_index]),
                    "donor_mapping_accuracy": float(
                        best_position_accuracy_by_loop[donor_index]
                    ),
                    "donor_resolved": bool(
                        resolved_position_by_loop[donor_index]
                    ),
                    "receiver_loop": receiver_index + 1,
                    "state_readout_accuracy": float(transplant_acc[donor_index, receiver_index, 0]),
                    "next_step_accuracy": float(transplant_acc[donor_index, receiver_index, 1]),
                }
            )
    _write_csv(run_dir / "state_transplant_rows.csv", transplant_rows)

    branch_grid = np.array(
        [
            [row["accuracy_drop"] for row in effective_branch_rows if row["loop"] == loop]
            for loop in range(1, trained_loops + 1)
        ]
    )
    patch_in_grid = np.array(
        [
            [row["patch_in_recovery"] for row in patch_rows if row["loop"] == loop]
            for loop in range(1, trained_loops + 1)
        ]
    )
    patch_out_grid = np.array(
        [
            [row["patch_out_effect"] for row in patch_rows if row["loop"] == loop]
            for loop in range(1, trained_loops + 1)
        ]
    )
    branch_labels = [_branch_label(branch) for branch in parameter_branches]
    for values, title, filename, label in (
        (branch_grid, "Effective branch necessity", "branch_accuracy_drop.png", "accuracy drop"),
        (patch_in_grid, "Clean-to-corrupt patch recovery", "branch_patch_in.png", "normalized recovery"),
        (patch_out_grid, "Corrupt-to-clean patch effect", "branch_patch_out.png", "normalized effect"),
    ):
        _heatmap(
            values,
            x_labels=branch_labels,
            y_labels=[f"loop {index + 1}" for index in range(trained_loops)],
            title=f"{name}: {title}",
            color_label=label,
            path=run_dir / filename,
        )
    _heatmap(
        transplant_acc[:, :, 1].detach().cpu().numpy(),
        x_labels=[str(index + 1) for index in range(trained_loops)],
        y_labels=[str(index + 1) for index in range(trained_loops)],
        title=f"{name}: cross-graph next-step transplant",
        color_label="accuracy",
        path=run_dir / "state_transplant_next_step.png",
        vmin=0.0,
        vmax=1.0,
    )

    chosen_labels = sorted(_branch_label(item) for item in chosen_set)
    chosen_minimality = []
    for branch in chosen_set:
        removal = chosen_set - {branch}
        row = subset_lookup[removal]
        chosen_minimality.append(
            {
                "removed": _branch_label(branch),
                "accuracy": float(row["accuracy"]),
                "margin_recovery": float(row["margin_recovery"]),
            }
        )
    summary = {
        "name": name,
        "checkpoint": str(checkpoint),
        "checkpoint_step": payload.get("step"),
        "config": asdict(cfg),
        "loss_mode": checkpoint_loss_mode(checkpoint),
        "query_depth": cfg.max_depth,
        "trained_loops": trained_loops,
        "analysis_overloops": analysis_loops,
        "behavior_metric": {
            "primary": "max-depth endpoint accuracy",
            "secondary": "correct-node logit margin",
        },
        "baseline": {
            "trained_endpoint_accuracy": base_final_acc,
            "trained_endpoint_margin": base_final_margin,
            "best_path_position_by_loop": [
                int(value) for value in best_position_by_loop.detach().cpu()
            ],
            "best_path_position_accuracy_by_loop": [
                float(value)
                for value in best_position_accuracy_by_loop.detach().cpu()
            ],
            "resolved_best_path_position_by_loop": [
                int(position) if bool(resolved) else None
                for position, resolved in zip(
                    best_position_by_loop.detach().cpu(),
                    resolved_position_by_loop.detach().cpu(),
                    strict=True,
                )
            ],
            "unrestricted_best_path_position_by_loop": [
                int(value)
                for value in unrestricted_best_position_by_loop.detach().cpu()
            ],
            "best_position_rule": (
                "argmax within f^0..f^D after excluding examples where a "
                "non-endpoint position aliases the queried endpoint; transplant "
                "requires mapping accuracy >= 0.80"
            ),
            "entropy_by_loop": [
                float(value) for value in baseline_entropy.detach().cpu()
            ],
        },
        "branch_circuit": {
            "granularity": "shared physical block attention/MLP branches",
            "selection_rule": "smallest subset with <=0.02 accuracy loss and >=0.90 margin recovery",
            "selected": chosen_labels,
            "circuit_only": {
                "accuracy": float(chosen_row["accuracy"]),
                "margin": float(chosen_row["margin"]),
                "margin_recovery": float(chosen_row["margin_recovery"]),
            },
            "complement_only": {
                "branches": sorted(_branch_label(item) for item in complement_set),
                "accuracy": float(complement_row["accuracy"]),
                "margin": float(complement_row["margin"]),
                "margin_recovery": float(complement_row["margin_recovery"]),
            },
            "minimality": chosen_minimality,
        },
        "attention_head_circuit": {
            "claim_scope": "attention sufficiency conditional on retaining every MLP branch",
            "selection_rule": "greedy removal with <=0.02 accuracy loss and >=0.90 attention-margin recovery",
            "selected": [
                f"B{block + 1}.H{head}"
                for block, head in sorted(selected_heads)
            ],
            "circuit_only": {
                **selected_head_metrics,
                "margin_recovery": head_margin_recovery(selected_head_metrics),
            },
            "complement_only": {
                "heads": [
                    f"B{block + 1}.H{head}"
                    for block, head in sorted(complement_heads)
                ],
                **complement_head_metrics,
                "margin_recovery": head_margin_recovery(complement_head_metrics),
            },
            "minimality": selected_head_minimality,
            "same_size_control_count": sum(
                row["size"] == len(selected_heads)
                for row in head_control_rows
            ),
        },
        "top_effective_branches_by_accuracy_drop": sorted(
            effective_branch_rows,
            key=lambda row: float(row["accuracy_drop"]),
            reverse=True,
        )[:8],
        "top_effective_patch_sites": sorted(
            patch_rows,
            key=lambda row: float(row["patch_in_recovery"])
            if np.isfinite(float(row["patch_in_recovery"]))
            else -float("inf"),
            reverse=True,
        )[:8],
        "top_tied_heads_by_accuracy_drop": sorted(
            tied_head_rows,
            key=lambda row: float(row["accuracy_drop"]),
            reverse=True,
        ),
        "schedule": schedule_rows,
        "transplant": {
            "evaluated_donor_loops": [
                index + 1
                for index, resolved in enumerate(
                    resolved_position_by_loop[:trained_loops].detach().cpu()
                )
                if bool(resolved)
            ],
            "mean_state_readout_accuracy": float(
                torch.nanmean(transplant_acc[:, :, 0])
            ),
            "mean_next_step_accuracy": float(
                torch.nanmean(transplant_acc[:, :, 1])
            ),
            "diagonal_next_step_accuracy": float(
                torch.nanmean(transplant_acc[:, :, 1].diagonal())
            ),
        },
        "controls": {
            "clean_corrupt_pair": "same permutation graph, different start node",
            "patch_control": "batch-shuffled clean branch activation",
            "circuit_control": "all same-size branch subsets are enumerated",
        },
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def parse_run_spec(text: str) -> tuple[str, Path]:
    if "=" not in text:
        raise argparse.ArgumentTypeError("run must be NAME=CHECKPOINT")
    name, raw_path = text.split("=", 1)
    if not name or not raw_path:
        raise argparse.ArgumentTypeError("run must be NAME=CHECKPOINT")
    return name, Path(raw_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Causal circuit analysis for explicit-depth graph composition checkpoints."
    )
    parser.add_argument("--run", action="append", type=parse_run_spec, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--batches", type=int, default=4)
    parser.add_argument("--overloops", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260725)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)
    summaries = {}
    for name, checkpoint in args.run:
        print(f"[depth-circuit] analyzing {name}: {checkpoint}", flush=True)
        summaries[name] = analyze_checkpoint(
            name=name,
            checkpoint=checkpoint,
            out_dir=args.out_dir,
            device=device,
            batch_size=args.batch_size,
            batches=args.batches,
            overloops=args.overloops,
            seed=args.seed,
        )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "comparison_summary.json").write_text(
        json.dumps(summaries, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
