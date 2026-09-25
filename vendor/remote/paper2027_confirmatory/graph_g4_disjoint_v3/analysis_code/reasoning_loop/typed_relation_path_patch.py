from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics
from typing import Any, Mapping

import torch
import torch.nn.functional as F

from reasoning_loop.typed_relation_composition import (
    RelationBatch,
    TypedRelationModel,
    TypedRelationWorkspaceCell,
    make_relation_batch,
    relation_visibility,
)
from reasoning_loop.typed_relation_circuit import load_checkpoint, run_intervened


def _batch_with_targets(
    *,
    f: torch.Tensor,
    g: torch.Tensor,
    query: torch.Tensor,
) -> RelationBatch:
    first = g.gather(1, query[:, None]).squeeze(1)
    endpoint = f.gather(1, first[:, None]).squeeze(1)
    return RelationBatch(
        f=f,
        g=g,
        query=query,
        targets=torch.stack((first, endpoint), dim=1),
        composition_order="g_then_f",
    )


def make_minimal_counterfactuals(
    clean: RelationBatch,
) -> dict[str, RelationBatch]:
    if clean.composition_order != "g_then_f":
        raise ValueError("minimal counterfactuals require g_then_f composition")
    node_count = clean.f.shape[1]
    row = torch.arange(clean.batch_size, device=clean.query.device)

    query_swap = (clean.query + 1) % node_count

    g_other = (clean.query + 1) % node_count
    g_swap = clean.g.clone()
    g_at_query = clean.g[row, clean.query].clone()
    g_at_other = clean.g[row, g_other].clone()
    g_swap[row, clean.query] = g_at_other
    g_swap[row, g_other] = g_at_query

    first = clean.targets[:, 0]
    f_other = (first + 1) % node_count
    f_swap = clean.f.clone()
    f_at_first = clean.f[row, first].clone()
    f_at_other = clean.f[row, f_other].clone()
    f_swap[row, first] = f_at_other
    f_swap[row, f_other] = f_at_first

    return {
        "query_swap": _batch_with_targets(
            f=clean.f,
            g=clean.g,
            query=query_swap,
        ),
        "g_at_x_swap": _batch_with_targets(
            f=clean.f,
            g=g_swap,
            query=clean.query,
        ),
        "f_at_y_swap": _batch_with_targets(
            f=f_swap,
            g=clean.g,
            query=clean.query,
        ),
    }


def _attention_parts(
    cell: TypedRelationWorkspaceCell,
    workspace: torch.Tensor,
    static_edges: torch.Tensor,
    visible_edges: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    attention = cell.attention
    query = cell.norm_attention(workspace).unsqueeze(1)
    edge_values = cell.norm_attention(static_edges)
    key_value = torch.cat((query, edge_values), dim=1)
    d_model = query.shape[-1]
    n_heads = attention.num_heads
    head_dim = d_model // n_heads
    q_weight, k_weight, v_weight = attention.in_proj_weight.chunk(3, dim=0)
    if attention.in_proj_bias is None:
        q_bias = k_bias = v_bias = None
    else:
        q_bias, k_bias, v_bias = attention.in_proj_bias.chunk(3, dim=0)

    def split_heads(value: torch.Tensor) -> torch.Tensor:
        return value.reshape(
            value.shape[0], value.shape[1], n_heads, head_dim
        ).transpose(1, 2)

    q = split_heads(F.linear(query, q_weight, q_bias))
    k = split_heads(F.linear(key_value, k_weight, k_bias))
    values = split_heads(F.linear(key_value, v_weight, v_bias))
    weights = _weights_from_qk(q, k, visible_edges)
    contexts = torch.matmul(weights, values)
    return q, k, values, weights, contexts


def _weights_from_qk(
    q: torch.Tensor,
    k: torch.Tensor,
    visible_edges: torch.Tensor,
) -> torch.Tensor:
    scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
    padding_mask = torch.cat(
        (
            torch.zeros(
                (q.shape[0], 1),
                dtype=torch.bool,
                device=q.device,
            ),
            ~visible_edges,
        ),
        dim=1,
    )
    return scores.masked_fill(
        padding_mask[:, None, None, :], -torch.inf
    ).softmax(dim=-1)


def _project_contexts(
    cell: TypedRelationWorkspaceCell,
    contexts: torch.Tensor,
) -> torch.Tensor:
    batch_size, n_heads, _, head_dim = contexts.shape
    joined = contexts.transpose(1, 2).reshape(
        batch_size,
        1,
        n_heads * head_dim,
    )
    return cell.attention.out_proj(joined).squeeze(1)


@torch.no_grad()
def run_path_patched(
    model: TypedRelationModel,
    *,
    receiver: RelationBatch,
    receiver_visibility: torch.Tensor,
    donor: RelationBatch,
    donor_visibility: torch.Tensor,
    head_patches: Mapping[int, Mapping[int, str]],
    mlp_patches: set[int] | None = None,
    workspace_patches_after: set[int] | None = None,
    stop_loop: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    if receiver.batch_size != donor.batch_size:
        raise ValueError("receiver and donor batch sizes must match")
    expected = (
        receiver.batch_size,
        model.cfg.loops,
        2 * model.cfg.node_count,
    )
    if receiver_visibility.shape != expected or donor_visibility.shape != expected:
        raise ValueError("visibility has the wrong shape")
    final_loop = model.cfg.loops if stop_loop is None else stop_loop
    if not 1 <= final_loop <= model.cfg.loops:
        raise ValueError("stop_loop must be within configured loops")
    allowed_patches = {
        "q_vector",
        "key_vectors",
        "qk_pattern",
        "value",
        "z_context",
    }
    mlp_patches = mlp_patches or set()
    workspace_patches_after = workspace_patches_after or set()
    for loop_index in mlp_patches | workspace_patches_after:
        if not 0 <= loop_index < final_loop:
            raise ValueError("component patch loop is out of range")
    for loop_index, patches in head_patches.items():
        if not 0 <= loop_index < final_loop:
            raise ValueError("patch loop is out of range")
        for head, patch_kind in patches.items():
            if not 0 <= head < model.cfg.n_heads:
                raise ValueError("patch head is out of range")
            if patch_kind not in allowed_patches:
                raise ValueError("unknown head patch kind")

    donor_edges = model.encode_edges(donor)
    receiver_edges = model.encode_edges(receiver)
    donor_workspace = model._initial_workspace(donor.query)
    receiver_workspace = model._initial_workspace(receiver.query)
    logits: list[torch.Tensor] = []
    states: list[torch.Tensor] = []
    trace: dict[str, torch.Tensor] = {}
    for loop_index in range(final_loop):
        cell = model.cell_for_loop(loop_index)
        (
            donor_q,
            donor_k,
            donor_values,
            donor_weights,
            donor_contexts,
        ) = _attention_parts(
            cell,
            donor_workspace,
            donor_edges,
            donor_visibility[:, loop_index],
        )
        donor_attention = _project_contexts(cell, donor_contexts)
        donor_residual_mid = donor_workspace + donor_attention
        donor_mlp = cell.mlp(cell.norm_mlp(donor_residual_mid))
        donor_workspace = donor_residual_mid + donor_mlp

        (
            receiver_q,
            receiver_k,
            receiver_values,
            receiver_weights,
            receiver_contexts,
        ) = _attention_parts(
            cell,
            receiver_workspace,
            receiver_edges,
            receiver_visibility[:, loop_index],
        )
        receiver_q = receiver_q.clone()
        receiver_k = receiver_k.clone()
        receiver_weights = receiver_weights.clone()
        receiver_values = receiver_values.clone()
        receiver_contexts = receiver_contexts.clone()
        for head, patch_kind in head_patches.get(loop_index, {}).items():
            if patch_kind == "q_vector":
                receiver_q[:, head] = donor_q[:, head]
                receiver_weights[:, head] = _weights_from_qk(
                    receiver_q[:, head : head + 1],
                    receiver_k[:, head : head + 1],
                    receiver_visibility[:, loop_index],
                )[:, 0]
                receiver_contexts[:, head] = torch.matmul(
                    receiver_weights[:, head], receiver_values[:, head]
                )
            elif patch_kind == "key_vectors":
                receiver_k[:, head] = donor_k[:, head]
                receiver_weights[:, head] = _weights_from_qk(
                    receiver_q[:, head : head + 1],
                    receiver_k[:, head : head + 1],
                    receiver_visibility[:, loop_index],
                )[:, 0]
                receiver_contexts[:, head] = torch.matmul(
                    receiver_weights[:, head], receiver_values[:, head]
                )
            elif patch_kind == "qk_pattern":
                receiver_weights[:, head] = donor_weights[:, head]
                receiver_contexts[:, head] = torch.matmul(
                    receiver_weights[:, head], receiver_values[:, head]
                )
            elif patch_kind == "value":
                receiver_values[:, head] = donor_values[:, head]
                receiver_contexts[:, head] = torch.matmul(
                    receiver_weights[:, head], receiver_values[:, head]
                )
            else:
                receiver_contexts[:, head] = donor_contexts[:, head]

        receiver_attention = _project_contexts(cell, receiver_contexts)
        receiver_residual_mid = receiver_workspace + receiver_attention
        receiver_mlp = cell.mlp(cell.norm_mlp(receiver_residual_mid))
        if loop_index in mlp_patches:
            receiver_mlp = donor_mlp
        receiver_workspace = receiver_residual_mid + receiver_mlp
        if loop_index in workspace_patches_after:
            receiver_workspace = donor_workspace
        logits.append(model.readout(model.readout_norm(receiver_workspace)))
        states.append(receiver_workspace)
        prefix = f"loop{loop_index}"
        trace[f"{prefix}.receiver_weights"] = receiver_weights.detach()
        trace[f"{prefix}.receiver_q"] = receiver_q.detach()
        trace[f"{prefix}.receiver_k"] = receiver_k.detach()
        trace[f"{prefix}.receiver_values"] = receiver_values.detach()
        trace[f"{prefix}.receiver_contexts"] = receiver_contexts.detach()
        trace[f"{prefix}.donor_weights"] = donor_weights.detach()
        trace[f"{prefix}.donor_q"] = donor_q.detach()
        trace[f"{prefix}.donor_k"] = donor_k.detach()
        trace[f"{prefix}.donor_values"] = donor_values.detach()
        trace[f"{prefix}.donor_contexts"] = donor_contexts.detach()
        trace[f"{prefix}.receiver_mlp_out"] = receiver_mlp.detach()
        trace[f"{prefix}.donor_mlp_out"] = donor_mlp.detach()
        trace[f"{prefix}.receiver_workspace_out"] = receiver_workspace.detach()
        trace[f"{prefix}.donor_workspace_out"] = donor_workspace.detach()
    return torch.stack(logits, dim=1), torch.stack(states, dim=1), trace


@torch.no_grad()
def analyze_checkpoint_path_patching(
    checkpoint: Path,
    out_dir: Path,
    *,
    device: torch.device,
    examples: int,
    seed: int,
) -> dict[str, Any]:
    if examples < 1:
        raise ValueError("examples must be positive")
    model, payload = load_checkpoint(checkpoint, device)
    train_loops = int(payload["train_loops"])
    if train_loops != 2:
        raise ValueError("path analysis requires exactly two trained loops")
    composition_order = payload.get("composition_order", "f_then_g")
    if composition_order != "g_then_f":
        raise ValueError("path analysis currently targets f(g(x))")
    generator = torch.Generator(device=device).manual_seed(seed)
    clean = make_relation_batch(
        batch_size=examples,
        node_count=model.cfg.node_count,
        device=device,
        generator=generator,
        composition_order=composition_order,
    )
    corruptions = make_minimal_counterfactuals(clean)
    visibility = relation_visibility(
        "full",
        batch_size=examples,
        loops=model.cfg.loops,
        node_count=model.cfg.node_count,
        device=device,
        composition_order=composition_order,
    )

    def scores(logits: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
        result: dict[str, float] = {}
        for name, readout_index, target_index in (
            ("stage1", 0, 0),
            ("endpoint", 1, 1),
        ):
            selected_logits = logits[:, readout_index]
            target = targets[:, target_index]
            prediction = selected_logits.argmax(dim=-1)
            target_logit = selected_logits.gather(1, target[:, None]).squeeze(1)
            distractors = selected_logits.masked_fill(
                F.one_hot(target, num_classes=selected_logits.shape[1]).bool(),
                -torch.inf,
            )
            result[f"{name}_accuracy"] = float(
                prediction.eq(target).float().mean().item()
            )
            result[f"{name}_margin"] = float(
                (target_logit - distractors.max(dim=1).values).mean().item()
            )
        return result

    clean_logits, _, _ = run_intervened(
        model,
        clean,
        visibility,
        stop_loop=train_loops,
    )
    clean_scores = scores(clean_logits, clean.targets)
    baselines: dict[str, Any] = {}
    patch_rows: list[dict[str, Any]] = []

    def effect_fraction(
        value: float,
        corrupt_value: float,
        clean_value: float,
        direction: str,
    ) -> float | None:
        denominator = clean_value - corrupt_value
        if abs(denominator) < 1e-12:
            return None
        if direction == "patch_in":
            return (value - corrupt_value) / denominator
        return (clean_value - value) / denominator

    for corruption_name, corrupt in corruptions.items():
        corrupt_logits, _, _ = run_intervened(
            model,
            corrupt,
            visibility,
            stop_loop=train_loops,
        )
        corrupt_clean_target_scores = scores(corrupt_logits, clean.targets)
        corrupt_own_target_scores = scores(corrupt_logits, corrupt.targets)
        baselines[corruption_name] = {
            "against_clean_target": corrupt_clean_target_scores,
            "against_corrupt_target": corrupt_own_target_scores,
        }

        def append_row(
            *,
            direction: str,
            scope: str,
            loop_index: int,
            head: int | None,
            patch_kind: str,
            logits: torch.Tensor,
        ) -> None:
            patched_scores = scores(logits, clean.targets)
            row: dict[str, Any] = {
                "direction": direction,
                "corruption": corruption_name,
                "scope": scope,
                "loop": loop_index + 1,
                "head": head,
                "patch_kind": patch_kind,
                **patched_scores,
            }
            for metric in (
                "stage1_accuracy",
                "stage1_margin",
                "endpoint_accuracy",
                "endpoint_margin",
            ):
                effect = effect_fraction(
                    patched_scores[metric],
                    corrupt_clean_target_scores[metric],
                    clean_scores[metric],
                    direction,
                )
                row[f"{metric}_effect_fraction"] = effect
                row[f"{metric}_recovery"] = effect
            patch_rows.append(row)

        for direction in ("patch_in", "patch_out"):
            receiver = corrupt if direction == "patch_in" else clean
            donor = clean if direction == "patch_in" else corrupt
            for loop_index in range(train_loops):
                for head in range(model.cfg.n_heads):
                    for patch_kind in (
                        "q_vector",
                        "key_vectors",
                        "qk_pattern",
                        "value",
                        "z_context",
                    ):
                        logits, _, _ = run_path_patched(
                            model,
                            receiver=receiver,
                            receiver_visibility=visibility,
                            donor=donor,
                            donor_visibility=visibility,
                            head_patches={loop_index: {head: patch_kind}},
                            stop_loop=train_loops,
                        )
                        append_row(
                            direction=direction,
                            scope="single_head",
                            loop_index=loop_index,
                            head=head,
                            patch_kind=patch_kind,
                            logits=logits,
                        )
                for patch_kind in (
                    "q_vector",
                    "key_vectors",
                    "qk_pattern",
                    "value",
                    "z_context",
                ):
                    logits, _, _ = run_path_patched(
                        model,
                        receiver=receiver,
                        receiver_visibility=visibility,
                        donor=donor,
                        donor_visibility=visibility,
                        head_patches={
                            loop_index: {
                                head: patch_kind for head in range(model.cfg.n_heads)
                            }
                        },
                        stop_loop=train_loops,
                    )
                    append_row(
                        direction=direction,
                        scope="all_heads",
                        loop_index=loop_index,
                        head=None,
                        patch_kind=patch_kind,
                        logits=logits,
                    )
                logits, _, _ = run_path_patched(
                    model,
                    receiver=receiver,
                    receiver_visibility=visibility,
                    donor=donor,
                    donor_visibility=visibility,
                    head_patches={},
                    mlp_patches={loop_index},
                    stop_loop=train_loops,
                )
                append_row(
                    direction=direction,
                    scope="component",
                    loop_index=loop_index,
                    head=None,
                    patch_kind="mlp_out",
                    logits=logits,
                )
                logits, _, _ = run_path_patched(
                    model,
                    receiver=receiver,
                    receiver_visibility=visibility,
                    donor=donor,
                    donor_visibility=visibility,
                    head_patches={},
                    workspace_patches_after={loop_index},
                    stop_loop=train_loops,
                )
                append_row(
                    direction=direction,
                    scope="component",
                    loop_index=loop_index,
                    head=None,
                    patch_kind="workspace_out",
                    logits=logits,
                )

    result = {
        "checkpoint": str(checkpoint.resolve()),
        "architecture": model.cfg.architecture,
        "model_seed": int(payload["seed"]),
        "analysis_seed": seed,
        "examples": examples,
        "clean_baseline": clean_scores,
        "baselines": baselines,
        "patch_rows": patch_rows,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "path_patches.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(patch_rows[0]))
        writer.writeheader()
        writer.writerows(patch_rows)
    (out_dir / "summary.json").write_text(
        json.dumps(result, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return result


def aggregate_path_patch_runs(
    run_dirs: list[Path],
    out_dir: Path,
) -> dict[str, Any]:
    if not run_dirs:
        raise ValueError("at least one run directory is required")
    per_seed: list[dict[str, Any]] = []
    run_summaries: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        summary = json.loads(
            (run_dir / "summary.json").read_text(encoding="utf-8")
        )
        run_summaries.append(
            {
                "architecture": summary["architecture"],
                "model_seed": int(summary["model_seed"]),
                "clean_baseline": summary["clean_baseline"],
                "baselines": summary["baselines"],
            }
        )
        for row in summary["patch_rows"]:
            per_seed.append(
                {
                    "architecture": summary["architecture"],
                    "seed": int(summary["model_seed"]),
                    **row,
                }
            )

    identity_fields = (
        "architecture",
        "direction",
        "corruption",
        "scope",
        "loop",
        "head",
        "patch_kind",
    )
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in per_seed:
        identity = tuple(row[field] for field in identity_fields)
        grouped.setdefault(identity, []).append(row)

    aggregate: list[dict[str, Any]] = []
    for identity in sorted(grouped, key=lambda value: tuple(str(item) for item in value)):
        selected = grouped[identity]
        aggregate_row = dict(zip(identity_fields, identity, strict=True))
        aggregate_row["seed_count"] = len({int(row["seed"]) for row in selected})
        metric_fields = [
            field
            for field, value in selected[0].items()
            if field not in {*identity_fields, "seed"}
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
        ]
        for field in metric_fields:
            values = [
                float(row[field])
                for row in selected
                if row.get(field) is not None
            ]
            if not values:
                continue
            aggregate_row[f"{field}_mean"] = statistics.mean(values)
            aggregate_row[f"{field}_std"] = (
                statistics.pstdev(values) if len(values) > 1 else 0.0
            )
        aggregate.append(aggregate_row)

    result = {
        "run_dirs": [str(path.resolve()) for path in run_dirs],
        "run_summaries": run_summaries,
        "per_seed": per_seed,
        "aggregate": aggregate,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    for path, rows in (
        (out_dir / "per_seed_path_patches.csv", per_seed),
        (out_dir / "aggregate_path_patches.csv", aggregate),
    ):
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (out_dir / "summary.json").write_text(
        json.dumps(result, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run counterfactual QK/value/context path patching."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--examples", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=93_001)
    parser.add_argument("--device", default="auto")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    device_name = (
        "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    )
    if device_name == "auto":
        device_name = "cpu"
    result = analyze_checkpoint_path_patching(
        args.checkpoint,
        args.out_dir,
        device=torch.device(device_name),
        examples=args.examples,
        seed=args.seed,
    )
    print(
        json.dumps(
            {
                "architecture": result["architecture"],
                "model_seed": result["model_seed"],
                "examples": result["examples"],
                "clean_baseline": result["clean_baseline"],
                "patch_rows": len(result["patch_rows"]),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
