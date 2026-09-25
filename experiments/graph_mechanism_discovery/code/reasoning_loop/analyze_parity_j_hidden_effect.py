"""Measure the trained parity J on real hidden states and attention inputs.

This is a frozen-model, no-gradient diagnostic.  The model follows the actual
full-J trajectory.  At every controlled step, the script also evaluates a
one-step no-J counterfactual from the exact same source state.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from reasoning_loop.paper_length_telomere import (
    answer_cross_entropy,
    exact_match,
    generate_paper_batch,
    load_backbone,
    load_controller,
    pick_device,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--controller", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--lengths", nargs="+", type=int, default=(20, 40, 50, 75, 100))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--batches", type=int, default=2)
    parser.add_argument("--seed", type=int, default=420001)
    parser.add_argument("--top-ranks", nargs="+", type=int, default=(1, 2, 4, 8, 16, 32, 48))
    parser.add_argument("--top-modes", type=int, default=8)
    parser.add_argument(
        "--paper-mode",
        action="store_true",
        help="reject checkpoints outside the corrected input-once Parity protocol",
    )
    return parser.parse_args()


class MeanTable:
    def __init__(self) -> None:
        self.sums: dict[tuple[Any, ...], dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        self.counts: dict[tuple[Any, ...], float] = defaultdict(float)

    def add(
        self,
        key: tuple[Any, ...],
        values: dict[str, float],
        *,
        count: float,
    ) -> None:
        self.counts[key] += count
        for name, value in values.items():
            self.sums[key][name] += float(value) * count

    def rows(self, key_names: tuple[str, ...]) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        for key in sorted(self.sums):
            count = self.counts[key]
            row = {name: value for name, value in zip(key_names, key, strict=True)}
            row["examples"] = count
            row.update(
                {name: value / count for name, value in self.sums[key].items()}
            )
            output.append(row)
        return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def selected_flat(value: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return value.flatten(1)
    batch = value.shape[0]
    return value[mask].reshape(batch, -1)


def norms(value: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    return torch.linalg.vector_norm(selected_flat(value.float(), mask), dim=1)


def mean_ratio(
    numerator: torch.Tensor,
    denominator: torch.Tensor,
) -> float:
    return float((numerator / denominator.clamp_min(1e-12)).mean().item())


def mean_cosine(
    left: torch.Tensor,
    right: torch.Tensor,
    mask: torch.Tensor | None,
) -> float:
    left_flat = selected_flat(left.float(), mask)
    right_flat = selected_flat(right.float(), mask)
    denominator = (
        torch.linalg.vector_norm(left_flat, dim=1)
        * torch.linalg.vector_norm(right_flat, dim=1)
    ).clamp_min(1e-12)
    return float(((left_flat * right_flat).sum(dim=1) / denominator).mean().item())


def attention_steps(length: int) -> set[int]:
    return {
        step
        for step in (
            2,
            5,
            10,
            20,
            max(2, length // 2),
            max(2, length - 10),
            max(2, length - 5),
            max(2, length - 2),
            max(2, length - 1),
            length,
            length + 1,
        )
        if 2 <= step <= length + 1
    }


@torch.no_grad()
def trace_single_layer(
    model,
    state: torch.Tensor,
    embedded: torch.Tensor,
) -> dict[str, torch.Tensor]:
    if len(model.layers) != 1:
        raise ValueError("parity trace expects one shared physical layer")
    layer = model.layers[0]
    pre_attention = state.float() + embedded.float()
    normalized = layer.attention_norm(pre_attention)
    batch, length, dimension = normalized.shape
    attention = layer.attention
    qkv = attention.qkv(normalized).view(
        batch,
        length,
        3,
        attention.n_heads,
        attention.head_dim,
    )
    query, key, value = qkv.unbind(dim=2)
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(
        attention.head_dim
    )
    causal_mask = torch.ones(
        length,
        length,
        dtype=torch.bool,
        device=state.device,
    ).triu(diagonal=1)
    scores = scores.masked_fill(causal_mask, float("-inf"))
    pattern = scores.softmax(dim=-1)
    head_output = torch.matmul(pattern, value)
    joined = head_output.transpose(1, 2).reshape(batch, length, dimension)
    attention_output = attention.output(joined)
    after_attention = pre_attention + attention_output
    mlp_output = layer.mlp(layer.mlp_norm(after_attention))
    before_final_norm = after_attention + mlp_output
    final = model.final_norm(before_final_norm)
    return {
        "query": query,
        "key": key,
        "value": value,
        "pattern": pattern,
        "head_output": head_output,
        "attention_output": attention_output,
        "mlp_output": mlp_output,
        "final": final,
    }


def head_metrics(
    *,
    model,
    base_trace: dict[str, torch.Tensor],
    full_trace: dict[str, torch.Tensor],
    answer_position: int,
) -> list[dict[str, float]]:
    layer = model.layers[0]
    attention = layer.attention
    keys = slice(0, answer_position + 1)
    output: list[dict[str, float]] = []
    for head in range(attention.n_heads):
        base_query = base_trace["query"][:, head, answer_position]
        full_query = full_trace["query"][:, head, answer_position]
        base_key = base_trace["key"][:, head, keys]
        full_key = full_trace["key"][:, head, keys]
        base_value = base_trace["value"][:, head, keys]
        full_value = full_trace["value"][:, head, keys]
        base_pattern = base_trace["pattern"][:, head, answer_position, keys]
        full_pattern = full_trace["pattern"][:, head, answer_position, keys]
        base_head_output = base_trace["head_output"][:, head, answer_position]
        full_head_output = full_trace["head_output"][:, head, answer_position]
        joined_start = head * attention.head_dim
        joined_end = joined_start + attention.head_dim
        output_weight = attention.output.weight[:, joined_start:joined_end]
        output_delta = (full_head_output - base_head_output) @ output_weight.T
        output.append(
            {
                "query_relative_change": mean_ratio(
                    torch.linalg.vector_norm(full_query - base_query, dim=1),
                    torch.linalg.vector_norm(base_query, dim=1),
                ),
                "key_relative_change": mean_ratio(
                    torch.linalg.vector_norm((full_key - base_key).flatten(1), dim=1),
                    torch.linalg.vector_norm(base_key.flatten(1), dim=1),
                ),
                "value_relative_change": mean_ratio(
                    torch.linalg.vector_norm((full_value - base_value).flatten(1), dim=1),
                    torch.linalg.vector_norm(base_value.flatten(1), dim=1),
                ),
                "answer_attention_total_variation": float(
                    (0.5 * (full_pattern - base_pattern).abs().sum(dim=1))
                    .mean()
                    .item()
                ),
                "head_output_relative_change": mean_ratio(
                    torch.linalg.vector_norm(full_head_output - base_head_output, dim=1),
                    torch.linalg.vector_norm(base_head_output, dim=1),
                ),
                "output_projected_delta_norm": float(
                    torch.linalg.vector_norm(output_delta, dim=1).mean().item()
                ),
            }
        )
    return output


def step_geometry(
    *,
    source: torch.Tensor,
    controlled: torch.Tensor,
    no_j_next: torch.Tensor,
    full_next: torch.Tensor,
    diagonal_delta: torch.Tensor,
    a: torch.Tensor,
    b_factor: torch.Tensor,
    bias: torch.Tensor,
    delta_u: torch.Tensor,
    delta_singular: torch.Tensor,
    delta_vh: torch.Tensor,
    top_ranks: Iterable[int],
    mask: torch.Tensor | None,
) -> dict[str, float]:
    d_effect = source.float() * diagonal_delta
    ab_effect = (source.float() @ a) @ b_factor
    bias_effect = bias.view(1, 1, -1).expand_as(source)
    weight_effect = d_effect + ab_effect
    correction = controlled.float() - source.float()
    executor_update = no_j_next.float() - source.float()
    post_f_effect = full_next.float() - no_j_next.float()
    source_norm = norms(source, mask)
    correction_norm = norms(correction, mask)
    weight_norm = norms(weight_effect, mask)
    d_norm = norms(d_effect, mask)
    ab_norm = norms(ab_effect, mask)
    bias_norm = norms(bias_effect, mask)
    executor_update_norm = norms(executor_update, mask)
    no_j_next_norm = norms(no_j_next, mask)
    post_f_norm = norms(post_f_effect, mask)
    values = {
        "source_norm": float(source_norm.mean().item()),
        "correction_norm": float(correction_norm.mean().item()),
        "correction_to_source": mean_ratio(correction_norm, source_norm),
        "weight_to_source": mean_ratio(weight_norm, source_norm),
        "D_to_source": mean_ratio(d_norm, source_norm),
        "AB_to_source": mean_ratio(ab_norm, source_norm),
        "bias_to_source": mean_ratio(bias_norm, source_norm),
        "D_to_correction": mean_ratio(d_norm, correction_norm),
        "AB_to_correction": mean_ratio(ab_norm, correction_norm),
        "bias_to_correction": mean_ratio(bias_norm, correction_norm),
        "weight_bias_cosine": mean_cosine(weight_effect, bias_effect, mask),
        "AB_bias_cosine": mean_cosine(ab_effect, bias_effect, mask),
        "correction_to_executor_update": mean_ratio(
            correction_norm, executor_update_norm
        ),
        "post_F_effect_norm": float(post_f_norm.mean().item()),
        "post_F_effect_to_no_J_state": mean_ratio(post_f_norm, no_j_next_norm),
        "post_F_gain_over_correction": mean_ratio(post_f_norm, correction_norm),
    }
    weight_squared = weight_norm.square().clamp_min(1e-20)
    for rank in top_ranks:
        coefficient = source.float() @ delta_u[:, :rank]
        top = (coefficient * delta_singular[:rank]) @ delta_vh[:rank]
        top_norm = norms(top, mask)
        values[f"top{rank}_weight_effect_energy"] = float(
            (top_norm.square() / weight_squared).mean().item()
        )
        values[f"top{rank}_to_full_correction"] = mean_ratio(
            top_norm, correction_norm
        )
    return values


def add_mode_metrics(
    table: MeanTable,
    *,
    length: int,
    output_step: int,
    source: torch.Tensor,
    targets: torch.Tensor,
    answer_position: int,
    delta_u: torch.Tensor,
    delta_singular: torch.Tensor,
    top_modes: int,
) -> None:
    coefficients = source[:, answer_position].float() @ delta_u[:, :top_modes]
    labels = targets[:, answer_position]
    for mode in range(top_modes):
        for label_name, selected in (
            ("all", torch.ones_like(labels, dtype=torch.bool)),
            ("parity0", labels.eq(0)),
            ("parity1", labels.eq(1)),
        ):
            if not bool(selected.any()):
                continue
            values = coefficients[selected, mode]
            table.add(
                (length, output_step, mode + 1, label_name),
                {
                    "coefficient_mean": float(values.mean().item()),
                    "coefficient_squared_mean": float(values.square().mean().item()),
                    "write_contribution_squared_mean": float(
                        (values * delta_singular[mode]).square().mean().item()
                    ),
                },
                count=float(selected.sum().item()),
            )


def validate_controller_backbone(
    controller_checkpoint: str, requested_checkpoint: Path
) -> None:
    if Path(controller_checkpoint).resolve() != requested_checkpoint.resolve():
        raise ValueError("controller belongs to a different backbone")


@torch.no_grad()
def run_diagnostic(args: argparse.Namespace) -> dict[str, Any]:
    device = pick_device(args.device)
    model, spec, backbone_payload = load_backbone(
        args.checkpoint, device=device, paper_mode=args.paper_mode
    )
    controller, controller_payload = load_controller(args.controller, device=device)
    if spec.name != "parity":
        raise ValueError("this diagnostic is fixed to parity")
    validate_controller_backbone(controller_payload["checkpoint"], args.checkpoint)
    if controller_payload["anchor_step"] != 1:
        raise ValueError("main diagnostic expects anchor 1")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in controller.parameters():
        parameter.requires_grad_(False)

    diagonal_delta = controller.diagonal.float() - 1.0
    a = controller.A.float()
    b_factor = controller.B.float()
    bias = controller.bias.float()
    delta_weight = torch.diag(diagonal_delta) + a @ b_factor
    delta_u, delta_singular, delta_vh = torch.linalg.svd(
        delta_weight, full_matrices=False
    )

    step_table = MeanTable()
    mode_table = MeanTable()
    head_table = MeanTable()
    trace_max_error = 0.0
    correction_reconstruction_error = 0.0
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)

    for length in args.lengths:
        for _ in range(args.batches):
            batch = generate_paper_batch(
                spec,
                batch_size=args.batch_size,
                min_length=length,
                max_length=length,
                fixed_length=length,
                generator=generator,
            ).to(device)
            state = torch.zeros_like(model.read_in(batch.inputs))
            for output_step in range(1, length + 2):
                embedded = model.input_embeddings(
                    batch.inputs, step_index=output_step
                )
                if output_step == 1:
                    state = model.recurrent_step(state, embedded)
                    continue
                source = state
                controlled = controller(source)
                correction_reconstruction_error = max(
                    correction_reconstruction_error,
                    float(
                        (
                            controlled
                            - source
                            - source.float() * diagonal_delta
                            - (source.float() @ a) @ b_factor
                            - bias
                        )
                        .abs()
                        .max()
                        .item()
                    ),
                )
                full_next = model.recurrent_step(controlled, embedded)
                no_j_next = model.recurrent_step(source, embedded)
                source_logits = model.decode(source).float()
                post_j_logits = model.decode(controlled).float()
                full_next_logits = model.decode(full_next).float()
                no_j_next_logits = model.decode(no_j_next).float()
                common = {
                    "source_exact_match": exact_match(source_logits, batch),
                    "post_J_exact_match": exact_match(post_j_logits, batch),
                    "full_next_exact_match": exact_match(full_next_logits, batch),
                    "no_J_next_exact_match": exact_match(no_j_next_logits, batch),
                    "full_minus_no_J_next_exact_match": (
                        exact_match(full_next_logits, batch)
                        - exact_match(no_j_next_logits, batch)
                    ),
                    "source_answer_nll": answer_cross_entropy(source_logits, batch),
                    "post_J_answer_nll": answer_cross_entropy(post_j_logits, batch),
                    "full_next_answer_nll": answer_cross_entropy(full_next_logits, batch),
                    "no_J_next_answer_nll": answer_cross_entropy(no_j_next_logits, batch),
                }
                for scope, mask in (
                    ("all_tokens", None),
                    ("answer_tokens", batch.answer_mask),
                ):
                    values = step_geometry(
                        source=source,
                        controlled=controlled,
                        no_j_next=no_j_next,
                        full_next=full_next,
                        diagonal_delta=diagonal_delta,
                        a=a,
                        b_factor=b_factor,
                        bias=bias,
                        delta_u=delta_u,
                        delta_singular=delta_singular,
                        delta_vh=delta_vh,
                        top_ranks=args.top_ranks,
                        mask=mask,
                    )
                    if scope == "answer_tokens":
                        answer_logit_delta = selected_flat(
                            full_next_logits - no_j_next_logits,
                            batch.answer_mask,
                        )
                        answer_logit_base = selected_flat(
                            no_j_next_logits,
                            batch.answer_mask,
                        )
                        values["answer_logit_relative_change"] = mean_ratio(
                            torch.linalg.vector_norm(answer_logit_delta, dim=1),
                            torch.linalg.vector_norm(answer_logit_base, dim=1),
                        )
                    values.update(common)
                    step_table.add(
                        (length, output_step, output_step - length, scope),
                        values,
                        count=args.batch_size,
                    )

                add_mode_metrics(
                    mode_table,
                    length=length,
                    output_step=output_step,
                    source=source,
                    targets=batch.targets,
                    answer_position=length,
                    delta_u=delta_u,
                    delta_singular=delta_singular,
                    top_modes=args.top_modes,
                )

                if output_step in attention_steps(length):
                    base_trace = trace_single_layer(model, source, embedded)
                    full_trace = trace_single_layer(model, controlled, embedded)
                    trace_max_error = max(
                        trace_max_error,
                        float((base_trace["final"] - no_j_next).abs().max().item()),
                        float((full_trace["final"] - full_next).abs().max().item()),
                    )
                    metrics = head_metrics(
                        model=model,
                        base_trace=base_trace,
                        full_trace=full_trace,
                        answer_position=length,
                    )
                    for head, values in enumerate(metrics):
                        head_table.add(
                            (length, output_step, output_step - length, head),
                            values,
                            count=args.batch_size,
                        )
                state = full_next

    step_rows = step_table.rows(("length", "output_step", "relative_to_target", "scope"))
    mode_rows = mode_table.rows(
        ("length", "output_step", "mode", "parity_class")
    )
    for row in mode_rows:
        row["coefficient_rms"] = math.sqrt(max(float(row["coefficient_squared_mean"]), 0.0))
        row["write_contribution_rms"] = math.sqrt(
            max(float(row["write_contribution_squared_mean"]), 0.0)
        )
    head_rows = head_table.rows(
        ("length", "output_step", "relative_to_target", "head")
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "step_metrics.csv", step_rows)
    write_csv(args.out_dir / "mode_metrics.csv", mode_rows)
    write_csv(args.out_dir / "attention_head_metrics.csv", head_rows)

    target_rows = [
        row
        for row in step_rows
        if row["scope"] == "answer_tokens"
        and row["output_step"] == row["length"]
    ]
    target_plus_one_rows = [
        row
        for row in step_rows
        if row["scope"] == "answer_tokens"
        and row["output_step"] == row["length"] + 1
    ]
    target_heads = [
        row
        for row in head_rows
        if row["output_step"] == row["length"]
    ]
    head_means: dict[int, dict[str, float]] = {}
    for head in range(model.config.n_heads):
        selected = [row for row in target_heads if row["head"] == head]
        if not selected:
            continue
        head_means[head] = {
            name: float(np.mean([float(row[name]) for row in selected]))
            for name in (
                "answer_attention_total_variation",
                "output_projected_delta_norm",
                "query_relative_change",
                "key_relative_change",
                "value_relative_change",
            )
        }
    top_heads = sorted(
        (
            {"head": head, **values}
            for head, values in head_means.items()
        ),
        key=lambda row: row["output_projected_delta_norm"],
        reverse=True,
    )[:12]
    summary = {
        "status": "complete",
        "execution": {
            "device": str(device),
            "diagnostic_kind": "local frozen-model no-gradient mechanistic diagnostic",
            "seed": args.seed,
            "batch_size": args.batch_size,
            "batches": args.batches,
            "examples_per_length": args.batch_size * args.batches,
            "paper_mode": bool(args.paper_mode),
        },
        "model": {
            "checkpoint": str(args.checkpoint),
            "checkpoint_step": backbone_payload["step"],
            "backbone_seed": backbone_payload["seed"],
            "task": spec.name,
            "shared_physical_layers": spec.block_layers,
            "d_model": model.config.d_model,
            "heads": model.config.n_heads,
            "trained_logical_lengths": [1, spec.train_max_length],
        },
        "controller": {
            "artifact": str(args.controller),
            "seed": controller_payload["seed"],
            "anchor_step": controller_payload["anchor_step"],
            "trained_logical_lengths": controller_payload[
                "controller_sampled_logical_lengths"
            ],
            "loss": controller_payload["loss"],
            "state_loss_weight": controller_payload["state_loss_weight"],
        },
        "evaluation_lengths": list(args.lengths),
        "intervention": (
            "Follow the full-J trajectory. At every controlled effective step, "
            "compare F(J(h),x) with one-step F(h,x) from the identical source h."
        ),
        "validation": {
            "maximum_manual_attention_trace_error": trace_max_error,
            "maximum_correction_reconstruction_error": correction_reconstruction_error,
        },
        "target_step_answer_metrics": target_rows,
        "target_plus_one_answer_metrics": target_plus_one_rows,
        "top_attention_heads_at_target_by_output_delta": top_heads,
        "evidence_boundary": (
            "Geometry and within-state one-step counterfactual localization. "
            "Head metrics are descriptive mediation candidates, not head ablations."
        ),
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    plot_results(step_rows, head_rows, args.out_dir)
    return summary


def plot_results(
    step_rows: list[dict[str, Any]],
    head_rows: list[dict[str, Any]],
    out_dir: Path,
) -> None:
    answer_rows = [row for row in step_rows if row["scope"] == "answer_tokens"]
    lengths = sorted({int(row["length"]) for row in answer_rows})
    figure, axes = plt.subplots(2, 3, figsize=(16, 9.2))
    for length in lengths:
        rows = [row for row in answer_rows if row["length"] == length]
        normalized_step = [row["output_step"] / length for row in rows]
        axes[0, 0].plot(
            normalized_step,
            [row["correction_to_source"] for row in rows],
            label=f"L{length}",
        )
        axes[0, 1].plot(
            normalized_step,
            [row["post_F_effect_to_no_J_state"] for row in rows],
            label=f"L{length}",
        )
        axes[0, 2].plot(
            normalized_step,
            [row["correction_to_executor_update"] for row in rows],
            label=f"L{length}",
        )
        axes[1, 0].plot(
            normalized_step,
            [row["top4_weight_effect_energy"] for row in rows],
            label=f"L{length}",
        )
        near = [row for row in rows if -12 <= row["relative_to_target"] <= 1]
        axes[1, 1].plot(
            [row["relative_to_target"] for row in near],
            [row["full_minus_no_J_next_exact_match"] for row in near],
            marker="o",
            markersize=2.5,
            label=f"L{length}",
        )
        axes[1, 2].plot(
            normalized_step,
            [row["bias_to_correction"] for row in rows],
            label=f"L{length}",
        )
    axes[0, 0].set_title("Actual ||J(h)-h|| / ||h||")
    axes[0, 1].set_title("Effect after F / no-J next-state norm")
    axes[0, 2].set_title("J correction / one-step executor update")
    axes[1, 0].set_title("Top-4 energy in actual weight correction")
    axes[1, 1].set_title("One-step exact-match gain near target")
    axes[1, 2].set_title("Bias norm / full correction norm")
    for axis in (axes[0, 0], axes[0, 1], axes[0, 2], axes[1, 0], axes[1, 2]):
        axis.set_xlabel("output step / target step")
    axes[1, 1].set_xlabel("output step minus target step")
    axes[0, 0].set_ylabel("ratio")
    axes[0, 1].set_ylabel("ratio")
    axes[0, 2].set_ylabel("ratio")
    axes[1, 0].set_ylabel("squared-energy fraction")
    axes[1, 1].set_ylabel("full-J minus one-step no-J accuracy")
    axes[1, 2].set_ylabel("ratio")
    for axis in axes.ravel():
        axis.axvline(1.0 if axis is not axes[1, 1] else 0.0, color="black", linestyle="--", linewidth=0.7)
        axis.grid(alpha=0.22)
    axes[0, 0].legend(fontsize=8)
    figure.suptitle("Parity anchor-1 J on real hidden states")
    figure.tight_layout()
    figure.savefig(out_dir / "hidden_effect_by_loop.png", dpi=190)
    plt.close(figure)

    target_heads = [row for row in head_rows if row["output_step"] == row["length"]]
    if not target_heads:
        return
    head_count = max(int(row["head"]) for row in target_heads) + 1
    tv = np.zeros((len(lengths), head_count))
    projected = np.zeros_like(tv)
    for length_index, length in enumerate(lengths):
        selected = [row for row in target_heads if row["length"] == length]
        for row in selected:
            head = int(row["head"])
            tv[length_index, head] = row["answer_attention_total_variation"]
            projected[length_index, head] = row["output_projected_delta_norm"]
    mean_projected = projected.mean(axis=0)
    top = np.argsort(mean_projected)[::-1][:12]
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    image = axes[0].imshow(tv, aspect="auto", cmap="magma")
    axes[0].set_title("Attention-pattern TV at target")
    axes[0].set_xlabel("head")
    axes[0].set_ylabel("length")
    axes[0].set_yticks(np.arange(len(lengths)), labels=[f"L{x}" for x in lengths])
    figure.colorbar(image, ax=axes[0], fraction=0.046)
    image = axes[1].imshow(projected, aspect="auto", cmap="viridis")
    axes[1].set_title("Per-head output-projected delta norm")
    axes[1].set_xlabel("head")
    axes[1].set_ylabel("length")
    axes[1].set_yticks(np.arange(len(lengths)), labels=[f"L{x}" for x in lengths])
    figure.colorbar(image, ax=axes[1], fraction=0.046)
    axes[2].bar([str(value) for value in top], mean_projected[top])
    axes[2].set_title("Top heads by J-induced output change")
    axes[2].set_xlabel("head")
    axes[2].set_ylabel("mean projected delta norm")
    axes[2].grid(axis="y", alpha=0.22)
    figure.suptitle("Descriptive attention mediation candidates at registered target")
    figure.tight_layout()
    figure.savefig(out_dir / "attention_head_effect_at_target.png", dpi=190)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    summary = run_diagnostic(args)
    compact = {
        "status": summary["status"],
        "device": summary["execution"]["device"],
        "examples_per_length": summary["execution"]["examples_per_length"],
        "trace_error": summary["validation"]["maximum_manual_attention_trace_error"],
        "reconstruction_error": summary["validation"][
            "maximum_correction_reconstruction_error"
        ],
    }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
