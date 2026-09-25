from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from reasoning_loop.graph_path_loop import (
    LoopedGraphPathTransformer,
    MultiHeadSelfAttention,
    TransformerBlock,
    pick_device,
)
from reasoning_loop.graph_path_stepwise import (
    StepwiseGraphPathConfig,
    make_stepwise_batch,
)
from reasoning_loop.graph_path_temporal_intervention import (
    cache_raw_states,
    logits_from_raw_state,
    replace_answer_state,
)


def token_position_sets(node_count: int) -> dict[str, list[int] | int]:
    if node_count < 1:
        raise ValueError("node_count must be positive")
    edge = [1 + 3 * node for node in range(node_count)]
    source = [position + 1 for position in edge]
    destination = [position + 2 for position in edge]
    query = 1 + 3 * node_count
    return {
        "edge": edge,
        "source": source,
        "destination": destination,
        "query": query,
        "start": query + 1,
        "answer": query + 2,
    }


def attention_alignment_metrics(
    attention_by_loop: torch.Tensor,
    *,
    node_count: int,
    start: torch.Tensor,
    targets_by_position: torch.Tensor,
) -> dict[str, torch.Tensor]:
    if attention_by_loop.ndim != 6:
        raise ValueError(
            "attention_by_loop must have [loop, block, batch, head, query, key] axes"
        )
    loops, _, batch, _, seq_len, key_len = attention_by_loop.shape
    if seq_len != key_len:
        raise ValueError("attention matrices must be square")
    if start.shape != (batch,):
        raise ValueError("start must have shape [batch]")
    if targets_by_position.ndim != 2 or targets_by_position.shape[0] != batch:
        raise ValueError("targets_by_position must have shape [batch, position]")
    if targets_by_position.shape[1] < max(0, loops - 1):
        raise ValueError("targets_by_position does not cover all loop inputs")

    positions = token_position_sets(node_count)
    source_positions = torch.tensor(
        positions["source"],
        device=attention_by_loop.device,
    )
    destination_positions = torch.tensor(
        positions["destination"],
        device=attention_by_loop.device,
    )
    answer_position = int(positions["answer"])
    if answer_position >= seq_len:
        raise ValueError("attention sequence is shorter than the graph token layout")

    paired = attention_by_loop[
        ...,
        destination_positions,
        source_positions,
    ]
    destination_to_source = paired.mean(dim=(2, 4))

    if loops == 1:
        current_by_loop = start[:, None]
    else:
        current_by_loop = torch.cat(
            [start[:, None], targets_by_position[:, : loops - 1]],
            dim=1,
        )
    current_by_loop = current_by_loop.transpose(0, 1)
    gather_index = current_by_loop[:, None, :, None, None].expand(
        loops,
        attention_by_loop.shape[1],
        batch,
        attention_by_loop.shape[3],
        1,
    )

    answer_to_destinations = attention_by_loop[
        ...,
        answer_position,
        destination_positions,
    ]
    relevant_destination = answer_to_destinations.gather(
        dim=-1,
        index=gather_index,
    ).squeeze(-1)
    answer_to_sources = attention_by_loop[
        ...,
        answer_position,
        source_positions,
    ]
    relevant_source = answer_to_sources.gather(
        dim=-1,
        index=gather_index,
    ).squeeze(-1)
    if node_count > 1:
        incorrect_destination = (
            answer_to_destinations.sum(dim=-1) - relevant_destination
        ) / (node_count - 1)
    else:
        incorrect_destination = torch.zeros_like(relevant_destination)
    return {
        "destination_to_paired_source": destination_to_source,
        "answer_to_current_destination": relevant_destination.mean(dim=2),
        "answer_to_current_source": relevant_source.mean(dim=2),
        "answer_to_incorrect_destination_mean": incorrect_destination.mean(dim=2),
        "answer_destination_selectivity": (
            relevant_destination - incorrect_destination
        ).mean(dim=2),
    }


def head_ablation_set(
    *,
    max_loops: int,
    block: int,
    head: int,
    loop: int | None,
) -> set[tuple[int, int, int]]:
    if max_loops < 1 or block < 0 or head < 0:
        raise ValueError("max_loops must be positive and indices must be nonnegative")
    if loop is not None and not 0 <= loop < max_loops:
        raise ValueError("loop index is out of range")
    selected_loops = range(max_loops) if loop is None else (loop,)
    return {(loop_index, block, head) for loop_index in selected_loops}


def explicit_attention(
    attention: MultiHeadSelfAttention,
    x: torch.Tensor,
    *,
    ablated_heads: Sequence[int] = (),
    return_weights: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    batch, seq_len, d_model = x.shape
    qkv = attention.qkv(x).view(
        batch,
        seq_len,
        3,
        attention.n_heads,
        attention.d_head,
    )
    query, key, value = qkv.unbind(dim=2)
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)
    weights: torch.Tensor | None = None
    if return_weights:
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(
            attention.d_head
        )
        causal_mask = torch.ones(
            seq_len,
            seq_len,
            dtype=torch.bool,
            device=x.device,
        ).triu(diagonal=1)
        scores = scores.masked_fill(causal_mask, float("-inf"))
        weights = scores.softmax(dim=-1)
        attended = torch.matmul(weights, value)
    else:
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=True,
        )
    if ablated_heads:
        head_mask = torch.ones(
            attention.n_heads,
            dtype=attended.dtype,
            device=attended.device,
        )
        for head in ablated_heads:
            if not 0 <= int(head) < attention.n_heads:
                raise ValueError(f"head index {head} is out of range")
            head_mask[int(head)] = 0
        attended = attended * head_mask.view(1, -1, 1, 1)
    attended = attended.transpose(1, 2).contiguous().view(
        batch,
        seq_len,
        d_model,
    )
    return attention.out_proj(attended), weights


def instrumented_block_forward(
    block: TransformerBlock,
    x: torch.Tensor,
    *,
    ablated_heads: Sequence[int] = (),
    return_attention: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if not isinstance(block.attn, MultiHeadSelfAttention):
        raise TypeError("component circuit analysis currently supports legacy attention")
    attention_update, weights = explicit_attention(
        block.attn,
        block.ln_1(x),
        ablated_heads=ablated_heads,
        return_weights=return_attention,
    )
    if block.inner_norm_style == "ouro_sandwich_rms":
        attention_update = block.attn_out_norm(attention_update)
    x = x + attention_update
    mlp_update = block.mlp(block.ln_2(x))
    if block.inner_norm_style == "ouro_sandwich_rms":
        mlp_update = block.mlp_out_norm(mlp_update)
    return x + mlp_update, weights


def apply_instrumented_stack(
    model: LoopedGraphPathTransformer,
    x: torch.Tensor,
    *,
    ablated_heads: set[tuple[int, int]] | None = None,
    loop_index: int = 0,
) -> torch.Tensor:
    ablations = ablated_heads or set()
    for block_index in model.active_block_indices(loop_index):
        block = model.blocks[block_index]
        if not isinstance(block, TransformerBlock):
            raise TypeError("component circuit analysis requires TransformerBlock")
        heads = sorted(
            head
            for selected_block, head in ablations
            if selected_block == block_index
        )
        x, _ = instrumented_block_forward(
            block,
            x,
            ablated_heads=heads,
            return_attention=False,
        )
    if model.outer_norm is not None:
        x = model.outer_norm(x)
    return x


@torch.no_grad()
def transplant_next_step_batch(
    *,
    model: LoopedGraphPathTransformer,
    donor_tokens: torch.Tensor,
    donor_targets: torch.Tensor,
    receiver_tokens: torch.Tensor,
    receiver_successors: torch.Tensor,
    donor_loop: int,
    receiver_loop: int,
    ablated_heads: set[tuple[int, int]] | None = None,
) -> dict[str, torch.Tensor]:
    if donor_loop < 1 or receiver_loop < 1:
        raise ValueError("donor_loop and receiver_loop must be positive")
    if donor_targets.shape[1] < donor_loop:
        raise ValueError("donor_targets do not cover donor_loop")
    donor_state = cache_raw_states(
        model,
        donor_tokens,
        max_loop=donor_loop,
    )[donor_loop - 1]
    receiver_state = cache_raw_states(
        model,
        receiver_tokens,
        max_loop=receiver_loop,
    )[receiver_loop - 1]
    patched = replace_answer_state(receiver_state, donor_state)
    updated = apply_instrumented_stack(
        model,
        patched,
        ablated_heads=ablated_heads,
        loop_index=receiver_loop,
    )
    logits = logits_from_raw_state(model, updated)
    donor_current = donor_targets[:, donor_loop - 1]
    target = receiver_successors.gather(
        1,
        donor_current[:, None],
    ).squeeze(1)
    probability = logits.softmax(dim=-1).gather(
        1,
        target[:, None],
    ).squeeze(1)
    return {
        "logits": logits,
        "target": target,
        "correct": logits.argmax(dim=-1).eq(target),
        "probability": probability,
    }


def instrumented_forward_all(
    model: LoopedGraphPathTransformer,
    tokens: torch.Tensor,
    *,
    max_loops: int,
    ablated_heads: set[tuple[int, int, int]] | None = None,
    return_attention: bool = True,
) -> dict[str, Any]:
    if model.block_style != "legacy":
        raise ValueError("component circuit analysis currently supports legacy blocks")
    if max_loops < 1:
        raise ValueError("max_loops must be positive")
    ablations = ablated_heads or set()
    x = model.token_embed(tokens) + model.pos_embed.unsqueeze(0)
    logits_by_loop: list[torch.Tensor] = []
    attention_by_loop: list[torch.Tensor] = []
    for loop_index in range(max_loops):
        attention_by_block: list[torch.Tensor] = []
        for block_index, block in enumerate(model.blocks):
            if not isinstance(block, TransformerBlock):
                raise TypeError("component circuit analysis requires TransformerBlock")
            if block_index not in model.active_block_indices(loop_index):
                if return_attention:
                    attention_by_block.append(
                        torch.zeros(
                            tokens.shape[0],
                            block.attn.n_heads,
                            tokens.shape[1],
                            tokens.shape[1],
                            dtype=x.dtype,
                            device=x.device,
                        )
                    )
                continue
            heads = sorted(
                head
                for selected_loop, selected_block, head in ablations
                if selected_loop == loop_index and selected_block == block_index
            )
            x, weights = instrumented_block_forward(
                block,
                x,
                ablated_heads=heads,
                return_attention=return_attention,
            )
            if return_attention:
                assert weights is not None
                attention_by_block.append(weights)
        if model.outer_norm is not None:
            x = model.outer_norm(x)
        final_state = model.ln_final(x[:, -1, :])
        logits_by_loop.append(model.unembed(final_state)[:, : model.cfg.node_count])
        if return_attention:
            attention_by_loop.append(torch.stack(attention_by_block, dim=0))
    output: dict[str, Any] = {
        "logits_by_loop": torch.stack(logits_by_loop, dim=1),
    }
    if return_attention:
        output["attention_by_loop"] = torch.stack(attention_by_loop, dim=0)
    return output


def _correct_by_loop(
    logits_by_loop: torch.Tensor,
    targets_by_position: torch.Tensor,
) -> torch.Tensor:
    loops = logits_by_loop.shape[1]
    if targets_by_position.shape[1] < loops:
        raise ValueError("targets_by_position does not cover every loop")
    prediction = logits_by_loop.argmax(dim=-1)
    return prediction.eq(targets_by_position[:, :loops]).sum(dim=0)


def _save_heatmap(
    values: np.ndarray,
    *,
    title: str,
    path: Path,
    row_prefix: str = "block",
    col_prefix: str = "head",
) -> None:
    fig, ax = plt.subplots(
        figsize=(max(5.5, 1.0 * values.shape[1] + 2.5), max(3.5, 0.8 * values.shape[0] + 2.0))
    )
    scale = max(float(np.abs(values).max()), 1e-6)
    image = ax.imshow(values, cmap="coolwarm", vmin=-scale, vmax=scale, aspect="auto")
    ax.set_xticks(range(values.shape[1]), [f"{col_prefix} {i}" for i in range(values.shape[1])])
    ax.set_yticks(range(values.shape[0]), [f"{row_prefix} {i}" for i in range(values.shape[0])])
    ax.set_title(title)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            ax.text(
                column,
                row,
                f"{values[row, column]:.3f}",
                ha="center",
                va="center",
                fontsize=8,
            )
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


@torch.no_grad()
def analyze_checkpoint(
    *,
    name: str,
    checkpoint: Path,
    out_dir: Path,
    device: torch.device,
    max_loops: int,
    batch_size: int,
    batches: int,
    transplant_loop: int | None = None,
) -> dict[str, Any]:
    if max_loops < 1 or batch_size < 1 or batches < 1:
        raise ValueError("loop and batch counts must be positive")
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = StepwiseGraphPathConfig(**payload["config"])
    model = LoopedGraphPathTransformer(cfg).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    blocks = len(model.blocks)
    heads = cfg.n_heads
    selected_transplant_loop = (
        min(4, max_loops) if transplant_loop is None else transplant_loop
    )
    if not 1 <= selected_transplant_loop <= max_loops:
        raise ValueError("transplant_loop must be covered by max_loops")

    baseline_correct = torch.zeros(max_loops, dtype=torch.float64, device=device)
    all_loop_correct = torch.zeros(
        blocks,
        heads,
        max_loops,
        dtype=torch.float64,
        device=device,
    )
    single_loop_correct = torch.zeros(
        max_loops,
        blocks,
        heads,
        max_loops,
        dtype=torch.float64,
        device=device,
    )
    alignment_sums: dict[str, torch.Tensor] | None = None
    transplant_baseline_correct = torch.zeros(
        (),
        dtype=torch.float64,
        device=device,
    )
    transplant_baseline_probability = torch.zeros_like(transplant_baseline_correct)
    transplant_head_correct = torch.zeros(
        blocks,
        heads,
        dtype=torch.float64,
        device=device,
    )
    transplant_head_probability = torch.zeros_like(transplant_head_correct)
    count = 0
    max_equivalence_error = 0.0
    for _ in range(batches):
        tokens, targets_by_position, _, start = make_stepwise_batch(
            cfg,
            batch_size,
            device,
            path_positions=max_loops,
        )
        baseline = instrumented_forward_all(
            model,
            tokens,
            max_loops=max_loops,
            return_attention=True,
        )
        original_logits = model.forward_all(tokens, max_loops=max_loops)[
            "logits_by_loop"
        ]
        max_equivalence_error = max(
            max_equivalence_error,
            float(
                (
                    baseline["logits_by_loop"] - original_logits
                ).abs().max().detach().cpu()
            ),
        )
        baseline_correct += _correct_by_loop(
            baseline["logits_by_loop"],
            targets_by_position,
        )
        alignment = attention_alignment_metrics(
            baseline["attention_by_loop"],
            node_count=cfg.node_count,
            start=start,
            targets_by_position=targets_by_position,
        )
        if alignment_sums is None:
            alignment_sums = {
                key: value.to(torch.float64) * batch_size
                for key, value in alignment.items()
            }
        else:
            for key, value in alignment.items():
                alignment_sums[key] += value.to(torch.float64) * batch_size

        receiver_tokens, _, receiver_successors, _ = make_stepwise_batch(
            cfg,
            batch_size,
            device,
            path_positions=max_loops,
        )
        donor_state = cache_raw_states(
            model,
            tokens,
            max_loop=selected_transplant_loop,
        )[selected_transplant_loop - 1]
        receiver_state = cache_raw_states(
            model,
            receiver_tokens,
            max_loop=selected_transplant_loop,
        )[selected_transplant_loop - 1]
        patched = replace_answer_state(receiver_state, donor_state)
        donor_current = targets_by_position[:, selected_transplant_loop - 1]
        transplant_target = receiver_successors.gather(
            1,
            donor_current[:, None],
        ).squeeze(1)
        baseline_transplant_logits = logits_from_raw_state(
            model,
            apply_instrumented_stack(
                model,
                patched,
                loop_index=selected_transplant_loop,
            ),
        )
        baseline_transplant_probability = baseline_transplant_logits.softmax(
            dim=-1
        ).gather(1, transplant_target[:, None]).squeeze(1)
        transplant_baseline_correct += baseline_transplant_logits.argmax(
            dim=-1
        ).eq(transplant_target).sum()
        transplant_baseline_probability += baseline_transplant_probability.sum()

        for block in range(blocks):
            for head in range(heads):
                head_transplant_logits = logits_from_raw_state(
                    model,
                    apply_instrumented_stack(
                        model,
                        patched,
                        ablated_heads={(block, head)},
                        loop_index=selected_transplant_loop,
                    ),
                )
                head_transplant_probability = head_transplant_logits.softmax(
                    dim=-1
                ).gather(1, transplant_target[:, None]).squeeze(1)
                transplant_head_correct[block, head] += (
                    head_transplant_logits.argmax(dim=-1)
                    .eq(transplant_target)
                    .sum()
                )
                transplant_head_probability[block, head] += (
                    head_transplant_probability.sum()
                )
                all_ablated = instrumented_forward_all(
                    model,
                    tokens,
                    max_loops=max_loops,
                    ablated_heads=head_ablation_set(
                        max_loops=max_loops,
                        block=block,
                        head=head,
                        loop=None,
                    ),
                    return_attention=False,
                )
                all_loop_correct[block, head] += _correct_by_loop(
                    all_ablated["logits_by_loop"],
                    targets_by_position,
                )
                for intervention_loop in range(max_loops):
                    single_ablated = instrumented_forward_all(
                        model,
                        tokens,
                        max_loops=max_loops,
                        ablated_heads=head_ablation_set(
                            max_loops=max_loops,
                            block=block,
                            head=head,
                            loop=intervention_loop,
                        ),
                        return_attention=False,
                    )
                    single_loop_correct[intervention_loop, block, head] += (
                        _correct_by_loop(
                            single_ablated["logits_by_loop"],
                            targets_by_position,
                        )
                    )
        count += batch_size

    assert alignment_sums is not None
    baseline_accuracy = baseline_correct / count
    all_loop_accuracy = all_loop_correct / count
    single_loop_accuracy = single_loop_correct / count
    all_loop_drop = baseline_accuracy.view(1, 1, -1) - all_loop_accuracy
    single_loop_drop = (
        baseline_accuracy.view(1, 1, 1, -1) - single_loop_accuracy
    )
    local_transition_drop = torch.stack(
        [
            single_loop_drop[loop, :, :, loop]
            for loop in range(max_loops)
        ],
        dim=0,
    )
    downstream_final_drop = single_loop_drop[:, :, :, -1]
    alignment_means = {
        key: value / count for key, value in alignment_sums.items()
    }
    transplant_baseline_accuracy = transplant_baseline_correct / count
    transplant_baseline_mean_probability = transplant_baseline_probability / count
    transplant_head_accuracy = transplant_head_correct / count
    transplant_head_mean_probability = transplant_head_probability / count
    transplant_head_accuracy_drop = (
        transplant_baseline_accuracy - transplant_head_accuracy
    )
    summary = {
        "name": name,
        "checkpoint": str(checkpoint),
        "checkpoint_step": payload.get("step"),
        "config": payload["config"],
        "examples": count,
        "max_loops": max_loops,
        "transplant_loop": selected_transplant_loop,
        "max_instrumented_logit_error": max_equivalence_error,
        "baseline_rolling_accuracy": baseline_accuracy.detach().cpu().tolist(),
        "all_loop_head_ablation_accuracy": all_loop_accuracy.detach().cpu().tolist(),
        "all_loop_head_ablation_drop": all_loop_drop.detach().cpu().tolist(),
        "single_loop_head_ablation_accuracy": single_loop_accuracy.detach().cpu().tolist(),
        "single_loop_head_ablation_drop": single_loop_drop.detach().cpu().tolist(),
        "local_transition_drop": local_transition_drop.detach().cpu().tolist(),
        "downstream_final_drop": downstream_final_drop.detach().cpu().tolist(),
        "attention_alignment": {
            key: value.detach().cpu().tolist()
            for key, value in alignment_means.items()
        },
        "transplant_next_step": {
            "baseline_accuracy": float(
                transplant_baseline_accuracy.detach().cpu()
            ),
            "baseline_mean_probability": float(
                transplant_baseline_mean_probability.detach().cpu()
            ),
            "head_ablation_accuracy": transplant_head_accuracy.detach()
            .cpu()
            .tolist(),
            "head_ablation_mean_probability": transplant_head_mean_probability.detach()
            .cpu()
            .tolist(),
            "head_ablation_accuracy_drop": transplant_head_accuracy_drop.detach()
            .cpu()
            .tolist(),
        },
    }
    run_dir = out_dir / name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    _save_heatmap(
        all_loop_drop[:, :, -1].detach().cpu().numpy(),
        title="Final f^L accuracy drop: head ablated at every loop",
        path=run_dir / "all_loop_final_drop.png",
    )
    _save_heatmap(
        local_transition_drop.mean(dim=0).detach().cpu().numpy(),
        title="Mean local transition drop: head ablated at one loop",
        path=run_dir / "single_loop_local_drop.png",
    )
    _save_heatmap(
        downstream_final_drop.mean(dim=0).detach().cpu().numpy(),
        title="Mean downstream final drop: head ablated at one loop",
        path=run_dir / "single_loop_downstream_drop.png",
    )
    selectivity = alignment_means["answer_destination_selectivity"].mean(dim=0)
    _save_heatmap(
        selectivity.detach().cpu().numpy(),
        title="Answer attention selectivity for the current edge destination",
        path=run_dir / "answer_destination_selectivity.png",
    )
    _save_heatmap(
        transplant_head_accuracy_drop.detach().cpu().numpy(),
        title="Receiver-successor transplant drop after one head ablation",
        path=run_dir / "transplant_next_step_head_drop.png",
    )
    return summary


def _parse_run_spec(text: str) -> tuple[str, Path]:
    if "=" not in text:
        raise ValueError("run spec must be NAME=/path/to/checkpoint.pt")
    name, path = text.split("=", 1)
    if not name:
        raise ValueError("run name must not be empty")
    return name, Path(path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Locate and causally ablate the repeated graph-successor circuit."
    )
    parser.add_argument("--run", action="append", required=True, help="NAME=checkpoint.pt")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-loops", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--batches", type=int, default=4)
    parser.add_argument("--transplant-loop", type=int)
    parser.add_argument("--device", default="auto")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)
    combined: list[dict[str, Any]] = []
    for spec in args.run:
        name, checkpoint = _parse_run_spec(spec)
        combined.append(
            analyze_checkpoint(
                name=name,
                checkpoint=checkpoint,
                out_dir=args.out_dir,
                device=device,
                max_loops=args.max_loops,
                batch_size=args.batch_size,
                batches=args.batches,
                transplant_loop=args.transplant_loop,
            )
        )
    (args.out_dir / "combined_summary.json").write_text(
        json.dumps(combined, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
