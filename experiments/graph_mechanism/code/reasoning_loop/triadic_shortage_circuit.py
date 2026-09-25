from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

from reasoning_loop.triadic_shortage import (
    StaticOperandWorkspaceCell,
    TriadicShortageModel,
    all_triples,
    make_visibility_mask,
)
from reasoning_loop.triadic_shortage_diagnostics import load_checkpoint
from reasoning_loop.triadic_shortage_train import pick_device


TensorPatch = Mapping[int, torch.Tensor]
HeadPatch = Mapping[int, Mapping[int, torch.Tensor]]
HeadAblation = Mapping[int, set[int]]


def manual_attention(
    cell: StaticOperandWorkspaceCell,
    workspace: torch.Tensor,
    static_operands: torch.Tensor,
    visible_operands: torch.Tensor,
    *,
    head_context_patch: Mapping[int, torch.Tensor] | None = None,
    ablate_heads: set[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluate MHA while exposing per-head pre-output-projection contexts."""
    attention = cell.attention
    if not attention._qkv_same_embed_dim:
        raise ValueError("manual_attention requires a shared q/k/v input dimension")
    query = cell.norm_attention(workspace).unsqueeze(1)
    operand_values = cell.norm_attention(static_operands)
    key_value = torch.cat((query, operand_values), dim=1)
    d_model = query.shape[-1]
    n_heads = attention.num_heads
    head_dim = d_model // n_heads
    weight = attention.in_proj_weight
    bias = attention.in_proj_bias
    q_weight, k_weight, v_weight = weight.chunk(3, dim=0)
    if bias is None:
        q_bias = k_bias = v_bias = None
    else:
        q_bias, k_bias, v_bias = bias.chunk(3, dim=0)
    q = F.linear(query, q_weight, q_bias)
    k = F.linear(key_value, k_weight, k_bias)
    v = F.linear(key_value, v_weight, v_bias)

    def split_heads(value: torch.Tensor) -> torch.Tensor:
        return value.reshape(value.shape[0], value.shape[1], n_heads, head_dim).transpose(1, 2)

    q = split_heads(q)
    k = split_heads(k)
    v = split_heads(v)
    scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(head_dim)
    padding_mask = torch.cat(
        (
            torch.zeros(
                (workspace.shape[0], 1),
                dtype=torch.bool,
                device=workspace.device,
            ),
            ~visible_operands,
        ),
        dim=1,
    )
    scores = scores.masked_fill(padding_mask[:, None, None, :], -torch.inf)
    weights = scores.softmax(dim=-1)
    contexts = torch.matmul(weights, v)
    if head_context_patch:
        contexts = contexts.clone()
        for head, value in head_context_patch.items():
            if not 0 <= head < n_heads:
                raise ValueError(f"head index {head} is out of range")
            if value.shape != contexts[:, head].shape:
                raise ValueError("patched head context has the wrong shape")
            contexts[:, head] = value
    if ablate_heads:
        contexts = contexts.clone()
        for head in ablate_heads:
            if not 0 <= head < n_heads:
                raise ValueError(f"head index {head} is out of range")
            contexts[:, head] = 0.0
    joined = contexts.transpose(1, 2).reshape(workspace.shape[0], 1, d_model)
    output = attention.out_proj(joined).squeeze(1)
    return output, weights, contexts


@torch.no_grad()
def run_intervened(
    model: TriadicShortageModel,
    operands: torch.Tensor,
    visibility: torch.Tensor,
    *,
    initial_workspace: torch.Tensor | None = None,
    start_loop: int = 0,
    stop_loop: int | None = None,
    zero_attention_loops: set[int] | None = None,
    zero_mlp_loops: set[int] | None = None,
    attention_patch: TensorPatch | None = None,
    mlp_patch: TensorPatch | None = None,
    head_context_patch: HeadPatch | None = None,
    ablate_heads: HeadAblation | None = None,
    mode_ids: torch.Tensor | None = None,
    return_trace: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    device = next(model.parameters()).device
    operands = operands.to(device)
    visibility = visibility.to(device)
    if mode_ids is not None:
        mode_ids = mode_ids.to(device)
    final_loop = model.cfg.loops if stop_loop is None else stop_loop
    if not 0 <= start_loop < final_loop <= model.cfg.loops:
        raise ValueError(
            "start_loop and stop_loop must define a nonempty configured range"
        )
    static_operands = model.encode_operands(operands)
    if initial_workspace is None:
        if start_loop != 0:
            raise ValueError(
                "initial_workspace is required when start_loop is nonzero"
            )
        workspace = model._initial_workspace(
            operands.shape[0],
            mode_ids=mode_ids,
        )
    else:
        if initial_workspace.shape != (
            operands.shape[0],
            model.cfg.d_model,
        ):
            raise ValueError("initial_workspace has the wrong shape")
        workspace = initial_workspace.to(device)
    logits: list[torch.Tensor] = []
    states: list[torch.Tensor] = []
    trace: dict[str, torch.Tensor] = {}
    zero_attention_loops = zero_attention_loops or set()
    zero_mlp_loops = zero_mlp_loops or set()
    attention_patch = attention_patch or {}
    mlp_patch = mlp_patch or {}
    head_context_patch = head_context_patch or {}
    ablate_heads = ablate_heads or {}
    for loop_index in range(start_loop, final_loop):
        cell = model.cell_for_loop(loop_index)
        attention_out, attention_weights, head_contexts = manual_attention(
            cell,
            workspace,
            static_operands,
            visibility[:, loop_index],
            head_context_patch=head_context_patch.get(loop_index),
            ablate_heads=ablate_heads.get(loop_index),
        )
        if loop_index in attention_patch:
            attention_out = attention_patch[loop_index]
        if loop_index in zero_attention_loops:
            attention_out = torch.zeros_like(attention_out)
        residual_mid = workspace + attention_out
        mlp_out = cell.mlp(cell.norm_mlp(residual_mid))
        if loop_index in mlp_patch:
            mlp_out = mlp_patch[loop_index]
        if loop_index in zero_mlp_loops:
            mlp_out = torch.zeros_like(mlp_out)
        workspace = residual_mid + mlp_out
        states.append(workspace)
        logits.append(model.readout(model.readout_norm(workspace)))
        if return_trace:
            prefix = f"loop{loop_index}"
            trace[f"{prefix}.attention_out"] = attention_out.detach()
            trace[f"{prefix}.attention_weights"] = attention_weights.detach()
            trace[f"{prefix}.head_contexts"] = head_contexts.detach()
            trace[f"{prefix}.residual_mid"] = residual_mid.detach()
            trace[f"{prefix}.mlp_out"] = mlp_out.detach()
            trace[f"{prefix}.workspace_out"] = workspace.detach()
    return torch.stack(logits, dim=1), torch.stack(states, dim=1), trace


def _accuracy(logits: torch.Tensor, target: torch.Tensor) -> float:
    return float(logits[:, -1].argmax(dim=-1).eq(target).float().mean().item())


@torch.no_grad()
def component_ablation_table(
    model: TriadicShortageModel,
    operands: torch.Tensor,
    labels: torch.Tensor,
    visibility: torch.Tensor,
) -> dict[str, Any]:
    baseline_logits, _, trace = run_intervened(
        model,
        operands,
        visibility,
        return_trace=True,
    )
    baseline = _accuracy(baseline_logits, labels)
    rows = []
    for loop_index in range(model.cfg.loops):
        attention_logits, _, _ = run_intervened(
            model,
            operands,
            visibility,
            zero_attention_loops={loop_index},
        )
        mlp_logits, _, _ = run_intervened(
            model,
            operands,
            visibility,
            zero_mlp_loops={loop_index},
        )
        row: dict[str, Any] = {
            "loop": loop_index + 1,
            "visible_operand": loop_index + 1 if loop_index < 3 else None,
            "baseline_accuracy": baseline,
            "zero_attention_accuracy": _accuracy(attention_logits, labels),
            "zero_mlp_accuracy": _accuracy(mlp_logits, labels),
            "attention_norm": float(
                trace[f"loop{loop_index}.attention_out"].norm(dim=-1).mean().item()
            ),
            "mlp_norm": float(
                trace[f"loop{loop_index}.mlp_out"].norm(dim=-1).mean().item()
            ),
            "attention_weights": trace[f"loop{loop_index}.attention_weights"]
            .mean(dim=0)
            .squeeze(1)
            .cpu()
            .tolist(),
        }
        row["attention_drop"] = baseline - row["zero_attention_accuracy"]
        row["mlp_drop"] = baseline - row["zero_mlp_accuracy"]
        rows.append(row)

    phase_rows = []
    for name, loops in (
        ("acquisition", set(range(min(3, model.cfg.loops)))),
        ("computation", set(range(3, model.cfg.loops))),
    ):
        for component in ("attention", "mlp"):
            kwargs = (
                {"zero_attention_loops": loops}
                if component == "attention"
                else {"zero_mlp_loops": loops}
            )
            logits, _, _ = run_intervened(model, operands, visibility, **kwargs)
            accuracy = _accuracy(logits, labels)
            phase_rows.append(
                {
                    "phase": name,
                    "component": component,
                    "ablated_loops": [loop + 1 for loop in sorted(loops)],
                    "accuracy": accuracy,
                    "drop": baseline - accuracy,
                }
            )
    return {"baseline_accuracy": baseline, "per_loop": rows, "per_phase": phase_rows}


def _all_nonempty_subsets(n_heads: int) -> list[tuple[int, ...]]:
    return [
        subset
        for size in range(1, n_heads + 1)
        for subset in itertools.combinations(range(n_heads), size)
    ]


@torch.no_grad()
def head_patch_table(
    model: TriadicShortageModel,
    receiver_operands: torch.Tensor,
    visibility: torch.Tensor,
    *,
    seed: int,
    pass_accuracy: float = 0.90,
    complement_max: float = 0.50,
) -> dict[str, Any]:
    device = next(model.parameters()).device
    receiver_operands = receiver_operands.to(device)
    visibility = visibility.to(device)
    generator = torch.Generator(device=device).manual_seed(seed)
    offsets = torch.randint(
        1,
        model.cfg.p,
        (receiver_operands.shape[0], 3),
        generator=generator,
        device=device,
    )
    receiver_target = receiver_operands.sum(dim=1).remainder(model.cfg.p)
    receiver_logits, _, _ = run_intervened(model, receiver_operands, visibility)
    rows = []
    for operand_index in range(3):
        donor_operands = receiver_operands.clone()
        donor_operands[:, operand_index] = (
            donor_operands[:, operand_index] + offsets[:, operand_index]
        ).remainder(model.cfg.p)
        donor_target = donor_operands.sum(dim=1).remainder(model.cfg.p)
        donor_logits, _, donor_trace = run_intervened(
            model,
            donor_operands,
            visibility,
            return_trace=True,
        )
        contexts = donor_trace[f"loop{operand_index}.head_contexts"]
        permutation = torch.roll(
            torch.arange(contexts.shape[0], device=device),
            shifts=1,
        )
        subset_rows = []
        all_heads = set(range(model.cfg.n_heads))
        for subset in _all_nonempty_subsets(model.cfg.n_heads):
            patch = {
                operand_index: {
                    head: contexts[:, head]
                    for head in subset
                }
            }
            patched_logits, _, _ = run_intervened(
                model,
                receiver_operands,
                visibility,
                head_context_patch=patch,
            )
            random_patch = {
                operand_index: {
                    head: contexts[permutation, head]
                    for head in subset
                }
            }
            random_logits, _, _ = run_intervened(
                model,
                receiver_operands,
                visibility,
                head_context_patch=random_patch,
            )
            complement = tuple(sorted(all_heads.difference(subset)))
            complement_patch = {
                operand_index: {
                    head: contexts[:, head]
                    for head in complement
                }
            }
            complement_logits, _, _ = run_intervened(
                model,
                receiver_operands,
                visibility,
                head_context_patch=complement_patch,
            )
            subset_rows.append(
                {
                    "heads": list(subset),
                    "size": len(subset),
                    "patch_accuracy": _accuracy(patched_logits, donor_target),
                    "random_patch_accuracy": _accuracy(random_logits, donor_target),
                    "complement_heads": list(complement),
                    "complement_patch_accuracy": _accuracy(
                        complement_logits,
                        donor_target,
                    ),
                }
            )
        candidates = [
            row
            for row in subset_rows
            if row["patch_accuracy"] >= pass_accuracy
            and row["complement_patch_accuracy"] <= complement_max
        ]
        selected = min(
            candidates,
            key=lambda row: (row["size"], -row["patch_accuracy"]),
            default=None,
        )
        rows.append(
            {
                "operand_index": operand_index,
                "loop": operand_index + 1,
                "receiver_accuracy": _accuracy(receiver_logits, receiver_target),
                "donor_accuracy": _accuracy(donor_logits, donor_target),
                "selected_circuit": selected,
                "subsets": subset_rows,
            }
        )
    return {
        "pass_accuracy": pass_accuracy,
        "complement_max": complement_max,
        "rows": rows,
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")


@torch.no_grad()
def analyze_circuit(
    checkpoint: Path,
    out_dir: Path,
    *,
    device: torch.device,
    sample_size: int = 1024,
    seed: int = 91_001,
) -> dict[str, Any]:
    model, payload = load_checkpoint(checkpoint, device)
    if model.cfg.loops < 3:
        raise ValueError("sequential circuit analysis requires at least three loops")
    operands, labels = all_triples(model.cfg.p, device=device)
    heldout_idx = payload["split"]["heldout_idx"].to(device)
    subset = heldout_idx[: min(sample_size, heldout_idx.numel())]
    operands = operands[subset]
    labels = labels[subset]
    visibility = make_visibility_mask(
        "sequential",
        operands.shape[0],
        model.cfg.loops,
        device,
    )
    direct_logits, direct_states = model(operands, visibility)
    manual_logits, manual_states, _ = run_intervened(model, operands, visibility)
    parity = {
        "max_logit_delta": float((direct_logits - manual_logits).abs().max().item()),
        "max_state_delta": float((direct_states - manual_states).abs().max().item()),
    }
    if parity["max_logit_delta"] > 2e-4 or parity["max_state_delta"] > 2e-4:
        raise RuntimeError(f"manual circuit evaluator failed parity: {parity}")
    ablations = component_ablation_table(model, operands, labels, visibility)
    patches = head_patch_table(
        model,
        operands,
        visibility,
        seed=seed,
    )
    selected = [row["selected_circuit"] for row in patches["rows"]]
    scorecard = {
        "checkpoint": str(checkpoint.resolve()),
        "condition": payload["condition"],
        "seed": int(payload["seed"]),
        "step": int(payload["step"]),
        "parity": parity,
        "baseline_accuracy": ablations["baseline_accuracy"],
        "selected_head_circuits": selected,
        "all_operands_have_selected_circuit": all(row is not None for row in selected),
    }
    _write_json(out_dir / "component_ablation.json", ablations)
    _write_json(out_dir / "head_patch.json", patches)
    _write_json(out_dir / "scorecard.json", scorecard)
    return scorecard


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Causal circuit analysis for triadic shortage.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--sample-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=91_001)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result = analyze_circuit(
        args.checkpoint,
        args.out_dir,
        device=pick_device(args.device),
        sample_size=args.sample_size,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
