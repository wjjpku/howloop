"""Map the shared compressed J subspace into the next loop's attention circuit."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from reasoning_loop.analyze_graph_path_j_anti_compression import build_stage_delta
from reasoning_loop.analyze_graph_path_j_compressed_subspace_circuit import (
    AGES,
    atomic_json,
    load_bank_and_bases,
)
from reasoning_loop.graph_path_depth_circuit import fixed_depth_batch, load_checkpoint
from reasoning_loop.graph_path_functional_circuit import (
    _attention_parts,
    _attention_pattern,
    explicit_depth_position_groups,
)
from reasoning_loop.graph_path_loop import TransformerBlock, pick_device, set_seed
from reasoning_loop.graph_path_telomere_overloop import advance_nodes


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--bank-artifact", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--examples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--graph-seeds", type=int, nargs="+", default=(849001, 849002))
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--singular-floor", type=float, default=0.5)
    parser.add_argument("--random-bases", type=int, default=16)
    parser.add_argument("--allow-relocated-checkpoint", action="store_true")
    return parser.parse_args(argv)


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def random_orthonormal(
    dimension: int, rank: int, rng: np.random.Generator
) -> np.ndarray:
    basis, _ = np.linalg.qr(rng.standard_normal((dimension, rank)))
    return basis[:, :rank]


def subspace_energy_fraction(value: torch.Tensor, basis: torch.Tensor) -> float:
    if basis.ndim != 2 or value.shape[-1] != basis.shape[0]:
        raise ValueError("basis dimension does not match value")
    projected = value.float() @ basis.float()
    return float(
        projected.square().sum()
        / value.float().square().sum().clamp_min(1e-12)
    )


def static_qkv_alignment(
    *, model, named_bases: dict[str, torch.Tensor], rank: int
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    chance = rank / model.cfg.d_model
    for block_index, block in enumerate(model.blocks):
        if not isinstance(block, TransformerBlock):
            raise TypeError("legacy TransformerBlock required")
        qkv = block.attn.qkv.weight.float().reshape(
            3, block.attn.n_heads, block.attn.d_head, model.cfg.d_model
        )
        for component_index, component in enumerate(("q", "k", "v")):
            for head in range(block.attn.n_heads):
                weight = qkv[component_index, head]
                for basis_name, basis in named_bases.items():
                    fraction = subspace_energy_fraction(weight, basis)
                    rows.append(
                        {
                            "block": block_index + 1,
                            "head": head,
                            "component": component,
                            "basis": basis_name,
                            "weight_energy_fraction": fraction,
                            "enrichment_over_rank_fraction": fraction / chance,
                        }
                    )
    return rows


def _b2_pattern_from_input(model, loop_input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    block1 = model.blocks[0]
    block2 = model.blocks[1]
    if not isinstance(block1, TransformerBlock) or not isinstance(block2, TransformerBlock):
        raise TypeError("two legacy blocks required")
    b2_input = block1(loop_input)
    q, k, _ = _attention_parts(block2.attn, block2.ln_1(b2_input))
    return _attention_pattern(q, k), b2_input


def _b2_pattern_direct(model, b2_input: torch.Tensor) -> torch.Tensor:
    block2 = model.blocks[1]
    if not isinstance(block2, TransformerBlock):
        raise TypeError("legacy block required")
    q, k, _ = _attention_parts(block2.attn, block2.ln_1(b2_input))
    return _attention_pattern(q, k)


def _mass(
    pattern: torch.Tensor,
    *, head: int,
    answer_position: int,
    destination: torch.Tensor,
) -> torch.Tensor:
    batch = torch.arange(pattern.shape[0], device=pattern.device)
    return pattern[batch, head, answer_position, destination]


def _gradient_rows(
    *,
    model,
    loop_input: torch.Tensor,
    destination: torch.Tensor,
    answer_position: int,
    position_groups: dict[str, tuple[int, ...]],
    named_bases: dict[str, torch.Tensor],
    rank: int,
    metadata: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[int, torch.Tensor], torch.Tensor]:
    rows: list[dict[str, Any]] = []
    chance = rank / model.cfg.d_model
    loop_live = loop_input.detach().requires_grad_(True)
    pattern, b2_from_loop = _b2_pattern_from_input(model, loop_live)
    loop_gradients: dict[int, torch.Tensor] = {}
    for head in range(model.cfg.n_heads):
        objective = _mass(
            pattern,
            head=head,
            answer_position=answer_position,
            destination=destination,
        ).clamp_min(1e-12).log().mean()
        gradient = torch.autograd.grad(
            objective,
            loop_live,
            retain_graph=head + 1 < model.cfg.n_heads,
        )[0]
        loop_gradients[head] = gradient.detach()
        for group_name, positions in position_groups.items():
            selected = gradient[:, list(positions)]
            for basis_name, basis in named_bases.items():
                fraction = subspace_energy_fraction(selected, basis)
                rows.append(
                    {
                        **metadata,
                        "gradient_site": "loop_input",
                        "head": head,
                        "position_group": group_name,
                        "basis": basis_name,
                        "gradient_energy_fraction": fraction,
                        "enrichment_over_rank_fraction": fraction / chance,
                        "correct_destination_mass": float(
                            _mass(
                                pattern,
                                head=head,
                                answer_position=answer_position,
                                destination=destination,
                            ).mean().detach()
                        ),
                    }
                )
    b2_live = b2_from_loop.detach().requires_grad_(True)
    direct_pattern = _b2_pattern_direct(model, b2_live)
    for head in range(model.cfg.n_heads):
        objective = _mass(
            direct_pattern,
            head=head,
            answer_position=answer_position,
            destination=destination,
        ).clamp_min(1e-12).log().mean()
        gradient = torch.autograd.grad(
            objective,
            b2_live,
            retain_graph=head + 1 < model.cfg.n_heads,
        )[0]
        for group_name, positions in position_groups.items():
            selected = gradient[:, list(positions)]
            for basis_name, basis in named_bases.items():
                fraction = subspace_energy_fraction(selected, basis)
                rows.append(
                    {
                        **metadata,
                        "gradient_site": "block2_input",
                        "head": head,
                        "position_group": group_name,
                        "basis": basis_name,
                        "gradient_energy_fraction": fraction,
                        "enrichment_over_rank_fraction": fraction / chance,
                        "correct_destination_mass": float(
                            _mass(
                                direct_pattern,
                                head=head,
                                answer_position=answer_position,
                                destination=destination,
                            ).mean().detach()
                        ),
                    }
                )
    return rows, loop_gradients, pattern.detach()


def _directional_cosine(gradient: torch.Tensor, delta: torch.Tensor) -> float:
    left = gradient.float().flatten(1)
    right = delta.float().flatten(1)
    numerator = (left * right).sum(-1)
    denominator = left.norm(dim=-1) * right.norm(dim=-1)
    return float((numerator / denominator.clamp_min(1e-12)).mean())


def _operator_effect_rows(
    *,
    model,
    pre_j: torch.Tensor,
    post_j: torch.Tensor,
    true_delta: torch.Tensor,
    random_delta: torch.Tensor,
    loop_gradients: dict[int, torch.Tensor],
    baseline_pattern: torch.Tensor,
    destination: torch.Tensor,
    answer_position: int,
    output_basis: torch.Tensor,
    metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    true_effect = pre_j.float() @ true_delta
    random_effect = pre_j.float() @ random_delta
    scale = true_effect.norm() / random_effect.norm().clamp_min(1e-12)
    random_effect = random_effect * scale
    states = {
        "bottom_anti": post_j.float() + true_effect,
        "random_state_effect_matched": post_j.float() + random_effect,
    }
    rows: list[dict[str, Any]] = []
    for condition, state in states.items():
        with torch.no_grad():
            pattern, _ = _b2_pattern_from_input(model, state.to(post_j.dtype))
        effect = true_effect if condition == "bottom_anti" else random_effect
        for head in range(model.cfg.n_heads):
            baseline_mass = _mass(
                baseline_pattern,
                head=head,
                answer_position=answer_position,
                destination=destination,
            )
            changed_mass = _mass(
                pattern,
                head=head,
                answer_position=answer_position,
                destination=destination,
            )
            rows.append(
                {
                    **metadata,
                    "condition": condition,
                    "head": head,
                    "state_effect_rms": float(effect.square().mean().sqrt()),
                    "effect_output_bottom_fraction": subspace_energy_fraction(
                        effect, output_basis
                    ),
                    "gradient_effect_cosine": _directional_cosine(
                        loop_gradients[head], effect
                    ),
                    "destination_mass_baseline": float(baseline_mass.mean()),
                    "destination_mass_changed": float(changed_mass.mean()),
                    "destination_mass_delta": float(
                        (changed_mass - baseline_mass).mean()
                    ),
                    "random_effect_scale": float(scale),
                }
            )
    return rows


def aggregate(rows: Sequence[dict[str, Any]], keys: Sequence[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(tuple(row[key] for key in keys), []).append(row)
    output: list[dict[str, Any]] = []
    for group, parts in sorted(groups.items(), key=lambda item: tuple(map(str, item[0]))):
        result = dict(zip(keys, group, strict=True))
        for key in sorted({key for row in parts for key in row} - set(keys)):
            values = [float(row[key]) for row in parts if isinstance(row.get(key), (int, float))]
            if values:
                result[key] = float(np.mean(values))
        result["rows"] = len(parts)
        output.append(result)
    return output


def plot_gradient(rows: Sequence[dict[str, Any]], path: Path) -> None:
    selected = [
        row for row in rows
        if row["gradient_site"] == "loop_input"
        and row["position_group"] == "answer"
    ]
    if not selected:
        return
    bases = sorted({str(row["basis"]) for row in selected})
    x = np.arange(4)
    width = 0.8 / len(bases)
    figure, axis = plt.subplots(figsize=(9, 5), dpi=180)
    for index, basis in enumerate(bases):
        chosen = {int(row["head"]): row for row in selected if row["basis"] == basis}
        axis.bar(
            x + (index - (len(bases) - 1) / 2) * width,
            [float(chosen[head]["enrichment_over_rank_fraction"]) for head in range(4)],
            width,
            label=basis,
        )
    axis.axhline(1.0, color="black", linewidth=0.8)
    axis.set(
        xticks=x,
        xticklabels=[f"B2H{head}" for head in range(4)],
        ylabel="gradient energy enrichment over rank/d",
        title="Which residual subspaces control destination routing?",
    )
    axis.legend(fontsize=8)
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.examples % args.batch_size:
        raise ValueError("examples must be divisible by batch_size")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(
        args.out_dir / "run_manifest.json",
        {"status": "running", "pid": os.getpid(), "checkpoint": str(args.checkpoint)},
    )
    device = pick_device(args.device)
    if device.type == "cuda":
        fraction = float(os.environ.get("TELOMERE_CUDA_MEMORY_FRACTION", "0.08"))
        torch.cuda.set_per_process_memory_fraction(
            fraction, device=torch.cuda.current_device()
        )
        torch.cuda.reset_peak_memory_stats(device)
    model, cfg, checkpoint_payload = load_checkpoint(args.checkpoint, device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    bank, weights, common_bases, _, bank_payload = load_bank_and_bases(
        args.bank_artifact,
        device,
        cfg.d_model,
        (args.rank,),
        expected_checkpoint=args.checkpoint,
        allow_relocated_checkpoint=args.allow_relocated_checkpoint,
    )
    rng = np.random.default_rng(849901)
    random_arrays = [
        random_orthonormal(cfg.d_model, args.rank, rng)
        for _ in range(args.random_bases)
    ]
    random_mean = np.mean(
        [basis @ basis.T for basis in random_arrays], axis=0
    )
    eigenvalues, eigenvectors = np.linalg.eigh(random_mean)
    random_basis = eigenvectors[:, np.argsort(eigenvalues)[-args.rank:]]
    named_bases = {
        "bottom_input_common": torch.as_tensor(
            common_bases["bottom_input"][args.rank], dtype=torch.float32, device=device
        ),
        "bottom_output_common": torch.as_tensor(
            common_bases["bottom_output"][args.rank], dtype=torch.float32, device=device
        ),
        "top_output_common": torch.as_tensor(
            common_bases["top_output"][args.rank], dtype=torch.float32, device=device
        ),
        "random_rank_matched": torch.as_tensor(
            random_basis, dtype=torch.float32, device=device
        ),
    }
    static_rows = static_qkv_alignment(
        model=model, named_bases=named_bases, rank=args.rank
    )
    groups = explicit_depth_position_groups(cfg.node_count)
    position_groups = {
        "answer": groups["answer"],
        "graph": groups["graph"],
        "query_metadata": groups["query_metadata"],
        "all": tuple(range(cfg.seq_len)),
    }
    true_deltas: dict[int, torch.Tensor] = {}
    random_deltas: dict[int, torch.Tensor] = {}
    for age in AGES:
        true, _ = build_stage_delta(
            weights[age], rank=args.rank, mode="bottom_floor",
            target=args.singular_floor,
        )
        random, _ = build_stage_delta(
            weights[age], rank=args.rank, mode="random_matched",
            target=args.singular_floor,
            rng=np.random.default_rng(850000 + age),
        )
        true_deltas[age] = torch.as_tensor(true, dtype=torch.float32, device=device)
        random_deltas[age] = torch.as_tensor(random, dtype=torch.float32, device=device)
    gradient_rows: list[dict[str, Any]] = []
    effect_rows: list[dict[str, Any]] = []
    for graph_seed in args.graph_seeds:
        set_seed(int(graph_seed))
        for batch_index in range(args.examples // args.batch_size):
            tokens, _, successors, start = fixed_depth_batch(
                cfg, args.batch_size, device, path_positions=cfg.max_depth
            )
            with torch.no_grad():
                raw = model.token_embed(tokens) + model.pos_embed.unsqueeze(0)
                state = raw
                current = start
                natural: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
                for age in range(1, max(AGES) + 1):
                    state = model.apply_loop(state, loop_index=age - 1)
                    current = advance_nodes(successors, current, steps=1)
                    natural[age] = (state, current)
            for source_age in AGES:
                pre_j, current = natural[source_age]
                with torch.no_grad():
                    post_j = bank.rollback(
                        pre_j, source_age=source_age,
                        positions=tuple(range(cfg.seq_len)),
                    )
                destination_positions = torch.as_tensor(
                    groups["destination"], device=device
                )
                destination = destination_positions[current]
                metadata = {
                    "graph_seed": int(graph_seed),
                    "batch": batch_index,
                    "source_age": source_age,
                }
                rows, loop_gradients, baseline_pattern = _gradient_rows(
                    model=model,
                    loop_input=post_j,
                    destination=destination,
                    answer_position=groups["answer"][0],
                    position_groups=position_groups,
                    named_bases=named_bases,
                    rank=args.rank,
                    metadata=metadata,
                )
                gradient_rows.extend(rows)
                effect_rows.extend(
                    _operator_effect_rows(
                        model=model,
                        pre_j=pre_j,
                        post_j=post_j,
                        true_delta=true_deltas[source_age],
                        random_delta=random_deltas[source_age],
                        loop_gradients=loop_gradients,
                        baseline_pattern=baseline_pattern,
                        destination=destination,
                        answer_position=groups["answer"][0],
                        output_basis=named_bases["bottom_output_common"],
                        metadata=metadata,
                    )
                )
    gradient_summary = aggregate(
        gradient_rows, ("gradient_site", "head", "position_group", "basis")
    )
    effect_summary = aggregate(effect_rows, ("condition", "head"))
    write_csv(args.out_dir / "static_qkv_alignment.csv", static_rows)
    write_csv(args.out_dir / "gradient_rows.csv", gradient_rows)
    write_csv(args.out_dir / "gradient_summary.csv", gradient_summary)
    write_csv(args.out_dir / "operator_effect_rows.csv", effect_rows)
    write_csv(args.out_dir / "operator_effect_summary.csv", effect_summary)
    plot_gradient(gradient_summary, args.out_dir / "attention_gradient_subspace.png")
    summary = {
        "status": "complete",
        "actual_device": str(device),
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": checkpoint_payload.get("step"),
        "bank_artifact": str(args.bank_artifact),
        "bank_kind": bank_payload.get("kind"),
        "rank": args.rank,
        "singular_floor": args.singular_floor,
        "graph_seeds": list(args.graph_seeds),
        "examples_per_graph_seed": args.examples,
        "random_basis_draws": args.random_bases,
        "peak_cuda_reserved_gib": (
            float(torch.cuda.max_memory_reserved(device) / 1024**3)
            if device.type == "cuda" else 0.0
        ),
        "claim_boundary": (
            "Static weight overlap is only architectural alignment. Gradient enrichment "
            "is local sensitivity. Causality comes from the separately reported operator "
            "surgery and clean/corrupt attention patching experiments."
        ),
    }
    atomic_json(args.out_dir / "summary.json", summary)
    atomic_json(args.out_dir / "run_manifest.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
