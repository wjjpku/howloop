"""Causal SVD interventions on delta_W = W - I for canonical boundary J.

Row-vector convention is used throughout: h maps to hW+b.  If
delta_W = U diag(s) Vh, columns of U are input-read directions and rows of Vh
are output-write directions.  Random replacements preserve the intervened
component's singular values and Frobenius norm, but are not claimed to
preserve the singular spectrum of the sum with the untouched tail.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from reasoning_loop.graph_path_depth_circuit import load_checkpoint
from reasoning_loop.graph_path_loop import pick_device
from reasoning_loop.graph_path_telomere_graph_leakage_audit import (
    reconstruct_primary_training_graphs,
    stratified_samples,
)
from reasoning_loop.graph_path_telomere_localized_query import VectorAffine
from reasoning_loop.graph_path_telomere_oracle_position_circuit import intervention_groups
from reasoning_loop.graph_path_telomere_task_mlp_audit import (
    _canonical_training_streams,
    evaluate_partition,
)


def affine(weight: torch.Tensor, bias: torch.Tensor) -> VectorAffine:
    dimension = int(weight.shape[0])
    return VectorAffine(
        weight=weight.float(),
        bias=bias.float(),
        update_rank=dimension,
        fit_dimension=dimension,
        retained_fit_energy=1.0,
    )


def random_frame(dimension: int, rank: int, seed: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    raw = torch.randn(dimension, rank, generator=generator, dtype=torch.float32)
    return torch.linalg.qr(raw, mode="reduced").Q.to(device)


def decompose_controller(item: dict[str, Any], device: torch.device):
    state = item["state_dict"]
    left = state["A"].to(device=device, dtype=torch.float32)
    right = state["B"].to(device=device, dtype=torch.float32)
    diagonal = state["diagonal_scale"].to(device=device, dtype=torch.float32)
    bias = state["bias"].to(device=device, dtype=torch.float32)
    weight = torch.diag(diagonal) + left @ right
    identity = torch.eye(weight.shape[0], device=device)
    delta = weight - identity
    u, singular, vh = torch.linalg.svd(delta, full_matrices=False)
    return weight, bias, delta, u, singular, vh


def build_intervention_weights(
    weight: torch.Tensor,
    delta: torch.Tensor,
    u: torch.Tensor,
    singular: torch.Tensor,
    vh: torch.Tensor,
    ranks: Sequence[int],
    random_seed: int,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    dimension = weight.shape[0]
    identity = torch.eye(dimension, device=weight.device)
    operators = {"identity_no_J": identity, "full_J": weight}
    total_energy = singular.square().sum().clamp_min(1e-20)
    metadata = {}
    for rank in sorted(set(int(value) for value in ranks)):
        if not 1 <= rank <= dimension:
            raise ValueError("all ranks must lie in [1, d_model]")
        top = (u[:, :rank] * singular[:rank]) @ vh[:rank]
        tail = delta - top
        input_frame = random_frame(dimension, rank, random_seed + 1009 * rank, weight.device)
        output_frame = random_frame(dimension, rank, random_seed + 2003 * rank, weight.device)
        random_input = (input_frame * singular[:rank]) @ vh[:rank]
        random_output = (u[:, :rank] * singular[:rank]) @ output_frame.T
        random_both = (input_frame * singular[:rank]) @ output_frame.T
        operators[f"keep_top_{rank}"] = identity + top
        operators[f"delete_top_{rank}"] = weight - top
        operators[f"replace_input_top_{rank}"] = identity + tail + random_input
        operators[f"replace_output_top_{rank}"] = identity + tail + random_output
        operators[f"replace_both_top_{rank}"] = identity + tail + random_both
        metadata[str(rank)] = {
            "cumulative_delta_energy": float(singular[:rank].square().sum() / total_energy),
            "top_component_frobenius_norm": float(top.norm()),
            "random_input_component_frobenius_norm": float(random_input.norm()),
            "random_output_component_frobenius_norm": float(random_output.norm()),
            "random_both_component_frobenius_norm": float(random_both.norm()),
        }
    return operators, metadata


def selected_metrics(curve: dict[str, Any]) -> dict[str, float]:
    values = curve["accuracy_by_cycle"]
    def auc(start: int, stop: int) -> float:
        part = values[start:stop]
        return float(sum(part) / len(part))
    return {
        "accuracy_loop1": float(values[0]),
        "accuracy_loop8": float(values[7]),
        "accuracy_loop32": float(values[31]),
        "accuracy_loop64": float(values[63]),
        "auc_1_24": auc(0, 24),
        "auc_25_48": auc(24, 48),
        "auc_49_64": auc(48, 64),
        "auc_1_64": auc(0, 64),
    }


def plot(result: dict[str, Any], path: Path) -> None:
    rows = result["selected_metrics"]
    singular = np.asarray(result["matrix_analysis"]["singular_values"])
    ranks = [int(value) for value in result["ranks"]]
    figure, axes = plt.subplots(1, 3, figsize=(11.7, 3.45), constrained_layout=True)

    axes[0].semilogy(np.arange(1, len(singular) + 1), singular, color="#2864A8")
    axes[0].set_xlabel("Singular-value index")
    axes[0].set_ylabel("Singular value of $W-I$")
    axes[0].set_title("A  $\\Delta W$ spectrum")
    axes[0].grid(alpha=0.2)

    for prefix, marker, color in (
        ("keep_top", "o", "#2864A8"),
        ("delete_top", "s", "#C44E52"),
    ):
        axes[1].plot(
            ranks,
            [rows[f"{prefix}_{rank}"]["auc_1_64"] for rank in ranks],
            marker=marker,
            color=color,
            label=prefix.replace("_", " "),
        )
    axes[1].axhline(rows["full_J"]["auc_1_64"], color="black", linestyle="--", label="full J")
    axes[1].set_xlabel("Intervened rank")
    axes[1].set_ylabel("Continuation AUC, loops 1--64")
    axes[1].set_ylim(-0.03, 1.03)
    axes[1].set_title("B  Keep/delete directions")
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].grid(alpha=0.2)

    for prefix, marker, color in (
        ("replace_input_top", "o", "#4C9F70"),
        ("replace_output_top", "s", "#D17C19"),
        ("replace_both_top", "^", "#8E63A9"),
    ):
        axes[2].plot(
            ranks,
            [rows[f"{prefix}_{rank}"]["auc_1_64"] for rank in ranks],
            marker=marker,
            color=color,
            label=prefix.replace("replace_", "").replace("_top", ""),
        )
    axes[2].axhline(rows["full_J"]["auc_1_64"], color="black", linestyle="--", label="full J")
    axes[2].set_xlabel("Replaced rank")
    axes[2].set_ylim(-0.03, 1.03)
    axes[2].set_title("C  Read/write subspace replacement")
    axes[2].legend(frameon=False, fontsize=8)
    axes[2].grid(alpha=0.2)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight")
    figure.savefig(path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(figure)


def write_metrics_csv(path: Path, rows: dict[str, dict[str, float]]) -> None:
    records = [{"operator": label, **metrics} for label, metrics in rows.items()]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    if args.continuation_loops < 64:
        raise ValueError("continuation-loops must be at least 64")
    device = pick_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(args.cuda_memory_fraction)
    model, cfg, checkpoint_payload = load_checkpoint(args.checkpoint, device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    phase_payload = json.loads(args.phase_summary.read_text(encoding="utf-8"))
    phase_positions = [int(value) for value in phase_payload["trajectory_positions_including_initial"]]
    positions = intervention_groups(cfg.node_count)["all"]

    payload = torch.load(args.artifact, map_location=device, weights_only=False)
    if payload.get("kind") != "graph_path_telomere_task_lora_j":
        raise ValueError("unexpected controller artifact kind")
    if payload["checkpoint"] != str(args.checkpoint):
        raise ValueError("controller and backbone checkpoints differ")
    if payload.get("placement") != "loop_boundary":
        raise ValueError("this audit is restricted to loop-boundary controllers")
    item = payload["modules"][args.label]
    if item.get("parameterization") != "diagonal_low_rank":
        raise ValueError("this audit requires diagonal_low_rank J")
    weight, bias, delta, u, singular, vh = decompose_controller(item, device)
    weight_operators, rank_metadata = build_intervention_weights(
        weight, delta, u, singular, vh, args.ranks, args.random_seed
    )
    operators = {label: affine(matrix, bias) for label, matrix in weight_operators.items()}

    streams = _canonical_training_streams(payload)
    seen, unique_after_stage, training_draws = reconstruct_primary_training_graphs(
        device=device, node_count=cfg.node_count, streams=streams
    )
    universe = set(itertools.permutations(range(cfg.node_count)))
    unseen = universe - seen
    _, sampled_unseen, distribution = stratified_samples(
        seen, unseen, count=args.sample_per_partition, seed=args.sample_seed
    )
    evaluation = evaluate_partition(
        permutations=sampled_unseen,
        operators=operators,
        model=model,
        cfg=cfg,
        phase_positions=phase_positions,
        positions=positions,
        device=device,
        batch_size=args.batch_size,
        continuation_loops=args.continuation_loops,
        placement="loop_boundary",
    )
    metrics = {label: selected_metrics(curve) for label, curve in evaluation["curves"].items()}
    result = {
        "status": "complete",
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": checkpoint_payload.get("step"),
        "backbone_loss_placement": payload.get("backbone_loss_description"),
        "trained_loops": cfg.max_loops,
        "shared_physical_blocks": cfg.n_layers,
        "effective_trained_depth": cfg.max_loops * cfg.n_layers,
        "controller_artifact": str(args.artifact),
        "controller_label": args.label,
        "controller_parameterization": "J(h)=hD+(hA)B+b",
        "controller_placement": "loop_boundary",
        "controller_training": "successor CE at every controlled continuation loop; no hidden-state loss",
        "row_vector_convention": "h -> hW+b; U columns read input directions; Vh rows write output directions",
        "ranks": sorted(set(args.ranks)),
        "rank_metadata": rank_metadata,
        "matrix_analysis": {
            "delta_definition": "W-I where W=D+AB",
            "delta_frobenius_norm": float(delta.norm()),
            "delta_operator_norm": float(singular[0]),
            "bias_norm": float(bias.norm()),
            "singular_values": singular.detach().cpu().tolist(),
        },
        "sampling": {
            "strictly_unseen_permutations": args.sample_per_partition,
            "all_starts_per_permutation": cfg.node_count,
            "examples": args.sample_per_partition * cfg.node_count,
            "sample_seed": args.sample_seed,
            "matched_cycle_type_distribution": distribution,
            "training_graph_draws": training_draws,
            "unique_training_graphs": len(seen),
            "unique_after_stage": unique_after_stage,
        },
        "selected_metrics": metrics,
        "strictly_unseen": evaluation,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    write_metrics_csv(args.out_dir / "selected_metrics.csv", metrics)
    plot(result, args.out_dir / "canonical_j_delta_svd_interventions.pdf")
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--phase-summary", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sample-per-partition", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--continuation-loops", type=int, default=64)
    parser.add_argument("--sample-seed", type=int, default=2026080902)
    parser.add_argument("--random-seed", type=int, default=2026080903)
    parser.add_argument("--ranks", type=int, nargs="+", default=(4, 8, 16, 32, 48))
    parser.add_argument("--cuda-memory-fraction", type=float, default=0.10)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    result = run_experiment(parse_args(argv))
    print(json.dumps({"status": result["status"], "selected_metrics": result["selected_metrics"]}, indent=2))


if __name__ == "__main__":
    main()
