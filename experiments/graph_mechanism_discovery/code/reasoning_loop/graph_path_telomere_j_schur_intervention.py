from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from collections import defaultdict
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F

from reasoning_loop.graph_path_depth_circuit import load_checkpoint
from reasoning_loop.graph_path_loop import pick_device
from reasoning_loop.graph_path_telomere_graph_leakage_audit import (
    PRIMARY_J_TRAINING_STREAMS,
    Permutation,
    _expand_all_starts,
    reconstruct_primary_training_graphs,
    stratified_samples,
)
from reasoning_loop.graph_path_telomere_localized_query import run_one_loop
from reasoning_loop.graph_path_telomere_oracle_position_circuit import (
    intervention_groups,
)
from reasoning_loop.graph_path_telomere_overloop import advance_nodes
from reasoning_loop.graph_path_telomere_simple_one_step import (
    _aligned_state_at_age,
)
from reasoning_loop.graph_path_telomere_unit_j import (
    exact_interfaces,
    load_unit_j_map,
)


Map = Callable[[torch.Tensor], torch.Tensor]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _project(value: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return (value @ basis) @ basis.T


def _matched_random_component(
    delta: torch.Tensor,
    target_basis: torch.Tensor,
    random_basis: torch.Tensor,
) -> torch.Tensor:
    target = _project(delta, target_basis)
    random = _project(delta, random_basis)
    target_norm = torch.linalg.vector_norm(target, dim=-1, keepdim=True)
    random_norm = torch.linalg.vector_norm(random, dim=-1, keepdim=True)
    return random * (target_norm / random_norm.clamp_min(1e-8))


def build_maps(
    *,
    age_map,
    schur_payload: dict,
    device: torch.device,
) -> tuple[dict[str, Map], dict[str, int]]:
    basis = schur_payload["basis"].to(device=device, dtype=torch.float32)
    dimension = basis.shape[0]
    if tuple(basis.shape) != (dimension, dimension):
        raise ValueError("Schur basis must be square")
    identity_error = torch.linalg.matrix_norm(
        basis.T @ basis - torch.eye(dimension, device=device)
    )
    if float(identity_error.item()) > 1e-4:
        raise ValueError(f"Schur basis is not orthogonal: {identity_error}")

    band_bases = {}
    band_dimensions = {}
    for label in ("zero", "one", "middle"):
        item = schur_payload["bands"][label]
        start, stop = int(item["start"]), int(item["stop"])
        band_bases[label] = basis[:, start:stop]
        band_dimensions[label] = stop - start

    def full_delta(value: torch.Tensor) -> torch.Tensor:
        source = value.float()
        return age_map(source) - source

    maps: dict[str, Map] = {
        "full_J": age_map,
        "no_J": lambda value: value.float(),
    }
    for label, band_basis in band_bases.items():
        maps[f"only_{label}"] = (
            lambda value, q=band_basis: value.float()
            + _project(full_delta(value), q)
        )
        maps[f"without_{label}"] = (
            lambda value, q=band_basis: age_map(value)
            - _project(full_delta(value), q)
        )

    for seed in schur_payload["random_seeds"]:
        random_full = schur_payload["random_bases"][str(seed)].to(
            device=device,
            dtype=torch.float32,
        )
        for label, band_basis in band_bases.items():
            band_dim = band_dimensions[label]
            random_basis = random_full[:, :band_dim]
            maps[f"random_only_{label}_seed{seed}"] = (
                lambda value, q=band_basis, r=random_basis: value.float()
                + _matched_random_component(full_delta(value), q, r)
            )
            maps[f"without_random_{label}_seed{seed}"] = (
                lambda value, q=band_basis, r=random_basis: age_map(value)
                - _matched_random_component(full_delta(value), q, r)
            )
    return maps, band_dimensions


def _relative_mse(source: torch.Tensor, target: torch.Tensor) -> float:
    denominator = (
        target.float()
        - target.float().mean(dim=(0, 1), keepdim=True)
    ).square().mean().clamp_min(1e-8)
    return float(
        ((source.float() - target.float()).square().mean() / denominator).item()
    )


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(
        F.cosine_similarity(
            left.float().flatten(1),
            right.float().flatten(1),
            dim=1,
        )
        .mean()
        .item()
    )


@torch.no_grad()
def evaluate_partition(
    *,
    label: str,
    permutations: list[Permutation],
    model,
    cfg,
    phase_positions: list[int],
    positions: tuple[int, ...],
    maps: dict[str, Map],
    device: torch.device,
    batch_size: int,
    continuation_loops: int,
    operating_age: int,
) -> dict:
    jump = phase_positions[3] - phase_positions[2]
    successors_all, starts_all = _expand_all_starts(
        permutations,
        device=device,
    )
    map_labels = list(maps)
    all_labels = map_labels + [f"exact_H{operating_age}"]
    label_to_index = {name: index for index, name in enumerate(all_labels)}
    correct = {
        name: [0] * continuation_loops for name in all_labels
    }
    nonendpoint_correct = {
        name: [0] * continuation_loops for name in all_labels
    }
    nonendpoint_counts = [0] * continuation_loops
    alignment_sums: dict[str, dict[str, float]] = {
        name: defaultdict(float) for name in all_labels
    }
    count = 0

    for offset in range(0, successors_all.shape[0], batch_size):
        successors = successors_all[offset : offset + batch_size]
        starts = starts_all[offset : offset + batch_size]
        current_batch = successors.shape[0]
        endpoint = advance_nodes(successors, starts, steps=cfg.max_depth)
        initial = _aligned_state_at_age(
            model=model,
            cfg=cfg,
            successors=successors,
            current=endpoint,
            age=8,
            phase_position=phase_positions[8],
        )
        initial_oracle = exact_interfaces(
            model=model,
            cfg=cfg,
            phase_positions=phase_positions,
            positions=positions,
            successors=successors,
            current=endpoint,
            ages=(operating_age,),
            loop_index=cfg.max_loops,
        )[operating_age]

        captured: dict[str, torch.Tensor] = {}

        def capture_identity(value: torch.Tensor) -> torch.Tensor:
            captured["source"] = value.detach()
            return value

        run_one_loop(
            model,
            initial,
            loop_index=cfg.max_loops,
            block2_position_transform=(positions, capture_identity),
        )
        source = captured["source"]
        full_update = maps["full_J"](source) - source.float()
        oracle_update = initial_oracle - source.float()
        full_norm = torch.linalg.vector_norm(
            full_update.flatten(1),
            dim=1,
        ).mean()

        for name in map_labels:
            transformed = maps[name](source)
            update = transformed - source.float()
            metrics = alignment_sums[name]
            metrics["relative_mse_to_exact_interface"] += (
                _relative_mse(transformed, initial_oracle) * current_batch
            )
            metrics["cosine_update_with_full_J"] += (
                _cosine(update, full_update) * current_batch
            )
            metrics["cosine_update_with_exact_reset"] += (
                _cosine(update, oracle_update) * current_batch
            )
            metrics["update_norm_ratio_to_full_J"] += (
                float(
                    (
                        torch.linalg.vector_norm(update.flatten(1), dim=1).mean()
                        / full_norm.clamp_min(1e-8)
                    ).item()
                )
                * current_batch
            )
        exact_metrics = alignment_sums[f"exact_H{operating_age}"]
        exact_metrics["relative_mse_to_exact_interface"] += 0.0
        exact_metrics["cosine_update_with_full_J"] += (
            _cosine(oracle_update, full_update) * current_batch
        )
        exact_metrics["cosine_update_with_exact_reset"] += 1.0 * current_batch
        exact_metrics["update_norm_ratio_to_full_J"] += (
            float(
                (
                    torch.linalg.vector_norm(oracle_update.flatten(1), dim=1).mean()
                    / full_norm.clamp_min(1e-8)
                ).item()
            )
            * current_batch
        )

        states = initial.unsqueeze(0).expand(
            len(all_labels),
            *initial.shape,
        ).clone()
        for cycle in range(1, continuation_loops + 1):
            current = advance_nodes(
                successors,
                endpoint,
                steps=jump * (cycle - 1),
            )
            target = advance_nodes(
                successors,
                endpoint,
                steps=jump * cycle,
            )
            nonendpoint = target.ne(endpoint)
            nonendpoint_counts[cycle - 1] += int(nonendpoint.sum().item())
            oracle = exact_interfaces(
                model=model,
                cfg=cfg,
                phase_positions=phase_positions,
                positions=positions,
                successors=successors,
                current=current,
                ages=(operating_age,),
                loop_index=cfg.max_loops + cycle - 1,
            )[operating_age]

            def batched_transform(value: torch.Tensor) -> torch.Tensor:
                grouped = value.reshape(
                    len(all_labels),
                    current_batch,
                    len(positions),
                    cfg.d_model,
                )
                transformed = []
                for name, group in zip(all_labels, grouped, strict=True):
                    if name == f"exact_H{operating_age}":
                        transformed.append(oracle)
                    else:
                        transformed.append(maps[name](group))
                return torch.stack(transformed).flatten(0, 1)

            step = run_one_loop(
                model,
                states.flatten(0, 1),
                loop_index=cfg.max_loops + cycle - 1,
                block2_position_transform=(positions, batched_transform),
            )
            predictions = step.logits.reshape(
                len(all_labels),
                current_batch,
                -1,
            ).argmax(dim=-1)
            for name, prediction in zip(
                all_labels,
                predictions,
                strict=True,
            ):
                is_correct = prediction.eq(target)
                correct[name][cycle - 1] += int(is_correct.sum().item())
                nonendpoint_correct[name][cycle - 1] += int(
                    (is_correct & nonendpoint).sum().item()
                )
            states = step.state.reshape(
                len(all_labels),
                current_batch,
                cfg.seq_len,
                cfg.d_model,
            )
        count += current_batch

    curves = {}
    for name in all_labels:
        accuracy = [value / count for value in correct[name]]
        nonendpoint_accuracy = [
            numerator / denominator
            for numerator, denominator in zip(
                nonendpoint_correct[name],
                nonendpoint_counts,
                strict=True,
            )
        ]
        curves[name] = {
            "accuracy_by_cycle": accuracy,
            "nonendpoint_accuracy_by_cycle": nonendpoint_accuracy,
            "auc_1_24": sum(accuracy[:24]) / min(24, len(accuracy)),
            "auc_25_48": (
                sum(accuracy[24:48]) / min(24, len(accuracy) - 24)
                if len(accuracy) > 24
                else None
            ),
            "auc_49_64": (
                sum(accuracy[48:64]) / min(16, len(accuracy) - 48)
                if len(accuracy) > 48
                else None
            ),
            "nonendpoint_auc_1_24": (
                sum(nonendpoint_accuracy[:24])
                / min(24, len(nonendpoint_accuracy))
            ),
            "nonendpoint_auc_25_48": (
                sum(nonendpoint_accuracy[24:48])
                / min(24, len(nonendpoint_accuracy) - 24)
                if len(nonendpoint_accuracy) > 24
                else None
            ),
            "nonendpoint_auc_49_64": (
                sum(nonendpoint_accuracy[48:64])
                / min(16, len(nonendpoint_accuracy) - 48)
                if len(nonendpoint_accuracy) > 48
                else None
            ),
        }

    return {
        "partition": label,
        "permutations": len(permutations),
        "examples_all_starts": count,
        "curves": curves,
        "first_application_alignment": {
            name: {
                metric: value / count
                for metric, value in metrics.items()
            }
            for name, metrics in alignment_sums.items()
        },
    }


def _load_training_streams(path: Path | None):
    if path is None:
        return PRIMARY_J_TRAINING_STREAMS
    stream_payload = json.loads(path.read_text(encoding="utf-8"))
    return tuple(
        (
            str(item["label"]),
            int(item["base_seed"]),
            int(item["stride"]),
            int(item["rounds"]),
            int(item["batch_size"]),
            int(item["batches"]),
        )
        for item in stream_payload
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--phase-summary", type=Path, required=True)
    parser.add_argument("--j-artifact", type=Path, required=True)
    parser.add_argument("--j-label", default="task")
    parser.add_argument("--schur-artifact", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sample-per-partition", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--continuation-loops", type=int, default=64)
    parser.add_argument("--sample-seed", type=int, default=296004)
    parser.add_argument("--operating-age", type=int, default=3)
    parser.add_argument(
        "--partitions",
        nargs="+",
        choices=("seen", "unseen"),
        default=("seen", "unseen"),
    )
    parser.add_argument("--training-streams-json", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)
    model, cfg, _ = load_checkpoint(args.checkpoint, device)
    if cfg.node_count != 8 or cfg.max_depth != 8 or cfg.max_loops != 8:
        raise ValueError("audit is fixed to the D8L8 N8 experiment")
    phase_summary = json.loads(args.phase_summary.read_text(encoding="utf-8"))
    phase_positions = [
        int(value)
        for value in phase_summary["trajectory_positions_including_initial"]
    ]
    positions = intervention_groups(cfg.node_count)["all"]
    age_map, map_checkpoint = load_unit_j_map(
        args.j_artifact,
        label=args.j_label,
        device=device,
    )
    if map_checkpoint != str(args.checkpoint):
        raise ValueError("J and frozen model checkpoints differ")
    schur_payload = torch.load(
        args.schur_artifact,
        map_location="cpu",
        weights_only=False,
    )
    if schur_payload.get("kind") != "graph_path_telomere_j_real_schur":
        raise ValueError("unexpected Schur artifact kind")
    if schur_payload["checkpoint"] != str(args.checkpoint):
        raise ValueError("Schur artifact and frozen model checkpoints differ")
    if schur_payload["source_j_sha256"] != _sha256(args.j_artifact):
        raise ValueError("Schur artifact and J artifact hashes differ")
    maps, band_dimensions = build_maps(
        age_map=age_map,
        schur_payload=schur_payload,
        device=device,
    )

    training_streams = _load_training_streams(args.training_streams_json)
    seen, unique_after_stage, total_draws = reconstruct_primary_training_graphs(
        device=device,
        node_count=cfg.node_count,
        streams=training_streams,
    )
    universe = set(itertools.permutations(range(cfg.node_count)))
    unseen = universe - seen
    sampled_seen, sampled_unseen, distribution = stratified_samples(
        seen,
        unseen,
        count=args.sample_per_partition,
        seed=args.sample_seed,
    )
    requested = []
    if "seen" in args.partitions:
        requested.append(("seen_during_J_training", sampled_seen))
    if "unseen" in args.partitions:
        requested.append(("strictly_unseen_by_J_training", sampled_unseen))
    results = [
        evaluate_partition(
            label=label,
            permutations=sample,
            model=model,
            cfg=cfg,
            phase_positions=phase_positions,
            positions=positions,
            maps=maps,
            device=device,
            batch_size=args.batch_size,
            continuation_loops=args.continuation_loops,
            operating_age=args.operating_age,
        )
        for label, sample in requested
    ]
    payload = {
        "status": "complete",
        "checkpoint": str(args.checkpoint),
        "loss_placement": (
            "final CE at loop 8 plus intermediate CE on "
            "p_min(2t,D), t=1..7"
        ),
        "trained_macro_loops": cfg.max_loops,
        "shared_physical_blocks": cfg.n_layers,
        "effective_training_depth": cfg.max_loops * cfg.n_layers,
        "evaluated_continuation_loops": args.continuation_loops,
        "J_artifact": str(args.j_artifact),
        "J_label": args.j_label,
        "schur_artifact": str(args.schur_artifact),
        "schur_bands": schur_payload["bands"],
        "band_dimensions": band_dimensions,
        "random_seeds": schur_payload["random_seeds"],
        "intervention": (
            "only_S(z)=z+(J(z)-z)P_S; "
            "without_S(z)=J(z)-(J(z)-z)P_S; "
            "random controls match per-token update L2 norm"
        ),
        "graph_universe": len(universe),
        "training_graph_draws": total_draws,
        "unique_training_graphs": len(seen),
        "strictly_unseen_graphs": len(unseen),
        "unique_after_stage": unique_after_stage,
        "sampling": {
            "per_partition": args.sample_per_partition,
            "all_starts_per_graph": cfg.node_count,
            "matched_cycle_type_distribution": distribution,
            "seed": args.sample_seed,
        },
        "graph_steps_per_continuation_loop": (
            phase_positions[3] - phase_positions[2]
        ),
        "results": results,
        "gpu_runtime": {
            "observed_peak_reserved_gib": (
                float(torch.cuda.max_memory_reserved(device) / 1024**3)
                if device.type == "cuda"
                else 0.0
            )
        },
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    temporary = args.out_dir / "summary.json.tmp"
    temporary.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.out_dir / "summary.json")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
