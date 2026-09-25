from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

from reasoning_loop.twohop_in_context import (
    PairedTwoHopBatch,
    TwoHopBatch,
    TwoHopConfig,
    TwoHopTransformer,
    build_twohop_model,
    make_paired_twohop_batch,
)
from reasoning_loop.twohop_standard_vs_looped import generator_for, pick_device


ROLE_NAMES = (
    "target_first_child",
    "target_second_parent",
    "target_second_child",
    "counterfactual_first_child",
    "counterfactual_second_parent",
    "counterfactual_second_child",
    "query",
)


@dataclass(frozen=True, order=True)
class Site:
    depth: int
    head: int
    role: str

    def to_dict(self, model: TwoHopTransformer) -> dict[str, Any]:
        return {
            "effective_depth": self.depth + 1,
            "parameter_block": model.parameter_index(self.depth),
            "head": self.head,
            "role": self.role,
            "effective_id": self.effective_id,
            "parameter_id": (
                f"P{model.parameter_index(self.depth)}H{self.head}:{self.role}"
            ),
        }

    @property
    def effective_id(self) -> str:
        return f"D{self.depth + 1}H{self.head}:{self.role}"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Discover and validate activation-patching circuits for trained "
            "two-hop standard/periodic Transformer checkpoints."
        )
    )
    parser.add_argument("--suite-dir", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=["S6", "P2x3"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--discovery-examples", type=int, default=256)
    parser.add_argument("--validation-examples", type=int, default=1024)
    parser.add_argument("--attention-examples", type=int, default=1024)
    parser.add_argument("--candidate-pool", type=int, default=48)
    parser.add_argument("--max-sites", type=int, default=16)
    parser.add_argument("--target-recovery", type=float, default=0.9)
    parser.add_argument("--random-controls", type=int, default=20)
    parser.add_argument("--behavior-threshold", type=float, default=0.98)
    parser.add_argument("--seed-base", type=int, default=47000)
    parser.add_argument("--overloop-depth", type=int, default=12)
    parser.add_argument("--cuda-memory-fraction", type=float, default=0.08)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def load_checkpoint(
    path: Path,
    device: torch.device,
) -> tuple[TwoHopTransformer, dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    cfg = TwoHopConfig.from_dict(checkpoint["config"])
    model = build_twohop_model(
        cfg,
        seed=int(checkpoint.get("init_seed", checkpoint["seed"])),
        device=device,
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


def _cache(
    result: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    cache = result.get("cache")
    if not isinstance(cache, dict):
        raise RuntimeError("forward result lacks an activation cache")
    return cache


def logit_metric(
    logits: torch.Tensor,
    clean_labels: torch.Tensor,
    corrupt_labels: torch.Tensor,
) -> torch.Tensor:
    clean = logits.gather(1, clean_labels[:, None]).squeeze(1)
    corrupt = logits.gather(1, corrupt_labels[:, None]).squeeze(1)
    return clean - corrupt


def normalized_recovery(
    patched_metric: float,
    *,
    clean_metric: float,
    corrupt_metric: float,
) -> float:
    denominator = clean_metric - corrupt_metric
    if abs(denominator) < 1e-8:
        return float("nan")
    return (patched_metric - corrupt_metric) / denominator


def _counterfactual_indices(pair: PairedTwoHopBatch) -> torch.Tensor:
    clean = pair.clean
    corrupt = pair.corrupt
    rows = torch.arange(clean.batch_size, device=clean.tokens.device)
    counterfactual_bridges = corrupt.chains[
        rows,
        corrupt.target_indices,
        1,
    ]
    matches = clean.chains[:, :, 1].eq(counterfactual_bridges[:, None])
    if not matches.any(dim=1).all():
        raise RuntimeError("could not locate counterfactual bridge chain")
    return matches.float().argmax(dim=1)


def role_positions(
    pair: PairedTwoHopBatch,
    role: str,
) -> list[torch.Tensor]:
    clean = pair.clean
    rows = torch.arange(clean.batch_size, device=clean.tokens.device)
    target = clean.target_indices
    counterfactual = _counterfactual_indices(pair)
    if role == "query":
        return [
            torch.full_like(
                target,
                clean.query_position,
            )
        ]
    index = target if role.startswith("target_") else counterfactual
    field = role.removeprefix("target_").removeprefix("counterfactual_")
    mapping = {
        "first_child": clean.first_child_positions,
        "second_parent": clean.second_parent_positions,
        "second_child": clean.second_child_positions,
    }
    if field not in mapping:
        raise ValueError(f"unknown semantic role: {role}")
    return [mapping[field][rows, index]]


def _site_mask(
    pair: PairedTwoHopBatch,
    site: Site,
    *,
    n_heads: int,
) -> torch.Tensor:
    clean = pair.clean
    mask = torch.zeros(
        (clean.batch_size, clean.tokens.shape[1], n_heads, 1),
        dtype=torch.bool,
        device=clean.tokens.device,
    )
    rows = torch.arange(clean.batch_size, device=clean.tokens.device)
    for positions in role_positions(pair, site.role):
        mask[rows, positions, site.head, 0] = True
    return mask


def interventions_for_sites(
    model: TwoHopTransformer,
    pair: PairedTwoHopBatch,
    sites: Iterable[Site],
    source_cache: Mapping[str, torch.Tensor],
) -> dict[int, dict[str, torch.Tensor]]:
    by_depth: dict[int, list[Site]] = {}
    for site in sites:
        by_depth.setdefault(site.depth, []).append(site)
    interventions: dict[int, dict[str, torch.Tensor]] = {}
    for depth, depth_sites in by_depth.items():
        patch = source_cache[f"depth{depth}.z"]
        mask = torch.zeros(
            (*patch.shape[:-1], 1),
            dtype=torch.bool,
            device=patch.device,
        )
        for site in depth_sites:
            mask |= _site_mask(
                pair,
                site,
                n_heads=model.cfg.n_heads,
            )
        interventions[depth] = {
            "z_patch": patch,
            "z_mask": mask,
        }
    return interventions


@torch.inference_mode()
def run_site_patch(
    model: TwoHopTransformer,
    pair: PairedTwoHopBatch,
    sites: Sequence[Site],
    *,
    direction: str,
    clean_cache: Mapping[str, torch.Tensor],
    corrupt_cache: Mapping[str, torch.Tensor],
    clean_metric: float,
    corrupt_metric: float,
) -> dict[str, float]:
    if direction == "patch_in":
        tokens = pair.corrupt.tokens
        source_cache = clean_cache
    elif direction == "patch_out":
        tokens = pair.clean.tokens
        source_cache = corrupt_cache
    else:
        raise ValueError("direction must be patch_in or patch_out")
    logits = model(
        tokens,
        interventions=interventions_for_sites(
            model,
            pair,
            sites,
            source_cache,
        ),
    )
    metric = float(
        logit_metric(
            logits,
            pair.clean.labels,
            pair.corrupt.labels,
        ).mean().item()
    )
    accuracy = float(
        logits.argmax(dim=1)
        .eq(pair.clean.labels)
        .float()
        .mean()
        .item()
    )
    recovery = normalized_recovery(
        metric,
        clean_metric=clean_metric,
        corrupt_metric=corrupt_metric,
    )
    if direction == "patch_out":
        necessity = normalized_recovery(
            clean_metric - metric + corrupt_metric,
            clean_metric=clean_metric,
            corrupt_metric=corrupt_metric,
        )
    else:
        necessity = float("nan")
    return {
        "metric": metric,
        "clean_answer_accuracy": accuracy,
        "recovery": recovery,
        "necessity": necessity,
    }


def baseline_pair(
    model: TwoHopTransformer,
    pair: PairedTwoHopBatch,
) -> dict[str, Any]:
    with torch.inference_mode():
        clean_result = model.forward_all(
            pair.clean.tokens,
            return_cache=True,
        )
        corrupt_result = model.forward_all(
            pair.corrupt.tokens,
            return_cache=True,
        )
    clean_logits_by_depth = clean_result["logits_by_depth"]
    corrupt_logits_by_depth = corrupt_result["logits_by_depth"]
    if not isinstance(clean_logits_by_depth, torch.Tensor) or not isinstance(
        corrupt_logits_by_depth,
        torch.Tensor,
    ):
        raise RuntimeError("missing depth logits")
    clean_final = clean_logits_by_depth[:, -1]
    corrupt_final = corrupt_logits_by_depth[:, -1]
    clean_metric = float(
        logit_metric(
            clean_final,
            pair.clean.labels,
            pair.corrupt.labels,
        ).mean().item()
    )
    corrupt_metric = float(
        logit_metric(
            corrupt_final,
            pair.clean.labels,
            pair.corrupt.labels,
        ).mean().item()
    )
    return {
        "clean_result": clean_result,
        "corrupt_result": corrupt_result,
        "clean_cache": _cache(clean_result),
        "corrupt_cache": _cache(corrupt_result),
        "clean_metric": clean_metric,
        "corrupt_metric": corrupt_metric,
        "clean_accuracy": float(
            clean_final.argmax(dim=1)
            .eq(pair.clean.labels)
            .float()
            .mean()
            .item()
        ),
        "corrupt_accuracy": float(
            corrupt_final.argmax(dim=1)
            .eq(pair.corrupt.labels)
            .float()
            .mean()
            .item()
        ),
        "prefix_clean_accuracy": clean_logits_by_depth.argmax(dim=2)
        .eq(pair.clean.labels[:, None])
        .float()
        .mean(dim=0)
        .tolist(),
        "prefix_clean_metric": [
            float(
                logit_metric(
                    clean_logits_by_depth[:, depth],
                    pair.clean.labels,
                    pair.corrupt.labels,
                )
                .mean()
                .item()
            )
            for depth in range(model.cfg.total_depth)
        ],
        "prefix_corrupt_metric": [
            float(
                logit_metric(
                    corrupt_logits_by_depth[:, depth],
                    pair.clean.labels,
                    pair.corrupt.labels,
                )
                .mean()
                .item()
            )
            for depth in range(model.cfg.total_depth)
        ],
    }


def all_sites(model: TwoHopTransformer) -> list[Site]:
    return [
        Site(depth=depth, head=head, role=role)
        for depth in range(model.cfg.total_depth)
        for head in range(model.cfg.n_heads)
        for role in ROLE_NAMES
    ]


def discover_sites(
    model: TwoHopTransformer,
    pair: PairedTwoHopBatch,
    baseline: Mapping[str, Any],
    *,
    candidate_pool: int,
    max_sites: int,
    target_recovery: float,
) -> tuple[list[Site], list[dict[str, Any]], list[dict[str, Any]]]:
    candidates = all_sites(model)
    individual_rows: list[dict[str, Any]] = []
    for site in candidates:
        result = run_site_patch(
            model,
            pair,
            [site],
            direction="patch_in",
            clean_cache=baseline["clean_cache"],
            corrupt_cache=baseline["corrupt_cache"],
            clean_metric=baseline["clean_metric"],
            corrupt_metric=baseline["corrupt_metric"],
        )
        individual_rows.append(
            {
                **site.to_dict(model),
                "individual_metric": result["metric"],
                "individual_recovery": result["recovery"],
                "individual_clean_answer_accuracy": result[
                    "clean_answer_accuracy"
                ],
            }
        )
    individual_rows.sort(
        key=lambda row: row["individual_recovery"],
        reverse=True,
    )
    by_id = {site.effective_id: site for site in candidates}
    pool = [
        by_id[row["effective_id"]]
        for row in individual_rows[: min(candidate_pool, len(individual_rows))]
    ]
    selected: list[Site] = []
    greedy_rows: list[dict[str, Any]] = []
    remaining = pool.copy()
    for selection_step in range(1, min(max_sites, len(pool)) + 1):
        trials: list[tuple[float, Site, dict[str, float]]] = []
        for site in remaining:
            result = run_site_patch(
                model,
                pair,
                [*selected, site],
                direction="patch_in",
                clean_cache=baseline["clean_cache"],
                corrupt_cache=baseline["corrupt_cache"],
                clean_metric=baseline["clean_metric"],
                corrupt_metric=baseline["corrupt_metric"],
            )
            trials.append((result["recovery"], site, result))
        trials.sort(key=lambda item: item[0], reverse=True)
        _, best_site, best_result = trials[0]
        selected.append(best_site)
        remaining.remove(best_site)
        greedy_rows.append(
            {
                "selection_step": selection_step,
                **best_site.to_dict(model),
                "joint_metric": best_result["metric"],
                "joint_recovery": best_result["recovery"],
                "joint_clean_answer_accuracy": best_result[
                    "clean_answer_accuracy"
                ],
            }
        )
        if best_result["recovery"] >= target_recovery:
            break
    return selected, individual_rows, greedy_rows


def _gather_attention(
    attention: torch.Tensor,
    destination: torch.Tensor,
    source: torch.Tensor,
) -> torch.Tensor:
    rows = torch.arange(attention.shape[0], device=attention.device)
    return attention[rows, :, destination, source]


def attention_route_table(
    model: TwoHopTransformer,
    pair: PairedTwoHopBatch,
    cache: Mapping[str, torch.Tensor],
) -> list[dict[str, Any]]:
    clean = pair.clean
    rows = torch.arange(clean.batch_size, device=clean.tokens.device)
    target = clean.target_indices
    counterfactual = _counterfactual_indices(pair)
    query = torch.full_like(target, clean.query_position)
    target_first_parent = clean.first_parent_positions[rows, target]
    target_first_child = clean.first_child_positions[rows, target]
    target_second_parent = clean.second_parent_positions[rows, target]
    target_second_child = clean.second_child_positions[rows, target]
    counterfactual_second_child = clean.second_child_positions[
        rows,
        counterfactual,
    ]
    first_copy_values: list[torch.Tensor] = []
    second_copy_values: list[torch.Tensor] = []
    for chain in range(clean.chain_count):
        first_copy_values.append(
            (clean.first_child_positions[:, chain], clean.first_parent_positions[:, chain])
        )
        second_copy_values.append(
            (
                clean.second_child_positions[:, chain],
                clean.second_parent_positions[:, chain],
            )
        )
    table: list[dict[str, Any]] = []
    for depth in range(model.cfg.total_depth):
        attention = cache[f"depth{depth}.attention"]
        first_copy = torch.stack(
            [
                _gather_attention(attention, destination, source)
                for destination, source in first_copy_values
            ],
            dim=0,
        ).mean(dim=(0, 1))
        second_copy = torch.stack(
            [
                _gather_attention(attention, destination, source)
                for destination, source in second_copy_values
            ],
            dim=0,
        ).mean(dim=(0, 1))
        query_to_first_child = _gather_attention(
            attention,
            query,
            target_first_child,
        ).mean(dim=0)
        query_to_second_parent = _gather_attention(
            attention,
            query,
            target_second_parent,
        ).mean(dim=0)
        query_to_target_end = _gather_attention(
            attention,
            query,
            target_second_child,
        ).mean(dim=0)
        query_to_counterfactual_end = _gather_attention(
            attention,
            query,
            counterfactual_second_child,
        ).mean(dim=0)
        query_to_source = _gather_attention(
            attention,
            query,
            target_first_parent,
        ).mean(dim=0)
        for head in range(model.cfg.n_heads):
            table.append(
                {
                    "effective_depth": depth + 1,
                    "parameter_block": model.parameter_index(depth),
                    "head": head,
                    "premise_first_hop_copy": float(first_copy[head].item()),
                    "premise_second_hop_copy": float(second_copy[head].item()),
                    "query_to_source": float(query_to_source[head].item()),
                    "query_to_target_first_child": float(
                        query_to_first_child[head].item()
                    ),
                    "query_to_target_second_parent": float(
                        query_to_second_parent[head].item()
                    ),
                    "query_to_target_end": float(
                        query_to_target_end[head].item()
                    ),
                    "query_to_counterfactual_end": float(
                        query_to_counterfactual_end[head].item()
                    ),
                    "target_end_selectivity": float(
                        (
                            query_to_target_end[head]
                            - query_to_counterfactual_end[head]
                        ).item()
                    ),
                }
            )
    return table


def residual_position_mask(
    pair: PairedTwoHopBatch,
    role: str,
    *,
    d_model: int,
) -> torch.Tensor:
    clean = pair.clean
    rows = torch.arange(clean.batch_size, device=clean.tokens.device)
    mask = torch.zeros(
        (clean.batch_size, clean.tokens.shape[1], 1),
        dtype=torch.bool,
        device=clean.tokens.device,
    )
    if role == "query":
        mask[:, clean.query_position] = True
        return mask
    target = clean.target_indices
    counterfactual = _counterfactual_indices(pair)
    index = target if role == "target_chain" else counterfactual
    for positions in (
        clean.first_parent_positions[rows, index],
        clean.first_child_positions[rows, index],
        clean.second_parent_positions[rows, index],
        clean.second_child_positions[rows, index],
    ):
        mask[rows, positions, 0] = True
    return mask


@torch.inference_mode()
def residual_patch_table(
    model: TwoHopTransformer,
    pair: PairedTwoHopBatch,
    baseline: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for depth in range(model.cfg.total_depth):
        for role in ("query", "target_chain", "counterfactual_chain"):
            mask = residual_position_mask(
                pair,
                role,
                d_model=model.cfg.d_model,
            )
            patch_in_logits = model(
                pair.corrupt.tokens,
                interventions={
                    depth: {
                        "resid_patch": baseline["clean_cache"][
                            f"depth{depth}.resid"
                        ],
                        "resid_mask": mask,
                    }
                },
            )
            patch_out_logits = model(
                pair.clean.tokens,
                interventions={
                    depth: {
                        "resid_patch": baseline["corrupt_cache"][
                            f"depth{depth}.resid"
                        ],
                        "resid_mask": mask,
                    }
                },
            )
            patch_in_metric = float(
                logit_metric(
                    patch_in_logits,
                    pair.clean.labels,
                    pair.corrupt.labels,
                ).mean().item()
            )
            patch_out_metric = float(
                logit_metric(
                    patch_out_logits,
                    pair.clean.labels,
                    pair.corrupt.labels,
                ).mean().item()
            )
            rows.append(
                {
                    "effective_depth": depth + 1,
                    "parameter_block": model.parameter_index(depth),
                    "role": role,
                    "patch_in_metric": patch_in_metric,
                    "patch_in_recovery": normalized_recovery(
                        patch_in_metric,
                        clean_metric=baseline["clean_metric"],
                        corrupt_metric=baseline["corrupt_metric"],
                    ),
                    "patch_out_metric": patch_out_metric,
                    "patch_out_necessity": (
                        baseline["clean_metric"] - patch_out_metric
                    )
                    / max(
                        1e-8,
                        baseline["clean_metric"]
                        - baseline["corrupt_metric"],
                    ),
                }
            )
    return rows


def validate_circuit(
    model: TwoHopTransformer,
    pair: PairedTwoHopBatch,
    selected: list[Site],
    baseline: Mapping[str, Any],
    *,
    individual_ranking: list[dict[str, Any]],
    random_controls: int,
    seed: int,
) -> dict[str, Any]:
    patch_in = run_site_patch(
        model,
        pair,
        selected,
        direction="patch_in",
        clean_cache=baseline["clean_cache"],
        corrupt_cache=baseline["corrupt_cache"],
        clean_metric=baseline["clean_metric"],
        corrupt_metric=baseline["corrupt_metric"],
    )
    patch_out = run_site_patch(
        model,
        pair,
        selected,
        direction="patch_out",
        clean_cache=baseline["clean_cache"],
        corrupt_cache=baseline["corrupt_cache"],
        clean_metric=baseline["clean_metric"],
        corrupt_metric=baseline["corrupt_metric"],
    )
    leave_one_out: list[dict[str, Any]] = []
    for removed in selected:
        remaining = [site for site in selected if site != removed]
        result = run_site_patch(
            model,
            pair,
            remaining,
            direction="patch_in",
            clean_cache=baseline["clean_cache"],
            corrupt_cache=baseline["corrupt_cache"],
            clean_metric=baseline["clean_metric"],
            corrupt_metric=baseline["corrupt_metric"],
        )
        leave_one_out.append(
            {
                "removed": removed.effective_id,
                "remaining_recovery": result["recovery"],
                "recovery_loss": patch_in["recovery"] - result["recovery"],
            }
        )
    rng = random.Random(seed)
    universe = all_sites(model)
    random_rows: list[dict[str, Any]] = []
    for control in range(random_controls):
        sampled = rng.sample(universe, k=len(selected))
        result = run_site_patch(
            model,
            pair,
            sampled,
            direction="patch_in",
            clean_cache=baseline["clean_cache"],
            corrupt_cache=baseline["corrupt_cache"],
            clean_metric=baseline["clean_metric"],
            corrupt_metric=baseline["corrupt_metric"],
        )
        random_rows.append(
            {
                "control": control,
                "recovery": result["recovery"],
                "clean_answer_accuracy": result["clean_answer_accuracy"],
                "sites": [site.effective_id for site in sampled],
            }
        )
    selected_ids = {site.effective_id for site in selected}
    by_id = {site.effective_id: site for site in universe}
    alternative = [
        by_id[row["effective_id"]]
        for row in individual_ranking
        if row["effective_id"] not in selected_ids
    ][: len(selected)]
    alternative_result = run_site_patch(
        model,
        pair,
        alternative,
        direction="patch_in",
        clean_cache=baseline["clean_cache"],
        corrupt_cache=baseline["corrupt_cache"],
        clean_metric=baseline["clean_metric"],
        corrupt_metric=baseline["corrupt_metric"],
    )
    parameter_sites = {
        (
            model.parameter_index(site.depth),
            site.head,
            site.role,
        )
        for site in selected
    }
    return {
        "patch_in_sufficiency": patch_in,
        "patch_out_necessity": patch_out,
        "leave_one_out": leave_one_out,
        "random_controls": random_rows,
        "random_recovery_mean": mean(
            row["recovery"] for row in random_rows
        )
        if random_rows
        else None,
        "random_recovery_std": pstdev(
            row["recovery"] for row in random_rows
        )
        if len(random_rows) > 1
        else 0.0,
        "alternative_disjoint_sites": [
            site.to_dict(model) for site in alternative
        ],
        "alternative_disjoint_result": alternative_result,
        "selected_effective_site_count": len(selected),
        "selected_parameter_site_count": len(parameter_sites),
        "parameter_reuse_factor": len(selected) / max(1, len(parameter_sites)),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@torch.inference_mode()
def overloop_curve(
    model: TwoHopTransformer,
    pair: PairedTwoHopBatch,
    max_depth: int,
) -> list[dict[str, float]]:
    if model.cfg.architecture != "periodic":
        return []
    result = model.forward_all(
        pair.clean.tokens,
        active_depth=max_depth,
    )
    logits = result["logits_by_depth"]
    if not isinstance(logits, torch.Tensor):
        raise RuntimeError("missing overloop logits")
    return [
        {
            "effective_depth": depth + 1,
            "accuracy": float(
                logits[:, depth]
                .argmax(dim=1)
                .eq(pair.clean.labels)
                .float()
                .mean()
                .item()
            ),
            "metric": float(
                logit_metric(
                    logits[:, depth],
                    pair.clean.labels,
                    pair.corrupt.labels,
                )
                .mean()
                .item()
            ),
        }
        for depth in range(max_depth)
    ]


def analyze_run(
    args: argparse.Namespace,
    run_summary: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    run_name = str(run_summary["run_name"])
    output_dir = args.suite_dir / "circuits" / run_name
    report_path = output_dir / "circuit_report.json"
    if report_path.exists() and not args.force:
        return json.loads(report_path.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(run_summary["checkpoint"])
    if not checkpoint_path.exists():
        checkpoint_path = args.suite_dir / "runs" / run_name / "best.pt"
    model, checkpoint = load_checkpoint(checkpoint_path, device)
    id_accuracy = float(
        run_summary["evaluation"]["topological"]["depth_accuracy"][-1]
    )
    if id_accuracy < args.behavior_threshold:
        report = {
            "run_name": run_name,
            "model": run_summary["model"],
            "seed": run_summary["seed"],
            "eligible": False,
            "reason": (
                f"ID accuracy {id_accuracy:.4f} is below circuit gate "
                f"{args.behavior_threshold:.4f}"
            ),
        }
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report

    discovery_pair = make_paired_twohop_batch(
        model.cfg,
        args.discovery_examples,
        device=device,
        generator=generator_for(
            device,
            args.seed_base + int(run_summary["seed"]),
        ),
        corruption_mode="bridge_swap",
    )
    discovery_baseline = baseline_pair(model, discovery_pair)
    selected, individual_rows, greedy_rows = discover_sites(
        model,
        discovery_pair,
        discovery_baseline,
        candidate_pool=args.candidate_pool,
        max_sites=args.max_sites,
        target_recovery=args.target_recovery,
    )
    discovery_baseline_summary = {
        key: value
        for key, value in discovery_baseline.items()
        if key
        not in {
            "clean_result",
            "corrupt_result",
            "clean_cache",
            "corrupt_cache",
        }
    }
    del discovery_pair, discovery_baseline
    if device.type == "cuda":
        torch.cuda.empty_cache()

    validation_pair = make_paired_twohop_batch(
        model.cfg,
        args.validation_examples,
        device=device,
        generator=generator_for(
            device,
            args.seed_base + 10000 + int(run_summary["seed"]),
        ),
        corruption_mode="bridge_swap",
    )
    validation_baseline = baseline_pair(model, validation_pair)
    validation = validate_circuit(
        model,
        validation_pair,
        selected,
        validation_baseline,
        individual_ranking=individual_rows,
        random_controls=args.random_controls,
        seed=args.seed_base + 20000 + int(run_summary["seed"]),
    )
    residual_rows = residual_patch_table(
        model,
        validation_pair,
        validation_baseline,
    )
    overloop = overloop_curve(
        model,
        validation_pair,
        args.overloop_depth,
    )
    validation_baseline_summary = {
        key: value
        for key, value in validation_baseline.items()
        if key
        not in {
            "clean_result",
            "corrupt_result",
            "clean_cache",
            "corrupt_cache",
        }
    }
    del validation_pair, validation_baseline
    if device.type == "cuda":
        torch.cuda.empty_cache()

    attention_pair = make_paired_twohop_batch(
        model.cfg,
        args.attention_examples,
        device=device,
        generator=generator_for(
            device,
            args.seed_base + 30000 + int(run_summary["seed"]),
        ),
        corruption_mode="bridge_swap",
    )
    attention_baseline = baseline_pair(model, attention_pair)
    attention_rows = attention_route_table(
        model,
        attention_pair,
        attention_baseline["clean_cache"],
    )
    selected_ids = {site.effective_id for site in selected}
    for row in attention_rows:
        matching = [
            site
            for site in selected
            if site.depth == row["effective_depth"] - 1
            and site.head == row["head"]
        ]
        row["selected_roles"] = ",".join(site.role for site in matching)
        row["selected"] = bool(matching)
    del attention_pair, attention_baseline
    if device.type == "cuda":
        torch.cuda.empty_cache()

    circuit_object = {
        "intervention_family": (
            "clean/corrupt attention-head z patching at semantic token positions"
        ),
        "corruption": (
            "swap first-hop bridge ownership between target and distractor "
            "while preserving query and token multiset"
        ),
        "selected_sites": [site.to_dict(model) for site in selected],
        "selected_effective_ids": sorted(selected_ids),
        "selected_parameter_ids": sorted(
            {
                site.to_dict(model)["parameter_id"]
                for site in selected
            }
        ),
        "semantic_path_order": [
            "target_first_child",
            "query",
            "target_second_child",
        ],
        "threshold": {
            "target_recovery": args.target_recovery,
            "max_sites": args.max_sites,
            "candidate_pool": args.candidate_pool,
        },
    }
    report = {
        "run_name": run_name,
        "model": run_summary["model"],
        "seed": run_summary["seed"],
        "eligible": True,
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": checkpoint["step"],
        "config": model.cfg.to_dict(),
        "behavior_id_accuracy": id_accuracy,
        "parameter_schedule": [
            model.parameter_index(depth)
            for depth in range(model.cfg.total_depth)
        ],
        "circuit": circuit_object,
        "discovery_baseline": discovery_baseline_summary,
        "greedy_discovery": greedy_rows,
        "validation_baseline": validation_baseline_summary,
        "validation": validation,
        "overloop": overloop,
        "claim_strength": (
            "A sparse, held-out activation-patching circuit under the "
            "bridge-swap intervention. It is not claimed to be unique or a "
            "complete weight-level algorithm."
        ),
        "limitations": [
            "The circuit granularity is effective attention-head output at semantic positions.",
            "Patch states may be off-manifold even though the corruption is token-natural.",
            "The alternative-circuit check is a disjoint ranked control, not exhaustive uniqueness proof.",
            "Attention routes support interpretation but are not counted as causal evidence by themselves.",
        ],
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_csv(output_dir / "individual_site_scores.csv", individual_rows)
    _write_csv(output_dir / "greedy_discovery.csv", greedy_rows)
    _write_csv(output_dir / "attention_routes.csv", attention_rows)
    _write_csv(output_dir / "residual_patching.csv", residual_rows)
    _write_csv(output_dir / "overloop.csv", overloop)
    _write_csv(
        output_dir / "leave_one_out.csv",
        validation["leave_one_out"],
    )
    _write_csv(
        output_dir / "random_controls.csv",
        [
            {
                **{key: value for key, value in row.items() if key != "sites"},
                "sites": json.dumps(row["sites"]),
            }
            for row in validation["random_controls"]
        ],
    )
    return report


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def aggregate_reports(
    args: argparse.Namespace,
    reports: list[dict[str, Any]],
) -> None:
    eligible = [report for report in reports if report.get("eligible")]
    rows: list[dict[str, Any]] = []
    for report in eligible:
        validation = report["validation"]
        selected = report["circuit"]["selected_sites"]
        roles = Counter(site["role"] for site in selected)
        rows.append(
            {
                "run_name": report["run_name"],
                "model": report["model"],
                "seed": report["seed"],
                "id_accuracy": report["behavior_id_accuracy"],
                "selected_effective_sites": validation[
                    "selected_effective_site_count"
                ],
                "selected_parameter_sites": validation[
                    "selected_parameter_site_count"
                ],
                "parameter_reuse_factor": validation[
                    "parameter_reuse_factor"
                ],
                "patch_in_recovery": validation[
                    "patch_in_sufficiency"
                ]["recovery"],
                "patch_in_accuracy": validation[
                    "patch_in_sufficiency"
                ]["clean_answer_accuracy"],
                "patch_out_necessity": validation[
                    "patch_out_necessity"
                ]["necessity"],
                "random_recovery_mean": validation["random_recovery_mean"],
                "alternative_recovery": validation[
                    "alternative_disjoint_result"
                ]["recovery"],
                "selected_roles": json.dumps(dict(roles), sort_keys=True),
                "selected_effective_ids": json.dumps(
                    report["circuit"]["selected_effective_ids"]
                ),
                "selected_parameter_ids": json.dumps(
                    report["circuit"]["selected_parameter_ids"]
                ),
            }
        )
    _write_csv(args.suite_dir / "circuit_summary.csv", rows)

    by_model: dict[str, list[dict[str, Any]]] = {}
    for report in eligible:
        by_model.setdefault(report["model"], []).append(report)
    stability: dict[str, Any] = {}
    for model_name, model_reports in by_model.items():
        effective_scores: list[float] = []
        parameter_scores: list[float] = []
        for left_index, left in enumerate(model_reports):
            for right in model_reports[left_index + 1 :]:
                effective_scores.append(
                    _jaccard(
                        set(left["circuit"]["selected_effective_ids"]),
                        set(right["circuit"]["selected_effective_ids"]),
                    )
                )
                parameter_scores.append(
                    _jaccard(
                        set(left["circuit"]["selected_parameter_ids"]),
                        set(right["circuit"]["selected_parameter_ids"]),
                    )
                )
        stability[model_name] = {
            "n_runs": len(model_reports),
            "effective_site_jaccard_mean": mean(effective_scores)
            if effective_scores
            else None,
            "parameter_site_jaccard_mean": mean(parameter_scores)
            if parameter_scores
            else None,
        }
    aggregate: dict[str, Any] = {}
    for model_name, model_rows in {
        name: [row for row in rows if row["model"] == name]
        for name in by_model
    }.items():
        numeric = (
            "selected_effective_sites",
            "selected_parameter_sites",
            "parameter_reuse_factor",
            "patch_in_recovery",
            "patch_in_accuracy",
            "patch_out_necessity",
            "random_recovery_mean",
            "alternative_recovery",
        )
        aggregate[model_name] = {
            "n_runs": len(model_rows),
            **{
                f"{field}_mean": mean(float(row[field]) for row in model_rows)
                for field in numeric
            },
            **{
                f"{field}_std": pstdev(
                    float(row[field]) for row in model_rows
                )
                if len(model_rows) > 1
                else 0.0
                for field in numeric
            },
        }
    payload = {
        "aggregate_by_model": aggregate,
        "cross_seed_stability": stability,
        "ineligible_runs": [
            {
                "run_name": report["run_name"],
                "reason": report["reason"],
            }
            for report in reports
            if not report.get("eligible")
        ],
        "claim_ledger": [
            {
                "claim": "standard model reproduces a sequential-query route",
                "status": "evaluate attention routes plus causal site order",
            },
            {
                "claim": "periodic model reuses parameter heads across effective roles",
                "status": "evaluate parameter reuse factor and role table",
            },
            {
                "claim": "the discovered circuit is unique",
                "status": "not claimed",
            },
        ],
    }
    (args.suite_dir / "circuit_aggregate.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    write_comparison_report(args, rows, aggregate, stability, reports)
    make_plots(args, rows, eligible)


def write_comparison_report(
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    aggregate: Mapping[str, Any],
    stability: Mapping[str, Any],
    reports: list[dict[str, Any]],
) -> None:
    lines = [
        "# Two-hop circuit comparison: standard vs periodic Transformer",
        "",
        "Behavior gate: only checkpoints with held-out ID accuracy at least "
        f"`{args.behavior_threshold:.2f}` enter the circuit comparison.",
        "",
        "Clean/corrupt pairs preserve the query and complete token multiset. "
        "They swap the first-hop bridge ownership between the queried chain and "
        "one distractor chain, changing the logically correct endpoint.",
        "",
        "| model | runs | effective sites | parameter sites | reuse | patch-in recovery | patch-out necessity | random recovery | alternative recovery |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model_name in args.models:
        values = aggregate.get(model_name)
        if values is None:
            continue
        lines.append(
            f"| {model_name} | {values['n_runs']} | "
            f"{values['selected_effective_sites_mean']:.2f} | "
            f"{values['selected_parameter_sites_mean']:.2f} | "
            f"{values['parameter_reuse_factor_mean']:.2f} | "
            f"{values['patch_in_recovery_mean']:.3f} | "
            f"{values['patch_out_necessity_mean']:.3f} | "
            f"{values['random_recovery_mean_mean']:.3f} | "
            f"{values['alternative_recovery_mean']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Stability",
            "",
            "| model | effective-site Jaccard | parameter-site Jaccard |",
            "|---|---:|---:|",
        ]
    )
    for model_name in args.models:
        values = stability.get(model_name)
        if values is None:
            continue
        effective = values["effective_site_jaccard_mean"]
        parameter = values["parameter_site_jaccard_mean"]
        lines.append(
            f"| {model_name} | "
            f"{'-' if effective is None else f'{effective:.3f}'} | "
            f"{'-' if parameter is None else f'{parameter:.3f}'} |"
        )
    lines.extend(
        [
            "",
            "## Evidence boundary",
            "",
            "- Selected sites were discovered on one paired dataset and validated on an independent paired dataset.",
            "- Patch-in is the circuit-only sufficiency test under the chosen intervention family.",
            "- Patch-out is the corresponding necessity test.",
            "- Same-size random sites and a disjoint ranked alternative are reported.",
            "- Attention routes are used to assign roles, not to establish causality.",
            "- The result is a faithful activation-patching circuit under this corruption; uniqueness and a complete weight-level algorithm are not claimed.",
            "",
        ]
    )
    ineligible = [report for report in reports if not report.get("eligible")]
    if ineligible:
        lines.extend(["## Behavior-gate failures", ""])
        for report in ineligible:
            lines.append(f"- `{report['run_name']}`: {report['reason']}")
        lines.append("")
    (args.suite_dir / "CIRCUIT_REPORT.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def make_plots(
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    reports: list[dict[str, Any]],
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as error:  # pragma: no cover - optional dependency
        (args.suite_dir / "PLOTTING_SKIPPED.txt").write_text(
            str(error),
            encoding="utf-8",
        )
        return
    plots = args.suite_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    model_names = [name for name in args.models if any(row["model"] == name for row in rows)]
    x = np.arange(len(model_names))
    width = 0.22
    metrics = (
        ("patch_in_recovery", "patch-in"),
        ("patch_out_necessity", "patch-out"),
        ("random_recovery_mean", "random"),
    )
    plt.figure(figsize=(8, 4.8))
    for offset, (field, label) in enumerate(metrics):
        values = [
            mean(float(row[field]) for row in rows if row["model"] == model)
            for model in model_names
        ]
        plt.bar(
            x + (offset - 1) * width,
            values,
            width=width,
            label=label,
        )
    plt.xticks(x, model_names)
    plt.ylabel("normalized clean-vs-corrupt recovery")
    plt.title("Held-out circuit validation")
    plt.axhline(0.0, color="black", linewidth=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots / "circuit_validation.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 4.8))
    for model in model_names:
        curves = [
            report["validation_baseline"]["prefix_clean_accuracy"]
            for report in reports
            if report["model"] == model
        ]
        array = np.asarray(curves, dtype=np.float64)
        depths = np.arange(1, array.shape[1] + 1)
        plt.plot(depths, array.mean(axis=0), marker="o", label=model)
        if array.shape[0] > 1:
            plt.fill_between(
                depths,
                array.mean(axis=0) - array.std(axis=0),
                array.mean(axis=0) + array.std(axis=0),
                alpha=0.18,
            )
    plt.xlabel("effective depth")
    plt.ylabel("clean accuracy")
    plt.title("Where two-hop behavior becomes readable")
    plt.ylim(0.0, 1.03)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots / "prefix_accuracy.png", dpi=180)
    plt.close()


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    device = pick_device(args.device)
    if device.type == "cuda":
        if not 0.0 < args.cuda_memory_fraction <= 1.0:
            raise ValueError("cuda-memory-fraction must be in (0, 1]")
        torch.cuda.set_per_process_memory_fraction(
            args.cuda_memory_fraction,
            device=torch.cuda.current_device(),
        )
    reports: list[dict[str, Any]] = []
    for model_name in args.models:
        for seed in args.seeds:
            run_name = f"{model_name}_seed{seed}"
            summary_path = args.suite_dir / "runs" / run_name / "summary.json"
            if not summary_path.exists():
                continue
            print(f"=== circuit {run_name} on {device} ===", flush=True)
            run_summary = json.loads(summary_path.read_text(encoding="utf-8"))
            report = analyze_run(args, run_summary, device)
            reports.append(report)
            if report.get("eligible"):
                validation = report["validation"]
                print(
                    f"{run_name}: sites="
                    f"{validation['selected_effective_site_count']} "
                    f"recovery={validation['patch_in_sufficiency']['recovery']:.3f} "
                    f"necessity={validation['patch_out_necessity']['necessity']:.3f}",
                    flush=True,
                )
            else:
                print(f"{run_name}: skipped ({report['reason']})", flush=True)
    if not reports:
        raise FileNotFoundError("no run summaries found")
    aggregate_reports(args, reports)
    print(f"wrote circuit outputs under {args.suite_dir}", flush=True)


if __name__ == "__main__":
    main()
