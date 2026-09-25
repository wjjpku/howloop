from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import torch

from reasoning_loop.graph_path_depth_circuit import (
    _apply_branch,
    fixed_depth_batch,
    load_checkpoint,
    target_margin,
)
from reasoning_loop.graph_path_loop import (
    GraphPathConfig,
    LoopedGraphPathTransformer,
    TransformerBlock,
    pick_device,
    set_seed,
)
from reasoning_loop.graph_path_temporal_intervention import (
    cache_raw_states,
    logits_from_raw_state,
    replace_answer_state,
)


Branch = Literal["attn", "mlp"]
InterventionMode = Literal["baseline", "skip", "repeat"]


@dataclass(frozen=True)
class BranchSite:
    loop_index: int
    block_index: int
    branch: Branch

    @property
    def label(self) -> str:
        return (
            f"L{self.loop_index + 1}.B{self.block_index + 1}."
            f"{self.branch}"
        )


def branch_sites(
    model: LoopedGraphPathTransformer,
    *,
    max_loops: int,
) -> list[BranchSite]:
    sites: list[BranchSite] = []
    for loop_index in range(max_loops):
        for block_index in model.active_block_indices(loop_index):
            for branch in ("attn", "mlp"):
                sites.append(BranchSite(loop_index, block_index, branch))
    return sites


@torch.no_grad()
def rollout_branch_intervention(
    model: LoopedGraphPathTransformer,
    tokens: torch.Tensor,
    *,
    max_loops: int,
    intervention_site: BranchSite | None = None,
    mode: InterventionMode = "baseline",
    cache_branch_states: bool = False,
) -> dict[str, Any]:
    """Run the shared stack while skipping or repeating one branch update.

    Repeat means recomputing the selected attention/MLP from the residual state
    produced by its first application. It is therefore a genuine extra branch
    application, not merely doubling a cached update vector.
    """
    if max_loops < 1:
        raise ValueError("max_loops must be positive")
    if mode not in {"baseline", "skip", "repeat"}:
        raise ValueError(f"unknown intervention mode: {mode}")
    if (mode == "baseline") != (intervention_site is None):
        raise ValueError(
            "baseline requires no site; skip/repeat require one site"
        )
    if model.block_style != "legacy":
        raise ValueError("compression analysis supports legacy blocks")

    valid_sites = set(branch_sites(model, max_loops=max_loops))
    if intervention_site is not None and intervention_site not in valid_sites:
        raise ValueError("intervention site is outside the rollout")

    x = model.token_embed(tokens) + model.pos_embed.unsqueeze(0)
    logits_by_loop: list[torch.Tensor] = []
    states: dict[BranchSite, dict[str, torch.Tensor]] = {}
    for loop_index in range(max_loops):
        for block_index in model.active_block_indices(loop_index):
            block = model.blocks[block_index]
            if not isinstance(block, TransformerBlock):
                raise TypeError("legacy TransformerBlock required")
            if block.residual_projector is not None:
                raise ValueError(
                    "branch skip/repeat does not support residual projectors"
                )
            for branch in ("attn", "mlp"):
                site = BranchSite(loop_index, block_index, branch)
                before = x
                applications = (
                    0
                    if site == intervention_site and mode == "skip"
                    else 2
                    if site == intervention_site and mode == "repeat"
                    else 1
                )
                for _ in range(applications):
                    x = x + _apply_branch(block, x, branch)
                if cache_branch_states:
                    states[site] = {"before": before, "after": x}
        if model.outer_norm is not None:
            x = model.outer_norm(x)
        logits_by_loop.append(logits_from_raw_state(model, x))
    return {
        "logits_by_loop": torch.stack(logits_by_loop, dim=1),
        "final_raw_state": x,
        "branch_states": states,
    }


def path_position_metrics(
    logits: torch.Tensor,
    *,
    start: torch.Tensor,
    targets: torch.Tensor,
    endpoint_position: int,
) -> dict[str, torch.Tensor]:
    """Score decoded path positions with an endpoint-collision control.

    A non-endpoint example is counted only when its node differs from the true
    endpoint node. This prevents permutation cycles from making an endpoint
    prediction look like evidence for an earlier semantic step.
    """
    if logits.ndim != 2 or targets.ndim != 2 or start.ndim != 1:
        raise ValueError("invalid logits/start/targets ranks")
    if logits.shape[0] != start.shape[0] or targets.shape[0] != start.shape[0]:
        raise ValueError("batch dimensions must match")
    all_targets = torch.cat((start[:, None], targets), dim=1)
    if not 0 <= endpoint_position < all_targets.shape[1]:
        raise ValueError("endpoint_position is outside the target range")
    endpoint = all_targets[:, endpoint_position]
    prediction = logits.argmax(dim=-1)
    probability = logits.softmax(dim=-1)
    correct: list[torch.Tensor] = []
    probability_sum: list[torch.Tensor] = []
    valid_count: list[torch.Tensor] = []
    for position in range(all_targets.shape[1]):
        target = all_targets[:, position]
        valid = (
            torch.ones_like(target, dtype=torch.bool)
            if position == endpoint_position
            else target.ne(endpoint)
        )
        correct.append((prediction.eq(target) & valid).sum())
        probability_sum.append(
            (
                probability.gather(1, target[:, None]).squeeze(1)
                * valid
            ).sum()
        )
        valid_count.append(valid.sum())
    return {
        "correct": torch.stack(correct),
        "probability_sum": torch.stack(probability_sum),
        "valid_count": torch.stack(valid_count),
    }


def _combine_path_metrics(
    items: list[dict[str, torch.Tensor]],
) -> tuple[list[float], list[float], list[int]]:
    correct = torch.stack([item["correct"].cpu() for item in items]).sum(0)
    probability = torch.stack(
        [item["probability_sum"].cpu() for item in items]
    ).sum(0)
    count = torch.stack([item["valid_count"].cpu() for item in items]).sum(0)
    denominator = count.clamp_min(1)
    accuracy = correct.float() / denominator
    mean_probability = probability.float() / denominator
    accuracy[count.eq(0)] = float("nan")
    mean_probability[count.eq(0)] = float("nan")
    return accuracy.tolist(), mean_probability.tolist(), count.tolist()


def _best_position(accuracies: list[float]) -> tuple[int, float]:
    finite = [
        (index, value)
        for index, value in enumerate(accuracies)
        if value == value
    ]
    return max(finite, key=lambda pair: pair[1]) if finite else (-1, float("nan"))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _calibrate_donor_positions(
    *,
    model: LoopedGraphPathTransformer,
    cfg: GraphPathConfig,
    device: torch.device,
    batch_size: int,
    path_positions: int,
) -> list[dict[str, Any]]:
    tokens, targets, _, start = fixed_depth_batch(
        cfg,
        batch_size,
        device,
        path_positions=path_positions,
    )
    states = cache_raw_states(model, tokens, max_loop=cfg.max_loops)
    resolved: list[dict[str, Any]] = []
    for loop_index, state in enumerate(states):
        metrics = path_position_metrics(
            logits_from_raw_state(model, state),
            start=start,
            targets=targets,
            endpoint_position=cfg.max_depth,
        )
        accuracy, _, _ = _combine_path_metrics([metrics])
        position, value = _best_position(accuracy)
        resolved.append(
            {
                "donor_loop": loop_index + 1,
                "decoded_position": position,
                "decoded_accuracy": value,
            }
        )
    return resolved


@torch.no_grad()
def analyze_checkpoint(
    *,
    name: str,
    checkpoint: Path,
    out_dir: Path,
    device: torch.device,
    batch_size: int,
    batches: int,
    seed: int,
    extra_loops: int,
) -> dict[str, Any]:
    set_seed(seed)
    model, cfg, payload = load_checkpoint(checkpoint, device)
    if cfg.block_schedule != "all_blocks":
        raise ValueError("matched compression analysis requires all_blocks")
    run_dir = out_dir / name
    run_dir.mkdir(parents=True, exist_ok=True)
    sites = branch_sites(model, max_loops=cfg.max_loops)
    evaluation_loops = cfg.max_loops + extra_loops
    path_positions = max(cfg.max_depth, evaluation_loops + 1)

    baseline_endpoint_correct = 0
    baseline_examples = 0
    baseline_margin_sum = 0.0
    unroll_metrics: list[list[dict[str, torch.Tensor]]] = [
        [] for _ in range(evaluation_loops)
    ]
    progression: dict[
        tuple[BranchSite, str], list[dict[str, torch.Tensor]]
    ] = {
        (site, stage): []
        for site in sites
        for stage in ("before", "after")
    }
    intervention_metrics: dict[
        tuple[BranchSite, InterventionMode], list[dict[str, torch.Tensor]]
    ] = {
        (site, mode): []
        for site in sites
        for mode in ("skip", "repeat")
    }
    intervention_endpoint: dict[
        tuple[BranchSite, InterventionMode], list[tuple[int, int, float]]
    ] = {
        key: [] for key in intervention_metrics
    }
    overloop_keys = [
        (
            horizon,
            BranchSite(horizon - 1, block_index, branch),
            mode,
        )
        for horizon in range(cfg.max_loops + 1, evaluation_loops + 1)
        for block_index in model.active_block_indices(horizon - 1)
        for branch in ("attn", "mlp")
        for mode in ("skip", "repeat")
    ]
    overloop_metrics: dict[
        tuple[int, BranchSite, InterventionMode],
        list[dict[str, torch.Tensor]],
    ] = {key: [] for key in overloop_keys}
    overloop_endpoint: dict[
        tuple[int, BranchSite, InterventionMode],
        list[tuple[int, int, float, int, float]],
    ] = {key: [] for key in overloop_keys}

    for _ in range(batches):
        tokens, targets, _, start = fixed_depth_batch(
            cfg,
            batch_size,
            device,
            path_positions=path_positions,
        )
        endpoint = targets[:, cfg.max_depth - 1]
        baseline = rollout_branch_intervention(
            model,
            tokens,
            max_loops=cfg.max_loops,
            cache_branch_states=True,
        )
        reference = model.forward_all(tokens, max_loops=cfg.max_loops)[
            "logits_by_loop"
        ]
        torch.testing.assert_close(
            baseline["logits_by_loop"], reference, rtol=1e-5, atol=1e-5
        )
        final_logits = baseline["logits_by_loop"][:, -1]
        baseline_endpoint_correct += int(
            final_logits.argmax(-1).eq(endpoint).sum()
        )
        baseline_examples += batch_size
        baseline_margin_sum += float(target_margin(final_logits, endpoint).sum())

        extended = model.forward_all(tokens, max_loops=evaluation_loops)[
            "logits_by_loop"
        ]
        for loop_index in range(evaluation_loops):
            unroll_metrics[loop_index].append(
                path_position_metrics(
                    extended[:, loop_index],
                    start=start,
                    targets=targets,
                    endpoint_position=cfg.max_depth,
                )
            )
        for horizon, site, mode in overloop_keys:
            result = rollout_branch_intervention(
                model,
                tokens,
                max_loops=horizon,
                intervention_site=site,
                mode=mode,
            )
            logits = result["logits_by_loop"][:, -1]
            baseline_extra = extended[:, horizon - 1]
            key = (horizon, site, mode)
            overloop_metrics[key].append(
                path_position_metrics(
                    logits,
                    start=start,
                    targets=targets,
                    endpoint_position=cfg.max_depth,
                )
            )
            overloop_endpoint[key].append(
                (
                    int(logits.argmax(-1).eq(endpoint).sum()),
                    batch_size,
                    float(target_margin(logits, endpoint).sum()),
                    int(baseline_extra.argmax(-1).eq(endpoint).sum()),
                    float(target_margin(baseline_extra, endpoint).sum()),
                )
            )

        for site, state_pair in baseline["branch_states"].items():
            for stage in ("before", "after"):
                progression[(site, stage)].append(
                    path_position_metrics(
                        logits_from_raw_state(model, state_pair[stage]),
                        start=start,
                        targets=targets,
                        endpoint_position=cfg.max_depth,
                    )
                )

        for site in sites:
            for mode in ("skip", "repeat"):
                result = rollout_branch_intervention(
                    model,
                    tokens,
                    max_loops=cfg.max_loops,
                    intervention_site=site,
                    mode=mode,
                )
                logits = result["logits_by_loop"][:, -1]
                key = (site, mode)
                intervention_metrics[key].append(
                    path_position_metrics(
                        logits,
                        start=start,
                        targets=targets,
                        endpoint_position=cfg.max_depth,
                    )
                )
                intervention_endpoint[key].append(
                    (
                        int(logits.argmax(-1).eq(endpoint).sum()),
                        batch_size,
                        float(target_margin(logits, endpoint).sum()),
                    )
                )

    baseline_accuracy = baseline_endpoint_correct / baseline_examples
    baseline_margin = baseline_margin_sum / baseline_examples

    unroll_rows: list[dict[str, Any]] = []
    for loop_index, metrics in enumerate(unroll_metrics):
        accuracy, probability, counts = _combine_path_metrics(metrics)
        best, best_accuracy = _best_position(accuracy)
        for position in range(len(accuracy)):
            unroll_rows.append(
                {
                    "loop": loop_index + 1,
                    "path_position": position,
                    "accuracy": accuracy[position],
                    "probability": probability[position],
                    "valid_count": counts[position],
                    "best_path_position": best,
                    "best_path_accuracy": best_accuracy,
                    "trained_horizon": int(loop_index + 1 == cfg.max_loops),
                }
            )
    _write_csv(run_dir / "unroll_curve_rows.csv", unroll_rows)

    progression_rows: list[dict[str, Any]] = []
    for site in sites:
        stage_values: dict[str, tuple[list[float], list[float], list[int]]] = {}
        for stage in ("before", "after"):
            stage_values[stage] = _combine_path_metrics(
                progression[(site, stage)]
            )
            accuracy, probability, counts = stage_values[stage]
            best, best_accuracy = _best_position(accuracy)
            for position in range(len(accuracy)):
                progression_rows.append(
                    {
                        "site": site.label,
                        "loop": site.loop_index + 1,
                        "block": site.block_index + 1,
                        "branch": site.branch,
                        "stage": stage,
                        "path_position": position,
                        "accuracy": accuracy[position],
                        "probability": probability[position],
                        "valid_count": counts[position],
                        "best_path_position": best,
                        "best_path_accuracy": best_accuracy,
                    }
                )
    _write_csv(run_dir / "branch_semantic_progression_rows.csv", progression_rows)

    intervention_rows: list[dict[str, Any]] = []
    for site in sites:
        for mode in ("skip", "repeat"):
            key = (site, mode)
            accuracy, probability, counts = _combine_path_metrics(
                intervention_metrics[key]
            )
            best, best_accuracy = _best_position(accuracy)
            endpoint_items = intervention_endpoint[key]
            correct = sum(item[0] for item in endpoint_items)
            examples = sum(item[1] for item in endpoint_items)
            margin = sum(item[2] for item in endpoint_items) / examples
            endpoint_accuracy = correct / examples
            for position in range(len(accuracy)):
                intervention_rows.append(
                    {
                        "site": site.label,
                        "loop": site.loop_index + 1,
                        "block": site.block_index + 1,
                        "branch": site.branch,
                        "mode": mode,
                        "path_position": position,
                        "accuracy": accuracy[position],
                        "probability": probability[position],
                        "valid_count": counts[position],
                        "best_path_position": best,
                        "best_path_accuracy": best_accuracy,
                        "baseline_endpoint_accuracy": baseline_accuracy,
                        "endpoint_accuracy": endpoint_accuracy,
                        "endpoint_accuracy_drop": (
                            baseline_accuracy - endpoint_accuracy
                        ),
                        "baseline_endpoint_margin": baseline_margin,
                        "endpoint_margin": margin,
                        "endpoint_margin_drop": baseline_margin - margin,
                    }
                )
    _write_csv(
        run_dir / "branch_skip_repeat_rows.csv", intervention_rows
    )

    overloop_rows: list[dict[str, Any]] = []
    for horizon, site, mode in overloop_keys:
        key = (horizon, site, mode)
        accuracy, probability, counts = _combine_path_metrics(
            overloop_metrics[key]
        )
        best, best_accuracy = _best_position(accuracy)
        items = overloop_endpoint[key]
        examples = sum(item[1] for item in items)
        endpoint_accuracy = sum(item[0] for item in items) / examples
        endpoint_margin = sum(item[2] for item in items) / examples
        baseline_extra_accuracy = sum(item[3] for item in items) / examples
        baseline_extra_margin = sum(item[4] for item in items) / examples
        for position in range(len(accuracy)):
            overloop_rows.append(
                {
                    "horizon": horizon,
                    "extra_loop": horizon - cfg.max_loops,
                    "site": site.label,
                    "block": site.block_index + 1,
                    "branch": site.branch,
                    "mode": mode,
                    "path_position": position,
                    "accuracy": accuracy[position],
                    "probability": probability[position],
                    "valid_count": counts[position],
                    "best_path_position": best,
                    "best_path_accuracy": best_accuracy,
                    "baseline_endpoint_accuracy": baseline_extra_accuracy,
                    "endpoint_accuracy": endpoint_accuracy,
                    "endpoint_accuracy_drop": (
                        baseline_extra_accuracy - endpoint_accuracy
                    ),
                    "baseline_endpoint_margin": baseline_extra_margin,
                    "endpoint_margin": endpoint_margin,
                    "endpoint_margin_drop": (
                        baseline_extra_margin - endpoint_margin
                    ),
                }
            )
    _write_csv(run_dir / "overloop_branch_rows.csv", overloop_rows)

    calibration = _calibrate_donor_positions(
        model=model,
        cfg=cfg,
        device=device,
        batch_size=batch_size,
        path_positions=path_positions,
    )
    transplant_sums: dict[
        tuple[int, int], dict[str, float]
    ] = {
        (donor["donor_loop"], receiver_loop): {
            "correct": 0.0,
            "shuffled_correct": 0.0,
            "current_correct": 0.0,
            "count": 0.0,
        }
        for donor in calibration
        for receiver_loop in range(1, cfg.max_loops + 1)
    }
    for _ in range(batches):
        start = torch.randint(
            0, cfg.node_count, (batch_size,), device=device
        )
        donor_tokens, donor_targets, _, _ = fixed_depth_batch(
            cfg,
            batch_size,
            device,
            path_positions=path_positions,
            start=start,
        )
        receiver_tokens, _, receiver_successors, _ = fixed_depth_batch(
            cfg,
            batch_size,
            device,
            path_positions=path_positions,
            start=start,
        )
        donor_states = cache_raw_states(
            model, donor_tokens, max_loop=cfg.max_loops
        )
        receiver_states = cache_raw_states(
            model, receiver_tokens, max_loop=cfg.max_loops
        )
        for donor in calibration:
            donor_loop = donor["donor_loop"]
            position = donor["decoded_position"]
            current = (
                start
                if position == 0
                else donor_targets[:, position - 1]
            )
            successor = receiver_successors.gather(
                1, current[:, None]
            ).squeeze(1)
            donor_state = donor_states[donor_loop - 1]
            for receiver_loop in range(1, cfg.max_loops + 1):
                receiver_state = receiver_states[receiver_loop - 1]
                patched = replace_answer_state(receiver_state, donor_state)
                current_logits = logits_from_raw_state(model, patched)
                advanced = model.apply_loop(
                    patched, loop_index=receiver_loop
                )
                advanced_logits = logits_from_raw_state(model, advanced)
                shuffled = receiver_state.clone()
                shuffled[:, -1, :] = donor_state[:, -1, :].roll(1, dims=0)
                shuffled_logits = logits_from_raw_state(
                    model,
                    model.apply_loop(shuffled, loop_index=receiver_loop),
                )
                item = transplant_sums[(donor_loop, receiver_loop)]
                item["current_correct"] += float(
                    current_logits.argmax(-1).eq(current).sum()
                )
                item["correct"] += float(
                    advanced_logits.argmax(-1).eq(successor).sum()
                )
                item["shuffled_correct"] += float(
                    shuffled_logits.argmax(-1).eq(successor).sum()
                )
                item["count"] += batch_size

    transplant_rows: list[dict[str, Any]] = []
    calibration_by_loop = {
        item["donor_loop"]: item for item in calibration
    }
    for (donor_loop, receiver_loop), sums in transplant_sums.items():
        count = sums["count"]
        transplant_rows.append(
            {
                **calibration_by_loop[donor_loop],
                "receiver_loop": receiver_loop,
                "patched_current_accuracy": sums["current_correct"] / count,
                "next_successor_accuracy": sums["correct"] / count,
                "shuffled_next_accuracy": (
                    sums["shuffled_correct"] / count
                ),
                "successor_specificity": (
                    sums["correct"] - sums["shuffled_correct"]
                )
                / count,
            }
        )
    _write_csv(run_dir / "within_model_transplant_rows.csv", transplant_rows)

    summary = {
        "name": name,
        "checkpoint": str(checkpoint),
        "checkpoint_step": payload.get("step"),
        "initialization_seed": payload.get("initialization_seed"),
        "data_seed": payload.get("data_seed"),
        "config": asdict(cfg),
        "loss_mode": "final_only",
        "trained_loops": cfg.max_loops,
        "physical_blocks": cfg.n_layers,
        "effective_branch_sites": len(sites),
        "evaluation_loops": evaluation_loops,
        "batches": batches,
        "batch_size": batch_size,
        "baseline_endpoint_accuracy": baseline_accuracy,
        "baseline_endpoint_margin": baseline_margin,
        "controls": {
            "path_position": (
                "non-endpoint positions exclude endpoint-token collisions"
            ),
            "branch_repeat": (
                "recompute branch from its updated residual state"
            ),
            "transplant": (
                "within-model only; shuffled donor-answer control"
            ),
        },
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
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
        description="Matched D8-L6 versus D8-L8 compression circuit analysis."
    )
    parser.add_argument(
        "--run", action="append", type=parse_run_spec, required=True
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--extra-loops", type=int, default=4)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)
    summaries: dict[str, Any] = {}
    for name, checkpoint in args.run:
        summaries[name] = analyze_checkpoint(
            name=name,
            checkpoint=checkpoint,
            out_dir=args.out_dir,
            device=device,
            batch_size=args.batch_size,
            batches=args.batches,
            seed=args.seed,
            extra_loops=args.extra_loops,
        )
        print(f"done {name}", flush=True)
    (args.out_dir / "combined_summary.json").write_text(
        json.dumps({"models": summaries}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
