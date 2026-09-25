from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from reasoning_loop.boolean_dag_data import (
    UNKNOWN,
    BooleanDAGBatch,
    BooleanDAGConfig,
    make_boolean_dag_batch,
)
from reasoning_loop.boolean_dag_macrostep import (
    HELDOUT_PROGRAMS,
    DepthConditionedBooleanDAGTransformer,
    macrostep_targets,
)
from reasoning_loop.boolean_dag_model import BooleanDAGModelConfig
from reasoning_loop.graph_path_loop import pick_device, set_seed


OVERLOOP_PROGRAMS: tuple[tuple[int, ...], ...] = (
    (2, 2, 2, 2, 2),
    (3, 3, 3, 3, 2),
    (2, 2, 2, 2, 2, 2),
    (1, 2, 3, 4, 3, 3),
)
PHASE_PAIR_SEED = 20_260_723
SAME_TOTAL_PAIR_SEED = 20_260_724


def canonical_prefix(cumulative_depth: int) -> tuple[int, ...]:
    if not 0 <= cumulative_depth <= 12:
        raise ValueError("cumulative_depth must lie in 0 through 12")
    quotient, remainder = divmod(cumulative_depth, 4)
    return (4,) * quotient + ((remainder,) if remainder else ())


def same_total_programs(total_depth: int) -> list[tuple[int, ...]]:
    if total_depth < 1 or total_depth > 16:
        raise ValueError("total_depth must lie in 1 through 16")
    programs = []
    for length in range(1, 5):
        programs.extend(
            program
            for program in itertools.product(range(1, 5), repeat=length)
            if sum(program) == total_depth
        )
    return programs


def realized_depth_from_predictions(
    predictions: torch.Tensor,
    batch: BooleanDAGBatch,
    *,
    max_depth: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if predictions.shape != (batch.batch_size, batch.node_count):
        raise ValueError("predictions must have shape [batch, node]")
    candidates = torch.arange(max_depth + 1, device=predictions.device)
    depths = candidates[None].expand(batch.batch_size, -1)
    targets = macrostep_targets(batch, depths)
    scores = predictions[:, None, :].eq(targets).float().mean(dim=-1)
    scores = scores.masked_fill(candidates[None] > batch.depths[:, None], -1.0)
    return scores.argmax(dim=1), scores


def affected_oracle_metrics(
    *,
    clean_logits: torch.Tensor,
    swapped_logits: torch.Tensor,
    clean_target: torch.Tensor,
    swapped_target: torch.Tensor,
) -> dict[str, float | int]:
    if clean_logits.shape != swapped_logits.shape:
        raise ValueError("clean and swapped logits must have identical shapes")
    if clean_logits.shape[:-1] != clean_target.shape or clean_target.shape != swapped_target.shape:
        raise ValueError("targets must match the non-class logit dimensions")
    affected = clean_target.ne(swapped_target)
    affected_count = int(affected.sum())
    clean_prediction = clean_logits.argmax(dim=-1)
    swapped_prediction = swapped_logits.argmax(dim=-1)
    clean_indices = clean_target.unsqueeze(-1)
    swapped_indices = swapped_target.unsqueeze(-1)
    clean_preference = clean_logits.gather(-1, swapped_indices).squeeze(-1) - clean_logits.gather(
        -1, clean_indices
    ).squeeze(-1)
    swapped_preference = swapped_logits.gather(-1, swapped_indices).squeeze(-1) - swapped_logits.gather(
        -1, clean_indices
    ).squeeze(-1)
    return {
        "affected_count": affected_count,
        "clean_follows_clean": float(clean_prediction.eq(clean_target)[affected].sum()),
        "swapped_follows_swapped": float(
            swapped_prediction.eq(swapped_target)[affected].sum()
        ),
        "swapped_keeps_clean": float(swapped_prediction.eq(clean_target)[affected].sum()),
        "swapped_target_margin_sum": float(swapped_preference[affected].sum()),
        "causal_margin_shift_sum": float(
            (swapped_preference - clean_preference)[affected].sum()
        ),
    }


def _program_label(program: Iterable[int]) -> str:
    return "+".join(str(value) for value in program)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _safe(correct: float, count: int) -> float:
    return correct / count if count else float("nan")


@torch.no_grad()
def _evaluate_program(
    model: DepthConditionedBooleanDAGTransformer,
    *,
    data_cfg: BooleanDAGConfig,
    program: tuple[int, ...],
    device: torch.device,
    batch_size: int,
    batches: int,
    amp: bool,
    generator: torch.Generator,
) -> list[dict[str, Any]]:
    total_depth = sum(program)
    if total_depth > data_cfg.eval_max_depth:
        raise ValueError("program exceeds the graph-depth limit")
    loop_totals = [
        {
            "state_correct": 0.0,
            "state_count": 0,
            "exact_correct": 0.0,
            "exact_count": 0,
            "frontier_correct": 0.0,
            "frontier_count": 0,
            "root_resolved_correct": 0.0,
            "root_resolved_count": 0,
            "root_unknown_correct": 0.0,
            "root_unknown_count": 0,
            "realized_sum": 0.0,
            "realized_count": 0,
        }
        for _ in program
    ]
    increments = torch.tensor(program, device=device)[None].expand(batch_size, -1)
    cumulative = increments.cumsum(dim=1)
    for _ in range(batches):
        batch = make_boolean_dag_batch(
            data_cfg,
            batch_size,
            device,
            depths=torch.full(
                (batch_size,),
                data_cfg.eval_max_depth,
                device=device,
            ),
            generator=generator,
        )
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=amp and device.type == "cuda",
        ):
            logits = model.forward_all(
                batch,
                depth_increments=increments,
            )["logits_by_loop"]
        predictions = logits.argmax(dim=-1)
        targets = macrostep_targets(batch, cumulative)
        root_indices = batch.root_mask.long().argmax(dim=1)
        batch_rows = torch.arange(batch_size, device=device)
        previous = 0
        for loop, cumulative_depth in enumerate(np.cumsum(program)):
            totals = loop_totals[loop]
            correct = predictions[:, loop].eq(targets[:, loop])
            totals["state_correct"] += float(correct.sum())
            totals["state_count"] += correct.numel()
            totals["exact_correct"] += float(correct.all(dim=1).sum())
            totals["exact_count"] += batch_size
            frontier = batch.levels.gt(previous) & batch.levels.le(cumulative_depth)
            totals["frontier_correct"] += float(correct[frontier].sum())
            totals["frontier_count"] += int(frontier.sum())
            root_predictions = logits[batch_rows, loop, root_indices].argmax(dim=-1)
            root_targets = targets[batch_rows, loop, root_indices]
            resolved_root = root_targets.ne(UNKNOWN)
            unknown_root = ~resolved_root
            totals["root_resolved_correct"] += float(
                root_predictions.eq(root_targets)[resolved_root].sum()
            )
            totals["root_resolved_count"] += int(resolved_root.sum())
            totals["root_unknown_correct"] += float(
                root_predictions.eq(root_targets)[unknown_root].sum()
            )
            totals["root_unknown_count"] += int(unknown_root.sum())
            realized, _ = realized_depth_from_predictions(
                predictions[:, loop],
                batch,
                max_depth=data_cfg.eval_max_depth,
            )
            totals["realized_sum"] += float(realized.sum())
            totals["realized_count"] += batch_size
            previous = cumulative_depth
    rows = []
    for loop, (increment, cumulative_depth, totals) in enumerate(
        zip(program, np.cumsum(program), loop_totals),
        start=1,
    ):
        rows.append(
            {
                "program": _program_label(program),
                "program_length": len(program),
                "loop": loop,
                "increment": increment,
                "cumulative_depth": int(cumulative_depth),
                "state_accuracy": _safe(totals["state_correct"], totals["state_count"]),
                "exact_state_accuracy": _safe(
                    totals["exact_correct"], totals["exact_count"]
                ),
                "frontier_accuracy": _safe(
                    totals["frontier_correct"], totals["frontier_count"]
                ),
                "root_resolved_accuracy": _safe(
                    totals["root_resolved_correct"], totals["root_resolved_count"]
                ),
                "root_unknown_accuracy": _safe(
                    totals["root_unknown_correct"], totals["root_unknown_count"]
                ),
                "mean_realized_depth": _safe(
                    totals["realized_sum"], totals["realized_count"]
                ),
                "realized_progress": _safe(
                    totals["realized_sum"], totals["realized_count"]
                )
                - (int(cumulative_depth) - increment),
            }
        )
    return rows


def _final_program_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    final = rows[-1]
    return {
        **final,
        "minimum_state_accuracy": min(row["state_accuracy"] for row in rows),
        "minimum_frontier_accuracy": min(row["frontier_accuracy"] for row in rows),
    }


@torch.no_grad()
def _final_outputs_by_batch(
    model: DepthConditionedBooleanDAGTransformer,
    *,
    data_cfg: BooleanDAGConfig,
    program: tuple[int, ...],
    device: torch.device,
    batch_size: int,
    batches: int,
    amp: bool,
    seed: int,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    generator = torch.Generator(device=device).manual_seed(seed)
    increments = torch.tensor(program, device=device)[None].expand(batch_size, -1)
    outputs = []
    for _ in range(batches):
        batch = make_boolean_dag_batch(
            data_cfg,
            batch_size,
            device,
            depths=torch.full(
                (batch_size,),
                data_cfg.eval_max_depth,
                device=device,
            ),
            generator=generator,
        )
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=amp and device.type == "cuda",
        ):
            result = model.forward_all(
                batch,
                depth_increments=increments,
                return_states=True,
            )
        outputs.append(
            (
                result["logits_by_loop"][:, -1].argmax(dim=-1).cpu(),
                result["states_by_loop"][:, -1].float().cpu(),
            )
        )
    return outputs


def _compare_final_outputs(
    outputs: list[tuple[torch.Tensor, torch.Tensor]],
    reference: list[tuple[torch.Tensor, torch.Tensor]],
) -> dict[str, float]:
    if len(outputs) != len(reference):
        raise ValueError("paired outputs must have the same number of batches")
    disagreement = 0.0
    prediction_count = 0
    cosine_sum = 0.0
    hidden_count = 0
    for (prediction, state), (reference_prediction, reference_state) in zip(
        outputs,
        reference,
    ):
        disagreement += float(prediction.ne(reference_prediction).sum())
        prediction_count += prediction.numel()
        cosine_sum += float(F.cosine_similarity(state, reference_state, dim=-1).sum())
        hidden_count += state.shape[0] * state.shape[1]
    return {
        "prediction_disagreement_from_reference": _safe(
            disagreement,
            prediction_count,
        ),
        "hidden_cosine_similarity_to_reference": _safe(cosine_sum, hidden_count),
    }


@torch.no_grad()
def _instruction_interchange(
    model: DepthConditionedBooleanDAGTransformer,
    *,
    data_cfg: BooleanDAGConfig,
    phases: list[int],
    device: torch.device,
    batch_size: int,
    batches: int,
    amp: bool,
    generator: torch.Generator,
) -> list[dict[str, Any]]:
    rows = []
    for phase in phases:
        prefix = canonical_prefix(phase)
        totals = {
            "clean_state_correct": 0.0,
            "swapped_state_follows_swapped": 0.0,
            "swapped_state_keeps_clean": 0.0,
            "shuffled_state_follows_shuffled": 0.0,
            "state_count": 0,
            "clean_affected_follows_clean": 0.0,
            "swapped_affected_follows_swapped": 0.0,
            "swapped_affected_keeps_clean": 0.0,
            "affected_count": 0,
            "swapped_target_margin_sum": 0.0,
            "causal_margin_shift_sum": 0.0,
            "unchanged_max_diff": 0.0,
            "swap_disagreement": 0.0,
            "prediction_count": 0,
        }
        for _ in range(batches):
            clean_increment = torch.arange(batch_size, device=device) % 4 + 1
            swapped_increment = clean_increment % 4 + 1
            shuffled_increment = clean_increment[
                torch.randperm(batch_size, device=device, generator=generator)
            ]
            batch = make_boolean_dag_batch(
                data_cfg,
                batch_size,
                device,
                depths=torch.full(
                    (batch_size,),
                    data_cfg.eval_max_depth,
                    device=device,
                ),
                generator=generator,
            )
            state = model.encode(batch)
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=amp and device.type == "cuda",
            ):
                for increment in prefix:
                    state = model.apply_macro_step(
                        state,
                        torch.full((batch_size,), increment, device=device),
                    )
                clean_state = model.apply_macro_step(state, clean_increment)
                swapped_state = model.apply_macro_step(state, swapped_increment)
                unchanged_state = model.apply_macro_step(state, clean_increment)
                shuffled_state = model.apply_macro_step(state, shuffled_increment)
                clean_logits = model._readout(clean_state)[0]
                swapped_logits = model._readout(swapped_state)[0]
                unchanged_logits = model._readout(unchanged_state)[0]
                shuffled_logits = model._readout(shuffled_state)[0]
            clean_target = macrostep_targets(
                batch,
                (phase + clean_increment)[:, None],
            )[:, 0]
            swapped_target = macrostep_targets(
                batch,
                (phase + swapped_increment)[:, None],
            )[:, 0]
            shuffled_target = macrostep_targets(
                batch,
                (phase + shuffled_increment)[:, None],
            )[:, 0]
            clean_prediction = clean_logits.argmax(dim=-1)
            swapped_prediction = swapped_logits.argmax(dim=-1)
            shuffled_prediction = shuffled_logits.argmax(dim=-1)
            totals["clean_state_correct"] += float(
                clean_prediction.eq(clean_target).sum()
            )
            totals["swapped_state_follows_swapped"] += float(
                swapped_prediction.eq(swapped_target).sum()
            )
            totals["swapped_state_keeps_clean"] += float(
                swapped_prediction.eq(clean_target).sum()
            )
            totals["shuffled_state_follows_shuffled"] += float(
                shuffled_prediction.eq(shuffled_target).sum()
            )
            totals["state_count"] += clean_prediction.numel()
            affected = affected_oracle_metrics(
                clean_logits=clean_logits,
                swapped_logits=swapped_logits,
                clean_target=clean_target,
                swapped_target=swapped_target,
            )
            for key in (
                "clean_follows_clean",
                "swapped_follows_swapped",
                "swapped_keeps_clean",
                "swapped_target_margin_sum",
                "causal_margin_shift_sum",
            ):
                destination = {
                    "clean_follows_clean": "clean_affected_follows_clean",
                    "swapped_follows_swapped": "swapped_affected_follows_swapped",
                    "swapped_keeps_clean": "swapped_affected_keeps_clean",
                }.get(key, key)
                totals[destination] += float(affected[key])
            totals["affected_count"] += int(affected["affected_count"])
            totals["unchanged_max_diff"] = max(
                totals["unchanged_max_diff"],
                float((clean_logits - unchanged_logits).abs().max()),
            )
            totals["swap_disagreement"] += float(
                swapped_prediction.ne(clean_prediction).sum()
            )
            totals["prediction_count"] += clean_prediction.numel()
        rows.append(
            {
                "phase": phase,
                "prefix": _program_label(prefix) if prefix else "fresh",
                "clean_state_accuracy": _safe(
                    totals["clean_state_correct"], totals["state_count"]
                ),
                "swapped_state_follows_swapped": _safe(
                    totals["swapped_state_follows_swapped"], totals["state_count"]
                ),
                "swapped_state_keeps_clean": _safe(
                    totals["swapped_state_keeps_clean"], totals["state_count"]
                ),
                "shuffled_state_follows_shuffled": _safe(
                    totals["shuffled_state_follows_shuffled"], totals["state_count"]
                ),
                "clean_affected_follows_clean": _safe(
                    totals["clean_affected_follows_clean"], totals["affected_count"]
                ),
                "swapped_affected_follows_swapped": _safe(
                    totals["swapped_affected_follows_swapped"],
                    totals["affected_count"],
                ),
                "swapped_affected_keeps_clean": _safe(
                    totals["swapped_affected_keeps_clean"], totals["affected_count"]
                ),
                "swapped_target_logit_margin": _safe(
                    totals["swapped_target_margin_sum"], totals["affected_count"]
                ),
                "causal_logit_margin_shift": _safe(
                    totals["causal_margin_shift_sum"], totals["affected_count"]
                ),
                "swap_prediction_disagreement": _safe(
                    totals["swap_disagreement"], totals["prediction_count"]
                ),
                "unchanged_max_abs_logit_difference": totals["unchanged_max_diff"],
            }
        )
    return rows


def _plot_heatmap(
    path: Path,
    matrix: np.ndarray,
    *,
    row_labels: list[int],
    column_labels: list[int],
    title: str,
    colorbar_label: str,
    value_format: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor("white")
    image = ax.imshow(matrix, vmin=np.nanmin(matrix), vmax=np.nanmax(matrix), cmap="viridis")
    ax.set(xticks=range(len(column_labels)), xticklabels=column_labels)
    ax.set(yticks=range(len(row_labels)), yticklabels=row_labels)
    ax.set(xlabel="requested increment", ylabel="prefix cumulative depth", title=title)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            if np.isfinite(matrix[row, column]):
                ax.text(
                    column,
                    row,
                    format(matrix[row, column], value_format),
                    ha="center",
                    va="center",
                    color="white" if matrix[row, column] < np.nanmean(matrix) else "black",
                    fontsize=8,
                )
    fig.colorbar(image, ax=ax, label=colorbar_label)
    fig.tight_layout()
    fig.savefig(path, dpi=180, facecolor="white")
    fig.savefig(path.with_suffix(".svg"), facecolor="white")
    plt.close(fig)


def _plot_program_bars(path: Path, rows: list[dict[str, Any]], *, title: str) -> None:
    labels = [row["program"] for row in rows]
    values = [row["state_accuracy"] for row in rows]
    fig, ax = plt.subplots(figsize=(max(8, 0.55 * len(rows)), 4.5))
    fig.patch.set_facecolor("white")
    ax.bar(range(len(rows)), values, color="#0072B2")
    ax.set(
        xticks=range(len(rows)),
        xticklabels=labels,
        ylabel="final cumulative-state accuracy",
        ylim=(0, 1.03),
        title=title,
    )
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=180, facecolor="white")
    fig.savefig(path.with_suffix(".svg"), facecolor="white")
    plt.close(fig)


def _plot_interchange(path: Path, rows: list[dict[str, Any]]) -> None:
    phases = [row["phase"] for row in rows]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    fig.patch.set_facecolor("white")
    for key, label, color in (
        ("clean_affected_follows_clean", "clean follows clean", "#0072B2"),
        ("swapped_affected_follows_swapped", "swapped follows swapped", "#009E73"),
        ("swapped_affected_keeps_clean", "swapped keeps clean", "#D55E00"),
    ):
        ax.plot(phases, [row[key] for row in rows], marker="o", label=label, color=color)
    ax.set(
        xlabel="prefix cumulative depth",
        ylabel="accuracy on oracle-difference nodes",
        ylim=(-0.03, 1.03),
        title="Instruction interchange at a fixed recurrent state",
    )
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=180, facecolor="white")
    fig.savefig(path.with_suffix(".svg"), facecolor="white")
    plt.close(fig)


@torch.no_grad()
def evaluate_macrostep_checkpoint(
    *,
    checkpoint: Path,
    out_dir: Path,
    device: torch.device,
    batch_size: int = 510,
    batches: int = 8,
    phase_values: list[int] | None = None,
    include_overloop: bool = True,
    amp: bool = True,
) -> dict[str, Any]:
    if batch_size < 2 or batch_size % 2:
        raise ValueError("batch_size must be positive even")
    checkpoint_data = torch.load(checkpoint, map_location=device, weights_only=False)
    data_cfg = BooleanDAGConfig(**checkpoint_data["data_config"])
    model_cfg = BooleanDAGModelConfig(**checkpoint_data["model_config"])
    model = DepthConditionedBooleanDAGTransformer(
        data_cfg,
        model_cfg,
        use_instruction=checkpoint_data["use_instruction"],
    ).to(device)
    model.load_state_dict(checkpoint_data["model"])
    model.eval()
    set_seed(20_260_722)
    generator = torch.Generator(device=device).manual_seed(20_260_722)
    phases = phase_values if phase_values is not None else list(range(13))
    out_dir.mkdir(parents=True, exist_ok=True)

    phase_rows = []
    for phase in phases:
        for increment in range(1, 5):
            if phase + increment > data_cfg.eval_max_depth:
                continue
            program = canonical_prefix(phase) + (increment,)
            rows = _evaluate_program(
                model,
                data_cfg=data_cfg,
                program=program,
                device=device,
                batch_size=batch_size,
                batches=batches,
                amp=amp,
                generator=torch.Generator(device=device).manual_seed(PHASE_PAIR_SEED),
            )
            final = rows[-1]
            phase_rows.append(
                {
                    "phase": phase,
                    "increment": increment,
                    "target_depth": phase + increment,
                    "pair_seed": PHASE_PAIR_SEED,
                    "program": final["program"],
                    "state_accuracy": final["state_accuracy"],
                    "exact_state_accuracy": final["exact_state_accuracy"],
                    "frontier_accuracy": final["frontier_accuracy"],
                    "root_resolved_accuracy": final["root_resolved_accuracy"],
                    "root_unknown_accuracy": final["root_unknown_accuracy"],
                    "mean_realized_depth": final["mean_realized_depth"],
                    "realized_progress": final["mean_realized_depth"] - phase,
                }
            )
    _write_csv(out_dir / "phase_increment.csv", phase_rows)

    same_total_rows = []
    for total_depth in (4, 8):
        pair_seed = SAME_TOTAL_PAIR_SEED + total_depth
        reference_program = canonical_prefix(total_depth)
        reference_outputs = _final_outputs_by_batch(
            model,
            data_cfg=data_cfg,
            program=reference_program,
            device=device,
            batch_size=batch_size,
            batches=batches,
            amp=amp,
            seed=pair_seed,
        )
        for program in same_total_programs(total_depth):
            rows = _evaluate_program(
                model,
                data_cfg=data_cfg,
                program=program,
                device=device,
                batch_size=batch_size,
                batches=batches,
                amp=amp,
                generator=torch.Generator(device=device).manual_seed(pair_seed),
            )
            outputs = _final_outputs_by_batch(
                model,
                data_cfg=data_cfg,
                program=program,
                device=device,
                batch_size=batch_size,
                batches=batches,
                amp=amp,
                seed=pair_seed,
            )
            same_total_rows.append(
                {
                    "total_depth": total_depth,
                    "reference_program": _program_label(reference_program),
                    "pair_seed": pair_seed,
                    **_final_program_row(rows),
                    **_compare_final_outputs(outputs, reference_outputs),
                }
            )
    _write_csv(out_dir / "same_total.csv", same_total_rows)

    heldout_rows = []
    heldout_by_loop_rows = []
    for program in HELDOUT_PROGRAMS:
        if sum(program) <= data_cfg.eval_max_depth:
            program_rows = _evaluate_program(
                model,
                data_cfg=data_cfg,
                program=program,
                device=device,
                batch_size=batch_size,
                batches=batches,
                amp=amp,
                generator=generator,
            )
            heldout_by_loop_rows.extend(program_rows)
            heldout_rows.append(
                _final_program_row(
                    program_rows
                )
            )
    _write_csv(out_dir / "heldout_programs.csv", heldout_rows)
    _write_csv(out_dir / "heldout_programs_by_loop.csv", heldout_by_loop_rows)

    overloop_rows = []
    overloop_by_loop_rows = []
    if include_overloop:
        for program in OVERLOOP_PROGRAMS:
            if sum(program) <= data_cfg.eval_max_depth:
                program_rows = _evaluate_program(
                    model,
                    data_cfg=data_cfg,
                    program=program,
                    device=device,
                    batch_size=batch_size,
                    batches=batches,
                    amp=amp,
                    generator=generator,
                )
                overloop_by_loop_rows.extend(program_rows)
                overloop_rows.append(
                    _final_program_row(
                        program_rows
                    )
                )
    _write_csv(out_dir / "overloop_programs.csv", overloop_rows)
    _write_csv(out_dir / "overloop_programs_by_loop.csv", overloop_by_loop_rows)

    interchange_rows = _instruction_interchange(
        model,
        data_cfg=data_cfg,
        phases=phases,
        device=device,
        batch_size=batch_size,
        batches=batches,
        amp=amp,
        generator=generator,
    )
    _write_csv(out_dir / "instruction_interchange.csv", interchange_rows)

    phase_index = {phase: index for index, phase in enumerate(phases)}
    accuracy = np.full((len(phases), 4), np.nan)
    progress = np.full_like(accuracy, np.nan)
    for row in phase_rows:
        accuracy[phase_index[row["phase"]], row["increment"] - 1] = row[
            "state_accuracy"
        ]
        progress[phase_index[row["phase"]], row["increment"] - 1] = row[
            "realized_progress"
        ]
    _plot_heatmap(
        out_dir / "phase_increment_accuracy.png",
        accuracy,
        row_labels=phases,
        column_labels=[1, 2, 3, 4],
        title="Cumulative-state accuracy by phase and requested increment",
        colorbar_label="state accuracy",
        value_format=".2f",
    )
    _plot_heatmap(
        out_dir / "requested_vs_realized.png",
        progress,
        row_labels=phases,
        column_labels=[1, 2, 3, 4],
        title="Requested increment versus realized wavefront progress",
        colorbar_label="mean realized progress",
        value_format=".1f",
    )
    _plot_program_bars(
        out_dir / "same_total.png",
        same_total_rows,
        title="Same total depth under different macro-step decompositions",
    )
    _plot_interchange(out_dir / "instruction_interchange.png", interchange_rows)

    finite_phase = [row["state_accuracy"] for row in phase_rows]
    summary = {
        "task_version": checkpoint_data.get("task_version"),
        "checkpoint": str(checkpoint),
        "checkpoint_step": checkpoint_data.get("step"),
        "condition": checkpoint_data["condition"],
        "use_instruction": checkpoint_data["use_instruction"],
        "phase_values": phases,
        "batch_size": batch_size,
        "batches": batches,
        "phase_increment_mean_state_accuracy": float(np.mean(finite_phase)),
        "phase_increment_min_state_accuracy": float(np.min(finite_phase)),
        "heldout_mean_final_state_accuracy": float(
            np.mean([row["state_accuracy"] for row in heldout_rows])
        ),
        "heldout_minimum_state_accuracy": float(
            min(row["minimum_state_accuracy"] for row in heldout_rows)
        ),
        "overloop_mean_final_state_accuracy": (
            float(np.mean([row["state_accuracy"] for row in overloop_rows]))
            if overloop_rows
            else None
        ),
        "interchange_mean_swapped_follows_swapped": float(
            np.mean(
                [row["swapped_affected_follows_swapped"] for row in interchange_rows]
            )
        ),
        "interchange_mean_swapped_keeps_clean": float(
            np.mean([row["swapped_affected_keeps_clean"] for row in interchange_rows])
        ),
        "interchange_mean_causal_logit_margin_shift": float(
            np.mean([row["causal_logit_margin_shift"] for row in interchange_rows])
        ),
        "interchange_max_unchanged_logit_difference": max(
            row["unchanged_max_abs_logit_difference"] for row in interchange_rows
        ),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Boolean-DAG macro-step checkpoints.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=510)
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-overloop", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate_macrostep_checkpoint(
        checkpoint=args.checkpoint,
        out_dir=args.out_dir,
        device=pick_device(args.device),
        batch_size=args.batch_size,
        batches=args.batches,
        include_overloop=args.include_overloop,
        amp=args.amp,
    )


if __name__ == "__main__":
    main()
