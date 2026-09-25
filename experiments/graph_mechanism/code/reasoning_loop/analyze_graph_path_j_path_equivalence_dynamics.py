"""Compare hidden dynamics of J banks on matched H1-start action schedules."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from reasoning_loop.graph_path_depth_circuit import fixed_depth_batch, load_checkpoint
from reasoning_loop.graph_path_loop import pick_device, set_seed
from reasoning_loop.graph_path_telomere_overloop import advance_nodes
from reasoning_loop.graph_path_telomere_simple_one_step import _aligned_state_at_age
from reasoning_loop.graph_path_temporal_intervention import logits_from_raw_state
from reasoning_loop.train_graph_path_age_specific_j_bank import AgeSpecificJBank
from reasoning_loop.train_graph_path_j_path_equivalence import (
    action_semantics,
    sample_equivalent_word_pair,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--phase-summary", type=Path, required=True)
    parser.add_argument("--bank-artifacts", type=Path, nargs="+", required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--age-probe", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=829001)
    parser.add_argument("--examples", type=int, default=768)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--cuda-memory-fraction", type=float, default=0.02)
    return parser.parse_args()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _load_bank(path: Path, device: torch.device) -> AgeSpecificJBank:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    bank = AgeSpecificJBank(
        dimension=256,
        rank=int(payload["rank"]),
        stage_rank=int(payload.get("stage_rank", payload["rank"])),
        map_architecture=str(payload["map_architecture"]),
    ).to(device)
    bank.load_state_dict(payload["state_dict"])
    return bank.frozen()


@torch.no_grad()
def _metrics(
    *,
    model,
    cfg,
    state: torch.Tensor,
    current: torch.Tensor,
    successors: torch.Tensor,
    symbolic_age: int,
    phase_positions: list[int],
    probe_weight: torch.Tensor,
    probe_bias: torch.Tensor,
) -> dict[str, float]:
    exact = _aligned_state_at_age(
        model=model,
        cfg=cfg,
        successors=successors,
        current=current,
        age=symbolic_age,
        phase_position=phase_positions[symbolic_age],
    )
    delta = state.float() - exact.float()
    predicted_age = state[:, -1].float() @ probe_weight + probe_bias
    continuation = state
    continuation_current = current
    for logical_age in range(symbolic_age, 8):
        continuation = model.apply_loop(continuation, loop_index=logical_age)
        continuation_current = advance_nodes(
            successors, continuation_current, steps=1
        )
    logits = logits_from_raw_state(model, continuation)
    return {
        "state_rms": float(state.float().square().mean().sqrt()),
        "answer_norm": float(state[:, -1].float().norm(dim=-1).mean()),
        "exact_state_relative_l2": float(
            delta.norm() / exact.float().norm().clamp_min(1e-12)
        ),
        "probe_age_mean": float(predicted_age.mean()),
        "probe_age_mae": float((predicted_age - symbolic_age).abs().mean()),
        "probe_rounded_accuracy": float(
            predicted_age.round().clamp(1, 8).eq(symbolic_age).float().mean()
        ),
        "continuation_to_H8_accuracy": float(
            logits.argmax(-1).eq(continuation_current).float().mean()
        ),
    }


def _schedule_specs() -> dict[str, tuple[int, ...]]:
    rng = np.random.default_rng(829117)
    k8_left, _ = sample_equivalent_word_pair(
        rng=rng, back_count=8, mandatory_source_age=8
    )
    k12_left, _ = sample_equivalent_word_pair(
        rng=rng, back_count=12, mandatory_source_age=8
    )
    return {
        "natural_F7": (1,) * 7,
        "F7J7": (1,) * 7 + (-1,) * 7,
        "FFJ_x6": (1, 1, -1) * 6,
        "FFFFJJJJ_x2": (1, 1, 1, 1, -1, -1, -1, -1) * 2,
        "equivalent_k8": k8_left,
        "equivalent_k12": k12_left,
    }


def _summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                row["bank"],
                row["schedule"],
                row["action_index"],
                row["action"],
                row["symbolic_age"],
            )
        ].append(row)
    metrics = (
        "state_rms",
        "answer_norm",
        "exact_state_relative_l2",
        "probe_age_mean",
        "probe_age_mae",
        "probe_rounded_accuracy",
        "continuation_to_H8_accuracy",
    )
    summary: list[dict[str, Any]] = []
    for key, values in grouped.items():
        bank, schedule, action_index, action, symbolic_age = key
        item: dict[str, Any] = {
            "bank": bank,
            "schedule": schedule,
            "action_index": action_index,
            "action": action,
            "symbolic_age": symbolic_age,
            "observations": len(values),
        }
        for metric in metrics:
            array = np.asarray([float(value[metric]) for value in values])
            item[f"{metric}_mean"] = float(array.mean())
            item[f"{metric}_sem"] = float(
                array.std(ddof=1) / np.sqrt(len(array))
                if len(array) > 1
                else 0.0
            )
        summary.append(item)
    return sorted(summary, key=lambda row: (row["schedule"], row["bank"], row["action_index"]))


def _plot(summary: list[dict[str, Any]], out_dir: Path) -> None:
    metrics = (
        ("continuation_to_H8_accuracy_mean", "continuation ACC", (0.0, 1.03)),
        ("probe_age_mean_mean", "natural-state age probe", None),
        ("state_rms_mean", "hidden RMS", None),
        ("exact_state_relative_l2_mean", "relative distance to natural H_age", None),
    )
    for schedule in sorted({row["schedule"] for row in summary}):
        figure, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=180)
        selected = [row for row in summary if row["schedule"] == schedule]
        for axis, (metric, title, ylim) in zip(axes.flat, metrics, strict=True):
            for bank in sorted({row["bank"] for row in selected}):
                rows = [row for row in selected if row["bank"] == bank]
                axis.plot(
                    [row["action_index"] for row in rows],
                    [row[metric] for row in rows],
                    marker="o",
                    markersize=2.5,
                    linewidth=1.2,
                    label=bank,
                )
            axis.set_title(title)
            axis.set_xlabel("action index")
            axis.grid(alpha=0.22)
            if ylim is not None:
                axis.set_ylim(*ylim)
        axes[0, 1].plot(
            [row["action_index"] for row in selected if row["bank"] == selected[0]["bank"]],
            [row["symbolic_age"] for row in selected if row["bank"] == selected[0]["bank"]],
            color="black",
            linestyle="--",
            linewidth=0.9,
            label="symbolic age",
        )
        handles, labels = axes[0, 0].get_legend_handles_labels()
        figure.legend(handles, labels, loc="lower center", ncol=max(1, len(labels)))
        figure.suptitle(schedule)
        figure.tight_layout(rect=(0, 0.06, 1, 0.97))
        figure.savefig(out_dir / f"dynamics_{schedule}.png", bbox_inches="tight")
        plt.close(figure)


@torch.no_grad()
def main(args: argparse.Namespace) -> None:
    if len(args.bank_artifacts) != len(args.labels):
        raise ValueError("bank artifacts and labels must have equal lengths")
    if args.examples % args.batch_size:
        raise ValueError("examples must be divisible by batch size")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(args.cuda_memory_fraction)
        torch.cuda.reset_peak_memory_stats()
    set_seed(args.seed)
    model, cfg, checkpoint_payload = load_checkpoint(args.checkpoint, device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    phase_payload = json.loads(args.phase_summary.read_text(encoding="utf-8"))
    phase_positions = [
        int(value)
        for value in phase_payload["trajectory_positions_including_initial"]
    ]
    probe_payload = np.load(args.age_probe)
    probe_weight = torch.as_tensor(
        probe_payload["baseline_weight"], device=device, dtype=torch.float32
    )
    probe_bias = torch.as_tensor(
        probe_payload["baseline_bias"], device=device, dtype=torch.float32
    )
    banks = {
        label: _load_bank(path, device)
        for label, path in zip(args.labels, args.bank_artifacts, strict=True)
    }
    positions = tuple(range(cfg.seq_len))
    schedules = _schedule_specs()
    rows: list[dict[str, Any]] = []
    for batch_index in range(args.examples // args.batch_size):
        tokens, _, successors, start = fixed_depth_batch(
            cfg, args.batch_size, device, path_positions=cfg.max_depth
        )
        raw = model.token_embed(tokens) + model.pos_embed.unsqueeze(0)
        h1 = model.apply_loop(raw, loop_index=0)
        h1_current = advance_nodes(successors, start, steps=1)
        for label, bank in banks.items():
            for schedule, actions in schedules.items():
                state = h1.clone()
                current = h1_current.clone()
                symbolic_age = 1
                rows.append(
                    {
                        "bank": label,
                        "schedule": schedule,
                        "batch": batch_index,
                        "action_index": 0,
                        "action": "initial_H1",
                        "source_age": None,
                        "symbolic_age": symbolic_age,
                        **_metrics(
                            model=model,
                            cfg=cfg,
                            state=state,
                            current=current,
                            successors=successors,
                            symbolic_age=symbolic_age,
                            phase_positions=phase_positions,
                            probe_weight=probe_weight,
                            probe_bias=probe_bias,
                        ),
                    }
                )
                for action_index, action in enumerate(actions, start=1):
                    source_age: int | None = None
                    if action == 1:
                        state = model.apply_loop(state, loop_index=symbolic_age)
                        current = advance_nodes(successors, current, steps=1)
                        symbolic_age += 1
                    else:
                        source_age = symbolic_age
                        state = bank.rollback(
                            state,
                            source_age=source_age,
                            positions=positions,
                        )
                        symbolic_age -= 1
                    rows.append(
                        {
                            "bank": label,
                            "schedule": schedule,
                            "batch": batch_index,
                            "action_index": action_index,
                            "action": "F" if action == 1 else "J",
                            "source_age": source_age,
                            "symbolic_age": symbolic_age,
                            **_metrics(
                                model=model,
                                cfg=cfg,
                                state=state,
                                current=current,
                                successors=successors,
                                symbolic_age=symbolic_age,
                                phase_positions=phase_positions,
                                probe_weight=probe_weight,
                                probe_bias=probe_bias,
                            ),
                        }
                    )
    summary = _summarize(rows)
    _write_csv(args.out_dir / "dynamics_per_batch.csv", rows)
    _write_csv(args.out_dir / "dynamics_summary.csv", summary)
    _plot(summary, args.out_dir)
    result = {
        "status": "complete",
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": checkpoint_payload.get("step"),
        "banks": {
            label: str(path)
            for label, path in zip(args.labels, args.bank_artifacts, strict=True)
        },
        "schedules": {
            name: {
                "actions": "".join("F" if action == 1 else "J" for action in actions),
                "semantics": action_semantics(actions).__dict__,
            }
            for name, actions in schedules.items()
        },
        "examples": args.examples,
        "peak_cuda_allocated_mib": (
            torch.cuda.max_memory_allocated() / 1024**2
            if device.type == "cuda"
            else None
        ),
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main(parse_args())
