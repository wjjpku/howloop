"""Probe whether J-controlled states retain a graph-independent logical-age code."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from reasoning_loop.graph_path_depth_circuit import fixed_depth_batch, load_checkpoint
from reasoning_loop.graph_path_loop import pick_device, set_seed
from reasoning_loop.graph_path_telomere_overloop import advance_nodes
from reasoning_loop.train_graph_path_age_specific_j_bank import AgeSpecificJBank
from reasoning_loop.train_graph_path_j_path_equivalence import sample_equivalent_word_pair


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--bank-artifacts", type=Path, nargs="+", required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--natural-age-probe", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=829301)
    parser.add_argument("--examples", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--cuda-memory-fraction", type=float, default=0.02)
    return parser.parse_args()


def load_bank(path: Path, device: torch.device, dimension: int) -> AgeSpecificJBank:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    bank = AgeSpecificJBank(
        dimension=dimension,
        rank=int(payload["rank"]),
        stage_rank=int(payload.get("stage_rank", payload["rank"])),
        map_architecture=str(payload["map_architecture"]),
    ).to(device)
    bank.load_state_dict(payload["state_dict"])
    return bank.frozen()


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


def fit_probe(features: np.ndarray, ages: np.ndarray, ridge: float) -> tuple[np.ndarray, float]:
    mean = features.mean(0)
    centered = features - mean
    gram = centered.T @ centered
    scale = float(np.trace(gram) / gram.shape[0])
    weight = np.linalg.solve(
        gram + np.eye(gram.shape[0]) * ridge * max(scale, 1e-12),
        centered.T @ ages,
    )
    bias = float(ages.mean() - mean @ weight)
    return weight, bias


def metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    residual = prediction - target
    return {
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "r2": float(1 - np.sum(residual**2) / np.sum((target - target.mean()) ** 2)),
        "rounded_accuracy": float(
            np.mean(np.clip(np.rint(prediction), 1, 8).astype(int) == target.astype(int))
        ),
        "prediction_mean": float(np.mean(prediction)),
        "target_mean": float(np.mean(target)),
    }


@torch.no_grad()
def collect(
    *, model, cfg, banks: dict[str, AgeSpecificJBank], examples: int,
    batch_size: int, device: torch.device, seed: int,
) -> tuple[
    dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    tuple[np.ndarray, np.ndarray, np.ndarray],
]:
    rng = np.random.default_rng(seed + 17)
    schedules = []
    for back_count in (2, 5, 8, 12):
        for mandatory_source_age in (2, 4, 6, 8):
            left, _ = sample_equivalent_word_pair(
                rng=rng,
                back_count=back_count,
                mandatory_source_age=mandatory_source_age,
            )
            schedules.append(left)
    controlled_x: dict[str, list[np.ndarray]] = {label: [] for label in banks}
    controlled_y: dict[str, list[np.ndarray]] = {label: [] for label in banks}
    controlled_split: dict[str, list[np.ndarray]] = {label: [] for label in banks}
    controlled_path_split: dict[str, list[np.ndarray]] = {
        label: [] for label in banks
    }
    natural_x: list[np.ndarray] = []
    natural_y: list[np.ndarray] = []
    natural_split: list[np.ndarray] = []
    for batch_index in range(examples // batch_size):
        tokens, _, successors, start = fixed_depth_batch(
            cfg, batch_size, device, path_positions=cfg.max_depth
        )
        raw = model.token_embed(tokens) + model.pos_embed.unsqueeze(0)
        state = raw
        current = start
        for age in range(1, 9):
            state = model.apply_loop(state, loop_index=age - 1)
            current = advance_nodes(successors, current, steps=1)
            natural_x.append(state[:, -1].float().cpu().numpy())
            natural_y.append(np.full(batch_size, age, dtype=np.float64))
            natural_split.append(np.full(batch_size, batch_index % 2, dtype=np.int64))
        h1 = model.apply_loop(raw, loop_index=0)
        positions = tuple(range(cfg.seq_len))
        for schedule_index, actions in enumerate(schedules):
            for label, bank in banks.items():
                state = h1.clone()
                logical_age = 1
                for action in actions:
                    if action == 1:
                        state = model.apply_loop(state, loop_index=logical_age)
                        logical_age += 1
                    else:
                        state = bank.rollback(
                            state, source_age=logical_age, positions=positions
                        )
                        logical_age -= 1
                        controlled_x[label].append(
                            state[:, -1].float().cpu().numpy()
                        )
                        controlled_y[label].append(
                            np.full(batch_size, logical_age, dtype=np.float64)
                        )
                        controlled_split[label].append(
                            np.full(batch_size, batch_index % 2, dtype=np.int64)
                        )
                        controlled_path_split[label].append(
                            np.full(batch_size, schedule_index % 2, dtype=np.int64)
                        )
    controlled = {
        label: (
            np.concatenate(controlled_x[label]),
            np.concatenate(controlled_y[label]),
            np.concatenate(controlled_split[label]),
            np.concatenate(controlled_path_split[label]),
        )
        for label in banks
    }
    natural = (
        np.concatenate(natural_x),
        np.concatenate(natural_y),
        np.concatenate(natural_split),
    )
    return controlled, natural


def main() -> None:
    args = parse_args()
    if len(args.bank_artifacts) != len(args.labels):
        raise ValueError("bank-artifacts and labels must have equal lengths")
    if args.examples % args.batch_size:
        raise ValueError("examples must be divisible by batch-size")
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
    banks = {
        label: load_bank(path, device, cfg.d_model)
        for label, path in zip(args.labels, args.bank_artifacts, strict=True)
    }
    controlled, natural = collect(
        model=model,
        cfg=cfg,
        banks=banks,
        examples=args.examples,
        batch_size=args.batch_size,
        device=device,
        seed=args.seed,
    )
    natural_payload = np.load(args.natural_age_probe)
    frozen_weight = natural_payload["baseline_weight"].reshape(-1)
    frozen_bias = float(np.asarray(natural_payload["baseline_bias"]).reshape(-1)[0])
    rows: list[dict[str, Any]] = []

    nx, ny, ns = natural
    natural_weight, natural_bias = fit_probe(nx[ns == 0], ny[ns == 0], args.ridge)
    for split_name, mask in (("train", ns == 0), ("held_out_graphs", ns == 1)):
        rows.append(
            {
                "probe": "refit_natural",
                "train_domain": "natural",
                "test_bank": "natural",
                "test_domain": split_name,
                "observations": int(mask.sum()),
                **metrics(nx[mask] @ natural_weight + natural_bias, ny[mask]),
            }
        )
    learned: dict[str, tuple[np.ndarray, float]] = {}
    for label, (x, y, split, path_split) in controlled.items():
        learned[label] = fit_probe(x[split == 0], y[split == 0], args.ridge)
        test = split == 1
        rows.append(
            {
                "probe": "frozen_natural",
                "train_domain": "natural_external",
                "test_bank": label,
                "test_domain": "post_J_held_out_graphs",
                "observations": int(test.sum()),
                **metrics(x[test] @ frozen_weight + frozen_bias, y[test]),
            }
        )
    for train_label, (weight, bias) in learned.items():
        for test_label, (x, y, split, path_split) in controlled.items():
            test = split == 1
            rows.append(
                {
                    "probe": f"controlled_{train_label}",
                    "train_domain": f"post_J_{train_label}",
                    "test_bank": test_label,
                    "test_domain": "post_J_held_out_graphs",
                    "observations": int(test.sum()),
                    **metrics(x[test] @ weight + bias, y[test]),
                }
            )
    joint_x = np.concatenate(
        [x[split == 0] for x, y, split, path_split in controlled.values()]
    )
    joint_y = np.concatenate(
        [y[split == 0] for x, y, split, path_split in controlled.values()]
    )
    joint_weight, joint_bias = fit_probe(joint_x, joint_y, args.ridge)
    for test_label, (x, y, split, path_split) in controlled.items():
        test = split == 1
        rows.append(
            {
                "probe": "controlled_joint_banks",
                "train_domain": "post_J_union_of_all_banks",
                "test_bank": test_label,
                "test_domain": "post_J_held_out_graphs",
                "observations": int(test.sum()),
                **metrics(x[test] @ joint_weight + joint_bias, y[test]),
            }
        )
    universal_domains = [(nx[ns == 0], ny[ns == 0])] + [
        (x[split == 0], y[split == 0])
        for x, y, split, path_split in controlled.values()
    ]
    balanced_count = min(len(domain_x) for domain_x, domain_y in universal_domains)
    universal_x = []
    universal_y = []
    for domain_x, domain_y in universal_domains:
        indices = np.linspace(0, len(domain_x) - 1, balanced_count, dtype=int)
        universal_x.append(domain_x[indices])
        universal_y.append(domain_y[indices])
    universal_weight, universal_bias = fit_probe(
        np.concatenate(universal_x), np.concatenate(universal_y), args.ridge
    )
    natural_test = ns == 1
    rows.append(
        {
            "probe": "universal_natural_and_banks",
            "train_domain": "balanced_union_natural_and_all_post_J_banks",
            "test_bank": "natural",
            "test_domain": "held_out_graphs",
            "observations": int(natural_test.sum()),
            **metrics(
                nx[natural_test] @ universal_weight + universal_bias,
                ny[natural_test],
            ),
        }
    )
    for test_label, (x, y, split, path_split) in controlled.items():
        test = split == 1
        rows.append(
            {
                "probe": "universal_natural_and_banks",
                "train_domain": "balanced_union_natural_and_all_post_J_banks",
                "test_bank": test_label,
                "test_domain": "post_J_held_out_graphs",
                "observations": int(test.sum()),
                **metrics(x[test] @ universal_weight + universal_bias, y[test]),
            }
        )
    path_learned: dict[str, tuple[np.ndarray, float]] = {}
    for label, (x, y, split, path_split) in controlled.items():
        train = (split == 0) & (path_split == 0)
        path_learned[label] = fit_probe(x[train], y[train], args.ridge)
    for train_label, (weight, bias) in path_learned.items():
        for test_label, (x, y, split, path_split) in controlled.items():
            test = (split == 1) & (path_split == 1)
            rows.append(
                {
                    "probe": f"controlled_unseen_paths_{train_label}",
                    "train_domain": f"post_J_{train_label}_path_half_A_graph_half_A",
                    "test_bank": test_label,
                    "test_domain": "post_J_unseen_action_words_and_graphs",
                    "observations": int(test.sum()),
                    **metrics(x[test] @ weight + bias, y[test]),
                }
            )
    joint_path_x = np.concatenate(
        [
            x[(split == 0) & (path_split == 0)]
            for x, y, split, path_split in controlled.values()
        ]
    )
    joint_path_y = np.concatenate(
        [
            y[(split == 0) & (path_split == 0)]
            for x, y, split, path_split in controlled.values()
        ]
    )
    joint_path_weight, joint_path_bias = fit_probe(
        joint_path_x, joint_path_y, args.ridge
    )
    for test_label, (x, y, split, path_split) in controlled.items():
        test = (split == 1) & (path_split == 1)
        rows.append(
            {
                "probe": "controlled_joint_banks_unseen_paths",
                "train_domain": "post_J_union_banks_path_half_A_graph_half_A",
                "test_bank": test_label,
                "test_domain": "post_J_unseen_action_words_and_graphs",
                "observations": int(test.sum()),
                **metrics(x[test] @ joint_path_weight + joint_path_bias, y[test]),
            }
        )
    write_csv(args.out_dir / "functional_age_probe.csv", rows)
    figure, axis = plt.subplots(figsize=(10, 5.5), dpi=180)
    labels = [f"{row['probe']}->{row['test_bank']}" for row in rows]
    axis.bar(np.arange(len(rows)), [row["rounded_accuracy"] for row in rows])
    axis.set_xticks(np.arange(len(rows)), labels, rotation=45, ha="right", fontsize=7)
    axis.set_ylim(0, 1.03)
    axis.set_ylabel("rounded age accuracy")
    axis.set_title("Natural and controller-specific linear age readouts")
    axis.grid(axis="y", alpha=0.22)
    figure.tight_layout()
    figure.savefig(args.out_dir / "functional_age_probe.png", bbox_inches="tight")
    plt.close(figure)
    summary = {
        "status": "complete",
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": checkpoint_payload.get("step"),
        "examples": args.examples,
        "split": (
            "graph-only rows use disjoint random graph batches; unseen-path rows also "
            "hold out half of the fixed k=2,5,8,12 action words"
        ),
        "features": "answer-token residual only",
        "controlled_domain": "states immediately after J on fixed unseen k=2,5,8,12 paths",
        "rows": rows,
        "claim_boundary": (
            "A successful probe establishes linearly decodable logical phase, not that "
            "the full hidden state lies on the natural H1..H8 manifold or that age is one-dimensional."
        ),
        "peak_cuda_allocated_mib": (
            torch.cuda.max_memory_allocated() / 1024**2 if device.type == "cuda" else None
        ),
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
