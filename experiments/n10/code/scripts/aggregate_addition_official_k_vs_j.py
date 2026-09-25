from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt

from reasoning_loop.evaluate_addition_normal_j_horizon import (
    exact_mcnemar_pvalue,
)


VARIANTS = ("raw", "k_1x", "k_2x", "k_3x", "inter_loop_j")
BANDS = {
    "id_1_19": (1, 19),
    "repair_20_40": (20, 40),
    "unseen_41_60": (41, 60),
    "far_61_80": (61, 80),
    "extreme_81_100": (81, 100),
}
COLORS = {
    "raw": "#202020",
    "k_1x": "#F59E0B",
    "k_2x": "#8B5CF6",
    "k_3x": "#2563EB",
    "inter_loop_j": "#10B981",
}
DISPLAY = {
    "raw": "raw",
    "k_1x": "K-weight 1x",
    "k_2x": "K-weight 2x",
    "k_3x": "K-weight 3x",
    "inter_loop_j": "inter-loop J",
}


def _as_int(row: dict[str, Any], key: str) -> int:
    return int(row[key])


def band_comparison(
    *,
    rows: Sequence[dict[str, Any]],
    paired_rows: Sequence[dict[str, Any]],
    candidate: str,
    reference: str,
    low: int,
    high: int,
) -> dict[str, Any]:
    candidate_rows = [
        row
        for row in rows
        if row["variant"] == candidate
        and low <= int(row["logical_length"]) <= high
    ]
    reference_rows = [
        row
        for row in rows
        if row["variant"] == reference
        and low <= int(row["logical_length"]) <= high
    ]
    if not candidate_rows or len(candidate_rows) != len(reference_rows):
        raise ValueError("band comparison lacks matched variant rows")
    candidate_correct = sum(_as_int(row, "correct") for row in candidate_rows)
    candidate_examples = sum(_as_int(row, "examples") for row in candidate_rows)
    reference_correct = sum(_as_int(row, "correct") for row in reference_rows)
    reference_examples = sum(_as_int(row, "examples") for row in reference_rows)
    if candidate_examples != reference_examples:
        raise ValueError("band comparison uses unequal sample counts")
    relevant_pairs = [
        row
        for row in paired_rows
        if row["candidate"] == candidate
        and row["reference"] == reference
        and low <= int(row["logical_length"]) <= high
    ]
    if len(relevant_pairs) != len(candidate_rows):
        raise ValueError("band comparison lacks paired outcome rows")
    pooled = {
        key: sum(_as_int(row, key) for row in relevant_pairs)
        for key in ("candidate_only", "reference_only", "both", "neither")
    }
    candidate_accuracy = candidate_correct / candidate_examples
    reference_accuracy = reference_correct / reference_examples
    return {
        "low": low,
        "high": high,
        "lengths": len(candidate_rows),
        "examples_per_variant": candidate_examples,
        "candidate_exact_match": candidate_accuracy,
        "reference_exact_match": reference_accuracy,
        "gain": candidate_accuracy - reference_accuracy,
        **pooled,
        "exact_mcnemar_pvalue": exact_mcnemar_pvalue(
            pooled["candidate_only"], pooled["reference_only"]
        ),
    }


def operator_cost_counts(
    *, d_model: int, block_layers: int, rank: int
) -> dict[str, float | int]:
    k_cost = block_layers * d_model * d_model
    j_cost = d_model + 2 * d_model * rank
    return {
        "k_extra_multiplications_per_token_loop": k_cost,
        "j_extra_multiplications_per_token_loop": j_cost,
        "k_to_j_multiplication_ratio": k_cost / j_cost,
        "note": "Counts cover only the added linear operators, not the shared frozen executor, attention quadratic cost, additions, or backward pass.",
    }


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _load_complete(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload["status"] != "complete":
        raise ValueError(f"incomplete artifact: {path}")
    return payload


def _variant_rows(
    rows: Sequence[dict[str, Any]], variant: str
) -> list[dict[str, Any]]:
    return sorted(
        (row for row in rows if row["variant"] == variant),
        key=lambda row: int(row["logical_length"]),
    )


def _pooled_accuracy(
    rows: Sequence[dict[str, Any]], *, variant: str, low: int, high: int
) -> float:
    selected = [
        row
        for row in rows
        if row["variant"] == variant
        and low <= int(row["logical_length"]) <= high
    ]
    return sum(_as_int(row, "correct") for row in selected) / sum(
        _as_int(row, "examples") for row in selected
    )


def _sample_recurrence_exposures(
    training_paths: Sequence[Path], *, batch_size: int
) -> int:
    total = 0
    for path in training_paths:
        for row in _read_csv(path):
            total += batch_size * (int(row["length"]) + 1)
    return total


def aggregate(
    *, experiment_root: Path, one_x_dir: Path, j_summary_path: Path
) -> dict[str, Any]:
    train_dir = experiment_root / "official_seed0_k_total3x"
    dense_dir = experiment_root / "comparison" / "dense_l1to100_n128"
    anchor_dir = experiment_root / "comparison" / "anchors_n512"
    out_dir = experiment_root / "aggregate"
    out_dir.mkdir(parents=True, exist_ok=True)

    train = _load_complete(train_dir / "summary.json")
    one_x = _load_complete(one_x_dir / "summary.json")
    dense = _load_complete(dense_dir / "summary.json")
    anchors = _load_complete(anchor_dir / "summary.json")
    j_summary = _load_complete(j_summary_path)
    if train["training"]["cumulative_updates"] != 16128:
        raise ValueError("K continuation did not reach the cumulative 3x budget")
    if sum(1 for _ in (train_dir / "training.jsonl").open()) != 10752:
        raise ValueError("K continuation training log is truncated")
    if dense["lengths"] != list(range(1, 101)):
        raise ValueError("dense comparison is not L1--100 consecutive")
    if dense["examples_per_length_variant"] != 128:
        raise ValueError("dense comparison does not use n=128 per length")
    if anchors["examples_per_length_variant"] != 512:
        raise ValueError("anchor comparison does not use n=512 per length")

    dense_rows = _read_csv(dense_dir / "by_length.csv")
    dense_pairs = _read_csv(dense_dir / "paired_vs_j.csv")
    anchor_rows = _read_csv(anchor_dir / "by_length.csv")
    anchor_pairs = _read_csv(anchor_dir / "paired_vs_j.csv")
    for variant in VARIANTS:
        if len(_variant_rows(dense_rows, variant)) != 100:
            raise ValueError(f"dense curve is incomplete for {variant}")

    comparisons: dict[str, dict[str, Any]] = {}
    table_rows: list[dict[str, Any]] = []
    for candidate in ("k_1x", "k_2x", "k_3x"):
        comparisons[candidate] = {}
        for band, (low, high) in BANDS.items():
            result = band_comparison(
                rows=dense_rows,
                paired_rows=dense_pairs,
                candidate=candidate,
                reference="inter_loop_j",
                low=low,
                high=high,
            )
            comparisons[candidate][band] = result
            table_rows.append({"candidate": candidate, "band": band, **result})

    with (out_dir / "band_comparison_vs_j.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table_rows[0]))
        writer.writeheader()
        writer.writerows(table_rows)

    anchor_table: list[dict[str, Any]] = []
    for length in anchors["lengths"]:
        row: dict[str, Any] = {"logical_length": length}
        for variant in VARIANTS:
            selected = [
                item
                for item in anchor_rows
                if item["variant"] == variant
                and int(item["logical_length"]) == length
            ]
            if len(selected) != 1:
                raise ValueError(f"missing anchor row for {variant} L{length}")
            row[f"{variant}_exact_match"] = float(selected[0]["exact_match"])
        for candidate in ("k_1x", "k_2x", "k_3x"):
            selected_pair = [
                item
                for item in anchor_pairs
                if item["candidate"] == candidate
                and int(item["logical_length"]) == length
            ]
            if len(selected_pair) != 1:
                raise ValueError(f"missing paired anchor for {candidate} L{length}")
            row[f"{candidate}_vs_j_pvalue"] = float(
                selected_pair[0]["exact_mcnemar_pvalue"]
            )
        anchor_table.append(row)
    with (out_dir / "anchor_comparison_n512.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(anchor_table[0]))
        writer.writeheader()
        writer.writerows(anchor_table)

    anchor_comparisons: dict[str, dict[str, Any]] = {}
    anchor_band_rows: list[dict[str, Any]] = []
    for candidate in ("k_1x", "k_2x", "k_3x"):
        anchor_comparisons[candidate] = {}
        for band, (low, high) in BANDS.items():
            result = band_comparison(
                rows=anchor_rows,
                paired_rows=anchor_pairs,
                candidate=candidate,
                reference="inter_loop_j",
                low=low,
                high=high,
            )
            anchor_comparisons[candidate][band] = result
            anchor_band_rows.append(
                {"candidate": candidate, "band": band, **result}
            )
    with (out_dir / "anchor_band_comparison_vs_j.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(anchor_band_rows[0]))
        writer.writeheader()
        writer.writerows(anchor_band_rows)

    batch_size = int(one_x["training"]["batch_size"])
    k_one_x_exposure = _sample_recurrence_exposures(
        [one_x_dir / "training.csv"], batch_size=batch_size
    )
    k_three_x_exposure = _sample_recurrence_exposures(
        [one_x_dir / "training.csv", train_dir / "training.csv"],
        batch_size=batch_size,
    )
    continuation_rows = _read_csv(train_dir / "training.csv")
    k_two_x_exposure = k_one_x_exposure + sum(
        batch_size * (int(row["length"]) + 1)
        for row in continuation_rows[:5376]
    )
    j_counts = {
        int(length): int(count)
        for length, count in j_summary["controller_logical_length_example_counts"].items()
    }
    j_recurrence_exposure = sum(
        count * (length + 1) for length, count in j_counts.items()
    )
    k_parameter_count = int(one_x["adapter"]["trainable_parameters"])
    d_model = math.isqrt(k_parameter_count)
    if d_model * d_model != k_parameter_count:
        raise ValueError("K adapter parameter count is not a square matrix")
    rank = int(dense["variants"]["inter_loop_j"]["rank"])
    compute = {
        "k_1x": {
            "optimizer_updates": int(one_x["training"]["updates"]),
            "training_examples": int(one_x["training"]["updates"]) * batch_size,
            "sample_recurrence_exposures": k_one_x_exposure,
            "trainable_parameters": int(one_x["adapter"]["trainable_parameters"]),
            "elapsed_seconds": float(one_x["elapsed_seconds"]),
        },
        "k_2x_budget_checkpoint": {
            "optimizer_updates": 10752,
            "training_examples": 10752 * batch_size,
            "sample_recurrence_exposures": k_two_x_exposure,
            "trainable_parameters": int(train["adapter"]["trainable_parameters"]),
            "selection": "exact cumulative-budget checkpoint; not selected on OOD evaluation",
        },
        "k_3x_cumulative": {
            "optimizer_updates": int(train["training"]["cumulative_updates"]),
            "training_examples": int(train["training"]["cumulative_updates"])
            * batch_size,
            "sample_recurrence_exposures": k_three_x_exposure,
            "trainable_parameters": int(train["adapter"]["trainable_parameters"]),
            "elapsed_seconds_sum": float(one_x["elapsed_seconds"])
            + float(train["elapsed_seconds"]),
            "optimizer_state_restored": train["training"][
                "optimizer_state_restored"
            ],
        },
        "inter_loop_j": {
            "optimizer_updates": int(
                j_summary["training_budget"]["total_optimizer_updates"]
            ),
            "training_examples": int(
                j_summary["training_budget"]["total_training_examples"]
            ),
            "sample_recurrence_exposures": j_recurrence_exposure,
            "trainable_parameters": int(j_summary["parameter_count"]),
            "elapsed_seconds": None,
        },
        "added_operator_cost": operator_cost_counts(
            d_model=d_model,
            block_layers=int(j_summary["shared_physical_block_layers"]),
            rank=rank,
        ),
    }

    aggregate_payload = {
        "status": "complete",
        "audit": {
            "k_training_complete": True,
            "k_continuation_lines": 10752,
            "dense_lengths": [1, 100],
            "dense_examples_per_length_variant": 128,
            "anchor_examples_per_length_variant": 512,
            "paired_examples": True,
            "selection_used_ood_results": False,
        },
        "checkpoint": dense["checkpoint"],
        "checkpoint_step": dense["checkpoint_step"],
        "target_loop_rule": dense["target_loop_rule"],
        "selected_updates": {
            "k_1x": one_x["selection"]["selected"]["update"],
            "k_2x_budget": 10752,
            "k_3x_local": train["selection"]["selected"]["update"],
            "k_3x_cumulative": train["selection"]["selected"][
                "cumulative_update"
            ],
        },
        "horizons": {
            variant: dense["metrics"][variant]["contiguous_accuracy_horizons"]
            for variant in VARIANTS
        },
        "dense_band_accuracy": {
            variant: {
                band: _pooled_accuracy(
                    dense_rows, variant=variant, low=low, high=high
                )
                for band, (low, high) in BANDS.items()
            }
            for variant in VARIANTS
        },
        "paired_comparison_vs_j": comparisons,
        "anchor_paired_comparison_vs_j": anchor_comparisons,
        "compute": compute,
        "backbone_equivalence_note": {
            "k_backbone_step": 100000,
            "j_training_backbone_step": 100001,
            "maximum_state_dict_abs_difference": 8.881784197001252e-16,
            "different_tensors": 1,
            "total_tensors": 42,
            "interpretation": "functionally identical to numerical precision; paired evaluation uses the K step-100000 checkpoint for every variant",
        },
    }
    (out_dir / "summary.json").write_text(
        json.dumps(aggregate_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    fig, axes = plt.subplots(1, 2, figsize=(15.0, 5.6), constrained_layout=True)
    left, right = axes
    for variant in VARIANTS:
        rows = _variant_rows(dense_rows, variant)
        left.plot(
            [int(row["logical_length"]) for row in rows],
            [float(row["exact_match"]) for row in rows],
            label=DISPLAY[variant],
            color=COLORS[variant],
            linewidth=2.1 if variant in {"k_3x", "inter_loop_j"} else 1.5,
            linestyle=(
                ":"
                if variant == "raw"
                else "--"
                if variant == "k_1x"
                else "-."
                if variant == "k_2x"
                else "-"
            ),
        )
    for boundary in (19.5, 40.5):
        left.axvline(boundary, color="#777777", linewidth=0.8)
        right.axvline(boundary, color="#777777", linewidth=0.8)
    left.axvspan(19.5, 40.5, color="#FDE68A", alpha=0.20)
    left.axvspan(40.5, 100.5, color="#BFDBFE", alpha=0.16)
    left.set_title("Paired dense horizon (n=128 per length)")
    left.set_xlabel("Logical input length")
    left.set_ylabel("Strict exact match")
    left.set_ylim(-0.03, 1.04)
    left.set_xlim(1, 100)
    left.grid(alpha=0.22)
    left.legend(fontsize=9)

    for candidate in ("k_1x", "k_2x", "k_3x"):
        candidate_rows = _variant_rows(dense_rows, candidate)
        reference_rows = _variant_rows(dense_rows, "inter_loop_j")
        lengths = [int(row["logical_length"]) for row in candidate_rows]
        gains = [
            float(crow["exact_match"]) - float(rrow["exact_match"])
            for crow, rrow in zip(candidate_rows, reference_rows, strict=True)
        ]
        right.plot(
            lengths,
            gains,
            color=COLORS[candidate],
            label=f"{DISPLAY[candidate]} minus J",
            linewidth=2.0,
            linestyle=(
                "--"
                if candidate == "k_1x"
                else "-."
                if candidate == "k_2x"
                else "-"
            ),
        )
        right.scatter(
            [int(row["logical_length"]) for row in anchor_table],
            [
                float(row[f"{candidate}_exact_match"])
                - float(row["inter_loop_j_exact_match"])
                for row in anchor_table
            ],
            color=COLORS[candidate],
            edgecolor="white",
            linewidth=0.5,
            s=30,
            zorder=3,
        )
    right.axhline(0.0, color="#202020", linewidth=1.0)
    right.set_title("K advantage over inter-loop J (anchors n=512)")
    right.set_xlabel("Logical input length")
    right.set_ylabel("Exact-match difference")
    right.set_xlim(1, 100)
    right.set_ylim(-1.02, 1.02)
    right.grid(alpha=0.22)
    right.legend(fontsize=9)
    fig.suptitle("Official Addition: internal K-weight adaptation vs canonical inter-loop J")
    fig.savefig(out_dir / "k_vs_j_generalization.png", dpi=180)
    plt.close(fig)
    return aggregate_payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_root", type=Path)
    parser.add_argument("one_x_dir", type=Path)
    parser.add_argument("j_summary", type=Path)
    args = parser.parse_args()
    aggregate(
        experiment_root=args.experiment_root.expanduser().resolve(),
        one_x_dir=args.one_x_dir.expanduser().resolve(),
        j_summary_path=args.j_summary.expanduser().resolve(),
    )


if __name__ == "__main__":
    main()
