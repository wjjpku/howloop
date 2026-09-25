from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _float(value: str | int | float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _median(values: list[float]) -> float:
    usable = [value for value in values if math.isfinite(value)]
    return statistics.median(usable) if usable else float("nan")


def _parse_family(name: str) -> tuple[str, int]:
    family, raw_seed = name.rsplit("_seed", 1)
    return family, int(raw_seed)


def _ids(value: str) -> set[int]:
    return {int(item) for item in value.split()} if value.strip() else set()


def _jaccard(left: set[int], right: set[int]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def _distance_medians(
    rows: list[dict[str, str]],
    field: str,
) -> tuple[float, float, float]:
    adjacent = [
        _float(row[field]) for row in rows if int(row["loop_distance"]) == 1
    ]
    middle = [
        _float(row[field])
        for row in rows
        if 1 < int(row["loop_distance"]) < 3
    ]
    far = [
        _float(row[field]) for row in rows if int(row["loop_distance"]) >= 3
    ]
    return _median(adjacent), _median(middle), _median(far)


def _task_specific_overlap(
    rows: list[dict[str, str]],
    *,
    block: int,
) -> tuple[float, float, int]:
    selected = sorted(
        (
            row
            for row in rows
            if row["condition"] == "top64" and int(row["block"]) == block
        ),
        key=lambda row: int(row["loop"]),
    )
    pairs: list[tuple[int, float]] = []
    for left_index, left in enumerate(selected):
        for right in selected[left_index + 1 :]:
            distance = int(right["loop"]) - int(left["loop"])
            pairs.append(
                (
                    distance,
                    _jaccard(
                        _ids(left["top_neuron_ids"]),
                        _ids(right["top_neuron_ids"]),
                    ),
                )
            )
    recoveries: dict[int, dict[str, float]] = defaultdict(dict)
    for row in rows:
        if int(row["block"]) == block and row["condition"] in {
            "top64",
            "random64",
        }:
            recoveries[int(row["loop"])][row["condition"]] = _float(
                row["recovery"]
            )
    positive_count = sum(
        values.get("top64", float("nan"))
        > values.get("random64", float("nan"))
        for values in recoveries.values()
    )
    return (
        _median([value for distance, value in pairs if distance == 1]),
        _median([value for distance, value in pairs if distance >= 3]),
        positive_count,
    )


def summarize_run(
    run_dir: Path,
    *,
    task_specific_rows: list[dict[str, str]] | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summary = json.loads((run_dir / "summary.json").read_text())
    name = summary["name"]
    family, seed = _parse_family(name)
    heads = _read_csv(run_dir / "head_loop_rows.csv")
    head_position_path = run_dir / "head_position_loop_rows.csv"
    head_positions = (
        _read_csv(head_position_path)
        if head_position_path.exists()
        else []
    )
    head_pairs = _read_csv(run_dir / "head_loop_pair_rows.csv")
    neuron = _read_csv(run_dir / "neuron_loop_rows.csv")
    neuron_pairs = _read_csv(run_dir / "neuron_loop_pair_rows.csv")
    transfer = _read_csv(run_dir / "neuron_transfer_rows.csv")
    detail_rows: list[dict[str, Any]] = []
    run_row: dict[str, Any] = {
        "run": name,
        "family": family,
        "seed": seed,
        "checkpoint_step": summary.get("checkpoint_step"),
        "trained_loops": summary["config"]["max_loops"],
        "baseline_accuracy": summary["baseline"]["endpoint_accuracy"],
        "baseline_margin": summary["baseline"]["endpoint_margin"],
    }
    for block in (1, 2):
        block_heads = [row for row in heads if int(row["block"]) == block]
        block_head_positions = [
            row
            for row in head_positions
            if int(row["block"]) == block
        ]
        by_loop: dict[int, list[dict[str, str]]] = defaultdict(list)
        for row in block_heads:
            by_loop[int(row["loop"])].append(row)
        dominant: list[int] = []
        activation_dominant: list[int] = []
        strong_sequence: list[str] = []
        strong_counts: list[int] = []
        for loop in sorted(by_loop):
            selected = by_loop[loop]
            dominant.append(
                int(
                    max(
                        selected,
                        key=lambda row: min(
                            _float(row["zero_margin_drop"]),
                            _float(row["shuffle_margin_drop"]),
                        ),
                    )["head"]
                )
            )
            activation_dominant.append(
                int(
                    max(
                        selected,
                        key=lambda row: _float(row["activation_rms"]),
                    )["head"]
                )
            )
            strong_sequence.append(
                ",".join(
                    f"H{row['head']}"
                    for row in selected
                    if int(row["strongly_used"])
                )
                or "-"
            )
            strong_counts.append(
                sum(int(row["strongly_used"]) for row in selected)
            )
        position_causal_heads: list[int] = []
        position_causal_scopes: list[str] = []
        position_causal_scores: list[float] = []
        position_strong_sequence: list[str] = []
        local_causal_heads: list[int] = []
        local_causal_scopes: list[str] = []
        local_causal_scores: list[float] = []
        local_strong_sequence: list[str] = []
        for loop in sorted(by_loop):
            selected_positions = [
                row
                for row in block_head_positions
                if int(row["loop"]) == loop
            ]
            if not selected_positions:
                continue
            strongest_position = max(
                selected_positions,
                key=lambda row: min(
                    _float(row["zero_margin_drop"]),
                    _float(row["shuffle_margin_drop"]),
                ),
            )
            position_causal_heads.append(int(strongest_position["head"]))
            position_causal_scopes.append(
                strongest_position["position_scope"]
            )
            position_causal_scores.append(
                min(
                    _float(strongest_position["zero_margin_drop"]),
                    _float(strongest_position["shuffle_margin_drop"]),
                )
            )
            position_strong_sequence.append(
                ",".join(
                    f"H{row['head']}@{row['position_scope']}"
                    for row in selected_positions
                    if int(row["strongly_used"])
                )
                or "-"
            )
            if "zero_local_path_accuracy_drop" in strongest_position:
                strongest_local = max(
                    selected_positions,
                    key=lambda row: min(
                        _float(row["zero_local_path_accuracy_drop"]),
                        _float(
                            row["shuffle_local_path_accuracy_drop"]
                        ),
                    ),
                )
                local_causal_heads.append(int(strongest_local["head"]))
                local_causal_scopes.append(
                    strongest_local["position_scope"]
                )
                local_causal_scores.append(
                    min(
                        _float(
                            strongest_local[
                                "zero_local_path_accuracy_drop"
                            ]
                        ),
                        _float(
                            strongest_local[
                                "shuffle_local_path_accuracy_drop"
                            ]
                        ),
                    )
                )
                local_strong_sequence.append(
                    ",".join(
                        f"H{row['head']}@{row['position_scope']}"
                        for row in selected_positions
                        if int(row["strongly_locally_used"])
                    )
                    or "-"
                )
        block_head_pairs = [
            row for row in head_pairs if int(row["block"]) == block
        ]
        head_adj_cos, _, head_far_cos = _distance_medians(
            block_head_pairs,
            "mean_context_cosine",
        )
        block_pairs = [
            row for row in neuron_pairs if int(row["block"]) == block
        ]
        adj_j, _, far_j = _distance_medians(
            block_pairs, "top_neuron_jaccard"
        )
        adj_cos, _, far_cos = _distance_medians(
            block_pairs, "activation_score_cosine"
        )
        block_neuron = [row for row in neuron if int(row["block"]) == block]
        positive_current = sum(
            _float(row["specific_margin_drop"]) > 0.0
            for row in block_neuron
        )
        block_transfer = [
            row for row in transfer if int(row["block"]) == block
        ]
        diagonal = [
            _float(row["specific_margin_drop"])
            for row in block_transfer
            if row["donor_loop"] == row["receiver_loop"]
        ]
        off_diagonal = [
            _float(row["specific_margin_drop"])
            for row in block_transfer
            if row["donor_loop"] != row["receiver_loop"]
        ]
        adjacent_transfer = [
            _float(row["specific_margin_drop"])
            for row in block_transfer
            if int(row["loop_distance"]) == 1
        ]
        far_transfer = [
            _float(row["specific_margin_drop"])
            for row in block_transfer
            if int(row["loop_distance"]) >= 3
        ]
        task_adj = float("nan")
        task_far = float("nan")
        task_positive = 0
        if task_specific_rows is not None:
            task_adj, task_far, task_positive = _task_specific_overlap(
                task_specific_rows,
                block=block,
            )
        run_row.update(
            {
                f"B{block}_dominant_head_sequence": " ".join(
                    str(item) for item in dominant
                ),
                f"B{block}_dominant_head_switch_count": sum(
                    left != right for left, right in zip(dominant, dominant[1:])
                ),
                f"B{block}_activation_dominant_head_sequence": " ".join(
                    str(item) for item in activation_dominant
                ),
                f"B{block}_activation_causal_head_agreement_count": sum(
                    active == causal
                    for active, causal in zip(activation_dominant, dominant)
                ),
                f"B{block}_strong_head_sequence": "|".join(strong_sequence),
                f"B{block}_strong_head_uses": sum(strong_counts),
                f"B{block}_loops_with_strong_head": sum(
                    count > 0 for count in strong_counts
                ),
                f"B{block}_position_causal_head_sequence": " ".join(
                    str(item) for item in position_causal_heads
                ),
                f"B{block}_position_causal_scope_sequence": " ".join(
                    position_causal_scopes
                ),
                f"B{block}_position_causal_score_sequence": " ".join(
                    f"{item:.6f}" for item in position_causal_scores
                ),
                f"B{block}_position_strong_head_sequence": "|".join(
                    position_strong_sequence
                ),
                f"B{block}_loops_with_strong_head_position": sum(
                    item != "-" for item in position_strong_sequence
                ),
                f"B{block}_local_causal_head_sequence": " ".join(
                    str(item) for item in local_causal_heads
                ),
                f"B{block}_local_causal_scope_sequence": " ".join(
                    local_causal_scopes
                ),
                f"B{block}_local_causal_score_sequence": " ".join(
                    f"{item:.6f}" for item in local_causal_scores
                ),
                f"B{block}_local_strong_head_sequence": "|".join(
                    local_strong_sequence
                ),
                f"B{block}_loops_with_locally_strong_head_position": sum(
                    item != "-" for item in local_strong_sequence
                ),
                f"B{block}_head_context_adjacent_cosine_median": (
                    head_adj_cos
                ),
                f"B{block}_head_context_far_cosine_median": head_far_cos,
                f"B{block}_raw_neuron_adjacent_jaccard_median": adj_j,
                f"B{block}_raw_neuron_far_jaccard_median": far_j,
                f"B{block}_raw_activation_adjacent_cosine_median": adj_cos,
                f"B{block}_raw_activation_far_cosine_median": far_cos,
                f"B{block}_task_neuron_adjacent_jaccard_median": task_adj,
                f"B{block}_task_neuron_far_jaccard_median": task_far,
                f"B{block}_task_top_minus_random_positive_loop_count": (
                    task_positive
                ),
                f"B{block}_top_minus_random_positive_loop_count": (
                    positive_current
                ),
                f"B{block}_transfer_diagonal_median": _median(diagonal),
                f"B{block}_transfer_off_diagonal_median": _median(
                    off_diagonal
                ),
                f"B{block}_transfer_adjacent_median": _median(
                    adjacent_transfer
                ),
                f"B{block}_transfer_far_median": _median(far_transfer),
                f"B{block}_transfer_diagonal_advantage": (
                    _median(diagonal) - _median(off_diagonal)
                ),
            }
        )
        detail_rows.append(
            {
                "run": name,
                "family": family,
                "seed": seed,
                "block": block,
                "dominant_head_sequence": " ".join(
                    str(item) for item in dominant
                ),
                "dominant_head_switch_count": run_row[
                    f"B{block}_dominant_head_switch_count"
                ],
                "activation_dominant_head_sequence": " ".join(
                    str(item) for item in activation_dominant
                ),
                "activation_causal_head_agreement_count": run_row[
                    f"B{block}_activation_causal_head_agreement_count"
                ],
                "strong_head_sequence": "|".join(strong_sequence),
                "loops_with_strong_head": run_row[
                    f"B{block}_loops_with_strong_head"
                ],
                "position_causal_head_sequence": " ".join(
                    str(item) for item in position_causal_heads
                ),
                "position_causal_scope_sequence": " ".join(
                    position_causal_scopes
                ),
                "position_strong_head_sequence": "|".join(
                    position_strong_sequence
                ),
                "loops_with_strong_head_position": run_row[
                    f"B{block}_loops_with_strong_head_position"
                ],
                "local_causal_head_sequence": " ".join(
                    str(item) for item in local_causal_heads
                ),
                "local_causal_scope_sequence": " ".join(
                    local_causal_scopes
                ),
                "local_strong_head_sequence": "|".join(
                    local_strong_sequence
                ),
                "loops_with_locally_strong_head_position": run_row[
                    f"B{block}_loops_with_locally_strong_head_position"
                ],
                "head_context_adjacent_cosine": head_adj_cos,
                "head_context_far_cosine": head_far_cos,
                "raw_adjacent_jaccard": adj_j,
                "raw_far_jaccard": far_j,
                "task_adjacent_jaccard": task_adj,
                "task_far_jaccard": task_far,
                "task_top_minus_random_positive_loop_count": task_positive,
                "activation_adjacent_cosine": adj_cos,
                "activation_far_cosine": far_cos,
                "transfer_diagonal_median": _median(diagonal),
                "transfer_adjacent_median": _median(adjacent_transfer),
                "transfer_far_median": _median(far_transfer),
                "transfer_diagonal_advantage": (
                    _median(diagonal) - _median(off_diagonal)
                ),
            }
        )
    return run_row, detail_rows


def _matrix(
    rows: list[dict[str, str]],
    *,
    size: int,
    x_field: str,
    y_field: str,
    value_field: str,
    symmetric: bool = False,
    diagonal: float | None = None,
) -> np.ndarray:
    values = np.full((size, size), np.nan)
    if diagonal is not None:
        np.fill_diagonal(values, diagonal)
    for row in rows:
        x = int(row[x_field]) - 1
        y = int(row[y_field]) - 1
        values[y, x] = _float(row[value_field])
        if symmetric:
            values[x, y] = _float(row[value_field])
    return values


def _plot_panels(
    *,
    raw_dirs: dict[str, Path],
    out_dir: Path,
) -> None:
    names = sorted(raw_dirs)
    for kind in (
        "head",
        "head_position",
        "head_local",
        "neuron_overlap",
        "neuron_transfer",
    ):
        diverging_map = matplotlib.colors.LinearSegmentedColormap.from_list(
            "blue_gray_red",
            ("#3568c8", "#e2e2e2", "#ba2b2b"),
        )
        diverging_map.set_bad("#4d4d4d")
        transfer_limit = None
        if kind == "neuron_transfer":
            transfer_values = []
            for name in names:
                for item in _read_csv(
                    raw_dirs[name] / "neuron_transfer_rows.csv"
                ):
                    value = _float(item["specific_margin_drop"])
                    if np.isfinite(value):
                        transfer_values.append(abs(value))
            transfer_limit = max(
                0.5,
                float(np.quantile(transfer_values, 0.95)),
            )
        figure, axes = plt.subplots(
            len(names),
            2,
            figsize=(14, 3.1 * len(names)),
            squeeze=False,
            constrained_layout=True,
        )
        images = []
        for row_index, name in enumerate(names):
            summary = json.loads(
                (raw_dirs[name] / "summary.json").read_text()
            )
            loops = int(summary["config"]["max_loops"])
            for block in (1, 2):
                axis = axes[row_index, block - 1]
                if kind == "head":
                    rows = [
                        row
                        for row in _read_csv(
                            raw_dirs[name] / "head_loop_rows.csv"
                        )
                        if int(row["block"]) == block
                    ]
                    values = np.full((4, loops), np.nan)
                    for item in rows:
                        values[int(item["head"]), int(item["loop"]) - 1] = min(
                            _float(item["zero_margin_drop"]),
                            _float(item["shuffle_margin_drop"]),
                        )
                    image = axis.imshow(
                        values,
                        aspect="auto",
                        cmap=diverging_map,
                        norm=matplotlib.colors.TwoSlopeNorm(
                            vmin=-1.0,
                            vcenter=0.0,
                            vmax=3.0,
                        ),
                    )
                    axis.set_ylabel("physical head")
                    axis.set_yticks(range(4))
                    title = "min(zero, shuffled) margin drop"
                elif kind == "head_local":
                    rows = [
                        row
                        for row in _read_csv(
                            raw_dirs[name] / "head_position_loop_rows.csv"
                        )
                        if int(row["block"]) == block
                        and row["position_scope"] == "answer"
                    ]
                    values = np.full((4, loops), np.nan)
                    strong = np.zeros((4, loops), dtype=bool)
                    for item in rows:
                        y = int(item["head"])
                        x = int(item["loop"]) - 1
                        values[y, x] = min(
                            _float(
                                item["zero_local_path_accuracy_drop"]
                            ),
                            _float(
                                item[
                                    "shuffle_local_path_accuracy_drop"
                                ]
                            ),
                        )
                        strong[y, x] = bool(
                            int(item["strongly_locally_used"])
                        )
                    image = axis.imshow(
                        values,
                        aspect="auto",
                        cmap=diverging_map,
                        norm=matplotlib.colors.TwoSlopeNorm(
                            vmin=-0.2,
                            vcenter=0.0,
                            vmax=0.8,
                        ),
                    )
                    for y in range(values.shape[0]):
                        for x in range(values.shape[1]):
                            if strong[y, x]:
                                axis.text(
                                    x,
                                    y,
                                    "●",
                                    ha="center",
                                    va="center",
                                    fontsize=7,
                                    color=(
                                        "white"
                                        if values[y, x] >= 0.45
                                        else "black"
                                    ),
                                )
                    axis.set_ylabel("physical head")
                    axis.set_yticks(range(4), [f"H{i}" for i in range(4)])
                    title = "immediate answer-state progress"
                elif kind == "head_position":
                    position_scopes = (
                        "edge_marker",
                        "source",
                        "destination",
                        "query",
                        "start",
                        "depth",
                        "answer",
                    )
                    rows = [
                        row
                        for row in _read_csv(
                            raw_dirs[name] / "head_position_loop_rows.csv"
                        )
                        if int(row["block"]) == block
                    ]
                    values = np.full((len(position_scopes), loops), np.nan)
                    labels = np.full(
                        (len(position_scopes), loops),
                        "",
                        dtype=object,
                    )
                    for scope_index, scope in enumerate(position_scopes):
                        for loop in range(1, loops + 1):
                            candidates = [
                                row
                                for row in rows
                                if row["position_scope"] == scope
                                and int(row["loop"]) == loop
                            ]
                            strongest = max(
                                candidates,
                                key=lambda row: min(
                                    _float(row["zero_margin_drop"]),
                                    _float(row["shuffle_margin_drop"]),
                                ),
                            )
                            values[scope_index, loop - 1] = min(
                                _float(strongest["zero_margin_drop"]),
                                _float(strongest["shuffle_margin_drop"]),
                            )
                            labels[scope_index, loop - 1] = (
                                f"H{strongest['head']}"
                                if int(strongest["strongly_used"])
                                else ""
                            )
                    image = axis.imshow(
                        values,
                        aspect="auto",
                        cmap=diverging_map,
                        norm=matplotlib.colors.TwoSlopeNorm(
                            vmin=-1.0,
                            vcenter=0.0,
                            vmax=3.0,
                        ),
                    )
                    for y in range(values.shape[0]):
                        for x in range(values.shape[1]):
                            if labels[y, x]:
                                axis.text(
                                    x,
                                    y,
                                    labels[y, x],
                                    ha="center",
                                    va="center",
                                    fontsize=6,
                                    color=(
                                        "white"
                                        if values[y, x] >= 1.5
                                        else "black"
                                    ),
                                )
                    axis.set_yticks(
                        range(len(position_scopes)),
                        position_scopes,
                    )
                    title = "best head per token group"
                elif kind == "neuron_overlap":
                    color_map = plt.get_cmap("viridis").copy()
                    color_map.set_bad("#bdbdbd")
                    rows = [
                        row
                        for row in _read_csv(
                            raw_dirs[name] / "neuron_loop_pair_rows.csv"
                        )
                        if int(row["block"]) == block
                    ]
                    values = _matrix(
                        rows,
                        size=loops,
                        x_field="right_loop",
                        y_field="left_loop",
                        value_field="top_neuron_jaccard",
                        symmetric=True,
                        diagonal=1.0,
                    )
                    image = axis.imshow(
                        values,
                        aspect="auto",
                        cmap=color_map,
                        vmin=0.0,
                        vmax=1.0,
                    )
                    title = "raw top-64 neuron Jaccard"
                else:
                    rows = [
                        row
                        for row in _read_csv(
                            raw_dirs[name] / "neuron_transfer_rows.csv"
                        )
                        if int(row["block"]) == block
                    ]
                    values = _matrix(
                        rows,
                        size=loops,
                        x_field="receiver_loop",
                        y_field="donor_loop",
                        value_field="specific_margin_drop",
                    )
                    image = axis.imshow(
                        values,
                        aspect="auto",
                        cmap=diverging_map,
                        norm=matplotlib.colors.TwoSlopeNorm(
                            vmin=-transfer_limit,
                            vcenter=0.0,
                            vmax=transfer_limit,
                        ),
                    )
                    axis.set_ylabel("donor loop")
                    title = "top-64 minus random transfer"
                images.append(image)
                axis.set_title(f"{name} B{block}: {title}", fontsize=9)
                axis.set_xlabel("loop" if kind != "neuron_transfer" else "receiver loop")
                axis.set_xticks(range(loops), range(1, loops + 1))
                if kind in {"neuron_overlap", "neuron_transfer"}:
                    axis.set_yticks(range(loops), range(1, loops + 1))
        figure.colorbar(
            images[-1],
            ax=axes.ravel().tolist(),
            shrink=0.65,
            pad=0.02,
        )
        figure.savefig(out_dir / f"{kind}_by_loop.png", dpi=180)
        plt.close(figure)


def _parse_task_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("task run must be NAME=CSV")
    name, path = value.split("=", 1)
    return name, Path(path)


def _summarize_trajectory(
    name: str,
    functional_dir: Path,
) -> dict[str, Any]:
    progression = _read_csv(functional_dir / "branch_progression_rows.csv")
    b2_outputs = sorted(
        (
            row
            for row in progression
            if int(row["block"]) == 2 and row["stage"] == "post_mlp"
        ),
        key=lambda row: int(row["loop"]),
    )
    ablations = _read_csv(functional_dir / "position_ablation_rows.csv")
    attention_groups: list[str] = []
    attention_drops: list[float] = []
    for loop in (int(row["loop"]) for row in b2_outputs):
        candidates = [
            row
            for row in ablations
            if int(row["loop"]) == loop
            and int(row["block"]) == 2
            and row["component"] == "attention_out"
        ]
        strongest = max(
            candidates,
            key=lambda row: _float(row["accuracy_drop"]),
        )
        attention_groups.append(strongest["position_group"])
        attention_drops.append(_float(strongest["accuracy_drop"]))
    family, seed = _parse_family(name)
    return {
        "run": name,
        "family": family,
        "seed": seed,
        "b2_post_mlp_best_path_sequence": " ".join(
            row["best_path_position"] for row in b2_outputs
        ),
        "b2_post_mlp_best_path_accuracy_sequence": " ".join(
            f"{_float(row['best_path_accuracy']):.6f}"
            for row in b2_outputs
        ),
        "b2_attention_strongest_position_sequence": " ".join(
            attention_groups
        ),
        "b2_attention_position_accuracy_drop_sequence": " ".join(
            f"{value:.6f}" for value in attention_drops
        ),
        "first_endpoint_loop": next(
            (
                int(row["loop"])
                for row in b2_outputs
                if int(row["best_path_position"]) == 8
                and _float(row["best_path_accuracy"]) >= 0.5
            ),
            None,
        ),
        "max_decoded_jump": max(
            (
                int(right["best_path_position"])
                - int(left["best_path_position"])
                for left, right in zip(b2_outputs, b2_outputs[1:])
            ),
            default=0,
        ),
    }


def _plot_trajectory(
    rows: list[dict[str, Any]],
    out_dir: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(8.5, 5.5), constrained_layout=True)
    for row in rows:
        positions = [
            int(item)
            for item in row["b2_post_mlp_best_path_sequence"].split()
        ]
        accuracies = [
            float(item)
            for item in row[
                "b2_post_mlp_best_path_accuracy_sequence"
            ].split()
        ]
        loops = np.arange(1, len(positions) + 1)
        (line,) = axis.plot(
            loops,
            positions,
            linewidth=2,
            alpha=0.7,
            label=row["run"],
        )
        axis.scatter(
            loops,
            positions,
            s=30 + 100 * np.asarray(accuracies),
            color=line.get_color(),
            zorder=3,
        )
    axis.axhline(8, color="#555555", linestyle="--", linewidth=1)
    axis.set_xlabel("loop")
    axis.set_ylabel("decoded path position after B2")
    axis.set_xticks(range(1, 9))
    axis.set_yticks(range(0, 9))
    axis.set_ylim(-0.25, 8.4)
    axis.set_title("Marker size indicates decoding accuracy")
    axis.grid(alpha=0.2)
    axis.legend(ncol=2, fontsize=9)
    figure.savefig(out_dir / "trajectory_by_loop.png", dpi=180)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate five-model loop activation and reuse results."
    )
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument(
        "--task-neuron-run",
        action="append",
        type=_parse_task_run,
        default=[],
    )
    parser.add_argument(
        "--functional-run",
        action="append",
        type=_parse_task_run,
        default=[],
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_dirs = {
        json.loads(path.read_text())["name"]: path.parent
        for path in args.raw_dir.glob("*/summary.json")
    }
    task_rows = {
        name: _read_csv(path) for name, path in args.task_neuron_run
    }
    functional_dirs = dict(args.functional_run)
    if not raw_dirs:
        raise FileNotFoundError(f"no run summaries under {args.raw_dir}")
    run_rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []
    for name in sorted(raw_dirs):
        run_row, block_rows = summarize_run(
            raw_dirs[name],
            task_specific_rows=task_rows.get(name),
        )
        run_rows.append(run_row)
        detail_rows.extend(block_rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out_dir / "run_summary.csv", run_rows)
    _write_csv(args.out_dir / "block_summary.csv", detail_rows)
    trajectory_rows = [
        _summarize_trajectory(name, functional_dirs[name])
        for name in sorted(functional_dirs)
    ]
    _write_csv(args.out_dir / "trajectory_summary.csv", trajectory_rows)
    _plot_panels(raw_dirs=raw_dirs, out_dir=args.out_dir)
    if trajectory_rows:
        _plot_trajectory(trajectory_rows, args.out_dir)
    payload = {
        "run_count": len(run_rows),
        "runs": run_rows,
        "random_top64_jaccard_expectation": 64 / (2 * 1024 - 64),
    }
    (args.out_dir / "aggregate_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
