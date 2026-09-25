from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _splits(
    y: np.ndarray,
    groups: np.ndarray,
    *,
    repeats: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    result: list[tuple[np.ndarray, np.ndarray]] = []
    dummy = np.zeros((len(y), 1), dtype=np.float64)
    for repeat in range(repeats):
        splitter = StratifiedGroupKFold(
            n_splits=5,
            shuffle=True,
            random_state=seed + repeat,
        )
        result.extend(splitter.split(dummy, y, groups))
    return result


def _score(
    features: np.ndarray,
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[float, float, float, float]:
    auc: list[float] = []
    balanced: list[float] = []
    for train, test in splits:
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=5000,
                class_weight="balanced",
                C=1.0,
            ),
        )
        model.fit(features[train], y[train])
        probability = model.predict_proba(features[test])[:, 1]
        prediction = probability >= 0.5
        auc.append(roc_auc_score(y[test], probability))
        balanced.append(balanced_accuracy_score(y[test], prediction))
    return (
        float(np.mean(auc)),
        float(np.std(auc)),
        float(np.mean(balanced)),
        float(np.std(balanced)),
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    per_example = [
        row
        for row in _read(args.mode_dir / "per_example.csv")
        if row["run"] == "J"
        and int(row["cycle"]) == args.cycle
        and int(row["current_cycle_length"]) > 1
    ]
    samples = sorted(int(row["sample"]) for row in per_example)
    metadata = {int(row["sample"]): row for row in per_example}
    y = np.asarray([int(metadata[sample]["correct"]) for sample in samples])
    groups = np.asarray([metadata[sample]["successors"] for sample in samples])
    splits = _splits(y, groups, repeats=args.repeats, seed=args.seed)

    modes = _read(args.mode_dir / "J_mode_per_example.csv")
    mode_values = {
        (int(row["sample"]), int(row["mode"])): float(
            row["left_coordinate"]
        )
        for row in modes
        if int(row["cycle"]) == args.cycle
    }
    mode_indices = sorted({mode for _, mode in mode_values})
    mode_matrix = np.asarray(
        [
            [mode_values[(sample, mode)] for mode in mode_indices]
            for sample in samples
        ]
    )
    norm_matrix = np.asarray(
        [
            [
                float(metadata[sample][field])
                for field in (
                    "loop_input_answer_norm",
                    "B1_output_answer_norm",
                    "B2_post_attention_answer_norm",
                    "B2_output_answer_norm",
                    "B2_MLP_hidden_norm",
                )
            ]
            for sample in samples
        ]
    )
    h0_attention = np.asarray(
        [
            [float(metadata[sample]["B2H0_current_destination_attention"])]
            for sample in samples
        ]
    )
    cycle_length = np.asarray(
        [[float(metadata[sample]["current_cycle_length"])] for sample in samples]
    )
    features = {
        "J_left_singular_48": mode_matrix,
        "J_left_singular_abs_48": np.abs(mode_matrix),
        "loop_input_norm_1": norm_matrix[:, :1],
        "all_stage_norms_5": norm_matrix,
        "B2H0_destination_attention_1": h0_attention,
        "cycle_length_1": cycle_length,
        "J_modes_plus_norms_53": np.column_stack((mode_matrix, norm_matrix)),
    }
    rows: list[dict[str, Any]] = []
    for label, matrix in features.items():
        auc_mean, auc_std, balanced_mean, balanced_std = _score(
            matrix, y, splits
        )
        rows.append(
            {
                "feature": label,
                "dimensions": matrix.shape[1],
                "draw": "",
                "roc_auc_mean": auc_mean,
                "roc_auc_std": auc_std,
                "balanced_accuracy_mean": balanced_mean,
                "balanced_accuracy_std": balanced_std,
            }
        )

    random_rows = _read(
        args.random_dir / "random_projection_per_example.csv"
    )
    draws = sorted({int(row["projection_draw"]) for row in random_rows})
    random_auc: list[float] = []
    random_balanced: list[float] = []
    for draw in draws:
        values = {
            (int(row["sample"]), int(row["coordinate"])): float(
                row["projection_value"]
            )
            for row in random_rows
            if int(row["cycle"]) == args.cycle
            and int(row["projection_draw"]) == draw
        }
        coordinates = sorted({coordinate for _, coordinate in values})
        matrix = np.asarray(
            [
                [values[(sample, coordinate)] for coordinate in coordinates]
                for sample in samples
            ]
        )
        auc_mean, auc_std, balanced_mean, balanced_std = _score(
            matrix, y, splits
        )
        random_auc.append(auc_mean)
        random_balanced.append(balanced_mean)
        rows.append(
            {
                "feature": "random_orthonormal_48",
                "dimensions": matrix.shape[1],
                "draw": draw,
                "roc_auc_mean": auc_mean,
                "roc_auc_std": auc_std,
                "balanced_accuracy_mean": balanced_mean,
                "balanced_accuracy_std": balanced_std,
            }
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write(args.out_dir / "probe_cv.csv", rows)
    summary_rows = [row for row in rows if row["draw"] == ""]
    summary_rows.append(
        {
            "feature": "random_orthonormal_48_mean_over_draws",
            "dimensions": 48,
            "draw": f"{len(draws)} draws",
            "roc_auc_mean": float(np.mean(random_auc)),
            "roc_auc_std": float(np.std(random_auc)),
            "balanced_accuracy_mean": float(np.mean(random_balanced)),
            "balanced_accuracy_std": float(np.std(random_balanced)),
        }
    )
    _write(args.out_dir / "probe_summary.csv", summary_rows)

    plotted = [
        next(row for row in summary_rows if row["feature"] == label)
        for label in (
            "cycle_length_1",
            "loop_input_norm_1",
            "random_orthonormal_48_mean_over_draws",
            "J_left_singular_48",
            "all_stage_norms_5",
            "B2H0_destination_attention_1",
            "J_modes_plus_norms_53",
        )
    ]
    labels = [
        "cycle length",
        "input norm",
        "random 48D",
        "J-mode 48D",
        "five norms",
        "B2 H0 attention",
        "J modes + norms",
    ]
    fig, ax = plt.subplots(figsize=(8.2, 4.3))
    ax.bar(
        np.arange(len(plotted)),
        [float(row["roc_auc_mean"]) for row in plotted],
        yerr=[float(row["roc_auc_std"]) for row in plotted],
        color=["#BAB0AC", "#9D755D", "#BAB0AC", "#4C78A8", "#F58518", "#E45756", "#54A24B"],
        capsize=3,
    )
    ax.set_xticks(np.arange(len(plotted)), labels, rotation=25, ha="right")
    ax.set(ylabel="Cross-validated ROC AUC", ylim=(0.4, 1.0))
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1)
    ax.set_title("Late-failure observability: J coordinates versus controls")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(args.out_dir / "probe_auc.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    result = {
        "status": "complete",
        "cycle": args.cycle,
        "subset": "current permutation cycle length greater than one",
        "examples": len(samples),
        "correct": int(y.sum()),
        "incorrect": int((1 - y).sum()),
        "unique_graphs": int(len(set(groups.tolist()))),
        "cross_validation": (
            f"{args.repeats} repeats of 5-fold stratified group CV; "
            "groups are successor permutations"
        ),
        "random_projection_draws": len(draws),
        "summary": summary_rows,
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cross-validated observability audit for rank-48 J modes."
    )
    parser.add_argument("--mode-dir", type=Path, required=True)
    parser.add_argument("--random-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--cycle", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260731)
    return parser.parse_args(argv)


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
