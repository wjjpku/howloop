from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def mean_and_sd(values: list[float]) -> tuple[float, float]:
    return (
        statistics.mean(values),
        statistics.stdev(values) if len(values) > 1 else 0.0,
    )


def summarize_conditions(
    rows: list[dict[str, str]],
    *,
    conditions: list[tuple[str, str]],
) -> list[dict[str, float | str]]:
    result: list[dict[str, float | str]] = []
    for direction in ("one_to_two", "two_to_one"):
        for condition, label in conditions:
            selected = [
                row
                for row in rows
                if row["direction"] == direction
                and row["condition"] == condition
            ]
            if not selected:
                raise KeyError(f"missing condition: {direction} {condition}")
            for metric in (
                "desired_accuracy",
                "undesired_accuracy",
                "endpoint_accuracy",
                "desired_minus_undesired_logit",
                "normalized_margin_recovery",
            ):
                mean, sd = mean_and_sd(
                    [float(row[metric]) for row in selected]
                )
                result.append(
                    {
                        "direction": direction,
                        "condition": condition,
                        "label": label,
                        "metric": metric,
                        "mean": mean,
                        "controller_seed_sd": sd,
                    }
                )
    return result


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def condition_metric(
    summary: list[dict[str, float | str]],
    *,
    direction: str,
    label: str,
    metric: str,
) -> tuple[float, float]:
    row = next(
        item
        for item in summary
        if item["direction"] == direction
        and item["label"] == label
        and item["metric"] == metric
    )
    return float(row["mean"]), float(row["controller_seed_sd"])


def aggregate_onehot(
    rows: list[dict[str, str]],
    *,
    mode: str,
    role: str,
    metric: str,
) -> tuple[float, float]:
    selected = [
        row
        for row in rows
        if row["mode"] == mode
        and row["block"] in {"2", "2.0"}
        and row["head"] in {"0", "0.0"}
        and row["source_role"] == role
    ]
    by_seed: dict[int, list[float]] = {}
    for row in selected:
        by_seed.setdefault(int(float(row["controller_seed"])), []).append(
            float(row[metric])
        )
    seed_means = [statistics.mean(values) for values in by_seed.values()]
    return mean_and_sd(seed_means)


def aggregate_routing(
    rows: list[dict[str, str]],
    *,
    mode: str,
    metric: str,
) -> tuple[float, float]:
    selected = [
        row
        for row in rows
        if row["mode"] == mode
        and row["block"] in {"2", "2.0"}
        and row["head"] in {"0", "0.0"}
    ]
    by_seed: dict[int, list[float]] = {}
    for row in selected:
        by_seed.setdefault(int(float(row["controller_seed"])), []).append(
            float(row[metric])
        )
    seed_means = [statistics.mean(values) for values in by_seed.values()]
    return mean_and_sd(seed_means)


def make_figure(
    *,
    condition_summary: list[dict[str, float | str]],
    onehot_rows: list[dict[str, str]],
    routing_rows: list[dict[str, str]],
    output: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    figure, axes = plt.subplots(1, 3, figsize=(17, 5.4))

    labels = [
        "receiver",
        "B1 registers",
        "B2 H0 Q",
        "B2 H0 pattern",
        "B2 H0 context",
        "B2 H1-3 context",
        "B2 MLP out",
        "B2 residual-mid",
        "donor",
    ]
    x = np.arange(len(labels))
    width = 0.38
    colors = {"one_to_two": "#D55E00", "two_to_one": "#0072B2"}
    for offset, direction, direction_label in (
        (-width / 2, "one_to_two", "one → two"),
        (width / 2, "two_to_one", "two → one"),
    ):
        values, errors = zip(
            *[
                condition_metric(
                    condition_summary,
                    direction=direction,
                    label=label,
                    metric="desired_accuracy",
                )
                for label in labels
            ],
            strict=True,
        )
        axes[0].bar(
            x + offset,
            values,
            width,
            yerr=errors,
            capsize=2,
            color=colors[direction],
            label=direction_label,
        )
    axes[0].set_xticks(x, labels, rotation=55, ha="right")
    axes[0].set_ylim(0, 1.08)
    axes[0].set_ylabel("desired-hop accuracy")
    axes[0].set_title("Bidirectional causal interchange")
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.2)

    semantic_cases = [
        ("J_one", "destination_current", r"$J_{one}$: read $c\to f(c)$"),
        ("J_one", "destination_one", r"$J_{one}$: read $f(c)\to f^2(c)$"),
        ("J_one", "destination_random", r"$J_{one}$: read random edge"),
        ("J_two", "destination_current", r"$J_{two}$: read $c\to f(c)$"),
        ("J_two", "destination_one", r"$J_{two}$: read $f(c)\to f^2(c)$"),
        ("J_two", "destination_random", r"$J_{two}$: read random edge"),
    ]
    x2 = np.arange(len(semantic_cases))
    for offset, metric, label, color in (
        (
            -width / 2,
            "distinct_one_accuracy",
            "one-hop output",
            "#009E73",
        ),
        (
            width / 2,
            "distinct_two_accuracy",
            "two-hop output",
            "#CC79A7",
        ),
    ):
        values, errors = zip(
            *[
                aggregate_onehot(
                    onehot_rows,
                    mode=mode,
                    role=role,
                    metric=metric,
                )
                for mode, role, _ in semantic_cases
            ],
            strict=True,
        )
        axes[1].bar(
            x2 + offset,
            values,
            width,
            yerr=errors,
            capsize=2,
            color=color,
            label=label,
        )
    axes[1].set_xticks(
        x2,
        [label for _, _, label in semantic_cases],
        rotation=55,
        ha="right",
    )
    axes[1].set_ylim(0, 1.08)
    axes[1].set_ylabel("collision-controlled accuracy")
    axes[1].set_title("Force Block-2 Head 0 to read an edge")
    axes[1].legend(frameon=False)
    axes[1].grid(axis="y", alpha=0.2)

    modes = ["J_one", "J_two", "oracle_one", "oracle_two"]
    mode_labels = [
        r"$J_{one}$",
        r"$J_{two}$",
        "one-hop oracle",
        "two-hop oracle",
    ]
    x3 = np.arange(len(modes))
    for offset, metric, label, color in (
        (
            -width / 2,
            "mass_destination_current",
            r"edge $c\to f(c)$",
            "#E69F00",
        ),
        (
            width / 2,
            "mass_destination_one",
            r"edge $f(c)\to f^2(c)$",
            "#56B4E9",
        ),
    ):
        values, errors = zip(
            *[
                aggregate_routing(
                    routing_rows,
                    mode=mode,
                    metric=metric,
                )
                for mode in modes
            ],
            strict=True,
        )
        axes[2].bar(
            x3 + offset,
            values,
            width,
            yerr=errors,
            capsize=2,
            color=color,
            label=label,
        )
    axes[2].set_xticks(x3, mode_labels, rotation=25, ha="right")
    axes[2].set_ylim(0, 0.32)
    axes[2].set_ylabel("mean attention mass")
    axes[2].set_title("Natural Block-2 Head-0 routing")
    axes[2].legend(frameon=False)
    axes[2].grid(axis="y", alpha=0.2)

    figure.suptitle(
        "D8L6 seed6: an affine controller selects hop size through "
        "a Block-2 Head-0 graph pointer",
        fontsize=14,
    )
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--causal-dir", type=Path, required=True)
    parser.add_argument("--routing-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    conditions = [
        ("receiver", "receiver"),
        ("B1.block_input.registers", "B1 registers"),
        ("B2.q.heads_0", "B2 H0 Q"),
        ("B2.attention_pattern.heads_0", "B2 H0 pattern"),
        ("B2.head_context.heads_0", "B2 H0 context"),
        ("B2.head_context.heads_123", "B2 H1-3 context"),
        ("B2.mlp_out.answer", "B2 MLP out"),
        ("B2.residual_mid.answer", "B2 residual-mid"),
        ("donor", "donor"),
    ]
    causal_rows = read_rows(args.causal_dir / "condition_summary.csv")
    condition_summary = summarize_conditions(
        causal_rows,
        conditions=conditions,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "key_causal_results.csv", condition_summary)
    make_figure(
        condition_summary=condition_summary,
        onehot_rows=read_rows(args.routing_dir / "onehot_rows.csv"),
        routing_rows=read_rows(args.routing_dir / "routing_rows.csv"),
        output=args.out_dir / "jump_selector_circuit.png",
    )


if __name__ == "__main__":
    main()
