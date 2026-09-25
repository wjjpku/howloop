from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def extract_signature(name: str, summary: dict[str, Any]) -> dict[str, Any]:
    cross = summary["cross_time"]
    schedule = summary["schedule"]
    condition_index = {
        condition: index for index, condition in enumerate(schedule["condition_names"])
    }
    for required in ("skip_1", "repeat_1"):
        if required not in condition_index:
            raise ValueError(f"schedule is missing {required}")

    final_slot = schedule["final_slot_accuracy"]
    skip_row = final_slot[condition_index["skip_1"]]
    repeat_row = final_slot[condition_index["repeat_1"]]
    cross_time_delta1 = float(cross["mean_transplant_acc_by_delta"][1])
    donor_final_by_loop = [
        float(np.mean([receiver[0] for receiver in donor]))
        for donor in cross["donor_final_acc"]
    ]
    answer_ready_loop = next(
        (loop for loop, value in enumerate(donor_final_by_loop, start=1) if value >= 0.95),
        None,
    )
    skip_f5 = float(skip_row[5])
    skip_f6 = float(skip_row[6])
    repeat_f6 = float(repeat_row[6])
    repeat_f7 = float(repeat_row[7])
    if cross_time_delta1 >= 0.9 and skip_f5 >= 0.9 and repeat_f7 >= 0.9:
        classification = "time_homogeneous_transition"
    elif cross_time_delta1 <= 0.3 and skip_f6 >= 0.9 and repeat_f6 >= 0.9:
        classification = "endpoint_attractor"
    else:
        classification = "mixed_or_phase_conditioned"

    return {
        "model": name,
        "family": re.sub(r"_seed\d+$", "", name),
        "classification": classification,
        "cross_time_delta1": cross_time_delta1,
        "skip_f5": skip_f5,
        "skip_f6": skip_f6,
        "repeat_f6": repeat_f6,
        "repeat_f7": repeat_f7,
        "endpoint_retention": 0.5 * (skip_f6 + repeat_f6),
        "answer_ready_loop_95": answer_ready_loop,
        "donor_final_by_loop": donor_final_by_loop,
    }


def aggregate_families(signatures: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for signature in signatures:
        grouped[signature["family"]].append(signature)
    metric_names = [
        "cross_time_delta1",
        "skip_f5",
        "repeat_f7",
        "endpoint_retention",
    ]
    rows: list[dict[str, Any]] = []
    for family, family_signatures in sorted(grouped.items()):
        row: dict[str, Any] = {
            "family": family,
            "seeds": len(family_signatures),
            "classifications": ",".join(
                sorted({signature["classification"] for signature in family_signatures})
            ),
        }
        for metric in metric_names:
            values = np.array([signature[metric] for signature in family_signatures])
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std())
        ready = [
            signature["answer_ready_loop_95"]
            for signature in family_signatures
            if signature["answer_ready_loop_95"] is not None
        ]
        row["answer_ready_loop_95_mean"] = float(np.mean(ready)) if ready else ""
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write empty rows")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_family_signatures(rows: list[dict[str, Any]], path: Path) -> None:
    metrics = [
        ("cross_time_delta1", "Cross-time continue"),
        ("skip_f5", "Skip tracks f^5"),
        ("repeat_f7", "Repeat tracks f^7"),
        ("endpoint_retention", "Retains f^6"),
    ]
    families = [row["family"] for row in rows]
    x = np.arange(len(metrics))
    width = 0.8 / max(1, len(families))
    fig, ax = plt.subplots(figsize=(11, 5.8))
    for family_idx, row in enumerate(rows):
        offset = (family_idx - (len(families) - 1) / 2) * width
        means = [row[f"{metric}_mean"] for metric, _ in metrics]
        stds = [row[f"{metric}_std"] for metric, _ in metrics]
        ax.bar(x + offset, means, width, yerr=stds, capsize=3, label=families[family_idx])
    ax.axhline(0.125, color="black", linestyle=":", linewidth=1, label="random N=8")
    ax.set_xticks(x, [label for _, label in metrics])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("accuracy")
    ax.set_title("Mechanism signatures: iterative transition vs endpoint attractor")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate temporal intervention signatures.")
    parser.add_argument("--input", action="append", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    signatures: list[dict[str, Any]] = []
    for path in args.input:
        data = json.loads(path.read_text(encoding="utf-8"))
        for name, summary in data["models"].items():
            signatures.append(extract_signature(name, summary))
    family_rows = aggregate_families(signatures)
    csv_signatures = [
        {key: value for key, value in row.items() if key != "donor_final_by_loop"}
        | {"donor_final_by_loop": ",".join(map(str, row["donor_final_by_loop"]))}
        for row in signatures
    ]
    write_csv(args.out_dir / "model_signatures.csv", csv_signatures)
    write_csv(args.out_dir / "family_signatures.csv", family_rows)
    plot_family_signatures(family_rows, args.out_dir / "mechanism_signature_comparison.png")
    (args.out_dir / "signature_summary.json").write_text(
        json.dumps({"models": signatures, "families": family_rows}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
