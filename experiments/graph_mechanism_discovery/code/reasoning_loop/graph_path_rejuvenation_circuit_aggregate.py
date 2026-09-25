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
import torch.nn.functional as F


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _row(
    rows: list[dict[str, str]],
    **matches: str,
) -> dict[str, str]:
    return next(
        row
        for row in rows
        if all(row[key] == value for key, value in matches.items())
    )


def aggregate(
    *,
    primary_dir: Path,
    replica_dir: Path,
    out_dir: Path,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    run_dirs = {
        "primary": primary_dir,
        "replica": replica_dir,
    }
    summaries = {
        name: json.loads((path / "summary.json").read_text())
        for name, path in run_dirs.items()
    }
    artifacts = {
        name: torch.load(
            path / "optimal_init_matrix.pt",
            map_location="cpu",
            weights_only=True,
        )
        for name, path in run_dirs.items()
    }

    stability_rows = []
    for rank in (8, 16, 24, 32):
        primary = artifacts["primary"]["age_directions"][:rank]
        replica = artifacts["replica"]["age_directions"][:rank]
        canonical_cosines = torch.linalg.svdvals(
            primary @ replica.T
        )
        stability_rows.append(
            {
                "rank": rank,
                "mean_squared_canonical_cosine": float(
                    canonical_cosines.square().mean()
                ),
                "mean_canonical_cosine": float(
                    canonical_cosines.mean()
                ),
                "minimum_canonical_cosine": float(
                    canonical_cosines.min()
                ),
            }
        )
    _write_csv(out_dir / "subspace_stability.csv", stability_rows)
    primary_matrix = artifacts["primary"]["full_update_matrix"].flatten()
    replica_matrix = artifacts["replica"]["full_update_matrix"].flatten()
    matrix_cosine = float(
        F.cosine_similarity(primary_matrix, replica_matrix, dim=0)
    )
    matrix_relative_difference = float(
        (primary_matrix - replica_matrix).norm()
        / primary_matrix.norm()
    )

    rank_rows = []
    for run_name, path in run_dirs.items():
        for row in _read_csv(path / "rank_intervention_rows.csv"):
            if row["condition"] not in {
                "age_subspace",
                "age_subspace_complement",
                "age_subspace_shuffled",
                "random_output_subspace",
            }:
                continue
            rank_rows.append({"run": run_name, **row})
    _write_csv(out_dir / "rank_curves.csv", rank_rows)

    component_rows = []
    for run_name, path in run_dirs.items():
        rows = _read_csv(path / "component_localization_rows.csv")
        for row in rows:
            if row["condition"] == "patch_out":
                component_rows.append({"run": run_name, **row})
    _write_csv(out_dir / "component_necessity.csv", component_rows)

    neuron_rows = []
    neuron_overlaps: dict[str, dict[str, float]] = {}
    for block in ("1", "2"):
        neuron_overlaps[block] = {}
        primary_rows = _read_csv(
            primary_dir / "mlp_neuron_rows.csv"
        )
        replica_rows = _read_csv(
            replica_dir / "mlp_neuron_rows.csv"
        )
        for count in (32, 64, 128, 256):
            condition = f"top{count}_removed"
            primary = _row(
                primary_rows,
                block=block,
                condition=condition,
            )
            replica = _row(
                replica_rows,
                block=block,
                condition=condition,
            )
            primary_set = set(map(int, primary["top_neurons"].split()))
            replica_set = set(map(int, replica["top_neurons"].split()))
            overlap = len(primary_set & replica_set)
            union = len(primary_set | replica_set)
            neuron_rows.append(
                {
                    "block": block,
                    "top_k": count,
                    "intersection": overlap,
                    "jaccard": overlap / union,
                }
            )
            neuron_overlaps[block][str(count)] = overlap / union
    _write_csv(out_dir / "mlp_neuron_stability.csv", neuron_rows)

    stage_rows = []
    for run_name, path in run_dirs.items():
        for row in _read_csv(path / "stage_readout_rows.csv"):
            stage_rows.append({"replication": run_name, **row})
    _write_csv(out_dir / "stage_readout.csv", stage_rows)

    attention_rows = []
    for run_name, path in run_dirs.items():
        rows = _read_csv(path / "attention_function_rows.csv")
        for state, head, role in (
            ("terminal", "2", "current_destination"),
            ("rejuvenated", "2", "current_destination"),
            ("oracle_young", "2", "current_destination"),
            ("terminal", "0", "answer_self"),
            ("rejuvenated", "0", "answer_self"),
            ("oracle_young", "0", "answer_self"),
        ):
            row = _row(
                rows,
                run=state,
                site="1",
                head=head,
                key_role=role,
            )
            attention_rows.append(
                {
                    "replication": run_name,
                    "state": state,
                    "block": 2,
                    "head": int(head),
                    "role": role,
                    "attention": float(row["attention"]),
                }
            )
    _write_csv(out_dir / "attention_roles.csv", attention_rows)

    selected_nodes = summaries["primary"]["selected_circuit_nodes"]
    circuit_object = {
        "model": "D8_L8_seed1",
        "behavior": (
            "restore f^2(current) at effective loop 9 from a terminal "
            "age-8 state"
        ),
        "upstream_intervention": {
            "node": "answer.age_subspace",
            "full_matrix_shape": [257, 257],
            "causal_subspace_dimension": 16,
            "scope": "terminal age8 to executable age2",
        },
        "effective_nodes": selected_nodes,
        "parameter_nodes": [
            "physical_B1.MLP",
            "physical_B2.H0",
            "physical_B2.H2",
            "physical_B2.MLP",
        ],
        "candidate_edges": [
            ["answer.age_subspace", "L9.B1.MLP.out@answer"],
            ["L9.B1.MLP.out@answer", "L9.B2.H0.context@answer"],
            ["L9.B1.MLP.out@answer", "L9.B2.H2.context@answer"],
            ["L9.B2.H0.context@answer", "L9.B2.MLP.out@answer"],
            ["L9.B2.H2.context@answer", "L9.B2.MLP.out@answer"],
            ["L9.B2.MLP.out@answer", "answer logits"],
        ],
        "edge_evidence": (
            "architectural ordering plus joint node interventions; "
            "individual edges were not separately path-patched"
        ),
        "intervention_family": (
            "answer-state age-subspace intervention and old/young "
            "activation replacement at loop9"
        ),
    }
    (out_dir / "circuit_object.json").write_text(
        json.dumps(circuit_object, indent=2),
        encoding="utf-8",
    )

    result = {
        "model": "D8_L8_seed1",
        "matrix_cosine_across_data_replications": matrix_cosine,
        "matrix_relative_difference": matrix_relative_difference,
        "rank16_subspace_stability": next(
            row for row in stability_rows if row["rank"] == 16
        ),
        "mlp_neuron_jaccard": neuron_overlaps,
        "runs": {
            name: {
                "selected_ridge": summary["selected_ridge"],
                "full_matrix_accuracy": summary[
                    "full_matrix_behavior"
                ]["accuracy"],
                "oracle_young_accuracy": summary[
                    "oracle_young_answer_behavior"
                ]["accuracy"],
                "rank16_accuracy": summary[
                    "minimal_rank_behavior"
                ]["accuracy"],
                "rank16_age_variance": summary[
                    "age_variance_at_minimal_rank"
                ],
                "selected_nodes": summary["selected_circuit_nodes"],
                "circuit_only_accuracy": summary[
                    "circuit_only"
                ]["accuracy"],
                "circuit_only_recovery": summary[
                    "circuit_only"
                ]["recovery"],
                "complement_only_accuracy": summary[
                    "complement_only"
                ]["accuracy"],
                "oracle_transfer_circuit_accuracy": summary[
                    "oracle_young_circuit_transfer"
                ]["circuit_only"]["accuracy"],
                "oracle_transfer_recovery": summary[
                    "oracle_young_circuit_transfer"
                ]["circuit_only"]["recovery"],
                "faithful_size4_subsets": summary[
                    "alternative_circuit_search"
                ]["faithful_subset_count"],
                "size4_subsets_evaluated": summary[
                    "alternative_circuit_search"
                ]["subsets_evaluated"],
                "best_disjoint_recovery": summary[
                    "alternative_circuit_search"
                ]["best_disjoint_recovery"],
            }
            for name, summary in summaries.items()
        },
    }
    (out_dir / "aggregate_summary.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    figure, axes = plt.subplots(2, 2, figsize=(12.0, 8.0))
    for run_name, marker in (("primary", "o"), ("replica", "s")):
        selected = [
            row
            for row in rank_rows
            if row["run"] == run_name
            and row["condition"] == "age_subspace"
        ]
        axes[0, 0].plot(
            [int(row["rank"]) for row in selected],
            [float(row["accuracy"]) for row in selected],
            marker=marker,
            label=run_name,
        )
    axes[0, 0].axhline(1 / 8, color="0.5", linestyle=":", label="chance")
    axes[0, 0].axvline(16, color="0.4", linestyle="--")
    axes[0, 0].set(
        xlabel="age-subspace dimension",
        ylabel="accuracy",
        title="Causal rank of the rejuvenation update",
        ylim=(-0.03, 1.04),
        xlim=(-5, 260),
    )
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.2)

    stages = [
        "input_to_loop9",
        "B1_post_attention",
        "B1_post_mlp",
        "B2_post_attention",
        "B2_post_mlp",
    ]
    for run_state, color in (
        ("terminal", "#777777"),
        ("oracle_young", "#ff7f0e"),
        ("rejuvenated", "#1f77b4"),
    ):
        values = []
        for stage in stages:
            matching = [
                float(row["accuracy"])
                for row in stage_rows
                if row["run"] == run_state
                and row["stage"] == stage
            ]
            values.append(float(np.mean(matching)))
        axes[0, 1].plot(
            range(len(stages)),
            values,
            marker="o",
            linewidth=2,
            color=color,
            label=run_state,
        )
    axes[0, 1].set(
        xticks=range(len(stages)),
        xticklabels=["input", "B1 attn", "B1 MLP", "B2 attn", "B2 MLP"],
        ylabel="intermediate-readout accuracy",
        title="Where the revived answer becomes correct",
        ylim=(-0.03, 1.04),
    )
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.2)

    labels = sorted(
        {row["node"] for row in component_rows},
        key=lambda label: (
            0 if "B1" in label else 1,
            label,
        ),
    )
    means = [
        float(
            np.mean(
                [
                    float(row["necessity"])
                    for row in component_rows
                    if row["node"] == label
                ]
            )
        )
        for label in labels
    ]
    short_labels = [
        label.replace("L9.", "").replace(".context@answer", "")
        .replace(".out@answer", "")
        for label in labels
    ]
    colors = [
        "#d62728" if label in selected_nodes else "#bdbdbd"
        for label in labels
    ]
    axes[1, 0].barh(short_labels, means, color=colors)
    axes[1, 0].invert_yaxis()
    axes[1, 0].set(
        xlabel="patch-out necessity",
        title="Downstream effective components",
    )
    axes[1, 0].grid(axis="x", alpha=0.2)

    role_order = [
        ("B2H2 current-edge destination", 2, "current_destination"),
        ("B2H0 answer self", 0, "answer_self"),
    ]
    states = ["terminal", "rejuvenated", "oracle_young"]
    x = np.arange(len(role_order))
    width = 0.24
    for offset, state in enumerate(states):
        values = []
        for _, head, role in role_order:
            values.append(
                float(
                    np.mean(
                        [
                            row["attention"]
                            for row in attention_rows
                            if row["state"] == state
                            and row["head"] == head
                            and row["role"] == role
                        ]
                    )
                )
            )
        axes[1, 1].bar(
            x + (offset - 1) * width,
            values,
            width,
            label=state,
        )
    axes[1, 1].set(
        xticks=x,
        xticklabels=[item[0] for item in role_order],
        ylabel="attention probability",
        title="Rejuvenation restores natural head routing",
        ylim=(0, 0.9),
    )
    axes[1, 1].legend()
    axes[1, 1].grid(axis="y", alpha=0.2)

    figure.tight_layout()
    figure.savefig(
        out_dir / "rejuvenation_circuit_summary.png",
        dpi=180,
    )
    plt.close(figure)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary-dir", type=Path, required=True)
    parser.add_argument("--replica-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    aggregate(
        primary_dir=args.primary_dir,
        replica_dir=args.replica_dir,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
