from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean, stdev

import matplotlib.pyplot as plt
import numpy as np
import torch

from reasoning_loop.graph_path_telomere_unit_j import load_unit_j_map


REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "results" / "graph_path_component_j_ce_only_20260731"
INITIAL_J = (
    REPO
    / "results"
    / "graph_path_prenorm_component_unit_j_direct_H3_curriculum64_20260731"
    / "remote_artifacts"
    / "h64"
    / "unit_j_maps.pt"
)
SEEDS = (404003, 411003, 421003)


def curve_metrics(curve: dict) -> dict[str, float | list[float]]:
    values = [float(value) for value in curve["nonendpoint_accuracy_by_cycle"]]
    return {
        "auc_1_24": float(curve["nonendpoint_auc_1_24"]),
        "auc_25_48": float(curve["nonendpoint_auc_25_48"]),
        "auc_49_64": float(curve["nonendpoint_auc_49_64"]),
        "auc_all": float(mean(values)),
        "loop64": values[63],
        "values": values,
    }


def vector_norm(value: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(value.float()).item())


def main() -> None:
    device = torch.device("cpu")
    initial_map, _ = load_unit_j_map(INITIAL_J, label="task", device=device)
    rows = []
    for seed in SEEDS:
        audit_path = ROOT / "remote_artifacts" / "paired_audit" / f"task{seed}" / "summary.json"
        ce_dir = ROOT / "remote_artifacts" / f"task{seed}" / "ce_only_h64"
        base_dir = ROOT / "matched_state01" / f"task{seed}"
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        ce_summary = json.loads((ce_dir / "summary.json").read_text(encoding="utf-8"))
        base_summary = json.loads((base_dir / "summary.json").read_text(encoding="utf-8"))
        result = audit["results"][0]
        base_curve = curve_metrics(result["curves"]["base_J"])
        ce_curve = curve_metrics(result["curves"]["full_J"])
        base_map, _ = load_unit_j_map(
            base_dir / "unit_j_maps.pt", label="task", device=device
        )
        ce_map, _ = load_unit_j_map(
            ce_dir / "unit_j_maps.pt", label="task", device=device
        )
        base_update = (base_map.weight - initial_map.weight).flatten().float()
        ce_update = (ce_map.weight - initial_map.weight).flatten().float()
        update_cosine = float(
            torch.dot(base_update, ce_update)
            / (
                torch.linalg.vector_norm(base_update)
                * torch.linalg.vector_norm(ce_update)
            )
        )
        base_training = base_summary["training"]["task_rows"]
        ce_training = ce_summary["training"]["task_rows"]
        row = {
            "task_seed": seed,
            "strictly_unseen_graphs": int(audit["strictly_unseen_graphs"]),
            "evaluated_examples_all_starts": int(result["examples_all_starts"]),
            "state01": base_curve,
            "ce_only": ce_curve,
            "paired_delta": {
                key: float(ce_curve[key] - base_curve[key])
                for key in ("auc_1_24", "auc_25_48", "auc_49_64", "auc_all", "loop64")
            },
            "h3_relative_mse_loop64": {
                "state01": float(
                    audit["trajectory_geometry_on_selected_partition"]["64"]
                    ["base_relative_mse_to_H3"]
                ),
                "ce_only": float(
                    audit["trajectory_geometry_on_selected_partition"]["64"]
                    ["target_relative_mse_to_H3"]
                ),
            },
            "matrix_update": {
                "ce_minus_state01_relative_to_initial_W": float(
                    audit["matrix_delta"]["relative_delta_weight_frobenius_norm"]
                ),
                "base_update_from_initial_relative": vector_norm(base_update)
                / vector_norm(initial_map.weight),
                "ce_update_from_initial_relative": vector_norm(ce_update)
                / vector_norm(initial_map.weight),
                "ce_vs_state01_update_cosine": update_cosine,
                "ce_minus_state01_over_state01_update": vector_norm(
                    ce_update - base_update
                )
                / vector_norm(base_update),
            },
            "training_last_round": {
                "state01_ce": float(base_training[-1]["task_ce"]),
                "ce_only_ce": float(ce_training[-1]["task_ce"]),
                "state01_accuracy": float(base_training[-1]["composition_accuracy"]),
                "ce_only_accuracy": float(ce_training[-1]["composition_accuracy"]),
                "state01_h3_relative_mse": float(
                    base_training[-1]["adjacent_power_relative_mse"]
                ),
                "ce_only_h3_relative_mse": float(
                    ce_training[-1]["adjacent_power_relative_mse"]
                ),
            },
        }
        rows.append(row)

    metric_names = ("auc_1_24", "auc_25_48", "auc_49_64", "auc_all", "loop64")
    aggregate = {}
    for metric in metric_names:
        base_values = [float(row["state01"][metric]) for row in rows]
        ce_values = [float(row["ce_only"][metric]) for row in rows]
        delta_values = [ce - base for base, ce in zip(base_values, ce_values)]
        aggregate[metric] = {
            "state01_mean": mean(base_values),
            "state01_sd": stdev(base_values),
            "ce_only_mean": mean(ce_values),
            "ce_only_sd": stdev(ce_values),
            "paired_delta_mean": mean(delta_values),
            "paired_delta_sd": stdev(delta_values),
            "paired_delta_values": delta_values,
        }
    payload = {
        "status": "complete",
        "comparison": "state loss weight 0.1 versus CE-only weight 0.0",
        "fixed_conditions": {
            "frozen_model_loss": "loop8 final CE plus loop1-7 intermediate CE on p_min(2t,D)",
            "trained_macro_loops": 8,
            "shared_physical_blocks": 2,
            "effective_depth": 16,
            "J": "one shared affine 256x256 matrix plus bias, once per continuation loop",
            "rollout_horizon": 64,
            "rounds": 8,
            "graphs_per_round": 288,
            "learning_rate": 1e-6,
            "time_weighting": "uniform",
            "detach_interval": 0,
            "strict_unseen_graphs_per_seed": 512,
            "all_starts_per_graph": 8,
        },
        "per_seed": rows,
        "aggregate": aggregate,
    }
    (ROOT / "analysis_summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )

    with (ROOT / "paired_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("task_seed", "metric", "state01", "ce_only", "paired_delta"))
        for row in rows:
            for metric in metric_names:
                writer.writerow(
                    (
                        row["task_seed"],
                        metric,
                        row["state01"][metric],
                        row["ce_only"][metric],
                        row["paired_delta"][metric],
                    )
                )

    loops = np.arange(1, 65)
    base_curves = np.asarray([row["state01"]["values"] for row in rows])
    ce_curves = np.asarray([row["ce_only"]["values"] for row in rows])
    delta_curves = ce_curves - base_curves
    figure, axes = plt.subplots(2, 1, figsize=(9.0, 7.2), sharex=True)
    axes[0].plot(loops, base_curves.mean(axis=0), label="state anchor 0.1", color="#1f77b4", linewidth=2)
    axes[0].fill_between(
        loops,
        base_curves.min(axis=0),
        base_curves.max(axis=0),
        color="#1f77b4",
        alpha=0.16,
    )
    axes[0].plot(loops, ce_curves.mean(axis=0), label="CE only", color="#d62728", linewidth=2)
    axes[0].fill_between(
        loops,
        ce_curves.min(axis=0),
        ce_curves.max(axis=0),
        color="#d62728",
        alpha=0.16,
    )
    axes[0].set_ylabel("Non-endpoint accuracy")
    axes[0].set_ylim(0.75, 1.01)
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="lower left")
    for index, seed in enumerate(SEEDS):
        axes[1].plot(loops, delta_curves[index], alpha=0.45, linewidth=1, label=f"seed {seed}")
    axes[1].plot(loops, delta_curves.mean(axis=0), color="black", linewidth=2.2, label="mean delta")
    axes[1].axhline(0.0, color="gray", linewidth=1)
    axes[1].set_xlabel("Continuation loop")
    axes[1].set_ylabel("CE-only minus 0.1")
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="upper left", ncol=2, fontsize=8)
    figure.suptitle("Matched strict-unseen comparison across three task seeds")
    figure.tight_layout()
    figure.savefig(ROOT / "ce_only_vs_state01_strict_unseen.png", dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
