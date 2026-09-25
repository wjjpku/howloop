from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _lifespan(curve: list[float], threshold: float) -> int:
    for index, value in enumerate(curve, start=1):
        if value < threshold:
            return index - 1
    return len(curve)


def _prefix_auc(curve: list[float], horizon: int) -> float:
    return float(np.mean(curve[: min(horizon, len(curve))]))


def _mean_curves(curves: Iterable[list[float]]) -> np.ndarray:
    arrays = [np.asarray(curve, dtype=float) for curve in curves]
    length = min(map(len, arrays))
    return np.stack([array[:length] for array in arrays])


def _lifespan_rows(root: Path) -> list[dict[str, Any]]:
    experiments = (
        (
            "natural",
            0,
            ("horizon16_primary", "horizon16_replica"),
            "no_intervention",
        ),
        (
            "state-MSE DAgger h=4",
            4,
            ("weighted_primary", "weighted_replica"),
            "shared_rw28_r256_round4",
        ),
        (
            "state-MSE DAgger h=8",
            8,
            ("horizon8_primary", "horizon8_replica"),
            "shared_rw28_r256_round4",
        ),
        (
            "state-MSE DAgger h=16",
            16,
            ("horizon16_primary", "horizon16_replica"),
            "shared_rw28_r256_round4",
        ),
        (
            "exact young interface",
            -1,
            ("horizon16_primary", "horizon16_replica"),
            "oracle_interface",
        ),
    )
    rows: list[dict[str, Any]] = []
    for method, train_horizon, directories, condition in experiments:
        for replicate, directory in enumerate(directories, start=1):
            summary = _load(root / directory / "summary.json")
            curve = summary["closed_loop"][condition]["accuracy"]
            rows.append(
                {
                    "method": method,
                    "training_horizon": train_horizon,
                    "replicate": replicate,
                    "evaluation_horizon": len(curve),
                    "auc8": _prefix_auc(curve, 8),
                    "auc16": _prefix_auc(curve, 16),
                    "auc32": _prefix_auc(curve, 32),
                    "auc64": _prefix_auc(curve, 64),
                    "auc128": _prefix_auc(curve, 128),
                    "lifespan_at_0.8": _lifespan(curve, 0.8),
                    "lifespan_at_0.7": _lifespan(curve, 0.7),
                    "lifespan_at_0.5": _lifespan(curve, 0.5),
                    "last_accuracy": curve[-1],
                }
            )
    return rows


def _position_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for replicate, directory in enumerate(
        ("complement_primary", "complement_replica"),
        start=1,
    ):
        curves = _load(root / directory / "summary.json")["curves"]
        for condition, values in curves.items():
            rows.append(
                {
                    "replicate": replicate,
                    "condition": condition,
                    "auc32": values["auc"],
                    "accuracy_cycle32": values["accuracy"][-1],
                    "q_cycle32": values["head2_q_answer_cosine"][-1],
                    "k_cycle32": values[
                        "head2_k_destination_cosine"
                    ][-1],
                    "v_cycle32": values[
                        "head2_v_destination_cosine"
                    ][-1],
                    "context_cycle32": values[
                        "head2_context_answer_cosine"
                    ][-1],
                    "mlp_cycle32": values[
                        "block2_mlp_answer_cosine"
                    ][-1],
                }
            )
    return rows


def _map_ablation_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for replicate, directory in enumerate(
        ("ablation_primary", "ablation_replica"),
        start=1,
    ):
        summary_path = root / directory / "summary.json"
        if not summary_path.exists():
            continue
        curves = _load(summary_path)["closed_loop"]
        for condition, values in curves.items():
            curve = values["accuracy"]
            rows.append(
                {
                    "replicate": replicate,
                    "condition": condition,
                    "auc32": _prefix_auc(curve, 32),
                    "auc64": _prefix_auc(curve, 64),
                    "lifespan_at_0.8": _lifespan(curve, 0.8),
                    "lifespan_at_0.7": _lifespan(curve, 0.7),
                    "lifespan_at_0.5": _lifespan(curve, 0.5),
                    "accuracy_cycle64": curve[-1],
                }
            )
    return rows


def _power_rows(root: Path) -> list[dict[str, Any]]:
    experiments = (
        (
            "DAgger h=8 map",
            ("power_primary", "power_replica"),
        ),
        (
            "pooled adjacent-age map",
            ("adjacent_power_primary", "adjacent_power_replica"),
        ),
    )
    rows: list[dict[str, Any]] = []
    for method, directories in experiments:
        for replicate, directory in enumerate(directories, start=1):
            summary = _load(root / directory / "summary.json")
            for condition, ages in summary["aggregate"].items():
                for age, values in ages.items():
                    rows.append(
                        {
                            "method": method,
                            "replicate": replicate,
                            "condition": condition,
                            "age": int(age),
                            "applications": int(age) - 2,
                            "accuracy": values["accuracy"],
                            "interface_relative_mse": values[
                                "interface_relative_mse"
                            ],
                            "answer_relative_mse": values[
                                "answer_relative_mse"
                            ],
                            "q_cosine": values[
                                "head2_q_answer_cosine"
                            ],
                            "k_cosine": values[
                                "head2_k_destination_cosine"
                            ],
                            "context_cosine": values[
                                "head2_context_answer_cosine"
                            ],
                        }
                    )
    return rows


def _component_patch_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for replicate, directory in enumerate(
        ("component_long_primary", "component_long_replica"),
        start=1,
    ):
        summary = _load(root / directory / "summary.json")
        for condition, cycles in summary["aggregate"].items():
            for cycle, values in cycles.items():
                rows.append(
                    {
                        "replicate": replicate,
                        "condition": condition,
                        "cycle": int(cycle),
                        "accuracy": values["accuracy"],
                        "head2_correct_destination_mass": values[
                            "head2_correct_destination_mass"
                        ],
                        "q_cosine": values["head2_q_answer_cosine"],
                        "k_cosine": values[
                            "head2_k_destination_cosine"
                        ],
                        "v_cosine": values[
                            "head2_v_destination_cosine"
                        ],
                        "context_cosine": values[
                            "head2_context_answer_cosine"
                        ],
                        "mlp_cosine": values[
                            "block2_mlp_answer_cosine"
                        ],
                    }
                )
    return rows


def _map_geometry(root: Path) -> dict[str, Any]:
    maps: list[tuple[torch.Tensor, torch.Tensor]] = []
    rows: list[dict[str, Any]] = []
    identity = torch.eye(256)
    for replicate, directory in enumerate(
        ("horizon8_primary", "horizon8_replica"),
        start=1,
    ):
        payload = torch.load(
            root / directory / "shared_position_maps.pt",
            map_location="cpu",
            weights_only=False,
        )
        item = payload["maps"]["w28_r256_round4"]
        weight = item["weight"].float()
        bias = item["bias"].float()
        update = weight - identity
        symmetric = (update + update.transpose(0, 1)) / 2
        skew = (update - update.transpose(0, 1)) / 2
        singular_values = torch.linalg.svdvals(weight)
        maps.append((update, bias))
        rows.append(
            {
                "replicate": replicate,
                "update_to_identity_frobenius": float(
                    torch.linalg.norm(update)
                    / torch.linalg.norm(identity)
                ),
                "symmetric_update_energy_fraction": float(
                    torch.linalg.norm(symmetric).square()
                    / torch.linalg.norm(update).square()
                ),
                "skew_update_energy_fraction": float(
                    torch.linalg.norm(skew).square()
                    / torch.linalg.norm(update).square()
                ),
                "orthogonality_defect": float(
                    torch.linalg.norm(
                        weight.transpose(0, 1) @ weight - identity
                    )
                    / torch.linalg.norm(identity)
                ),
                "bias_l2": float(torch.linalg.norm(bias)),
                "min_singular": float(singular_values.min()),
                "max_singular": float(singular_values.max()),
                "spectral_radius": float(
                    torch.linalg.eigvals(weight).abs().max()
                ),
            }
        )
    update_cosine = float(
        torch.nn.functional.cosine_similarity(
            maps[0][0].flatten(),
            maps[1][0].flatten(),
            dim=0,
        )
    )
    bias_cosine = float(
        torch.nn.functional.cosine_similarity(
            maps[0][1],
            maps[1][1],
            dim=0,
        )
    )
    return {
        "rows": rows,
        "cross_replication_update_cosine": update_cosine,
        "cross_replication_bias_cosine": bias_cosine,
    }


def _role_action_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for replicate, directory in enumerate(
        ("role_primary", "role_replica"),
        start=1,
    ):
        summary = _load(root / directory / "summary.json")
        for role, cycles in summary["aggregate"].items():
            for cycle, values in cycles.items():
                rows.append(
                    {
                        "replicate": replicate,
                        "role": role,
                        "cycle": int(cycle),
                        **values,
                    }
                )
    return rows


def _rank_sweep_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for replicate, directory in enumerate(
        ("rank_primary", "rank_replica"),
        start=1,
    ):
        summary = _load(root / directory / "summary.json")
        artifact = torch.load(
            root / directory / "shared_position_maps.pt",
            map_location="cpu",
            weights_only=False,
        )
        for rank in (8, 16, 32, 64, 128, 256):
            label = f"w28_r{rank}_round4"
            curve = summary["closed_loop"][f"shared_r{label}"]["accuracy"]
            training = next(
                row
                for row in summary["training_rows"]
                if int(row["rank"]) == rank and int(row["round"]) == 4
            )
            item = artifact["maps"][label]
            parameter_count = (
                256 * 256 + 256
                if rank >= 256
                else 2 * 256 * rank + 256
            )
            rows.append(
                {
                    "replicate": replicate,
                    "rank": rank,
                    "factor_parameter_count": parameter_count,
                    "retained_fitted_update_energy": item[
                        "retained_fit_energy"
                    ],
                    "heldout_natural_relative_mse": training[
                        "heldout_natural_relative_mse"
                    ],
                    "heldout_answer_relative_mse": training[
                        "heldout_answer_relative_mse"
                    ],
                    "auc32": _prefix_auc(curve, 32),
                    "auc64": _prefix_auc(curve, 64),
                    "lifespan_at_0.8": _lifespan(curve, 0.8),
                    "lifespan_at_0.7": _lifespan(curve, 0.7),
                    "lifespan_at_0.5": _lifespan(curve, 0.5),
                }
            )
    return rows


def _initialization_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for replicate, directory in enumerate(
        ("initialization_primary", "initialization_replica"),
        start=1,
    ):
        summary = _load(root / directory / "summary.json")
        for condition, values in summary["closed_loop"].items():
            curve = values["accuracy"]
            for cycle, accuracy in enumerate(curve, start=1):
                rows.append(
                    {
                        "replicate": replicate,
                        "condition": condition,
                        "cycle": cycle,
                        "accuracy": accuracy,
                    }
                )
    return rows


def _learned_initializer_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for replicate, directory in enumerate(
        ("learned_initializer_primary", "learned_initializer_replica"),
        start=1,
    ):
        summary = _load(root / directory / "summary.json")
        for condition, values in summary["closed_loop"].items():
            for cycle, accuracy in enumerate(values["accuracy"], start=1):
                rows.append(
                    {
                        "replicate": replicate,
                        "condition": condition,
                        "cycle": cycle,
                        "accuracy": accuracy,
                    }
                )
    return rows


def _initializer_geometry(root: Path) -> dict[str, Any]:
    identity = torch.eye(256)
    rows: list[dict[str, Any]] = []
    initializers: list[tuple[torch.Tensor, torch.Tensor]] = []
    for replicate, (initializer_dir, feedback_dir) in enumerate(
        (
            ("learned_initializer_primary", "horizon8_primary"),
            ("learned_initializer_replica", "horizon8_replica"),
        ),
        start=1,
    ):
        initializer_item = torch.load(
            root / initializer_dir / "learned_initializer.pt",
            map_location="cpu",
            weights_only=False,
        )["map"]
        feedback_item = torch.load(
            root / feedback_dir / "shared_position_maps.pt",
            map_location="cpu",
            weights_only=False,
        )["maps"]["w28_r256_round4"]
        init_update = initializer_item["weight"].float() - identity
        feedback_update = feedback_item["weight"].float() - identity
        init_bias = initializer_item["bias"].float()
        feedback_bias = feedback_item["bias"].float()
        initializers.append((init_update, init_bias))
        rows.append(
            {
                "replicate": replicate,
                "initializer_to_feedback_update_cosine": float(
                    torch.nn.functional.cosine_similarity(
                        init_update.flatten(),
                        feedback_update.flatten(),
                        dim=0,
                    )
                ),
                "initializer_to_feedback_bias_cosine": float(
                    torch.nn.functional.cosine_similarity(
                        init_bias,
                        feedback_bias,
                        dim=0,
                    )
                ),
                "initializer_update_to_identity_frobenius": float(
                    torch.linalg.norm(init_update)
                    / torch.linalg.norm(identity)
                ),
                "feedback_update_to_identity_frobenius": float(
                    torch.linalg.norm(feedback_update)
                    / torch.linalg.norm(identity)
                ),
                "initializer_bias_l2": float(torch.linalg.norm(init_bias)),
                "feedback_bias_l2": float(torch.linalg.norm(feedback_bias)),
            }
        )
    return {
        "rows": rows,
        "cross_replication_initializer_update_cosine": float(
            torch.nn.functional.cosine_similarity(
                initializers[0][0].flatten(),
                initializers[1][0].flatten(),
                dim=0,
            )
        ),
        "cross_replication_initializer_bias_cosine": float(
            torch.nn.functional.cosine_similarity(
                initializers[0][1],
                initializers[1][1],
                dim=0,
            )
        ),
    }


def _joint_single_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for replicate, directory in enumerate(
        ("joint_single_primary", "joint_single_replica"),
        start=1,
    ):
        summary = _load(root / directory / "summary.json")
        for condition, values in summary["closed_loop"].items():
            for cycle, accuracy in enumerate(values["accuracy"], start=1):
                rows.append(
                    {
                        "replicate": replicate,
                        "condition": condition,
                        "cycle": cycle,
                        "accuracy": accuracy,
                    }
                )
    return rows


def _joint_weight_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for weight in (1, 2, 4):
        for replicate, suffix in enumerate(
            ("primary", "replica"),
            start=1,
        ):
            summary = _load(
                root / f"joint_weight{weight}_{suffix}" / "summary.json"
            )
            curve = summary["closed_loop"]["joint_single_map"]["accuracy"]
            rows.append(
                {
                    "initializer_repeat": weight,
                    "replicate": replicate,
                    "first_cycle_accuracy": curve[0],
                    "second_cycle_accuracy": curve[1],
                    "auc8": _prefix_auc(curve, 8),
                    "auc32": _prefix_auc(curve, 32),
                    "auc64": _prefix_auc(curve, 64),
                    "lifespan_at_0.8": _lifespan(curve, 0.8),
                    "lifespan_at_0.5": _lifespan(curve, 0.5),
                    "initializer_relative_mse": summary["training_rows"][
                        -1
                    ]["initializer_relative_mse"],
                    "local_relative_mse": summary["training_rows"][-1][
                        "local_relative_mse"
                    ],
                }
            )
    return rows


def _answer_weight_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for replicate, suffix in enumerate(("primary", "replica"), start=1):
        summary_path = (
            root / f"answer_weight_h8_{suffix}" / "summary.json"
        )
        if not summary_path.exists():
            continue
        summary = _load(summary_path)
        for answer_weight in (14, 28, 42, 56):
            condition = f"shared_rw{answer_weight}_r256_round4"
            curve = summary["closed_loop"][condition]["accuracy"]
            training = next(
                row
                for row in summary["training_rows"]
                if int(row["answer_repeat"]) == answer_weight
                and int(row["rank"]) == 256
                and int(row["round"]) == 4
            )
            rows.append(
                {
                    "answer_weight": answer_weight,
                    "replicate": replicate,
                    "auc8": _prefix_auc(curve, 8),
                    "auc16": _prefix_auc(curve, 16),
                    "auc32": _prefix_auc(curve, 32),
                    "auc64": _prefix_auc(curve, 64),
                    "lifespan_at_0.8": _lifespan(curve, 0.8),
                    "lifespan_at_0.5": _lifespan(curve, 0.5),
                    "accuracy_cycle64": curve[-1],
                    "heldout_natural_relative_mse": training[
                        "heldout_natural_relative_mse"
                    ],
                    "heldout_answer_relative_mse": training[
                        "heldout_answer_relative_mse"
                    ],
                }
            )
    return rows


def _map_strength_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    directories = (
        (1, "coarse", "map_strength_primary"),
        (2, "coarse", "map_strength_replica"),
        (1, "fine", "map_strength_fine_primary"),
        (2, "fine", "map_strength_fine_replica"),
    )
    for replicate, scan, directory in directories:
        summary_path = root / directory / "summary.json"
        if not summary_path.exists():
            continue
        summary = _load(summary_path)
        curves = summary["closed_loop"]
        for strength in summary["strengths"]:
            strength = float(strength)
            label = (
                f"{strength:g}".replace("-", "neg").replace(".", "p")
            )
            condition = f"map_strength_{label}"
            values = curves[condition]
            curve = values["accuracy"]
            rows.append(
                {
                    "strength": strength,
                    "replicate": replicate,
                    "scan": scan,
                    "source_directory": directory,
                    "condition": condition,
                    "auc8": _prefix_auc(curve, 8),
                    "auc16": _prefix_auc(curve, 16),
                    "auc32": _prefix_auc(curve, 32),
                    "auc64": _prefix_auc(curve, 64),
                    "lifespan_at_0.8": _lifespan(curve, 0.8),
                    "lifespan_at_0.5": _lifespan(curve, 0.5),
                    "accuracy_cycle64": curve[-1],
                    "q_cosine_cycle64": values[
                        "head2_q_answer_cosine"
                    ][-1],
                    "k_cosine_cycle64": values[
                        "head2_k_destination_cosine"
                    ][-1],
                    "v_cosine_cycle64": values[
                        "head2_v_destination_cosine"
                    ][-1],
                    "context_cosine_cycle64": values[
                        "head2_context_answer_cosine"
                    ][-1],
                    "mlp_cosine_cycle64": values[
                        "block2_mlp_answer_cosine"
                    ][-1],
                }
            )
    return rows


def _probe_dose_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for replicate, suffix in enumerate(("primary", "replica"), start=1):
        path = root / f"probe_dose_{suffix}" / "probe_dose_summary.csv"
        if not path.exists():
            continue
        for source in _read_csv(path):
            row: dict[str, Any] = dict(source)
            row["replicate"] = replicate
            rows.append(row)
    return rows


def _periodic_booster_data(root: Path) -> dict[str, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    geometry: list[dict[str, Any]] = []
    identity = torch.eye(256)
    specifications = (
        (1, "primary", "late32", 32, "periodic_booster_primary"),
        (2, "replica", "late32", 32, "periodic_booster_replica"),
        (1, "primary", "early24", 24, "periodic_booster24_primary"),
        (2, "replica", "early24", 24, "periodic_booster24_replica"),
        (
            1,
            "primary",
            "early24_dagger",
            24,
            "periodic_booster24_dagger_primary",
        ),
        (
            2,
            "replica",
            "early24_dagger",
            24,
            "periodic_booster24_dagger_replica",
        ),
    )
    for replicate, suffix, training, reset_cycle, directory_name in specifications:
        directory = root / directory_name
        summary_path = directory / "summary.json"
        if not summary_path.exists():
            continue
        summary = _load(summary_path)
        for condition, values in summary["closed_loop"].items():
            curve = [float(value) for value in values["accuracy"]]
            rows.append(
                {
                    "replicate": replicate,
                    "training": training,
                    "reset_cycle": reset_cycle,
                    "condition": condition,
                    "auc32": _prefix_auc(curve, 32),
                    "auc64": _prefix_auc(curve, 64),
                    "auc128": _prefix_auc(curve, 128),
                    "lifespan_at_0.8": _lifespan(curve, 0.8),
                    "lifespan_at_0.5": _lifespan(curve, 0.5),
                    "accuracy_cycle32": curve[31],
                    "accuracy_cycle64": curve[63],
                    "accuracy_cycle96": curve[95],
                    "accuracy_cycle128": curve[127],
                    "heldout_relative_mse": summary["heldout"][
                        "relative_mse"
                    ],
                }
            )
        booster = torch.load(
            directory / "periodic_booster.pt",
            map_location="cpu",
            weights_only=False,
        )["map"]
        feedback = torch.load(
            root
            / f"horizon8_{suffix}"
            / "shared_position_maps.pt",
            map_location="cpu",
            weights_only=False,
        )["maps"]["w28_r256_round4"]
        initializer = torch.load(
            root
            / f"learned_initializer_{suffix}"
            / "learned_initializer.pt",
            map_location="cpu",
            weights_only=False,
        )["map"]
        maps = {
            "feedback": feedback,
            "h8_initializer": initializer,
            "late_booster": booster,
        }
        updates = {
            name: values["weight"].float() - identity
            for name, values in maps.items()
        }
        biases = {
            name: values["bias"].float()
            for name, values in maps.items()
        }
        for left, right in (
            ("late_booster", "feedback"),
            ("late_booster", "h8_initializer"),
            ("feedback", "h8_initializer"),
        ):
            geometry.append(
                {
                    "replicate": replicate,
                    "training": training,
                    "reset_cycle": reset_cycle,
                    "left": left,
                    "right": right,
                    "update_cosine": float(
                        torch.nn.functional.cosine_similarity(
                            updates[left].flatten(),
                            updates[right].flatten(),
                            dim=0,
                        )
                    ),
                    "bias_cosine": float(
                        torch.nn.functional.cosine_similarity(
                            biases[left],
                            biases[right],
                            dim=0,
                        )
                    ),
                    "left_update_norm": float(
                        torch.linalg.norm(updates[left])
                    ),
                    "right_update_norm": float(
                        torch.linalg.norm(updates[right])
                    ),
                    "left_bias_norm": float(
                        torch.linalg.norm(biases[left])
                    ),
                    "right_bias_norm": float(
                        torch.linalg.norm(biases[right])
                    ),
                }
            )
    return {"summary": rows, "geometry": geometry}


def _reset_age_cross_data(
    root: Path,
) -> dict[str, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    geometry: list[dict[str, Any]] = []
    paired: list[dict[str, Any]] = []
    trend: list[dict[str, Any]] = []
    for replicate, suffix in enumerate(("primary", "replica"), start=1):
        directory = root / f"reset_age_cross_{suffix}"
        summary_path = directory / "summary.json"
        table_path = directory / "reset_age_cross_summary.csv"
        if not summary_path.exists() or not table_path.exists():
            continue
        for source in _read_csv(table_path):
            row: dict[str, Any] = {"replicate": replicate, **source}
            rows.append(row)
        geometry.append(
            {
                "replicate": replicate,
                **_load(summary_path)["map_geometry"],
            }
        )
        trend_path = directory / "reset_age_trend_summary.csv"
        if trend_path.exists():
            trend.extend(
                {
                    "replicate": replicate,
                    **row,
                }
                for row in _read_csv(trend_path)
            )
        raw_path = directory / "reset_age_cross_rows.csv"
        if not raw_path.exists():
            continue
        raw = _read_csv(raw_path)
        batches = sorted({int(row["batch"]) for row in raw})
        cycles = sorted({int(row["cycle"]) for row in raw})

        def batch_value(
            *,
            batch: int,
            cycle: int,
            condition: str,
        ) -> float:
            matches = [
                float(row["accuracy"])
                for row in raw
                if int(row["batch"]) == batch
                and int(row["cycle"]) == cycle
                and row["condition"] == condition
            ]
            if len(matches) != 1:
                raise ValueError("expected one cross-age row per batch")
            return matches[0]

        comparisons = (
            (
                "reset24_minus_feedback",
                "booster_trained_cycle24",
                "feedback",
            ),
            (
                "reset32_minus_feedback",
                "booster_trained_cycle32",
                "feedback",
            ),
            (
                "reset32_minus_reset24",
                "booster_trained_cycle32",
                "booster_trained_cycle24",
            ),
            (
                "age_linear_minus_feedback",
                "linear_age_extrapolation",
                "feedback",
            ),
            (
                "age_linear_minus_reset32",
                "linear_age_extrapolation",
                "booster_trained_cycle32",
            ),
        )
        for cycle in cycles:
            for index, (label, treatment, control) in enumerate(comparisons):
                differences = np.asarray(
                    [
                        batch_value(
                            batch=batch,
                            cycle=cycle,
                            condition=treatment,
                        )
                        - batch_value(
                            batch=batch,
                            cycle=cycle,
                            condition=control,
                        )
                        for batch in batches
                    ],
                    dtype=float,
                )
                rng = np.random.default_rng(
                    166000 + 1000 * replicate + 10 * cycle + index
                )
                samples = rng.choice(
                    differences,
                    size=(20000, len(differences)),
                    replace=True,
                ).mean(axis=1)
                paired.append(
                    {
                        "replicate": replicate,
                        "cycle": cycle,
                        "comparison": label,
                        "paired_batches": len(differences),
                        "positive_batch_clusters": int(
                            np.sum(differences > 0)
                        ),
                        "mean_accuracy_difference": float(
                            differences.mean()
                        ),
                        "bootstrap_ci95_low": float(
                            np.quantile(samples, 0.025)
                        ),
                        "bootstrap_ci95_high": float(
                            np.quantile(samples, 0.975)
                        ),
                    }
                )
    return {
        "summary": rows,
        "geometry": geometry,
        "paired": paired,
        "trend": trend,
    }


def _exact_renewal_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for replicate, suffix in enumerate(("primary", "replica"), start=1):
        directory = root / f"exact_renewal_scan_{suffix}"
        path = directory / "summary.json"
        if not path.exists():
            continue
        summary = _load(path)
        raw = _read_csv(directory / "exact_renewal_rows.csv")
        for condition, values in summary["closed_loop"].items():
            curve = [float(value) for value in values["accuracy"]]
            worst_cycle = int(np.argmin(curve)) + 1
            parts = [
                row
                for row in raw
                if row["condition"] == condition
                and int(row["cycle"]) == worst_cycle
            ]
            worst_count = sum(
                int(float(row["valid_count"])) for row in parts
            )
            worst_correct = round(
                sum(
                    float(row["accuracy"])
                    * int(float(row["valid_count"]))
                    for row in parts
                )
            )
            proportion = worst_correct / worst_count
            z = 1.959963984540054
            denominator = 1 + z**2 / worst_count
            center = (
                proportion + z**2 / (2 * worst_count)
            ) / denominator
            half_width = (
                z
                * np.sqrt(
                    proportion * (1 - proportion) / worst_count
                    + z**2 / (4 * worst_count**2)
                )
                / denominator
            )
            period = (
                int(condition.rsplit("_", 1)[1])
                if condition.startswith("exact_period_")
                else None
            )
            rows.append(
                {
                    "replicate": replicate,
                    "condition": condition,
                    "period": period,
                    "auc128": _prefix_auc(curve, 128),
                    "minimum_accuracy": min(curve),
                    "worst_cycle": worst_cycle,
                    "worst_valid_count": worst_count,
                    "worst_accuracy_wilson_low": center - half_width,
                    "worst_accuracy_wilson_high": center + half_width,
                    "lifespan_at_0.8": _lifespan(curve, 0.8),
                    "lifespan_at_0.5": _lifespan(curve, 0.5),
                    "accuracy_cycle128": curve[127],
                }
            )
    return rows


def _periodic_booster_component_rows(
    root: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    metrics = (
        "accuracy",
        "head2_q_answer_cosine",
        "head2_k_destination_cosine",
        "head2_v_destination_cosine",
        "head2_context_answer_cosine",
        "block2_mlp_answer_cosine",
    )
    conditions = (
        "feedback_only",
        "learned_booster_periodic",
        "exact_interface_periodic",
    )
    for replicate, suffix in enumerate(("primary", "replica"), start=1):
        path = (
            root
            / f"periodic_booster24_{suffix}"
            / "closed_loop_rows.csv"
        )
        if not path.exists():
            continue
        source = _read_csv(path)
        for cycle in (24, 48, 72, 96, 120):
            for condition in conditions:
                parts = [
                    row
                    for row in source
                    if int(row["cycle"]) == cycle
                    and row["condition"] == condition
                ]
                if not parts:
                    continue
                total = sum(float(row["valid_count"]) for row in parts)
                result: dict[str, Any] = {
                    "replicate": replicate,
                    "cycle": cycle,
                    "condition": condition,
                    "valid_count": total,
                }
                for metric in metrics:
                    result[metric] = sum(
                        float(row[metric]) * float(row["valid_count"])
                        for row in parts
                    ) / total
                rows.append(result)
    return rows


def _role_booster_data(root: Path) -> dict[str, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    heldout_rows: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []
    metrics = (
        "accuracy",
        "head2_q_answer_cosine",
        "head2_k_destination_cosine",
        "head2_v_destination_cosine",
        "head2_context_answer_cosine",
        "block2_mlp_answer_cosine",
    )
    specifications = (
        (1, "small", "role_booster24_primary"),
        (2, "small", "role_booster24_replica"),
        (1, "large", "role_booster24_large_primary"),
        (2, "large", "role_booster24_large_replica"),
        (1, "rank25", "role_booster24_rank25_primary"),
        (2, "rank25", "role_booster24_rank25_replica"),
    )
    for replicate, training, directory_name in specifications:
        directory = root / directory_name
        summary_path = directory / "summary.json"
        if not summary_path.exists():
            continue
        summary = _load(summary_path)
        for condition, values in summary["closed_loop"].items():
            curve = [float(value) for value in values["accuracy"]]
            rows.append(
                {
                    "replicate": replicate,
                    "training": training,
                    "calibration_graphs": summary["booster"]["graphs"],
                    "role_rank": summary["booster"].get(
                        "role_update_rank",
                        256,
                    ),
                    "role_parameter_count": summary["booster"][
                        "parameter_count"
                    ],
                    "matched_shared_parameter_count": summary["booster"].get(
                        "matched_shared_parameter_count",
                        65792,
                    ),
                    "condition": condition,
                    "auc32": _prefix_auc(curve, 32),
                    "auc64": _prefix_auc(curve, 64),
                    "auc128": _prefix_auc(curve, 128),
                    "lifespan_at_0.8": _lifespan(curve, 0.8),
                    "lifespan_at_0.5": _lifespan(curve, 0.5),
                    "accuracy_cycle24": curve[23],
                    "accuracy_cycle48": curve[47],
                    "accuracy_cycle72": curve[71],
                    "accuracy_cycle96": curve[95],
                    "accuracy_cycle120": curve[119],
                }
            )
        for role, values in summary["heldout"].items():
            heldout_rows.append(
                {
                    "replicate": replicate,
                    "training": training,
                    "calibration_graphs": summary["booster"]["graphs"],
                    "role_rank": summary["booster"].get(
                        "role_update_rank",
                        256,
                    ),
                    "role": role,
                    **values,
                }
            )
        raw = _read_csv(directory / "closed_loop_rows.csv")
        for cycle in (24, 48, 72, 96, 120):
            for condition in (
                "feedback_only",
                "matched_shared_booster_periodic",
                "learned_booster_periodic",
                "exact_interface_periodic",
            ):
                parts = [
                    row
                    for row in raw
                    if int(row["cycle"]) == cycle
                    and row["condition"] == condition
                ]
                if not parts:
                    continue
                total = sum(float(row["valid_count"]) for row in parts)
                result: dict[str, Any] = {
                    "replicate": replicate,
                    "training": training,
                    "cycle": cycle,
                    "condition": condition,
                    "valid_count": total,
                }
                for metric in metrics:
                    result[metric] = sum(
                        float(row[metric]) * float(row["valid_count"])
                        for row in parts
                    ) / total
                component_rows.append(result)
    return {
        "summary": rows,
        "heldout": heldout_rows,
        "components": component_rows,
    }


def _role_booster_paired_effect(root: Path) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for training in ("rank25", "large"):
        for replicate, suffix in enumerate(
            ("primary", "replica"),
            start=1,
        ):
            path = (
                root
                / f"role_booster24_{training}_{suffix}"
                / "closed_loop_rows.csv"
            )
            if not path.exists():
                continue
            rows = _read_csv(path)
            differences = []
            for batch in sorted({int(row["batch"]) for row in rows}):
                def batch_auc(condition: str) -> float:
                    values = [
                        float(row["accuracy"])
                        for row in rows
                        if int(row["batch"]) == batch
                        and row["condition"] == condition
                    ]
                    return float(np.mean(values))

                differences.append(
                    batch_auc("learned_booster_periodic")
                    - batch_auc("matched_shared_booster_periodic")
                )
            rng = np.random.default_rng(150000 + 10 * replicate)
            values = np.asarray(differences)
            samples = rng.choice(
                values,
                size=(20000, len(values)),
                replace=True,
            ).mean(axis=1)
            results.append(
                {
                    "training": training,
                    "replicate": replicate,
                    "batch_clusters": len(values),
                    "positive_batch_clusters": int((values > 0).sum()),
                    "auc128_difference": float(values.mean()),
                    "batch_difference_min": float(values.min()),
                    "batch_difference_max": float(values.max()),
                    "cluster_bootstrap_95_low": float(
                        np.quantile(samples, 0.025)
                    ),
                    "cluster_bootstrap_95_high": float(
                        np.quantile(samples, 0.975)
                    ),
                }
            )
    return results


def _role_hybrid_data(root: Path) -> dict[str, list[dict[str, Any]]]:
    summary_rows: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    metrics = (
        "accuracy",
        "head2_q_answer_cosine",
        "head2_k_destination_cosine",
        "head2_v_destination_cosine",
        "head2_context_answer_cosine",
        "block2_mlp_answer_cosine",
    )
    for replicate, suffix in enumerate(("primary", "replica"), start=1):
        directory = root / f"role_hybrid24_{suffix}"
        summary_path = directory / "summary.json"
        rows_path = directory / "role_hybrid_rows.csv"
        if not summary_path.exists() or not rows_path.exists():
            continue
        summary = _load(summary_path)
        for condition, values in summary["closed_loop"].items():
            curve = [float(value) for value in values["accuracy"]]
            summary_rows.append(
                {
                    "replicate": replicate,
                    "condition": condition,
                    "auc32": _prefix_auc(curve, 32),
                    "auc64": _prefix_auc(curve, 64),
                    "auc128": _prefix_auc(curve, 128),
                    "lifespan_at_0.8": _lifespan(curve, 0.8),
                    "lifespan_at_0.5": _lifespan(curve, 0.5),
                    "accuracy_cycle24": curve[23],
                    "accuracy_cycle48": curve[47],
                    "accuracy_cycle72": curve[71],
                    "accuracy_cycle96": curve[95],
                    "accuracy_cycle120": curve[119],
                }
            )
        raw = _read_csv(rows_path)
        for cycle in (24, 48, 72, 96, 120):
            for condition in summary["closed_loop"]:
                parts = [
                    row
                    for row in raw
                    if int(row["cycle"]) == cycle
                    and row["condition"] == condition
                ]
                if not parts:
                    continue
                total = sum(float(row["valid_count"]) for row in parts)
                result: dict[str, Any] = {
                    "replicate": replicate,
                    "cycle": cycle,
                    "condition": condition,
                    "valid_count": total,
                }
                for metric in metrics:
                    result[metric] = sum(
                        float(row[metric]) * float(row["valid_count"])
                        for row in parts
                    ) / total
                component_rows.append(result)

        batches = sorted({int(row["batch"]) for row in raw})

        def batch_auc(batch: int, condition: str) -> float:
            values = [
                float(row["accuracy"])
                for row in raw
                if int(row["batch"]) == batch
                and row["condition"] == condition
            ]
            return float(np.mean(values))

        for condition in summary["closed_loop"]:
            if condition in {
                "shared_all",
                "feedback_only",
                "exact_interface",
            }:
                continue
            differences = np.asarray(
                [
                    batch_auc(batch, condition)
                    - batch_auc(batch, "shared_all")
                    for batch in batches
                ],
                dtype=float,
            )
            rng = np.random.default_rng(
                158000 + 100 * replicate + len(paired_rows)
            )
            samples = rng.choice(
                differences,
                size=(20000, len(differences)),
                replace=True,
            ).mean(axis=1)
            paired_rows.append(
                {
                    "replicate": replicate,
                    "condition": condition,
                    "paired_batches": len(differences),
                    "mean_auc128_difference_vs_shared": float(
                        differences.mean()
                    ),
                    "bootstrap_ci95_low": float(
                        np.quantile(samples, 0.025)
                    ),
                    "bootstrap_ci95_high": float(
                        np.quantile(samples, 0.975)
                    ),
                    "positive_batch_clusters": int(
                        np.sum(differences > 0)
                    ),
                }
            )
    return {
        "summary": summary_rows,
        "components": component_rows,
        "paired_effect": paired_rows,
    }


def _factorial_shapley(values: dict[int, float], role_count: int) -> list[float]:
    denominator = math.factorial(role_count)
    contributions = []
    full_mask = (1 << role_count) - 1
    for role_index in range(role_count):
        role_bit = 1 << role_index
        contribution = 0.0
        for mask in range(full_mask + 1):
            if mask & role_bit:
                continue
            size = mask.bit_count()
            weight = (
                math.factorial(size)
                * math.factorial(role_count - size - 1)
                / denominator
            )
            contribution += weight * (
                values[mask | role_bit] - values[mask]
            )
        contributions.append(float(contribution))
    return contributions


def _role_factorial_data(root: Path) -> dict[str, list[dict[str, Any]]]:
    role_names = (
        "edge_marker",
        "source",
        "destination",
        "query_metadata",
        "answer",
    )
    summary_rows: list[dict[str, Any]] = []
    shapley_rows: list[dict[str, Any]] = []
    interaction_rows: list[dict[str, Any]] = []
    specifications = (
        ("rank25", "role_factorial24"),
        ("full", "role_factorial24_full"),
    )
    runs = [
        (training, prefix, replicate, suffix)
        for training, prefix in specifications
        for replicate, suffix in enumerate(("primary", "replica"), start=1)
    ]
    for training, prefix, replicate, suffix in runs:
        directory = root / f"{prefix}_{suffix}"
        summary_path = directory / "summary.json"
        rows_path = directory / "role_hybrid_rows.csv"
        if not summary_path.exists() or not rows_path.exists():
            continue
        summary = _load(summary_path)
        values: dict[int, float] = {}
        for mask in range(1 << len(role_names)):
            condition = f"subset_{mask:02d}"
            curve = [
                float(value)
                for value in summary["closed_loop"][condition]["accuracy"]
            ]
            auc64 = _prefix_auc(curve, 64)
            values[mask] = auc64
            summary_rows.append(
                {
                    "training": training,
                    "replicate": replicate,
                    "mask": mask,
                    "roles": "+".join(
                        role
                        for index, role in enumerate(role_names)
                        if mask & (1 << index)
                    )
                    or "shared_all",
                    "auc64": auc64,
                    "lifespan_at_0.8": _lifespan(curve, 0.8),
                    "lifespan_at_0.5": _lifespan(curve, 0.5),
                    "accuracy_cycle24": curve[23],
                    "accuracy_cycle48": curve[47],
                    "accuracy_cycle64": curve[63],
                }
            )

        raw = _read_csv(rows_path)
        batches = sorted({int(row["batch"]) for row in raw})
        batch_shapley: list[list[float]] = []
        for batch in batches:
            batch_values = {
                mask: float(
                    np.mean(
                        [
                            float(row["accuracy"])
                            for row in raw
                            if int(row["batch"]) == batch
                            and row["condition"] == f"subset_{mask:02d}"
                        ]
                    )
                )
                for mask in range(1 << len(role_names))
            }
            batch_shapley.append(
                _factorial_shapley(batch_values, len(role_names))
            )
        batch_array = np.asarray(batch_shapley, dtype=float)
        aggregate_shapley = _factorial_shapley(values, len(role_names))
        for role_index, role in enumerate(role_names):
            rng = np.random.default_rng(
                160000 + 100 * replicate + role_index
                + (1000 if training == "full" else 0)
            )
            samples = rng.choice(
                batch_array[:, role_index],
                size=(20000, len(batches)),
                replace=True,
            ).mean(axis=1)
            shapley_rows.append(
                {
                    "training": training,
                    "replicate": replicate,
                    "role": role,
                    "aggregate_auc64_shapley": aggregate_shapley[
                        role_index
                    ],
                    "batch_mean_auc64_shapley": float(
                        batch_array[:, role_index].mean()
                    ),
                    "bootstrap_ci95_low": float(
                        np.quantile(samples, 0.025)
                    ),
                    "bootstrap_ci95_high": float(
                        np.quantile(samples, 0.975)
                    ),
                    "positive_batch_clusters": int(
                        np.sum(batch_array[:, role_index] > 0)
                    ),
                    "paired_batches": len(batches),
                }
            )

        for first in range(len(role_names)):
            for second in range(first + 1, len(role_names)):
                first_bit = 1 << first
                second_bit = 1 << second
                effects = []
                for mask in range(1 << len(role_names)):
                    if mask & (first_bit | second_bit):
                        continue
                    effects.append(
                        values[mask | first_bit | second_bit]
                        - values[mask | first_bit]
                        - values[mask | second_bit]
                        + values[mask]
                    )
                interaction_rows.append(
                    {
                        "training": training,
                        "replicate": replicate,
                        "role_a": role_names[first],
                        "role_b": role_names[second],
                        "uniform_subset_interaction_auc64": float(
                            np.mean(effects)
                        ),
                        "min_context_interaction": float(np.min(effects)),
                        "max_context_interaction": float(np.max(effects)),
                    }
                )
    return {
        "summary": summary_rows,
        "shapley": shapley_rows,
        "interactions": interaction_rows,
    }


def _role_rank_sweep_data(
    root: Path,
) -> dict[str, list[dict[str, Any]]]:
    summary_rows: list[dict[str, Any]] = []
    heldout_rows: list[dict[str, Any]] = []
    batch_values: dict[tuple[int, int], np.ndarray] = {}
    shared_values: dict[tuple[int, int], np.ndarray] = {}
    for rank in (0, 8, 16, 25, 32, 64, 128, 256):
        for replicate, suffix in enumerate(
            ("primary", "replica"),
            start=1,
        ):
            path = (
                root
                / f"role_rank_sweep24_r{rank}_{suffix}"
                / "summary.json"
            )
            if not path.exists():
                continue
            summary = _load(path)
            artifact_path = path.parent / "role_conditioned_booster.pt"
            artifact = (
                torch.load(
                    artifact_path,
                    map_location="cpu",
                    weights_only=False,
                )
                if artifact_path.exists()
                else None
            )
            for condition in (
                "matched_shared_booster_periodic",
                "learned_booster_periodic",
                "exact_interface_periodic",
            ):
                curve = [
                    float(value)
                    for value in summary["closed_loop"][condition]["accuracy"]
                ]
                summary_rows.append(
                    {
                        "replicate": replicate,
                        "rank": rank,
                        "effective_parameter_count": summary["booster"][
                            "parameter_count"
                        ],
                        "condition": condition,
                        "auc32": _prefix_auc(curve, 32),
                        "auc64": _prefix_auc(curve, 64),
                        "lifespan_at_0.8": _lifespan(curve, 0.8),
                        "lifespan_at_0.5": _lifespan(curve, 0.5),
                        "accuracy_cycle24": curve[23],
                        "accuracy_cycle48": curve[47],
                        "accuracy_cycle64": curve[63],
                    }
                )
            for role, values in summary["heldout"].items():
                heldout_rows.append(
                    {
                        "replicate": replicate,
                        "rank": rank,
                        "role": role,
                        "retained_fit_energy": (
                            artifact["maps"][role][
                                "retained_fit_energy"
                            ]
                            if artifact is not None
                            else float("nan")
                        ),
                        **values,
                    }
                )
            raw_path = path.parent / "closed_loop_rows.csv"
            if raw_path.exists():
                raw = _read_csv(raw_path)
                batches = sorted({int(row["batch"]) for row in raw})

                def batch_auc(condition: str, batch: int) -> float:
                    return float(
                        np.mean(
                            [
                                float(row["accuracy"])
                                for row in raw
                                if int(row["batch"]) == batch
                                and row["condition"] == condition
                            ]
                        )
                    )

                batch_values[(replicate, rank)] = np.asarray(
                    [
                        batch_auc("learned_booster_periodic", batch)
                        for batch in batches
                    ]
                )
                shared_values[(replicate, rank)] = np.asarray(
                    [
                        batch_auc(
                            "matched_shared_booster_periodic",
                            batch,
                        )
                        for batch in batches
                    ]
                )
    paired_rows: list[dict[str, Any]] = []
    for (replicate, rank), values in sorted(batch_values.items()):
        comparisons = {
            "matched_shared_full": shared_values[(replicate, rank)]
        }
        if (replicate, 256) in batch_values:
            comparisons["rank256_role_maps"] = batch_values[
                (replicate, 256)
            ]
        for comparison, reference in comparisons.items():
            differences = values - reference
            rng = np.random.default_rng(
                164000
                + 1000 * replicate
                + 10 * rank
                + (1 if comparison == "rank256_role_maps" else 0)
            )
            samples = rng.choice(
                differences,
                size=(20000, len(differences)),
                replace=True,
            ).mean(axis=1)
            paired_rows.append(
                {
                    "replicate": replicate,
                    "rank": rank,
                    "comparison": comparison,
                    "paired_batches": len(differences),
                    "mean_auc64_difference": float(differences.mean()),
                    "bootstrap_ci95_low": float(
                        np.quantile(samples, 0.025)
                    ),
                    "bootstrap_ci95_high": float(
                        np.quantile(samples, 0.975)
                    ),
                    "positive_batch_clusters": int(
                        np.sum(differences > 0)
                    ),
                }
            )
    return {
        "summary": summary_rows,
        "heldout": heldout_rows,
        "paired_effect": paired_rows,
    }


def _observability_data(root: Path) -> dict[str, list[dict[str, Any]]]:
    summary_rows: list[dict[str, Any]] = []
    per_node_rows: list[dict[str, Any]] = []
    circuit_rows: list[dict[str, Any]] = []
    circuit_per_node_rows: list[dict[str, Any]] = []
    for replicate, directory in enumerate(
        ("observability_primary", "observability_replica"),
        start=1,
    ):
        summary_path = root / directory / "summary.json"
        if not summary_path.exists():
            continue
        summary = _load(summary_path)
        for row in summary["natural_readout"]:
            summary_rows.append(
                {
                    "replicate": replicate,
                    "metric": "natural_readout",
                    **row,
                }
            )
        for metric, key in (
            ("matched_age2_current_readout", "matched_age2_current_readout"),
            ("matched_age2_next_executor", "matched_age2_next_executor"),
        ):
            values = summary[key]
            summary_rows.append(
                {
                    "replicate": replicate,
                    "metric": metric,
                    "age": 2,
                    "expected_path_position": (
                        4 if "current" in metric else 6
                    ),
                    "collision_controlled": (
                        metric == "matched_age2_next_executor"
                    ),
                    **{
                        field: values[field]
                        for field in (
                            "accuracy",
                            "ci95_low",
                            "ci95_high",
                            "mean_probability",
                            "mean_margin",
                            "correct",
                            "valid_count",
                        )
                    },
                }
            )
        for row in _read_csv(root / directory / "per_node_rows.csv"):
            per_node_rows.append({"replicate": replicate, **row})
        for row in _read_csv(
            root / directory / "cycle3_component_rows.csv"
        ):
            circuit_rows.append({"replicate": replicate, **row})
        for row in _read_csv(
            root / directory / "cycle3_component_per_node_rows.csv"
        ):
            circuit_per_node_rows.append(
                {"replicate": replicate, **row}
            )
    return {
        "summary": summary_rows,
        "per_node": per_node_rows,
        "circuit": circuit_rows,
        "circuit_per_node": circuit_per_node_rows,
    }


def _orbit_control_data(root: Path) -> dict[str, list[dict[str, Any]]]:
    data: dict[str, list[dict[str, Any]]] = {
        "summary": [],
        "curves": [],
        "paired": [],
        "components": [],
        "patches": [],
        "lifespans": [],
    }
    runs = (
        (8, 1, "orbit_control_primary"),
        (8, 2, "orbit_control_replica"),
        (16, 1, "orbit_control_h16_primary"),
        (16, 2, "orbit_control_h16_replica"),
    )
    for training_horizon, replicate, directory in runs:
        base = root / directory
        if not (base / "summary.json").exists():
            continue
        summary = _load(base / "summary.json")
        for row in summary["condition_summaries"]:
            data["summary"].append(
                {
                    "training_horizon": training_horizon,
                    "replicate": replicate,
                    **row,
                }
            )
        for key, filename in (
            ("curves", "orbit_control_curves.csv"),
            ("paired", "orbit_control_paired.csv"),
            ("components", "orbit_control_component_curves.csv"),
            ("patches", "orbit8_component_patch_curves.csv"),
        ):
            path = base / filename
            if path.exists():
                data[key].extend(
                    {
                        "training_horizon": training_horizon,
                        "replicate": replicate,
                        **row,
                    }
                    for row in _read_csv(path)
                )

    groups: dict[
        tuple[int, int, int, int], list[dict[str, Any]]
    ] = {}
    for row in data["curves"]:
        key = (
            int(row["training_horizon"]),
            int(row["replicate"]),
            int(row["orbit_length"]),
            int(row["relative_phase"]),
        )
        groups.setdefault(key, []).append(row)
    for (
        training_horizon,
        replicate,
        orbit_length,
        relative_phase,
    ), rows in groups.items():
        oracle = {
            int(row["cycle"]): float(row["accuracy"])
            for row in rows
            if row["condition"] == "exact_interface"
        }
        for condition in ("no_control", "feedback_map"):
            curve = sorted(
                (
                    row
                    for row in rows
                    if row["condition"] == condition
                ),
                key=lambda row: int(row["cycle"]),
            )
            ratios = [
                float(row["accuracy"])
                / max(oracle[int(row["cycle"])], 1e-12)
                for row in curve
            ]
            successful = _lifespan(ratios, 0.8)
            data["lifespans"].append(
                {
                    "training_horizon": training_horizon,
                    "replicate": replicate,
                    "condition": condition,
                    "orbit_length": orbit_length,
                    "relative_phase": relative_phase,
                    "visits": len(curve),
                    "successful_visits_at_0.8_of_oracle": successful,
                    "last_successful_cycle": (
                        int(curve[successful - 1]["cycle"])
                        if successful
                        else 0
                    ),
                    "first_failure_cycle": (
                        int(curve[successful]["cycle"])
                        if successful < len(curve)
                        else ""
                    ),
                }
            )
    return data


def _per_node_data(root: Path) -> dict[str, list[dict[str, Any]]]:
    data: dict[str, list[dict[str, Any]]] = {
        "lifespan_summary": [],
        "lifespan_curves": [],
        "component_patches": [],
        "component_similarities": [],
    }
    for replicate, suffix in enumerate(("primary", "replica"), start=1):
        lifespan_dir = root / f"per_node_lifespan_{suffix}"
        component_dir = root / f"per_node_component_{suffix}"
        for key, path in (
            (
                "lifespan_summary",
                lifespan_dir / "per_node_summary.csv",
            ),
            (
                "lifespan_curves",
                lifespan_dir / "per_node_cycle_rows.csv",
            ),
            (
                "component_patches",
                component_dir / "per_node_component_patch_rows.csv",
            ),
            (
                "component_similarities",
                component_dir
                / "per_node_component_similarity_rows.csv",
            ),
        ):
            if path.exists():
                data[key].extend(
                    {"replicate": replicate, **row}
                    for row in _read_csv(path)
                )
    return data


def _orbit_component_drift_rows(
    data: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    if not data["components"]:
        return []
    identifiers = {
        "training_horizon",
        "replicate",
        "orbit_length",
        "relative_phase",
        "cycle",
        "visit_index",
    }
    fields = [
        field
        for field in data["components"][0]
        if field not in identifiers
    ]
    accuracy = {
        (
            int(row["training_horizon"]),
            int(row["replicate"]),
            int(row["orbit_length"]),
            int(row["relative_phase"]),
            int(row["cycle"]),
        ): float(row["accuracy"])
        for row in data["curves"]
        if row["condition"] == "feedback_map"
    }
    groups: dict[
        tuple[int, int, int, int], list[dict[str, Any]]
    ] = {}
    for row in data["components"]:
        key = (
            int(row["training_horizon"]),
            int(row["replicate"]),
            int(row["orbit_length"]),
            int(row["relative_phase"]),
        )
        groups.setdefault(key, []).append(row)
    cohort_metrics: dict[
        tuple[int, int, str], list[dict[str, float]]
    ] = {}
    for (
        training_horizon,
        replicate,
        orbit_length,
        relative_phase,
    ), rows in groups.items():
        ordered = sorted(rows, key=lambda row: int(row["cycle"]))
        behavior = np.asarray(
            [
                accuracy[
                    (
                        training_horizon,
                        replicate,
                        orbit_length,
                        relative_phase,
                        int(row["cycle"]),
                    )
                ]
                for row in ordered
            ],
            dtype=float,
        )
        for field in fields:
            values = np.asarray(
                [float(row[field]) for row in ordered],
                dtype=float,
            )
            correlation = float("nan")
            if values.std() > 1e-12 and behavior.std() > 1e-12:
                correlation = float(np.corrcoef(values, behavior)[0, 1])
            key = (training_horizon, replicate, field)
            cohort_metrics.setdefault(key, []).append(
                {
                    "first": float(values[0]),
                    "last": float(values[-1]),
                    "drop": float(values[0] - values[-1]),
                    "nonincreasing_fraction": float(
                        np.mean(values[1:] <= values[:-1] + 1e-6)
                    ),
                    "accuracy_correlation": correlation,
                    "below_0.9_at_last": float(values[-1] < 0.9),
                }
            )
    rows: list[dict[str, Any]] = []
    for (training_horizon, replicate, field), metrics in sorted(
        cohort_metrics.items()
    ):
        correlations = np.asarray(
            [metric["accuracy_correlation"] for metric in metrics],
            dtype=float,
        )
        rows.append(
            {
                "training_horizon": training_horizon,
                "replicate": replicate,
                "component": field,
                "cohorts": len(metrics),
                "mean_first_cosine": float(
                    np.mean([metric["first"] for metric in metrics])
                ),
                "mean_last_cosine": float(
                    np.mean([metric["last"] for metric in metrics])
                ),
                "mean_first_to_last_drop": float(
                    np.mean([metric["drop"] for metric in metrics])
                ),
                "mean_nonincreasing_transition_fraction": float(
                    np.mean(
                        [
                            metric["nonincreasing_fraction"]
                            for metric in metrics
                        ]
                    )
                ),
                "mean_within_cohort_accuracy_correlation": float(
                    np.nanmean(correlations)
                ),
                "cohorts_below_0.9_at_last": int(
                    sum(metric["below_0.9_at_last"] for metric in metrics)
                ),
            }
        )
    return rows


def _plot_lifespan(root: Path, figure_dir: Path) -> None:
    specs = (
        (
            "No intervention",
            ("horizon16_primary", "horizon16_replica"),
            "no_intervention",
            "#777777",
        ),
        (
            "R, train horizon 4",
            ("weighted_primary", "weighted_replica"),
            "shared_rw28_r256_round4",
            "#E69F00",
        ),
        (
            "R, train horizon 8",
            ("horizon8_primary", "horizon8_replica"),
            "shared_rw28_r256_round4",
            "#0072B2",
        ),
        (
            "R, train horizon 16",
            ("horizon16_primary", "horizon16_replica"),
            "shared_rw28_r256_round4",
            "#009E73",
        ),
        (
            "Exact young interface",
            ("horizon16_primary", "horizon16_replica"),
            "oracle_interface",
            "#CC79A7",
        ),
    )
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for label, directories, condition, color in specs:
        arrays = _mean_curves(
            _load(root / directory / "summary.json")["closed_loop"][
                condition
            ]["accuracy"]
            for directory in directories
        )
        x = np.arange(1, arrays.shape[1] + 1)
        mean = arrays.mean(axis=0)
        ax.plot(x, mean, label=label, color=color, linewidth=2)
        ax.fill_between(
            x,
            arrays.min(axis=0),
            arrays.max(axis=0),
            color=color,
            alpha=0.14,
        )
    ax.axhline(0.125, color="black", linestyle=":", linewidth=1)
    ax.axhline(0.8, color="black", linestyle="--", linewidth=0.8)
    ax.set(
        xlabel="Extra recurrent cycle",
        ylabel="Collision-controlled accuracy",
        xlim=(1, 128),
        ylim=(0, 1.02),
    )
    ax.legend(frameon=False, ncol=2, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(figure_dir / "lifespan_scaling.png", dpi=180)
    plt.close(fig)


def _plot_position_circuit(
    rows: list[dict[str, Any]],
    figure_dir: Path,
) -> None:
    labels = (
        ("no_intervention", "None"),
        ("b2_answer", "Answer"),
        ("b2_answer_destination", "Answer+dest"),
        ("b2_answer_graph", "Answer+graph"),
        ("b2_answer_graph_metadata", "Full interface"),
        ("b2_all_except_answer", "All except answer"),
        ("b2_all_except_graph", "All except graph"),
        ("b2_all_except_metadata", "All except metadata"),
    )
    means, lows, highs = [], [], []
    for condition, _ in labels:
        values = [
            float(row["auc32"])
            for row in rows
            if row["condition"] == condition
        ]
        means.append(np.mean(values))
        lows.append(np.mean(values) - np.min(values))
        highs.append(np.max(values) - np.mean(values))
    fig, ax = plt.subplots(figsize=(9.4, 4.7))
    x = np.arange(len(labels))
    ax.bar(
        x,
        means,
        yerr=np.asarray([lows, highs]),
        capsize=3,
        color="#56B4E9",
        edgecolor="white",
    )
    ax.axhline(0.125, color="black", linestyle=":", linewidth=1)
    ax.set_xticks(x, [label for _, label in labels], rotation=24, ha="right")
    ax.set(ylabel="32-cycle AUC", ylim=(0, 1.0))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(figure_dir / "position_circuit_auc.png", dpi=180)
    plt.close(fig)


def _plot_layer_boundary(
    rows: list[dict[str, Any]],
    figure_dir: Path,
) -> None:
    labels = (
        ("no_intervention", "None"),
        ("loop_answer", "Loop input:\nanswer"),
        ("b2_answer", "Block2 input:\nanswer"),
        ("loop_answer_graph", "Loop input:\nanswer+graph"),
        ("b2_answer_graph", "Block2 input:\nanswer+graph"),
        ("b2_answer_graph_metadata", "Block2 input:\nfull interface"),
    )
    means, lows, highs = [], [], []
    for condition, _ in labels:
        values = [
            float(row["auc32"])
            for row in rows
            if row["condition"] == condition
        ]
        means.append(np.mean(values))
        lows.append(np.mean(values) - np.min(values))
        highs.append(np.max(values) - np.mean(values))
    fig, ax = plt.subplots(figsize=(8.9, 4.5))
    x = np.arange(len(labels))
    ax.bar(
        x,
        means,
        yerr=np.asarray([lows, highs]),
        capsize=3,
        color=("#999999", "#56B4E9", "#0072B2", "#E69F00", "#D55E00", "#009E73"),
    )
    ax.axhline(0.125, color="black", linestyle=":", linewidth=1)
    ax.set_xticks(x, [label for _, label in labels])
    ax.set(
        ylabel="32-cycle AUC",
        title="Causal repair before versus after physical Block 1",
        ylim=(0, 1.0),
    )
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(figure_dir / "layer_boundary_patch.png", dpi=180)
    plt.close(fig)


def _plot_power(rows: list[dict[str, Any]], figure_dir: Path) -> None:
    specs = (
        ("DAgger h=8 map", "map_power", "DAgger map $R^{a-2}$", "#0072B2"),
        (
            "pooled adjacent-age map",
            "map_power",
            "Adjacent-age map $R^{a-2}$",
            "#E69F00",
        ),
        ("DAgger h=8 map", "natural_aged", "Natural aged", "#777777"),
        (
            "DAgger h=8 map",
            "oracle_interface_on_aged",
            "Exact interface on aged state",
            "#009E73",
        ),
    )
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    for method, condition, label, color in specs:
        by_age: dict[int, list[float]] = {}
        for row in rows:
            if row["method"] == method and row["condition"] == condition:
                by_age.setdefault(int(row["age"]), []).append(
                    float(row["accuracy"])
                )
        ages = sorted(by_age)
        values = np.asarray([by_age[age] for age in ages])
        ax.plot(
            ages,
            values.mean(axis=1),
            marker="o",
            linewidth=2,
            label=label,
            color=color,
        )
        ax.fill_between(
            ages,
            values.min(axis=1),
            values.max(axis=1),
            alpha=0.15,
            color=color,
        )
    ax.axhline(0.125, color="black", linestyle=":", linewidth=1)
    ax.set(
        xlabel="Natural hidden-state age",
        ylabel="Two-hop continuation accuracy",
        xticks=range(3, 9),
        ylim=(0, 1.02),
    )
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(figure_dir / "power_law_failure.png", dpi=180)
    plt.close(fig)


def _plot_map_ablation(
    rows: list[dict[str, Any]],
    figure_dir: Path,
) -> None:
    labels = (
        ("no_intervention", "None"),
        ("map_full", "Full affine R"),
        ("map_full_no_bias", "R without bias"),
        ("map_full_bias_only", "Bias only"),
        ("map_answer", "R on answer"),
        ("map_answer_graph", "R on answer+graph"),
        ("map_answer_metadata", "R on answer+metadata"),
        ("map_graph", "R on graph"),
    )
    means, lows, highs = [], [], []
    for condition, _ in labels:
        values = [
            float(row["auc32"])
            for row in rows
            if row["condition"] == condition
        ]
        means.append(np.mean(values))
        lows.append(np.mean(values) - np.min(values))
        highs.append(np.max(values) - np.mean(values))
    fig, ax = plt.subplots(figsize=(9.4, 4.7))
    x = np.arange(len(labels))
    colors = [
        "#999999",
        "#009E73",
        "#E69F00",
        "#F0E442",
        "#56B4E9",
        "#0072B2",
        "#CC79A7",
        "#D55E00",
    ]
    ax.bar(
        x,
        means,
        yerr=np.asarray([lows, highs]),
        capsize=3,
        color=colors,
        edgecolor="white",
    )
    ax.axhline(0.125, color="black", linestyle=":", linewidth=1)
    ax.set_xticks(x, [label for _, label in labels], rotation=25, ha="right")
    ax.set(ylabel="32-cycle AUC", ylim=(0, 1.0))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(figure_dir / "learned_map_ablation.png", dpi=180)
    plt.close(fig)


def _plot_components(root: Path, figure_dir: Path) -> None:
    fields = (
        ("head2_q_answer_cosine", "Q(answer)", "#D55E00"),
        ("head2_k_destination_cosine", "K(destination)", "#0072B2"),
        ("head2_v_destination_cosine", "V(destination)", "#56B4E9"),
        ("head2_context_answer_cosine", "Attention context", "#CC79A7"),
        ("block2_mlp_answer_cosine", "MLP(answer)", "#009E73"),
    )
    summaries = [
        _load(root / directory / "summary.json")["closed_loop"][
            "shared_rw28_r256_round4"
        ]
        for directory in ("horizon8_primary", "horizon8_replica")
    ]
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    x = np.arange(1, 65)
    for field, label, color in fields:
        values = np.asarray([summary[field] for summary in summaries])
        ax.plot(x, values.mean(axis=0), label=label, color=color, linewidth=2)
        ax.fill_between(
            x,
            values.min(axis=0),
            values.max(axis=0),
            color=color,
            alpha=0.12,
        )
    accuracy = np.asarray([summary["accuracy"] for summary in summaries])
    ax.plot(
        x,
        accuracy.mean(axis=0),
        color="black",
        linestyle="--",
        linewidth=1.5,
        label="Accuracy",
    )
    ax.set(
        xlabel="Extra recurrent cycle",
        ylabel="Cosine to exact-young circuit / accuracy",
        xlim=(1, 64),
        ylim=(0, 1.02),
    )
    ax.legend(frameon=False, ncol=2, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(figure_dir / "component_decay_horizon8.png", dpi=180)
    plt.close(fig)


def _plot_component_patches(
    rows: list[dict[str, Any]],
    figure_dir: Path,
) -> None:
    specs = (
        ("baseline_map", "Learned R", "#777777"),
        ("q_head2", "Patch Q", "#D55E00"),
        ("k_head2", "Patch K", "#0072B2"),
        ("v_head2", "Patch V", "#56B4E9"),
        ("context_head2", "Patch head2 context", "#CC79A7"),
        ("mlp_out_answer", "Patch MLP output", "#E69F00"),
        ("exact_interface", "Patch full interface", "#009E73"),
    )
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for condition, label, color in specs:
        by_cycle: dict[int, list[float]] = {}
        for row in rows:
            if row["condition"] == condition:
                by_cycle.setdefault(int(row["cycle"]), []).append(
                    float(row["accuracy"])
                )
        cycles = sorted(by_cycle)
        values = np.asarray([by_cycle[cycle] for cycle in cycles])
        ax.plot(
            cycles,
            values.mean(axis=1),
            marker="o",
            linewidth=2,
            label=label,
            color=color,
        )
        ax.fill_between(
            cycles,
            values.min(axis=1),
            values.max(axis=1),
            color=color,
            alpha=0.12,
        )
    ax.axhline(0.125, color="black", linestyle=":", linewidth=1)
    ax.set(
        xlabel="Extra recurrent cycle",
        ylabel="Immediate patched accuracy",
        xticks=(16, 24, 32, 40, 48, 64),
        ylim=(0, 1.02),
    )
    ax.legend(frameon=False, ncol=2, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(figure_dir / "long_horizon_component_patching.png", dpi=180)
    plt.close(fig)


def _plot_role_action(
    rows: list[dict[str, Any]],
    figure_dir: Path,
) -> None:
    specs = (
        ("answer", "Answer", "#D55E00"),
        ("edge_marker", "Edge marker", "#E69F00"),
        ("source", "Source", "#0072B2"),
        ("destination", "Destination", "#56B4E9"),
        ("metadata", "Metadata", "#CC79A7"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.5))
    for role, label, color in specs:
        by_cycle: dict[int, list[dict[str, Any]]] = {}
        for row in rows:
            if row["role"] == role:
                by_cycle.setdefault(int(row["cycle"]), []).append(row)
        cycles = sorted(by_cycle)
        cosine = np.asarray(
            [
                [
                    float(item["correction_to_desired_cosine"])
                    for item in by_cycle[cycle]
                ]
                for cycle in cycles
            ]
        )
        error_ratio = np.asarray(
            [
                [
                    float(item["post_relative_mse"])
                    / max(float(item["pre_relative_mse"]), 1e-12)
                    for item in by_cycle[cycle]
                ]
                for cycle in cycles
            ]
        )
        axes[0].plot(
            cycles,
            cosine.mean(axis=1),
            marker="o",
            color=color,
            linewidth=2,
            label=label,
        )
        axes[1].plot(
            cycles,
            error_ratio.mean(axis=1),
            marker="o",
            color=color,
            linewidth=2,
            label=label,
        )
    axes[0].set(
        xlabel="Extra recurrent cycle",
        ylabel="Cosine(map correction, exact correction)",
        ylim=(0.45, 1.02),
    )
    axes[1].set(
        xlabel="Extra recurrent cycle",
        ylabel="Post-map / pre-map relative MSE",
        ylim=(0, 1.0),
    )
    axes[0].legend(frameon=False, fontsize=9)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(figure_dir / "shared_map_role_action.png", dpi=180)
    plt.close(fig)


def _plot_rank_sweep(
    rows: list[dict[str, Any]],
    figure_dir: Path,
) -> None:
    ranks = sorted({int(row["rank"]) for row in rows})

    def values(field: str) -> np.ndarray:
        return np.asarray(
            [
                [
                    float(row[field])
                    for row in rows
                    if int(row["rank"]) == rank
                ]
                for rank in ranks
            ]
        )

    auc32 = values("auc32")
    auc64 = values("auc64")
    energy = values("retained_fitted_update_energy")
    life80 = values("lifespan_at_0.8")
    life50 = values("lifespan_at_0.5")
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.5))
    axes[0].plot(
        ranks,
        auc32.mean(axis=1),
        marker="o",
        linewidth=2,
        label="32-cycle AUC",
        color="#0072B2",
    )
    axes[0].plot(
        ranks,
        auc64.mean(axis=1),
        marker="o",
        linewidth=2,
        label="64-cycle AUC",
        color="#D55E00",
    )
    energy_axis = axes[0].twinx()
    energy_axis.plot(
        ranks,
        energy.mean(axis=1),
        marker="s",
        linestyle=":",
        linewidth=1.5,
        label="Retained update energy",
        color="#999999",
    )
    axes[0].set(
        xscale="log",
        xlabel="Rank of learned update",
        ylabel="Behavior AUC",
        ylim=(0.1, 0.9),
        xticks=ranks,
    )
    axes[0].set_xticklabels([str(rank) for rank in ranks])
    energy_axis.set(ylabel="Retained fitted-update energy", ylim=(0.55, 1.01))
    handles, labels = axes[0].get_legend_handles_labels()
    handles2, labels2 = energy_axis.get_legend_handles_labels()
    axes[0].legend(
        handles + handles2,
        labels + labels2,
        frameon=False,
        fontsize=8.5,
        loc="lower right",
    )
    axes[1].plot(
        ranks,
        life80.mean(axis=1),
        marker="o",
        linewidth=2,
        label="Lifespan at 80%",
        color="#009E73",
    )
    axes[1].plot(
        ranks,
        life50.mean(axis=1),
        marker="o",
        linewidth=2,
        label="Lifespan at 50%",
        color="#CC79A7",
    )
    axes[1].set(
        xscale="log",
        xlabel="Rank of learned update",
        ylabel="Continuous successful cycles",
        xticks=ranks,
        ylim=(0, 40),
    )
    axes[1].set_xticklabels([str(rank) for rank in ranks])
    axes[1].legend(frameon=False, fontsize=9)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    energy_axis.spines["top"].set_visible(False)
    fig.tight_layout()
    fig.savefig(figure_dir / "behavioral_rank_sweep.png", dpi=180)
    plt.close(fig)


def _plot_initialization_gate(
    rows: list[dict[str, Any]],
    figure_dir: Path,
) -> None:
    specs = (
        ("raw_h8_no_control", "Raw H8, no control", "#777777"),
        ("raw_h8_R_each_cycle", "Raw H8, R each cycle", "#D55E00"),
        (
            "raw_h8_R6_first_then_R",
            "Raw H8, $R^6$ then R",
            "#E69F00",
        ),
        (
            "raw_h8_exact_answer_first_then_R",
            "Exact answer once, then R",
            "#56B4E9",
        ),
        (
            "raw_h8_exact_answer_graph_first_then_R",
            "Exact answer+graph once, then R",
            "#0072B2",
        ),
        (
            "raw_h8_exact_interface_first_then_R",
            "Exact interface once, then R",
            "#009E73",
        ),
        (
            "raw_h8_shuffled_interface_first_then_R",
            "Shuffled interface once, then R",
            "#CC79A7",
        ),
    )
    fig, ax = plt.subplots(figsize=(8.5, 4.9))
    for condition, label, color in specs:
        by_cycle: dict[int, list[float]] = {}
        for row in rows:
            if row["condition"] == condition:
                by_cycle.setdefault(int(row["cycle"]), []).append(
                    float(row["accuracy"])
                )
        cycles = sorted(by_cycle)
        values = np.asarray([by_cycle[cycle] for cycle in cycles])
        ax.plot(
            cycles,
            values.mean(axis=1),
            linewidth=2,
            label=label,
            color=color,
        )
        ax.fill_between(
            cycles,
            values.min(axis=1),
            values.max(axis=1),
            color=color,
            alpha=0.12,
        )
    ax.axhline(0.125, color="black", linestyle=":", linewidth=1)
    ax.set(
        xlabel="Extra recurrent cycle from the original H8 endpoint",
        ylabel="Collision-controlled accuracy",
        xlim=(1, 64),
        ylim=(0, 1.02),
    )
    ax.legend(frameon=False, ncol=2, fontsize=8.5)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(figure_dir / "initialization_gate.png", dpi=180)
    plt.close(fig)


def _plot_learned_initializer(
    rows: list[dict[str, Any]],
    figure_dir: Path,
) -> None:
    specs = (
        ("raw_h8_no_control", "Raw H8, no control", "#777777"),
        (
            "raw_h8_feedback_each_cycle",
            "Feedback R only",
            "#D55E00",
        ),
        ("learned_init_only", "Learned init only", "#E69F00"),
        (
            "learned_init_then_feedback",
            "Learned init + feedback R",
            "#0072B2",
        ),
        (
            "shuffled_learned_init_then_feedback",
            "Shuffled init + feedback R",
            "#CC79A7",
        ),
        (
            "exact_interface_then_feedback",
            "Exact interface + feedback R",
            "#009E73",
        ),
    )
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    for condition, label, color in specs:
        by_cycle: dict[int, list[float]] = {}
        for row in rows:
            if row["condition"] == condition:
                by_cycle.setdefault(int(row["cycle"]), []).append(
                    float(row["accuracy"])
                )
        cycles = sorted(by_cycle)
        values = np.asarray([by_cycle[cycle] for cycle in cycles])
        ax.plot(
            cycles,
            values.mean(axis=1),
            linewidth=2,
            label=label,
            color=color,
        )
        ax.fill_between(
            cycles,
            values.min(axis=1),
            values.max(axis=1),
            color=color,
            alpha=0.12,
        )
    ax.axhline(0.125, color="black", linestyle=":", linewidth=1)
    ax.set(
        xlabel="Extra recurrent cycle from the original H8 endpoint",
        ylabel="Collision-controlled accuracy",
        xlim=(1, 64),
        ylim=(0, 1.02),
    )
    ax.legend(frameon=False, ncol=2, fontsize=8.8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(figure_dir / "learned_two_stage_rejuvenation.png", dpi=180)
    plt.close(fig)


def _plot_joint_single_map(
    rows: list[dict[str, Any]],
    figure_dir: Path,
) -> None:
    specs = (
        ("raw_h8_no_control", "Raw H8, no control", "#777777"),
        ("joint_single_map", "One joint affine map", "#D55E00"),
        ("exact_h2_joint_local", "Joint map from exact H2", "#E69F00"),
        ("learned_two_stage", "Two learned affine maps", "#0072B2"),
        (
            "shuffled_joint_single_map",
            "Shuffled joint-map output",
            "#CC79A7",
        ),
        (
            "exact_interface_every_cycle",
            "Exact interface every cycle",
            "#009E73",
        ),
    )
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    for condition, label, color in specs:
        by_cycle: dict[int, list[float]] = {}
        for row in rows:
            if row["condition"] == condition:
                by_cycle.setdefault(int(row["cycle"]), []).append(
                    float(row["accuracy"])
                )
        cycles = sorted(by_cycle)
        values = np.asarray([by_cycle[cycle] for cycle in cycles])
        ax.plot(
            cycles,
            values.mean(axis=1),
            linewidth=2,
            label=label,
            color=color,
        )
        ax.fill_between(
            cycles,
            values.min(axis=1),
            values.max(axis=1),
            color=color,
            alpha=0.12,
        )
    ax.axhline(0.125, color="black", linestyle=":", linewidth=1)
    ax.set(
        xlabel="Extra recurrent cycle",
        ylabel="Collision-controlled accuracy",
        xlim=(1, 64),
        ylim=(0, 1.02),
    )
    ax.legend(frameon=False, ncol=2, fontsize=8.8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(figure_dir / "joint_single_vs_two_stage.png", dpi=180)
    plt.close(fig)


def _plot_joint_weight_tradeoff(
    rows: list[dict[str, Any]],
    learned_initializer_rows: list[dict[str, Any]],
    figure_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 5.0))
    colors = {1: "#56B4E9", 2: "#E69F00", 4: "#D55E00"}
    for weight in (1, 2, 4):
        parts = [
            row for row in rows if int(row["initializer_repeat"]) == weight
        ]
        x = np.asarray([float(row["first_cycle_accuracy"]) for row in parts])
        y = np.asarray([float(row["auc32"]) for row in parts])
        ax.errorbar(
            x.mean(),
            y.mean(),
            xerr=[[x.mean() - x.min()], [x.max() - x.mean()]],
            yerr=[[y.mean() - y.min()], [y.max() - y.mean()]],
            marker="o",
            markersize=8,
            capsize=3,
            color=colors[weight],
            label=f"One map, init weight {weight}",
        )
    two_stage_by_rep: dict[int, list[float]] = {}
    for row in learned_initializer_rows:
        if row["condition"] == "learned_init_then_feedback":
            two_stage_by_rep.setdefault(int(row["replicate"]), []).append(
                float(row["accuracy"])
            )
    two_first = np.asarray(
        [values[0] for _, values in sorted(two_stage_by_rep.items())]
    )
    two_auc32 = np.asarray(
        [np.mean(values[:32]) for _, values in sorted(two_stage_by_rep.items())]
    )
    ax.errorbar(
        two_first.mean(),
        two_auc32.mean(),
        xerr=[
            [two_first.mean() - two_first.min()],
            [two_first.max() - two_first.mean()],
        ],
        yerr=[
            [two_auc32.mean() - two_auc32.min()],
            [two_auc32.max() - two_auc32.mean()],
        ],
        marker="*",
        markersize=13,
        capsize=3,
        color="#009E73",
        label="Two learned maps",
    )
    ax.axvline(0.8, color="black", linestyle="--", linewidth=0.8)
    ax.set(
        xlabel="First-cycle accuracy from raw H8",
        ylabel="32-cycle AUC",
        xlim=(0.45, 1.0),
        ylim=(0.35, 0.95),
    )
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(figure_dir / "joint_weight_tradeoff.png", dpi=180)
    plt.close(fig)


def _plot_observability(
    data: dict[str, list[dict[str, Any]]],
    figure_dir: Path,
) -> None:
    natural = [
        row
        for row in data["summary"]
        if row["metric"] == "natural_readout"
    ]
    ages = sorted({int(row["age"]) for row in natural})
    natural_values = np.asarray(
        [
            [
                float(row["accuracy"])
                for row in natural
                if int(row["age"]) == age
            ]
            for age in ages
        ]
    )
    executor = [
        float(row["accuracy"])
        for row in data["summary"]
        if row["metric"] == "matched_age2_next_executor"
    ]
    per_node = [
        row
        for row in data["per_node"]
        if row["metric"] == "age2_executor"
    ]
    nodes = sorted({int(row["target_node"]) for row in per_node})
    node_values = np.asarray(
        [
            [
                float(row["accuracy"])
                for row in per_node
                if int(row["target_node"]) == node
            ]
            for node in nodes
        ]
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.5))
    axes[0].plot(
        ages,
        natural_values.mean(axis=1),
        marker="o",
        linewidth=2,
        color="#0072B2",
        label="Natural current-node readout",
    )
    axes[0].fill_between(
        ages,
        natural_values.min(axis=1),
        natural_values.max(axis=1),
        color="#0072B2",
        alpha=0.15,
    )
    axes[0].axhline(
        float(np.mean(executor)),
        color="#D55E00",
        linestyle="--",
        linewidth=1.8,
        label="Matched age-2 two-hop executor",
    )
    axes[0].axhline(0.125, color="black", linestyle=":", linewidth=1)
    axes[0].set(
        xlabel="Natural recurrent age",
        ylabel="Collision-controlled accuracy",
        xticks=ages,
        ylim=(0, 1.02),
    )
    axes[0].legend(frameon=False, fontsize=8.5)
    width = 0.34
    x = np.asarray(nodes, dtype=float)
    for replicate in (1, 2):
        values = [
            float(
                next(
                    row["accuracy"]
                    for row in per_node
                    if int(row["target_node"]) == node
                    and int(row["replicate"]) == replicate
                )
            )
            for node in nodes
        ]
        axes[1].bar(
            x + (replicate - 1.5) * width,
            values,
            width=width,
            color=("#56B4E9" if replicate == 1 else "#E69F00"),
            label=f"Data replication {replicate}",
        )
    axes[1].axhline(0.125, color="black", linestyle=":", linewidth=1)
    axes[1].set(
        xlabel="Two-hop target node",
        ylabel="Matched age-2 executor accuracy",
        xticks=nodes,
        ylim=(0, 1.02),
    )
    axes[1].legend(frameon=False, fontsize=8.5)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(figure_dir / "intermediate_observability.png", dpi=180)
    plt.close(fig)


def _plot_cycle3_scope(
    rows: list[dict[str, Any]],
    figure_dir: Path,
) -> None:
    specs = (
        ("baseline", "Baseline"),
        ("q_head2", "Q"),
        ("k_head2", "K"),
        ("v_head2", "V"),
        ("context_head2", "Context"),
        ("mlp_out_answer", "MLP"),
        ("exact_interface", "Full interface"),
    )
    trajectories = (
        ("raw_no_control", "Raw no-control trajectory", "#777777"),
        (
            "one_step_answer_map",
            "One-step answer-map trajectory",
            "#0072B2",
        ),
    )
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    x = np.arange(len(specs))
    width = 0.36
    for index, (trajectory, label, color) in enumerate(trajectories):
        means = []
        lows = []
        highs = []
        for condition, _ in specs:
            values = np.asarray(
                [
                    float(row["accuracy"])
                    for row in rows
                    if row["trajectory"] == trajectory
                    and row["condition"] == condition
                ]
            )
            means.append(values.mean())
            lows.append(values.mean() - values.min())
            highs.append(values.max() - values.mean())
        ax.bar(
            x + (index - 0.5) * width,
            means,
            width=width,
            yerr=np.asarray([lows, highs]),
            capsize=3,
            color=color,
            label=label,
        )
    ax.axhline(0.125, color="black", linestyle=":", linewidth=1)
    ax.set_xticks(
        x,
        [label for _, label in specs],
        rotation=20,
        ha="right",
    )
    ax.set(
        ylabel="Cycle-3 immediate patched accuracy",
        ylim=(0, 1.02),
    )
    ax.legend(frameon=False, fontsize=8.5)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(figure_dir / "cycle3_circuit_scope.png", dpi=180)
    plt.close(fig)


def _plot_orbit_control(
    data: dict[str, list[dict[str, Any]]],
    figure_dir: Path,
) -> None:
    if not data["components"] or not data["patches"]:
        return
    orbit_length = 8
    relative_phase = 2
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.6))

    behavior_specs = (
        ("no_control", 8, "No control", "#777777"),
        ("feedback_map", 8, "Affine feedback, train h=8", "#0072B2"),
        ("feedback_map", 16, "Affine feedback, train h=16", "#009E73"),
        ("exact_interface", 8, "Exact young interface", "#CC79A7"),
    )
    for condition, training_horizon, label, color in behavior_specs:
        selected = [
            row
            for row in data["curves"]
            if row["condition"] == condition
            and int(row["training_horizon"]) == training_horizon
            and int(row["orbit_length"]) == orbit_length
            and int(row["relative_phase"]) == relative_phase
        ]
        cycles = sorted({int(row["cycle"]) for row in selected})
        values = np.asarray(
            [
                [
                    float(row["accuracy"])
                    for row in selected
                    if int(row["cycle"]) == cycle
                ]
                for cycle in cycles
            ]
        )
        axes[0].plot(
            cycles,
            values.mean(axis=1),
            color=color,
            linewidth=2,
            label=label,
        )
        axes[0].fill_between(
            cycles,
            values.min(axis=1),
            values.max(axis=1),
            color=color,
            alpha=0.14,
        )
    axes[0].axhline(0.125, color="black", linestyle=":", linewidth=1)
    axes[0].axhline(0.8, color="black", linestyle="--", linewidth=0.8)
    axes[0].set(
        xlabel="Extra recurrent cycle",
        ylabel="Accuracy on the same orbit phase",
        title="Fixed task semantics",
        ylim=(0, 1.02),
    )
    axes[0].legend(frameon=False, fontsize=8)

    component_specs = (
        ("q_cosine", "Q", "#D55E00"),
        ("k_cosine", "K", "#009E73"),
        ("v_cosine", "V", "#56B4E9"),
        ("context_cosine", "Context", "#CC79A7"),
        ("mlp_cosine", "MLP", "#E69F00"),
    )
    selected_components = [
        row
        for row in data["components"]
        if int(row["training_horizon"]) == 8
        and int(row["orbit_length"]) == orbit_length
        and int(row["relative_phase"]) == relative_phase
    ]
    cycles = sorted(
        {int(row["cycle"]) for row in selected_components}
    )
    for field, label, color in component_specs:
        values = np.asarray(
            [
                [
                    float(row[field])
                    for row in selected_components
                    if int(row["cycle"]) == cycle
                ]
                for cycle in cycles
            ]
        )
        axes[1].plot(
            cycles,
            values.mean(axis=1),
            color=color,
            linewidth=1.8,
            label=label,
        )
    axes[1].set(
        xlabel="Extra recurrent cycle",
        ylabel="Cosine to exact-young component",
        title="Component drift",
        ylim=(0, 1.02),
    )
    axes[1].legend(frameon=False, fontsize=8, ncol=2)

    patch_specs = (
        ("feedback_baseline", "Baseline", "#777777"),
        ("q_head2", "Q patch", "#D55E00"),
        ("qkv_head2", "QKV patch", "#009E73"),
        ("context_head2", "Context patch", "#CC79A7"),
        ("mlp_out_answer", "MLP patch", "#E69F00"),
        ("exact_interface", "Full interface", "#0072B2"),
    )
    for condition, label, color in patch_specs:
        selected = [
            row
            for row in data["patches"]
            if row["condition"] == condition
            and int(row["training_horizon"]) == 8
            and int(row["relative_phase"]) == relative_phase
        ]
        cycles = sorted({int(row["cycle"]) for row in selected})
        values = np.asarray(
            [
                [
                    float(row["accuracy"])
                    for row in selected
                    if int(row["cycle"]) == cycle
                ]
                for cycle in cycles
            ]
        )
        axes[2].plot(
            cycles,
            values.mean(axis=1),
            color=color,
            linewidth=1.8,
            label=label,
        )
    axes[2].axhline(0.125, color="black", linestyle=":", linewidth=1)
    axes[2].set(
        xlabel="Extra recurrent cycle",
        ylabel="Immediate patched accuracy",
        title="Causal component rescue",
        ylim=(0, 1.02),
    )
    axes[2].legend(frameon=False, fontsize=7.5, ncol=2)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "D8L8 orbit-length 8, fixed relative phase 2",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(figure_dir / "orbit_controlled_aging.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    horizon_styles = ((8, "-"), (16, "--"))
    focused_components = (
        ("q_cosine", "Q", "#D55E00"),
        ("context_cosine", "Context", "#CC79A7"),
        ("mlp_cosine", "MLP", "#E69F00"),
    )
    for training_horizon, linestyle in horizon_styles:
        selected = [
            row
            for row in data["components"]
            if int(row["training_horizon"]) == training_horizon
            and int(row["orbit_length"]) == orbit_length
            and int(row["relative_phase"]) == relative_phase
        ]
        cycles = sorted({int(row["cycle"]) for row in selected})
        for field, label, color in focused_components:
            values = np.asarray(
                [
                    [
                        float(row[field])
                        for row in selected
                        if int(row["cycle"]) == cycle
                    ]
                    for cycle in cycles
                ]
            )
            axes[0].plot(
                cycles,
                values.mean(axis=1),
                color=color,
                linestyle=linestyle,
                linewidth=2,
                label=f"{label}, train h={training_horizon}",
            )
    axes[0].set(
        xlabel="Extra recurrent cycle",
        ylabel="Cosine to exact-young component",
        title="Which components are kept young",
        ylim=(0, 1.02),
    )
    axes[0].legend(frameon=False, fontsize=8, ncol=2)

    focused_patches = (
        ("feedback_baseline", "Baseline", "#777777"),
        ("q_head2", "Q patch", "#D55E00"),
        ("context_head2", "Context patch", "#CC79A7"),
        ("exact_interface", "Full interface", "#0072B2"),
    )
    for training_horizon, linestyle in horizon_styles:
        for condition, label, color in focused_patches:
            selected = [
                row
                for row in data["patches"]
                if row["condition"] == condition
                and int(row["training_horizon"]) == training_horizon
                and int(row["relative_phase"]) == relative_phase
            ]
            cycles = sorted({int(row["cycle"]) for row in selected})
            values = np.asarray(
                [
                    [
                        float(row["accuracy"])
                        for row in selected
                        if int(row["cycle"]) == cycle
                    ]
                    for cycle in cycles
                ]
            )
            axes[1].plot(
                cycles,
                values.mean(axis=1),
                color=color,
                linestyle=linestyle,
                linewidth=2,
                label=f"{label}, train h={training_horizon}",
            )
    axes[1].axhline(0.125, color="black", linestyle=":", linewidth=1)
    axes[1].set(
        xlabel="Extra recurrent cycle",
        ylabel="Immediate patched accuracy",
        title="How the causal bottleneck is delayed",
        ylim=(0, 1.02),
    )
    axes[1].legend(frameon=False, fontsize=7.5, ncol=2)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "Longer closure training preserves the executable interface",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(
        figure_dir / "orbit_controller_component_comparison.png",
        dpi=180,
    )
    plt.close(fig)


def _plot_per_node_lifespan(
    data: dict[str, list[dict[str, Any]]],
    figure_dir: Path,
) -> None:
    rows = [
        row
        for row in data["lifespan_summary"]
        if row["condition"] == "feedback_map"
    ]
    if not rows:
        return
    nodes = sorted({int(row["target_node"]) for row in rows})
    auc = np.asarray(
        [
            [
                float(row["auc_fraction_of_oracle"])
                for row in rows
                if int(row["target_node"]) == node
            ]
            for node in nodes
        ]
    )
    lifespan = np.asarray(
        [
            [
                float(row["lifespan_at_0.8_of_oracle"])
                for row in rows
                if int(row["target_node"]) == node
            ]
            for node in nodes
        ]
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.4))
    for axis, values, ylabel in (
        (
            axes[0],
            auc,
            "64-cycle AUC / exact-interface AUC",
        ),
        (
            axes[1],
            lifespan,
            "Prefix lifespan at 80% of oracle",
        ),
    ):
        axis.bar(
            nodes,
            values.mean(axis=1),
            color="#0072B2",
            yerr=np.asarray(
                [
                    values.mean(axis=1) - values.min(axis=1),
                    values.max(axis=1) - values.mean(axis=1),
                ]
            ),
            capsize=3,
        )
        axis.set(xlabel="Target node", ylabel=ylabel, xticks=nodes)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].axhline(1.0, color="black", linestyle=":", linewidth=1)
    fig.tight_layout()
    fig.savefig(figure_dir / "per_node_lifespan.png", dpi=180)
    plt.close(fig)


def _plot_answer_weight(
    root: Path,
    rows: list[dict[str, Any]],
    figure_dir: Path,
) -> None:
    if not rows:
        return
    weights = sorted({int(row["answer_weight"]) for row in rows})
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4))
    metric_axis = axes[0].twinx()
    for metric, label, color, axis in (
        ("auc64", "64-cycle AUC", "#0072B2", axes[0]),
        (
            "lifespan_at_0.8",
            "80% prefix lifespan",
            "#D55E00",
            metric_axis,
        ),
    ):
        values = np.asarray(
            [
                [
                    float(row[metric])
                    for row in rows
                    if int(row["answer_weight"]) == weight
                ]
                for weight in weights
            ]
        )
        axis.plot(
            weights,
            values.mean(axis=1),
            marker="o",
            linewidth=2,
            color=color,
            label=label,
        )
        axis.fill_between(
            weights,
            values.min(axis=1),
            values.max(axis=1),
            color=color,
            alpha=0.14,
        )
    axes[0].set(
        xlabel="Answer role weight in state-MSE regression",
        ylabel="64-cycle AUC",
        xticks=weights,
        title="Circuit-guided weighting",
    )
    metric_axis.set_ylabel("80% prefix lifespan")
    handles_left, labels_left = axes[0].get_legend_handles_labels()
    handles_right, labels_right = metric_axis.get_legend_handles_labels()
    axes[0].legend(
        handles_left + handles_right,
        labels_left + labels_right,
        frameon=False,
        fontsize=8.5,
    )

    colors = ("#56B4E9", "#0072B2", "#E69F00", "#D55E00")
    for weight, color in zip(weights, colors, strict=True):
        curves = []
        for suffix in ("primary", "replica"):
            summary = _load(
                root / f"answer_weight_h8_{suffix}" / "summary.json"
            )
            condition = f"shared_rw{weight}_r256_round4"
            curves.append(summary["closed_loop"][condition]["accuracy"])
        array = _mean_curves(curves)
        cycles = np.arange(1, array.shape[1] + 1)
        axes[1].plot(
            cycles,
            array.mean(axis=0),
            color=color,
            linewidth=2,
            label=f"answer weight {weight}",
        )
    axes[1].axhline(0.8, color="black", linestyle="--", linewidth=0.8)
    axes[1].axhline(0.125, color="black", linestyle=":", linewidth=1)
    axes[1].set(
        xlabel="Extra recurrent cycle",
        ylabel="Collision-controlled accuracy",
        title="Closed-loop behavior",
        ylim=(0, 1.02),
    )
    axes[1].legend(frameon=False, fontsize=8)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    metric_axis.spines["top"].set_visible(False)
    fig.tight_layout()
    fig.savefig(figure_dir / "answer_weight_tradeoff.png", dpi=180)
    plt.close(fig)


def _plot_map_strength(
    root: Path,
    rows: list[dict[str, Any]],
    figure_dir: Path,
) -> None:
    if not rows:
        return
    rows = [row for row in rows if row["scan"] == "coarse"]
    strengths = sorted({float(row["strength"]) for row in rows})
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4))
    metric_axis = axes[0].twinx()
    for metric, label, color, axis in (
        ("auc64", "64-cycle AUC", "#0072B2", axes[0]),
        (
            "lifespan_at_0.8",
            "80% prefix lifespan",
            "#D55E00",
            metric_axis,
        ),
    ):
        values = np.asarray(
            [
                [
                    float(row[metric])
                    for row in rows
                    if float(row["strength"]) == strength
                ]
                for strength in strengths
            ]
        )
        axis.plot(
            strengths,
            values.mean(axis=1),
            marker="o",
            linewidth=2,
            color=color,
            label=label,
        )
        axis.fill_between(
            strengths,
            values.min(axis=1),
            values.max(axis=1),
            color=color,
            alpha=0.14,
        )
    axes[0].axvline(0.0, color="black", linestyle=":", linewidth=1)
    axes[0].axvline(1.0, color="black", linestyle="--", linewidth=0.8)
    axes[0].set(
        xlabel=r"Controller strength $\alpha$",
        ylabel="64-cycle AUC",
        xticks=strengths,
        title="Signed dose response",
    )
    metric_axis.set_ylabel("80% prefix lifespan")
    handles_left, labels_left = axes[0].get_legend_handles_labels()
    handles_right, labels_right = metric_axis.get_legend_handles_labels()
    axes[0].legend(
        handles_left + handles_right,
        labels_left + labels_right,
        frameon=False,
        fontsize=8.5,
    )

    colors = plt.cm.viridis(np.linspace(0.05, 0.95, len(strengths)))
    labels = {
        -1.0: "reverse",
        0.0: "identity",
        1.0: "learned R",
    }
    for strength, color in zip(strengths, colors, strict=True):
        curves = []
        selected_rows = sorted(
            (
                row
                for row in rows
                if float(row["strength"]) == strength
            ),
            key=lambda row: int(row["replicate"]),
        )
        for row in selected_rows:
            summary = _load(
                root / str(row["source_directory"]) / "summary.json"
            )
            curves.append(
                summary["closed_loop"][str(row["condition"])]["accuracy"]
            )
        array = _mean_curves(curves)
        cycles = np.arange(1, array.shape[1] + 1)
        axes[1].plot(
            cycles,
            array.mean(axis=0),
            color=color,
            linewidth=(2.4 if strength in labels else 1.4),
            label=labels.get(strength, f"α={strength:g}"),
        )
    axes[1].axhline(0.8, color="black", linestyle="--", linewidth=0.8)
    axes[1].axhline(0.125, color="black", linestyle=":", linewidth=1)
    axes[1].set(
        xlabel="Extra recurrent cycle",
        ylabel="Collision-controlled accuracy",
        title="Closed-loop dose trajectories",
        ylim=(0, 1.02),
    )
    axes[1].legend(frameon=False, fontsize=7.5, ncol=2)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    metric_axis.spines["top"].set_visible(False)
    fig.tight_layout()
    fig.savefig(figure_dir / "map_strength_dose_response.png", dpi=180)
    plt.close(fig)


def _plot_map_strength_fine(
    rows: list[dict[str, Any]],
    figure_dir: Path,
) -> None:
    rows = [row for row in rows if row["scan"] == "fine"]
    if not rows:
        return
    strengths = sorted({float(row["strength"]) for row in rows})
    fig, ax = plt.subplots(figsize=(7.6, 4.5))
    lifespan_axis = ax.twinx()
    for metric, label, color, axis in (
        ("auc64", "64-cycle AUC", "#0072B2", ax),
        (
            "lifespan_at_0.8",
            "80% prefix lifespan",
            "#D55E00",
            lifespan_axis,
        ),
    ):
        values = np.asarray(
            [
                [
                    float(row[metric])
                    for row in rows
                    if float(row["strength"]) == strength
                ]
                for strength in strengths
            ]
        )
        axis.plot(
            strengths,
            values.mean(axis=1),
            marker="o",
            linewidth=2,
            color=color,
            label=label,
        )
        axis.fill_between(
            strengths,
            values.min(axis=1),
            values.max(axis=1),
            color=color,
            alpha=0.14,
        )
    ax.axvline(1.0, color="black", linestyle="--", linewidth=0.8)
    ax.set(
        xlabel=r"Controller strength $\alpha$",
        ylabel="64-cycle AUC",
        xticks=strengths,
        title="Fine dose response on a matched evaluation split",
    )
    lifespan_axis.set_ylabel("80% prefix lifespan")
    left_handles, left_labels = ax.get_legend_handles_labels()
    right_handles, right_labels = lifespan_axis.get_legend_handles_labels()
    ax.legend(
        left_handles + right_handles,
        left_labels + right_labels,
        frameon=False,
        fontsize=8.5,
    )
    ax.spines[["top", "right"]].set_visible(False)
    lifespan_axis.spines["top"].set_visible(False)
    fig.tight_layout()
    fig.savefig(figure_dir / "map_strength_fine_response.png", dpi=180)
    plt.close(fig)


def _plot_probe_dose(
    rows: list[dict[str, Any]],
    figure_dir: Path,
) -> None:
    if not rows:
        return
    strength_rows = [
        row for row in rows if row["condition"] != "oracle_interface"
    ]
    cycles = sorted({int(row["cycle"]) for row in strength_rows})
    strengths = sorted({float(row["strength"]) for row in strength_rows})
    accuracy = np.asarray(
        [
            [
                np.mean(
                    [
                        float(row["accuracy"])
                        for row in strength_rows
                        if int(row["cycle"]) == cycle
                        and float(row["strength"]) == strength
                    ]
                )
                for cycle in cycles
            ]
            for strength in strengths
        ]
    )
    standard_index = strengths.index(1.0)
    delta = accuracy - accuracy[standard_index][None, :]
    oracle = np.asarray(
        [
            np.mean(
                [
                    float(row["accuracy"])
                    for row in rows
                    if int(row["cycle"]) == cycle
                    and row["condition"] == "oracle_interface"
                ]
            )
            for cycle in cycles
        ]
    )
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.5))
    limit = max(abs(float(delta.min())), abs(float(delta.max())), 0.01)
    image = axes[0].imshow(
        delta,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit,
    )
    axes[0].set_xticks(np.arange(len(cycles)), cycles)
    axes[0].set_yticks(
        np.arange(len(strengths)),
        [f"{strength:g}" for strength in strengths],
    )
    axes[0].set(
        xlabel="Probe cycle on the standard alpha=1 trajectory",
        ylabel=r"One-step probe strength $\alpha$",
        title=r"Immediate accuracy change vs $\alpha=1$",
    )
    fig.colorbar(image, ax=axes[0], label="Accuracy delta")
    best = accuracy.max(axis=0)
    axes[1].plot(
        cycles,
        accuracy[standard_index],
        marker="o",
        linewidth=2,
        label=r"Standard $\alpha=1$",
        color="#0072B2",
    )
    axes[1].plot(
        cycles,
        best,
        marker="o",
        linewidth=2,
        label="Best one-step dose in grid",
        color="#D55E00",
    )
    axes[1].plot(
        cycles,
        oracle,
        marker="o",
        linewidth=2,
        label="Exact interface",
        color="#009E73",
    )
    axes[1].axhline(0.125, color="black", linestyle=":", linewidth=1)
    axes[1].set(
        xlabel="Probe cycle",
        ylabel="Collision-controlled accuracy",
        title="Late failure is not rescued by dose retuning",
        ylim=(0, 1.02),
    )
    axes[1].legend(frameon=False, fontsize=8.5)
    axes[1].spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(figure_dir / "probe_dose_expiry.png", dpi=180)
    plt.close(fig)


def _plot_periodic_booster(
    root: Path,
    rows: list[dict[str, Any]],
    figure_dir: Path,
) -> None:
    if not rows:
        return
    specs = (
        (
            "periodic_booster24",
            "feedback_only",
            "Feedback only",
            "#999999",
        ),
        (
            "periodic_booster24",
            "learned_booster_once",
            "Cycle-24 affine reset once",
            "#0072B2",
        ),
        (
            "periodic_booster24",
            "learned_booster_periodic",
            "Same affine reset every 24 cycles",
            "#D55E00",
        ),
        (
            "periodic_booster24_dagger",
            "learned_booster_periodic",
            "Periodic affine + state DAgger",
            "#CC79A7",
        ),
        (
            "periodic_booster24",
            "exact_interface_periodic",
            "Exact interface every 24 cycles",
            "#009E73",
        ),
    )
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    for directory_prefix, condition, label, color in specs:
        curves = []
        for suffix in ("primary", "replica"):
            summary = _load(
                root / f"{directory_prefix}_{suffix}" / "summary.json"
            )
            curves.append(summary["closed_loop"][condition]["accuracy"])
        array = _mean_curves(curves)
        cycles = np.arange(1, array.shape[1] + 1)
        ax.plot(
            cycles,
            array.mean(axis=0),
            linewidth=2,
            label=label,
            color=color,
        )
        ax.fill_between(
            cycles,
            array.min(axis=0),
            array.max(axis=0),
            color=color,
            alpha=0.12,
        )
    for cycle in (24, 48, 72, 96, 120):
        ax.axvline(cycle, color="black", linestyle=":", linewidth=0.6)
    ax.axhline(0.8, color="black", linestyle="--", linewidth=0.8)
    ax.axhline(0.125, color="black", linestyle=":", linewidth=1)
    ax.set(
        xlabel="Extra recurrent cycle",
        ylabel="Collision-controlled accuracy",
        title="Preventive interface renewal: learned affine vs exact reset",
        xlim=(1, 128),
        ylim=(0, 1.02),
    )
    ax.legend(frameon=False, fontsize=8.5, ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(figure_dir / "periodic_booster.png", dpi=180)
    plt.close(fig)


def _plot_reset_age_cross(
    data: dict[str, list[dict[str, Any]]],
    figure_dir: Path,
) -> None:
    rows = data["summary"]
    if not rows:
        return
    condition_specs = (
        ("feedback", "Standard feedback", "#999999"),
        (
            "booster_trained_cycle24",
            "Reset learned at cycle 24",
            "#0072B2",
        ),
        (
            "booster_trained_cycle32",
            "Reset learned at cycle 32",
            "#D55E00",
        ),
        (
            "linear_age_extrapolation",
            "Linear age extrapolation",
            "#CC79A7",
        ),
        ("exact_interface", "Exact young interface", "#009E73"),
    )
    metric_specs = (
        ("accuracy", "Accuracy"),
        ("head2_q_answer_cosine", "Head-2 query cosine"),
        ("head2_context_answer_cosine", "Head-2 context cosine"),
        ("block2_mlp_answer_cosine", "Block-2 MLP cosine"),
    )
    cycles = sorted({int(row["cycle"]) for row in rows})
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.4), sharex=True)
    for axis, (metric, title) in zip(
        axes.flatten(),
        metric_specs,
        strict=True,
    ):
        for condition, label, color in condition_specs:
            values = np.asarray(
                [
                    [
                        float(row[metric])
                        for row in rows
                        if int(row["cycle"]) == cycle
                        and row["condition"] == condition
                    ]
                    for cycle in cycles
                ]
            )
            axis.plot(
                cycles,
                values.mean(axis=1),
                marker="o",
                linewidth=2,
                label=label,
                color=color,
            )
            axis.fill_between(
                cycles,
                values.min(axis=1),
                values.max(axis=1),
                color=color,
                alpha=0.12,
            )
        axis.axhline(
            0.125 if metric == "accuracy" else 0.0,
            color="black",
            linestyle=":",
            linewidth=0.8,
        )
        axis.set(
            title=title,
            ylabel=title,
            ylim=(0, 1.02),
            xticks=cycles,
        )
        axis.spines[["top", "right"]].set_visible(False)
    for axis in axes[1]:
        axis.set_xlabel("Probe cycle on the same feedback trajectory")
    axes[0, 0].legend(frameon=False, fontsize=8.5)
    fig.suptitle(
        "Cross-age reuse: cycle-24 and cycle-32 affine resets share a "
        "direction but expire together",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(figure_dir / "reset_age_cross_eval.png", dpi=180)
    plt.close(fig)


def _plot_reset_age_functional_trend(
    data: dict[str, list[dict[str, Any]]],
    figure_dir: Path,
) -> None:
    rows = data["trend"]
    if not rows:
        return
    cycles = sorted({int(row["cycle"]) for row in rows})
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.5))
    for key, label, color in (
        (
            "reset24_to_reset32_update_cosine",
            "Reset-24 vs reset-32",
            "#999999",
        ),
        (
            "reset24_to_exact_update_cosine",
            "Reset-24 vs exact-needed",
            "#0072B2",
        ),
        (
            "reset32_to_exact_update_cosine",
            "Reset-32 vs exact-needed",
            "#D55E00",
        ),
        (
            "age_linear_to_exact_update_cosine",
            "Age-linear vs exact-needed",
            "#CC79A7",
        ),
    ):
        values = np.asarray(
            [
                [
                    float(row[key])
                    for row in rows
                    if int(row["cycle"]) == cycle
                ]
                for cycle in cycles
            ]
        )
        axes[0].plot(
            cycles,
            values.mean(axis=1),
            marker="o",
            linewidth=2,
            label=label,
            color=color,
        )
        axes[0].fill_between(
            cycles,
            values.min(axis=1),
            values.max(axis=1),
            color=color,
            alpha=0.12,
        )
    axes[0].set(
        xlabel="Probe cycle",
        ylabel="Mean per-graph correction cosine",
        title="Reset maps stay mutually aligned,\nbut rotate away from exact need",
        xticks=cycles,
        ylim=(0.65, 1.01),
    )
    axes[0].legend(frameon=False, fontsize=8.2)
    for key, label, color in (
        ("reset24_update_norm", "Reset-24", "#0072B2"),
        ("reset32_update_norm", "Reset-32", "#D55E00"),
        ("age_linear_update_norm", "Age-linear", "#CC79A7"),
        ("exact_update_norm", "Exact-needed", "#009E73"),
    ):
        values = np.asarray(
            [
                [
                    float(row[key])
                    for row in rows
                    if int(row["cycle"]) == cycle
                ]
                for cycle in cycles
            ]
        )
        axes[1].plot(
            cycles,
            values.mean(axis=1),
            marker="o",
            linewidth=2,
            label=label,
            color=color,
        )
        axes[1].fill_between(
            cycles,
            values.min(axis=1),
            values.max(axis=1),
            color=color,
            alpha=0.12,
        )
    axes[1].set(
        xlabel="Probe cycle",
        ylabel="Mean flattened interface-update norm",
        title="Matching correction magnitude is not sufficient",
        xticks=cycles,
    )
    axes[1].legend(frameon=False, fontsize=8.2)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(
        figure_dir / "reset_age_functional_trend.png",
        dpi=180,
    )
    plt.close(fig)


def _plot_exact_renewal(
    root: Path,
    rows: list[dict[str, Any]],
    figure_dir: Path,
) -> None:
    period_rows = [row for row in rows if row["period"] is not None]
    if not period_rows:
        return
    periods = sorted({int(row["period"]) for row in period_rows})
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.5))
    for metric, label, color in (
        ("auc128", "128-cycle AUC", "#0072B2"),
        ("minimum_accuracy", "Worst-cycle accuracy", "#D55E00"),
    ):
        values = np.asarray(
            [
                [
                    float(row[metric])
                    for row in period_rows
                    if int(row["period"]) == period
                ]
                for period in periods
            ]
        )
        axes[0].plot(
            periods,
            values.mean(axis=1),
            marker="o",
            linewidth=2,
            label=label,
            color=color,
        )
        axes[0].fill_between(
            periods,
            values.min(axis=1),
            values.max(axis=1),
            color=color,
            alpha=0.14,
        )
    axes[0].axhline(0.8, color="black", linestyle="--", linewidth=0.8)
    axes[0].axhline(0.125, color="black", linestyle=":", linewidth=1)
    axes[0].set(
        xlabel="Exact renewal period",
        ylabel="Collision-controlled accuracy",
        xticks=periods,
        title="Maximum safe interface-renewal interval",
        ylim=(0, 1.02),
    )
    axes[0].legend(frameon=False, fontsize=8.5)
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, 5))
    for period, color in zip((16, 22, 24, 26, 32), colors, strict=True):
        curves = []
        for suffix in ("primary", "replica"):
            summary = _load(
                root / f"exact_renewal_scan_{suffix}" / "summary.json"
            )
            curves.append(
                summary["closed_loop"][f"exact_period_{period}"]["accuracy"]
            )
        array = _mean_curves(curves)
        cycles = np.arange(1, array.shape[1] + 1)
        axes[1].plot(
            cycles,
            array.mean(axis=0),
            linewidth=1.8,
            label=f"period={period}",
            color=color,
        )
    axes[1].axhline(0.8, color="black", linestyle="--", linewidth=0.8)
    axes[1].axhline(0.125, color="black", linestyle=":", linewidth=1)
    axes[1].set(
        xlabel="Extra recurrent cycle",
        ylabel="Collision-controlled accuracy",
        title="Exact-renewal trajectories",
        xlim=(1, 128),
        ylim=(0, 1.02),
    )
    axes[1].legend(frameon=False, fontsize=8.5, ncol=2)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(figure_dir / "exact_renewal_period_scan.png", dpi=180)
    plt.close(fig)


def _plot_periodic_booster_components(
    rows: list[dict[str, Any]],
    figure_dir: Path,
) -> None:
    if not rows:
        return
    metrics = (
        ("head2_q_answer_cosine", "Head2 answer Q"),
        ("head2_context_answer_cosine", "Head2 context"),
        ("block2_mlp_answer_cosine", "Block2 MLP output"),
    )
    conditions = (
        ("feedback_only", "Feedback only", "#999999"),
        (
            "learned_booster_periodic",
            "Learned affine reset",
            "#D55E00",
        ),
        (
            "exact_interface_periodic",
            "Exact interface reset",
            "#009E73",
        ),
    )
    cycles = sorted({int(row["cycle"]) for row in rows})
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.9), sharey=True)
    for axis, (metric, title) in zip(axes, metrics, strict=True):
        for condition, label, color in conditions:
            values = np.asarray(
                [
                    [
                        float(row[metric])
                        for row in rows
                        if int(row["cycle"]) == cycle
                        and row["condition"] == condition
                    ]
                    for cycle in cycles
                ]
            )
            axis.plot(
                cycles,
                values.mean(axis=1),
                marker="o",
                linewidth=2,
                label=label,
                color=color,
            )
            axis.fill_between(
                cycles,
                values.min(axis=1),
                values.max(axis=1),
                color=color,
                alpha=0.12,
            )
        axis.set(
            xlabel="Scheduled reset cycle",
            title=title,
            xticks=cycles,
            ylim=(0, 1.03),
        )
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Cosine to exact-young component")
    axes[-1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(
        figure_dir / "periodic_booster_component_reset.png",
        dpi=180,
    )
    plt.close(fig)


def _plot_role_booster(
    root: Path,
    rows: list[dict[str, Any]],
    figure_dir: Path,
) -> None:
    if not rows:
        return
    specs = (
        (
            "role_booster24_rank25",
            "feedback_only",
            "Feedback only",
            "#999999",
        ),
        (
            "role_booster24_rank25",
            "matched_shared_booster_periodic",
            "One shared full affine (65.8k)",
            "#0072B2",
        ),
        (
            "role_booster24_rank25",
            "learned_booster_periodic",
            "Five rank-25 role affines (62.2k)",
            "#D55E00",
        ),
        (
            "role_booster24_large",
            "learned_booster_periodic",
            "Five full role affines (328.9k)",
            "#CC79A7",
        ),
        (
            "role_booster24_rank25",
            "exact_interface_periodic",
            "Exact interface reset",
            "#009E73",
        ),
    )
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    for prefix, condition, label, color in specs:
        if not all(
            (root / f"{prefix}_{suffix}" / "summary.json").exists()
            for suffix in ("primary", "replica")
        ):
            continue
        curves = []
        for suffix in ("primary", "replica"):
            summary = _load(
                root / f"{prefix}_{suffix}" / "summary.json"
            )
            curves.append(summary["closed_loop"][condition]["accuracy"])
        array = _mean_curves(curves)
        cycles = np.arange(1, array.shape[1] + 1)
        ax.plot(
            cycles,
            array.mean(axis=0),
            linewidth=2,
            label=label,
            color=color,
        )
        ax.fill_between(
            cycles,
            array.min(axis=0),
            array.max(axis=0),
            color=color,
            alpha=0.12,
        )
    for cycle in (24, 48, 72, 96, 120):
        ax.axvline(cycle, color="black", linestyle=":", linewidth=0.6)
    ax.axhline(0.8, color="black", linestyle="--", linewidth=0.8)
    ax.axhline(0.125, color="black", linestyle=":", linewidth=1)
    ax.set(
        xlabel="Extra recurrent cycle",
        ylabel="Collision-controlled accuracy",
        title="Does separating token roles solve affine renewal?",
        xlim=(1, 128),
        ylim=(0, 1.02),
    )
    ax.legend(frameon=False, fontsize=8.5, ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(figure_dir / "role_conditioned_booster.png", dpi=180)
    plt.close(fig)


def _plot_role_hybrid(
    root: Path,
    rows: list[dict[str, Any]],
    figure_dir: Path,
) -> None:
    if not rows:
        return
    specs = (
        ("shared_all", "Shared affine at every role", "#0072B2"),
        ("answer_only", "Role-specific answer only", "#E69F00"),
        ("metadata_only", "Role-specific metadata only", "#56B4E9"),
        ("graph_roles", "Role-specific graph roles", "#CC79A7"),
        ("all_roles", "All five role-specific", "#D55E00"),
        ("exact_interface", "Exact interface reset", "#009E73"),
    )
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    for condition, label, color in specs:
        curves = []
        for suffix in ("primary", "replica"):
            path = root / f"role_hybrid24_{suffix}" / "summary.json"
            if not path.exists():
                continue
            curves.append(
                _load(path)["closed_loop"][condition]["accuracy"]
            )
        if not curves:
            continue
        array = _mean_curves(curves)
        cycles = np.arange(1, array.shape[1] + 1)
        ax.plot(
            cycles,
            array.mean(axis=0),
            linewidth=2,
            label=label,
            color=color,
        )
        ax.fill_between(
            cycles,
            array.min(axis=0),
            array.max(axis=0),
            color=color,
            alpha=0.12,
        )
    for cycle in (24, 48, 72, 96, 120):
        ax.axvline(cycle, color="black", linestyle=":", linewidth=0.6)
    ax.axhline(0.8, color="black", linestyle="--", linewidth=0.8)
    ax.axhline(0.125, color="black", linestyle=":", linewidth=1)
    ax.set(
        xlabel="Extra recurrent cycle",
        ylabel="Collision-controlled accuracy",
        title="Which token-role separations produce renewal gain?",
        xlim=(1, 128),
        ylim=(0, 1.02),
    )
    ax.legend(frameon=False, fontsize=8.3, ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(figure_dir / "role_hybrid_ablation.png", dpi=180)
    plt.close(fig)


def _plot_role_factorial(
    data: dict[str, list[dict[str, Any]]],
    figure_dir: Path,
) -> None:
    if not data["shapley"]:
        return
    roles = (
        "edge_marker",
        "source",
        "destination",
        "query_metadata",
        "answer",
    )
    labels = ("Edge marker", "Source", "Destination", "Metadata", "Answer")
    trainings = [
        training
        for training in ("rank25", "full")
        if any(row["training"] == training for row in data["shapley"])
    ]
    fig, axes = plt.subplots(
        1,
        1 + len(trainings),
        figsize=(6.1 + 4.6 * len(trainings), 4.5),
        squeeze=False,
        layout="constrained",
    )
    axes = axes[0]
    x = np.arange(len(roles))
    width = 0.34 if len(trainings) == 2 else 0.55
    colors = {"rank25": "#0072B2", "full": "#D55E00"}
    labels_by_training = {
        "rank25": "Rank-25 per role",
        "full": "Full-rank per role",
    }
    for training_index, training in enumerate(trainings):
        values = np.asarray(
            [
                [
                    float(row["aggregate_auc64_shapley"])
                    for row in data["shapley"]
                    if row["training"] == training
                    and row["role"] == role
                ]
                for role in roles
            ],
            dtype=float,
        )
        means = values.mean(axis=1)
        errors = np.stack(
            [means - values.min(axis=1), values.max(axis=1) - means]
        )
        offset = (
            (training_index - (len(trainings) - 1) / 2) * width
        )
        axes[0].bar(
            x + offset,
            means,
            width=width,
            yerr=errors,
            capsize=3,
            color=colors[training],
            label=labels_by_training[training],
        )
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_xticks(
        x,
        labels,
        rotation=25,
        ha="right",
    )
    axes[0].set(
        ylabel="AUC64 Shapley contribution",
        title="Average marginal value of role separation",
    )
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].spines[["top", "right"]].set_visible(False)

    matrices = []
    for training in trainings:
        matrix = np.zeros((len(roles), len(roles)), dtype=float)
        for row in data["interactions"]:
            if row["training"] != training:
                continue
            first = roles.index(row["role_a"])
            second = roles.index(row["role_b"])
            matrix[first, second] += (
                float(row["uniform_subset_interaction_auc64"]) / 2
            )
            matrix[second, first] = matrix[first, second]
        np.fill_diagonal(matrix, np.nan)
        matrices.append(matrix)
    limit = max(
        max(float(np.nanmax(np.abs(matrix))), 1e-6)
        for matrix in matrices
    )
    interaction_cmap = plt.get_cmap("RdBu_r").copy()
    interaction_cmap.set_bad("#D9D9D9")
    for axis, training, matrix in zip(
        axes[1:],
        trainings,
        matrices,
        strict=True,
    ):
        image = axis.imshow(
            matrix,
            cmap=interaction_cmap,
            vmin=-limit,
            vmax=limit,
        )
        for row_index in range(len(roles)):
            for column_index in range(len(roles)):
                if row_index == column_index:
                    axis.text(
                        column_index,
                        row_index,
                        "self",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="#555555",
                    )
                    continue
                axis.text(
                    column_index,
                    row_index,
                    f"{matrix[row_index, column_index]:+.3f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="black",
                )
        axis.set_xticks(
            np.arange(len(roles)),
            labels,
            rotation=25,
            ha="right",
        )
        axis.set_yticks(np.arange(len(roles)), labels)
        axis.set_title(
            f"{labels_by_training[training]}\npair interaction"
        )
    fig.colorbar(image, ax=axes[1:].tolist(), fraction=0.025, pad=0.03)
    fig.savefig(figure_dir / "role_factorial_decomposition.png", dpi=180)
    plt.close(fig)


def _plot_role_rank_sweep(
    data: dict[str, list[dict[str, Any]]],
    figure_dir: Path,
) -> None:
    if not data["summary"]:
        return
    ranks = sorted(
        {
            int(row["rank"])
            for row in data["summary"]
            if row["condition"] == "learned_booster_periodic"
        }
    )
    rank_positions = np.arange(len(ranks))
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15.6, 4.5),
        layout="constrained",
    )
    auc = np.asarray(
        [
            [
                float(row["auc64"])
                for row in data["summary"]
                if int(row["rank"]) == rank
                and row["condition"] == "learned_booster_periodic"
            ]
            for rank in ranks
        ]
    )
    axes[0].plot(
        rank_positions,
        auc.mean(axis=1),
        color="#D55E00",
        marker="o",
        linewidth=2,
        label="Five role-specific maps",
    )
    axes[0].fill_between(
        rank_positions,
        auc.min(axis=1),
        auc.max(axis=1),
        color="#D55E00",
        alpha=0.14,
    )
    shared = np.mean(
        [
            float(row["auc64"])
            for row in data["summary"]
            if row["condition"] == "matched_shared_booster_periodic"
        ]
    )
    exact = np.mean(
        [
            float(row["auc64"])
            for row in data["summary"]
            if row["condition"] == "exact_interface_periodic"
        ]
    )
    axes[0].axhline(
        shared,
        color="#0072B2",
        linestyle="--",
        label="Matched shared full affine",
    )
    axes[0].axhline(
        exact,
        color="#009E73",
        linestyle=":",
        label="Exact interface",
    )
    axes[0].set(
        xlabel="Update rank per role",
        ylabel="AUC64",
        title="Behavioral rank requirement",
        xticks=rank_positions,
    )
    axes[0].set_xticklabels(ranks)
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].spines[["top", "right"]].set_visible(False)

    colors = {
        "edge_marker": "#999999",
        "source": "#56B4E9",
        "destination": "#0072B2",
        "query_metadata": "#D55E00",
        "answer": "#E69F00",
    }
    labels = {
        "edge_marker": "Edge marker",
        "source": "Source",
        "destination": "Destination",
        "query_metadata": "Metadata",
        "answer": "Answer",
    }
    for role in colors:
        values = np.asarray(
            [
                [
                    float(row["relative_mse"])
                    for row in data["heldout"]
                    if int(row["rank"]) == rank
                    and row["role"] == role
                ]
                for rank in ranks
            ]
        )
        axes[1].plot(
            rank_positions,
            values.mean(axis=1),
            color=colors[role],
            marker="o",
            linewidth=1.8,
            label=labels[role],
        )
        axes[1].fill_between(
            rank_positions,
            values.min(axis=1),
            values.max(axis=1),
            color=colors[role],
            alpha=0.1,
        )
    axes[1].set(
        xlabel="Update rank per role",
        ylabel="Held-out relative state MSE",
        title="Different role corrections need different capacity",
        xticks=rank_positions,
    )
    axes[1].set_xticklabels(ranks)
    axes[1].legend(frameon=False, fontsize=8, ncol=2)
    axes[1].spines[["top", "right"]].set_visible(False)
    for role in colors:
        values = np.asarray(
            [
                [
                    float(row["retained_fit_energy"])
                    for row in data["heldout"]
                    if int(row["rank"]) == rank
                    and row["role"] == role
                ]
                for rank in ranks
            ]
        )
        axes[2].plot(
            rank_positions,
            values.mean(axis=1),
            color=colors[role],
            marker="o",
            linewidth=1.8,
            label=labels[role],
        )
        axes[2].fill_between(
            rank_positions,
            values.min(axis=1),
            values.max(axis=1),
            color=colors[role],
            alpha=0.1,
        )
    axes[2].set(
        xlabel="Update rank per role",
        ylabel="Retained fitted-update energy",
        title="Energy retention is not a behavioral dimension",
        xticks=rank_positions,
        ylim=(-0.02, 1.02),
    )
    axes[2].set_xticklabels(ranks)
    axes[2].spines[["top", "right"]].set_visible(False)
    fig.savefig(figure_dir / "role_rank_sweep.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    figure_dir = root / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    lifespan_rows = _lifespan_rows(root)
    position_rows = _position_rows(root)
    map_ablation_rows = _map_ablation_rows(root)
    power_rows = _power_rows(root)
    component_patch_rows = _component_patch_rows(root)
    map_geometry = _map_geometry(root)
    role_action_rows = _role_action_rows(root)
    rank_sweep_rows = _rank_sweep_rows(root)
    initialization_rows = _initialization_rows(root)
    learned_initializer_rows = _learned_initializer_rows(root)
    initializer_geometry = _initializer_geometry(root)
    joint_single_rows = _joint_single_rows(root)
    joint_weight_rows = _joint_weight_rows(root)
    answer_weight_rows = _answer_weight_rows(root)
    map_strength_rows = _map_strength_rows(root)
    probe_dose_rows = _probe_dose_rows(root)
    periodic_booster = _periodic_booster_data(root)
    reset_age_cross = _reset_age_cross_data(root)
    exact_renewal_rows = _exact_renewal_rows(root)
    periodic_booster_components = _periodic_booster_component_rows(root)
    role_booster = _role_booster_data(root)
    role_booster_paired = _role_booster_paired_effect(root)
    role_hybrid = _role_hybrid_data(root)
    role_factorial = _role_factorial_data(root)
    role_rank_sweep = _role_rank_sweep_data(root)
    observability = _observability_data(root)
    orbit_control = _orbit_control_data(root)
    per_node = _per_node_data(root)
    orbit_component_drift = _orbit_component_drift_rows(orbit_control)
    _write_csv(root / "lifespan_scaling.csv", lifespan_rows)
    _write_csv(root / "position_circuit_summary.csv", position_rows)
    if map_ablation_rows:
        _write_csv(root / "learned_map_ablation_summary.csv", map_ablation_rows)
    _write_csv(root / "power_law_summary.csv", power_rows)
    _write_csv(
        root / "long_component_patch_summary.csv",
        component_patch_rows,
    )
    _write_csv(root / "map_geometry_summary.csv", map_geometry["rows"])
    _write_csv(root / "shared_map_role_action_summary.csv", role_action_rows)
    _write_csv(root / "behavioral_rank_sweep.csv", rank_sweep_rows)
    _write_csv(root / "initialization_gate_summary.csv", initialization_rows)
    _write_csv(
        root / "learned_initializer_summary.csv",
        learned_initializer_rows,
    )
    _write_csv(
        root / "initializer_feedback_geometry.csv",
        initializer_geometry["rows"],
    )
    _write_csv(root / "joint_single_map_summary.csv", joint_single_rows)
    _write_csv(root / "joint_weight_tradeoff.csv", joint_weight_rows)
    _write_csv(root / "answer_weight_tradeoff.csv", answer_weight_rows)
    _write_csv(root / "map_strength_dose_response.csv", map_strength_rows)
    _write_csv(root / "probe_dose_response.csv", probe_dose_rows)
    _write_csv(
        root / "periodic_booster_summary.csv",
        periodic_booster["summary"],
    )
    _write_csv(
        root / "periodic_booster_geometry.csv",
        periodic_booster["geometry"],
    )
    _write_csv(
        root / "reset_age_cross_summary.csv",
        reset_age_cross["summary"],
    )
    _write_csv(
        root / "reset_age_cross_geometry.csv",
        reset_age_cross["geometry"],
    )
    _write_csv(
        root / "reset_age_cross_paired_effect.csv",
        reset_age_cross["paired"],
    )
    _write_csv(
        root / "reset_age_functional_trend.csv",
        reset_age_cross["trend"],
    )
    _write_csv(root / "exact_renewal_period_scan.csv", exact_renewal_rows)
    _write_csv(
        root / "periodic_booster_component_summary.csv",
        periodic_booster_components,
    )
    _write_csv(root / "role_booster_summary.csv", role_booster["summary"])
    _write_csv(
        root / "role_booster_heldout.csv",
        role_booster["heldout"],
    )
    _write_csv(
        root / "role_booster_component_summary.csv",
        role_booster["components"],
    )
    _write_csv(
        root / "role_booster_paired_effect.csv",
        role_booster_paired,
    )
    _write_csv(root / "role_hybrid_summary.csv", role_hybrid["summary"])
    _write_csv(
        root / "role_hybrid_component_summary.csv",
        role_hybrid["components"],
    )
    _write_csv(
        root / "role_hybrid_paired_effect.csv",
        role_hybrid["paired_effect"],
    )
    _write_csv(
        root / "role_factorial_summary.csv",
        role_factorial["summary"],
    )
    _write_csv(
        root / "role_factorial_shapley.csv",
        role_factorial["shapley"],
    )
    _write_csv(
        root / "role_factorial_interactions.csv",
        role_factorial["interactions"],
    )
    _write_csv(
        root / "role_rank_sweep_summary.csv",
        role_rank_sweep["summary"],
    )
    _write_csv(
        root / "role_rank_sweep_heldout.csv",
        role_rank_sweep["heldout"],
    )
    _write_csv(
        root / "role_rank_sweep_paired_effect.csv",
        role_rank_sweep["paired_effect"],
    )
    _write_csv(
        root / "intermediate_observability_summary.csv",
        observability["summary"],
    )
    _write_csv(
        root / "intermediate_observability_per_node.csv",
        observability["per_node"],
    )
    _write_csv(
        root / "cycle3_circuit_scope.csv",
        observability["circuit"],
    )
    _write_csv(
        root / "cycle3_circuit_per_node.csv",
        observability["circuit_per_node"],
    )
    for key, filename in (
        ("summary", "orbit_control_summary.csv"),
        ("paired", "orbit_control_paired.csv"),
        ("lifespans", "orbit_control_lifespan.csv"),
        ("components", "orbit_control_component_curves.csv"),
        ("patches", "orbit8_component_patch_curves.csv"),
    ):
        _write_csv(root / filename, orbit_control[key])
    _write_csv(
        root / "orbit_component_drift_summary.csv",
        orbit_component_drift,
    )
    for key, filename in (
        ("lifespan_summary", "per_node_lifespan_summary.csv"),
        ("component_patches", "per_node_component_patch_summary.csv"),
        (
            "component_similarities",
            "per_node_component_similarity_summary.csv",
        ),
    ):
        _write_csv(root / filename, per_node[key])
    _plot_lifespan(root, figure_dir)
    _plot_position_circuit(position_rows, figure_dir)
    _plot_layer_boundary(position_rows, figure_dir)
    _plot_power(power_rows, figure_dir)
    if map_ablation_rows:
        _plot_map_ablation(map_ablation_rows, figure_dir)
    _plot_components(root, figure_dir)
    _plot_component_patches(component_patch_rows, figure_dir)
    _plot_role_action(role_action_rows, figure_dir)
    _plot_rank_sweep(rank_sweep_rows, figure_dir)
    _plot_initialization_gate(initialization_rows, figure_dir)
    _plot_learned_initializer(learned_initializer_rows, figure_dir)
    _plot_joint_single_map(joint_single_rows, figure_dir)
    _plot_joint_weight_tradeoff(
        joint_weight_rows,
        learned_initializer_rows,
        figure_dir,
    )
    _plot_observability(observability, figure_dir)
    _plot_cycle3_scope(observability["circuit"], figure_dir)
    _plot_orbit_control(orbit_control, figure_dir)
    _plot_per_node_lifespan(per_node, figure_dir)
    _plot_answer_weight(root, answer_weight_rows, figure_dir)
    _plot_map_strength(root, map_strength_rows, figure_dir)
    _plot_map_strength_fine(map_strength_rows, figure_dir)
    _plot_probe_dose(probe_dose_rows, figure_dir)
    _plot_periodic_booster(
        root,
        periodic_booster["summary"],
        figure_dir,
    )
    _plot_reset_age_cross(reset_age_cross, figure_dir)
    _plot_reset_age_functional_trend(reset_age_cross, figure_dir)
    _plot_exact_renewal(root, exact_renewal_rows, figure_dir)
    _plot_periodic_booster_components(
        periodic_booster_components,
        figure_dir,
    )
    _plot_role_booster(root, role_booster["summary"], figure_dir)
    _plot_role_hybrid(root, role_hybrid["summary"], figure_dir)
    _plot_role_factorial(role_factorial, figure_dir)
    _plot_role_rank_sweep(role_rank_sweep, figure_dir)

    summary = {
        "status": "complete",
        "replications": 2,
        "chance_accuracy": 0.125,
        "lifespan_scaling": lifespan_rows,
        "position_circuit": position_rows,
        "learned_map_ablation": map_ablation_rows,
        "power_law": power_rows,
        "long_component_patching": component_patch_rows,
        "map_geometry": map_geometry,
        "shared_map_role_action": role_action_rows,
        "behavioral_rank_sweep": rank_sweep_rows,
        "initialization_gate": initialization_rows,
        "learned_initializer": learned_initializer_rows,
        "initializer_feedback_geometry": initializer_geometry,
        "joint_single_map": joint_single_rows,
        "joint_weight_tradeoff": joint_weight_rows,
        "answer_weight_tradeoff": answer_weight_rows,
        "map_strength_dose_response": map_strength_rows,
        "probe_dose_response": probe_dose_rows,
        "periodic_booster": periodic_booster,
        "reset_age_cross": reset_age_cross,
        "exact_renewal_period_scan": exact_renewal_rows,
        "periodic_booster_components": periodic_booster_components,
        "role_booster": role_booster,
        "role_booster_paired_effect": role_booster_paired,
        "role_hybrid": role_hybrid,
        "role_factorial": role_factorial,
        "role_rank_sweep": role_rank_sweep,
        "intermediate_observability": observability,
        "orbit_control": orbit_control,
        "orbit_component_drift": orbit_component_drift,
        "per_node": per_node,
        "figures": sorted(path.name for path in figure_dir.glob("*.png")),
    }
    (root / "FINAL_SUMMARY.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps({"status": "complete", "root": str(root)}, indent=2))


if __name__ == "__main__":
    main()
