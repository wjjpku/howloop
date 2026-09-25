from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from reasoning_loop.graph_path_functional_circuit_aggregate import (
    _localized_transition_executor,
    aggregate_run,
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0])
    known = set(fieldnames)
    for row in rows[1:]:
        for key in row:
            if key not in known:
                fieldnames.append(key)
                known.add(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _float(value: str | float | int | None) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")


def _passes(value: float, threshold: float) -> bool:
    return math.isfinite(value) and value > threshold


def _run_parts(name: str) -> tuple[str, int]:
    match = re.fullmatch(r"(D\d+_L\d+)_seed(\d+)", name)
    if match is None:
        raise ValueError(f"unexpected run name: {name}")
    return match.group(1), int(match.group(2))


def _endpoint_rows(
    rows: list[dict[str, str]],
    *,
    endpoint_position: int,
) -> list[dict[str, str]]:
    return [
        row
        for row in rows
        if int(row["path_position"]) == endpoint_position
    ]


def _component_label(block: int, branch: str) -> str:
    return f"B{block}.{'attn' if branch == 'attention' else 'mlp'}"


def summarize_compression_run(run_dir: Path) -> tuple[
    dict[str, Any], dict[str, dict[str, Any]]
]:
    summary = json.loads((run_dir / "summary.json").read_text())
    name = summary["name"]
    family, seed = _run_parts(name)
    endpoint = int(summary["config"]["max_depth"])
    trained_loops = int(summary["trained_loops"])

    unroll = _read_csv(run_dir / "unroll_curve_rows.csv")
    unroll_by_loop: dict[int, dict[str, str]] = {}
    for row in unroll:
        unroll_by_loop.setdefault(int(row["loop"]), row)
    best_sequence = [
        int(unroll_by_loop[loop]["best_path_position"])
        for loop in sorted(unroll_by_loop)
    ]
    accuracy_sequence = [
        _float(unroll_by_loop[loop]["best_path_accuracy"])
        for loop in sorted(unroll_by_loop)
    ]
    endpoint_accuracy_by_loop = {
        int(row["loop"]): _float(row["accuracy"])
        for row in unroll
        if int(row["path_position"]) == endpoint
    }
    earliest_endpoint_loop = next(
        (
            loop
            for loop in sorted(endpoint_accuracy_by_loop)
            if endpoint_accuracy_by_loop[loop] >= 0.80
        ),
        -1,
    )
    confident_sequence = [
        position if accuracy >= 0.50 else -1
        for position, accuracy in zip(
            best_sequence, accuracy_sequence, strict=True
        )
    ]
    if confident_sequence[:4] == [2, 4, 6, 8]:
        trajectory_subtype = "two_step_stride"
    elif confident_sequence[: min(8, trained_loops)] == list(
        range(1, min(8, trained_loops) + 1)
    ):
        trajectory_subtype = "one_step_stride"
    elif any(
        current >= 0
        and previous >= 0
        and current - previous > 2
        for previous, current in zip(
            confident_sequence, confident_sequence[1:], strict=False
        )
    ):
        trajectory_subtype = "abrupt_jump"
    else:
        trajectory_subtype = "mixed_or_hidden"
    run_row: dict[str, Any] = {
        "run": name,
        "family": family,
        "seed": seed,
        "trained_loops": trained_loops,
        "physical_blocks": summary["physical_blocks"],
        "loss_mode": summary["loss_mode"],
        "baseline_endpoint_accuracy": summary[
            "baseline_endpoint_accuracy"
        ],
        "baseline_endpoint_margin": summary["baseline_endpoint_margin"],
        "best_path_sequence": " ".join(map(str, best_sequence)),
        "confident_path_sequence": " ".join(
            map(str, confident_sequence)
        ),
        "best_path_accuracy_sequence": " ".join(
            f"{value:.4f}" for value in accuracy_sequence
        ),
        "trajectory_subtype": trajectory_subtype,
        "earliest_endpoint_loop_acc80": earliest_endpoint_loop,
        "unused_trained_loops_after_acc80": (
            trained_loops - earliest_endpoint_loop
            if earliest_endpoint_loop > 0
            else -1
        ),
        "first_loop_best_position": best_sequence[0],
        "first_loop_best_accuracy": accuracy_sequence[0],
        "preterminal_best_position": best_sequence[trained_loops - 2],
        "preterminal_best_accuracy": accuracy_sequence[trained_loops - 2],
        "trained_horizon_best_position": best_sequence[trained_loops - 1],
        "trained_horizon_best_accuracy": accuracy_sequence[
            trained_loops - 1
        ],
    }
    for extra in range(1, len(best_sequence) - trained_loops + 1):
        run_row[f"extra{extra}_best_position"] = best_sequence[
            trained_loops + extra - 1
        ]
        run_row[f"extra{extra}_best_accuracy"] = accuracy_sequence[
            trained_loops + extra - 1
        ]
        endpoint_row = next(
            row
            for row in unroll
            if int(row["loop"]) == trained_loops + extra
            and int(row["path_position"]) == endpoint
        )
        run_row[f"extra{extra}_endpoint_accuracy"] = _float(
            endpoint_row["accuracy"]
        )

    transplant = _read_csv(run_dir / "within_model_transplant_rows.csv")
    if transplant:
        best_transition = max(
            transplant, key=lambda row: _float(row["next_successor_accuracy"])
        )
        best_specific = max(
            transplant, key=lambda row: _float(row["successor_specificity"])
        )
        best_donor = int(best_transition["donor_loop"])
        same_donor = [
            row
            for row in transplant
            if int(row["donor_loop"]) == best_donor
        ]
        receiver_values = [
            _float(row["next_successor_accuracy"]) for row in same_donor
        ]
        run_row.update(
            {
                "transplant_decoded_positions": " ".join(
                    str(int(row["decoded_position"]))
                    for row in transplant
                    if int(row["receiver_loop"]) == 1
                ),
                "transplant_max_next_accuracy": _float(
                    best_transition["next_successor_accuracy"]
                ),
                "transplant_best_donor_loop": int(
                    best_transition["donor_loop"]
                ),
                "transplant_best_receiver_loop": int(
                    best_transition["receiver_loop"]
                ),
                "transplant_max_specificity": _float(
                    best_specific["successor_specificity"]
                ),
                "transplant_transition_gate": int(
                    any(
                        _float(row["next_successor_accuracy"]) >= 0.40
                        and _float(row["successor_specificity"]) >= 0.10
                        for row in transplant
                    )
                ),
                "transplant_best_donor_receiver_range": (
                    max(receiver_values) - min(receiver_values)
                ),
            }
        )

    progression = _read_csv(
        run_dir / "branch_semantic_progression_rows.csv"
    )
    progression_sites: dict[str, dict[str, dict[str, str]]] = {}
    for row in progression:
        progression_sites.setdefault(row["site"], {}).setdefault(
            row["stage"], row
        )

    effects = _endpoint_rows(
        _read_csv(run_dir / "branch_skip_repeat_rows.csv"),
        endpoint_position=endpoint,
    )
    by_component: dict[str, list[dict[str, str]]] = {}
    for row in effects:
        component = f"B{row['block']}.{row['branch']}"
        by_component.setdefault(component, []).append(row)

    overloop = _endpoint_rows(
        _read_csv(run_dir / "overloop_branch_rows.csv"),
        endpoint_position=endpoint,
    )
    overloop_by_component: dict[str, list[dict[str, str]]] = {}
    for row in overloop:
        component = f"B{row['block']}.{row['branch']}"
        overloop_by_component.setdefault(component, []).append(row)

    components: dict[str, dict[str, Any]] = {}
    for component, rows in by_component.items():
        skip_rows = [row for row in rows if row["mode"] == "skip"]
        repeat_rows = [row for row in rows if row["mode"] == "repeat"]
        skip_drops = [_float(row["endpoint_accuracy_drop"]) for row in skip_rows]
        repeat_drops = [
            _float(row["endpoint_accuracy_drop"]) for row in repeat_rows
        ]
        semantic_shift_sites: list[str] = []
        for site in sorted({row["site"] for row in rows}):
            skip = next(
                row
                for row in skip_rows
                if row["site"] == site
            )
            repeat = next(
                row
                for row in repeat_rows
                if row["site"] == site
            )
            skip_step = (
                int(skip["best_path_position"]) == endpoint - 1
                and _float(skip["best_path_accuracy"]) >= 0.50
                and _float(skip["endpoint_accuracy_drop"]) > 0.10
            )
            repeat_step = (
                int(repeat["best_path_position"]) == endpoint + 1
                and _float(repeat["best_path_accuracy"]) >= 0.50
                and _float(repeat["endpoint_accuracy_drop"]) > 0.10
            )
            if skip_step or repeat_step:
                semantic_shift_sites.append(site)

        jumps: list[tuple[str, int, float, float]] = []
        for site, stages in progression_sites.items():
            if not site.endswith(
                "." + component.split(".", 1)[1]
            ):
                continue
            if f".B{component[1]}." not in site:
                continue
            before = stages["before"]
            after = stages["after"]
            before_accuracy = _float(before["best_path_accuracy"])
            after_accuracy = _float(after["best_path_accuracy"])
            jump = int(after["best_path_position"]) - int(
                before["best_path_position"]
            )
            jumps.append(
                (site, jump, before_accuracy, after_accuracy)
            )
        reliable_jumps = [
            item
            for item in jumps
            if item[2] >= 0.50 and item[3] >= 0.50
        ]
        max_jump = (
            max(reliable_jumps, key=lambda item: item[1])
            if reliable_jumps
            else ("", 0, float("nan"), float("nan"))
        )

        extra_rows = overloop_by_component.get(component, [])
        extra_skip = [
            row for row in extra_rows if row["mode"] == "skip"
        ]
        extra_repeat = [
            row for row in extra_rows if row["mode"] == "repeat"
        ]
        components[component] = {
            "run": name,
            "family": family,
            "seed": seed,
            "component": component,
            "max_skip_endpoint_drop": max(skip_drops),
            "max_skip_site": skip_rows[
                max(range(len(skip_rows)), key=lambda i: skip_drops[i])
            ]["site"],
            "skip_drop_range": max(skip_drops) - min(skip_drops),
            "max_repeat_endpoint_drop": max(repeat_drops),
            "max_repeat_site": repeat_rows[
                max(range(len(repeat_rows)), key=lambda i: repeat_drops[i])
            ]["site"],
            "repeat_drop_range": max(repeat_drops) - min(repeat_drops),
            "semantic_shift_site_count": len(semantic_shift_sites),
            "semantic_shift_sites": " ".join(semantic_shift_sites),
            "max_reliable_decoded_jump": max_jump[1],
            "max_reliable_jump_site": max_jump[0],
            "max_overloop_skip_drop": (
                max(_float(row["endpoint_accuracy_drop"]) for row in extra_skip)
                if extra_skip
                else float("nan")
            ),
            "max_overloop_skip_improvement": (
                max(
                    -_float(row["endpoint_accuracy_drop"])
                    for row in extra_skip
                )
                if extra_skip
                else float("nan")
            ),
            "max_overloop_repeat_drop": (
                max(
                    _float(row["endpoint_accuracy_drop"])
                    for row in extra_repeat
                )
                if extra_repeat
                else float("nan")
            ),
        }
    return run_row, components


def functional_roles(
    run_dir: Path,
    *,
    top_k: int,
) -> dict[str, dict[str, Any]]:
    summary = json.loads((run_dir / "summary.json").read_text())
    name = summary["name"]
    family, seed = _run_parts(name)
    rows = aggregate_run(run_dir, top_k=top_k)
    transition_rows = _read_csv(run_dir / "transition_function_rows.csv")
    executor = (
        _localized_transition_executor(transition_rows)
        if summary["functional_tests"]["transition"]["status"]
        == "passed_heldout_transition_gate"
        else None
    )
    position_rows = _read_csv(run_dir / "position_ablation_rows.csv")
    output: dict[str, dict[str, Any]] = {}
    for block in range(1, int(summary["physical_blocks"]) + 1):
        for branch, raw_branch, component_name in (
            ("attention", "attn", "attention_out"),
            ("mlp", "mlp", "mlp_out"),
        ):
            selected = [
                row
                for row in rows
                if row["block"] == block and row["branch"] == branch
            ]
            raw_position = [
                row
                for row in position_rows
                if int(row["block"]) == block
                and row["component"] == component_name
            ]
            roles: list[str] = []
            evidence: dict[str, float | int | str] = {}
            if branch == "attention":
                graph_content = max(
                    max(
                        _float(row["graph_k_specific_recovery"]),
                        _float(row["graph_v_specific_recovery"]),
                    )
                    for row in selected
                )
                query_routing = max(
                    _float(row["query_q_specific_recovery"])
                    for row in selected
                )
                evidence["graph_content_max"] = graph_content
                evidence["query_routing_max"] = query_routing
                if graph_content > 0.10:
                    roles.append("graph_content")
                if query_routing > 0.10:
                    roles.append("answer_query_routing")
                if executor is not None and executor["block"] == block:
                    roles.append("edge_specific_successor")
            else:
                query_transform = max(
                    _float(row["query_mlp_specific_recovery"])
                    for row in selected
                )
                neuron_specificity = max(
                    _float(row["neuron_top_specificity"])
                    for row in selected
                )
                transition_support = max(
                    _float(row["transition_output_accuracy_drop"])
                    for row in selected
                )
                evidence["query_transform_max"] = query_transform
                evidence["neuron_specificity_max"] = neuron_specificity
                evidence["transition_output_drop_max"] = transition_support
                if query_transform > 0.10 and neuron_specificity > 0.02:
                    roles.append("answer_nonlinear_transform")
                if (
                    summary["functional_tests"]["transition"]["status"]
                    == "passed_heldout_transition_gate"
                    and transition_support > 0.10
                    and neuron_specificity > 0.02
                ):
                    roles.append("transition_nonlinear_support")

            graph_token_effect = max(
                (
                    _float(row["accuracy_drop"])
                    if _float(row["accuracy_drop"]) > 0.03
                    else _float(row["margin_drop"]) / 10.0
                    if _float(row["margin_drop"]) > 0.5
                    else 0.0
                )
                for row in raw_position
                if row["position_group"]
                in {"edge_marker", "source", "destination"}
            )
            depth_effect = max(
                (
                    _float(row["accuracy_drop"])
                    if _float(row["accuracy_drop"]) > 0.03
                    else _float(row["margin_drop"]) / 10.0
                    if _float(row["margin_drop"]) > 0.5
                    else 0.0
                )
                for row in raw_position
                if row["position_group"] == "depth"
            )
            evidence["graph_token_effect"] = graph_token_effect
            evidence["depth_effect"] = depth_effect
            if graph_token_effect > 0:
                roles.append("graph_token_representation")
            if depth_effect > 0:
                roles.append("depth_control")
            component = _component_label(block, branch)
            output[component] = {
                "run": name,
                "family": family,
                "seed": seed,
                "component": component,
                "functional_roles": roles,
                **evidence,
            }
    return output


def _median(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return statistics.median(finite) if finite else float("nan")


def _plot_trajectories(rows: list[dict[str, Any]], path: Path) -> None:
    families = sorted({row["family"] for row in rows})
    figure, axes = plt.subplots(
        len(families),
        1,
        figsize=(10, 2.2 + 1.1 * len(families)),
        squeeze=False,
    )
    for axis_index, family in enumerate(families):
        selected = sorted(
            (row for row in rows if row["family"] == family),
            key=lambda row: int(row["seed"]),
        )
        sequences = [
            [int(value) for value in row["confident_path_sequence"].split()]
            for row in selected
        ]
        width = max(len(sequence) for sequence in sequences)
        values = np.full((len(sequences), width), np.nan)
        for row_index, sequence in enumerate(sequences):
            values[row_index, : len(sequence)] = sequence
        values[values < 0] = np.nan
        axis = axes[axis_index, 0]
        colormap = plt.get_cmap("viridis").copy()
        colormap.set_bad("#d9d9d9")
        image = axis.imshow(
            values,
            vmin=0,
            vmax=8,
            cmap=colormap,
            aspect="auto",
        )
        axis.set_title(
            f"{family}: decoded path position by loop "
            "(gray ? = no position reaches 50% accuracy)"
        )
        axis.set_xlabel("effective loop")
        axis.set_ylabel("seed")
        axis.set_xticks(
            range(width), [str(index + 1) for index in range(width)]
        )
        axis.set_yticks(
            range(len(selected)),
            [str(row["seed"]) for row in selected],
        )
        for y in range(values.shape[0]):
            for x in range(values.shape[1]):
                if np.isfinite(values[y, x]):
                    axis.text(
                        x,
                        y,
                        str(int(values[y, x])),
                        ha="center",
                        va="center",
                        color="white",
                        fontsize=8,
                    )
                else:
                    axis.text(
                        x,
                        y,
                        "?",
                        ha="center",
                        va="center",
                        color="#555555",
                        fontsize=8,
                    )
        figure.colorbar(image, ax=axis, label="best decoded path position")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_component_effects(
    rows: list[dict[str, Any]], path: Path
) -> None:
    families = sorted({row["family"] for row in rows})
    components = ["B1.attn", "B1.mlp", "B2.attn", "B2.mlp"]
    figure, axes = plt.subplots(
        1, len(families), figsize=(5.2 * len(families), 4.3), squeeze=False
    )
    for family_index, family in enumerate(families):
        selected = {
            row["component"]: row
            for row in rows
            if row["family"] == family
        }
        axis = axes[0, family_index]
        x = np.arange(len(components))
        width = 0.36
        axis.bar(
            x - width / 2,
            [
                _float(selected[item]["max_skip_drop_median"])
                for item in components
            ],
            width,
            label="skip",
        )
        axis.bar(
            x + width / 2,
            [
                _float(selected[item]["max_repeat_drop_median"])
                for item in components
            ],
            width,
            label="repeat",
        )
        axis.set_xticks(x, components, rotation=25, ha="right")
        axis.set_ylim(0, 1.02)
        axis.set_ylabel("median maximum endpoint accuracy drop")
        axis.set_title(family)
        axis.grid(axis="y", alpha=0.25)
        axis.legend()
    figure.suptitle("Where skip/repeat interventions damage the circuit")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate matched graph loop-compression circuits."
    )
    parser.add_argument("--compression-dir", type=Path, required=True)
    parser.add_argument("--functional-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument(
        "--paired-seeds",
        action="store_true",
        help=(
            "Declare that equal seed IDs across loop-count families share "
            "the same initialization and training-batch stream."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    compression_dirs = {
        json.loads(path.read_text())["name"]: path.parent
        for path in args.compression_dir.rglob("summary.json")
    }
    functional_dirs = {
        json.loads(path.read_text())["name"]: path.parent
        for path in args.functional_dir.rglob("summary.json")
        if (path.parent / "internal_patching_rows.csv").exists()
    }
    shared = sorted(set(compression_dirs) & set(functional_dirs))
    if not shared:
        raise FileNotFoundError("no matching compression/functional runs")
    if args.paired_seeds:
        paired_metadata: dict[tuple[int, int], list[tuple[Any, Any]]] = {}
        for name in shared:
            summary = json.loads(
                (compression_dirs[name] / "summary.json").read_text()
            )
            _, seed = _run_parts(name)
            depth = int(summary["config"]["max_depth"])
            paired_metadata.setdefault((depth, seed), []).append(
                (
                    summary.get("initialization_seed"),
                    summary.get("data_seed"),
                )
            )
        if not any(len(values) >= 2 for values in paired_metadata.values()):
            raise ValueError("--paired-seeds requires at least one seed pair")
        for key, values in paired_metadata.items():
            if len(values) < 2:
                continue
            if any(item[0] is None or item[1] is None for item in values):
                raise ValueError(
                    f"missing paired seed metadata for depth/seed {key}"
                )
            if len(set(values)) != 1:
                raise ValueError(
                    f"mismatched initialization/data seeds for {key}: "
                    f"{values}"
                )

    run_rows: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []
    for name in shared:
        run_row, compression_components = summarize_compression_run(
            compression_dirs[name]
        )
        roles = functional_roles(functional_dirs[name], top_k=args.top_k)
        run_rows.append(run_row)
        for component in sorted(compression_components):
            row = {**compression_components[component]}
            role_names = list(roles[component]["functional_roles"])
            if int(row["semantic_shift_site_count"]) > 0:
                role_names.append("horizon_step_control")
            if _passes(_float(row["max_overloop_skip_drop"]), 0.10):
                role_names.append("endpoint_maintenance")
            overloop_destabilizer = int(
                _passes(
                    _float(row["max_overloop_skip_improvement"]), 0.10
                )
                or _passes(_float(row["max_overloop_repeat_drop"]), 0.10)
            )
            role_names = sorted(set(role_names))
            role_detail = {
                key: value
                for key, value in roles[component].items()
                if key
                not in {
                    "run",
                    "family",
                    "seed",
                    "component",
                    "functional_roles",
                }
            }
            component_rows.append(
                {
                    **row,
                    **role_detail,
                    "roles": " | ".join(role_names),
                    "role_count": len(role_names),
                    "multifunctional": int(len(role_names) >= 2),
                    "overloop_destabilizer": overloop_destabilizer,
                }
            )

    family_rows: list[dict[str, Any]] = []
    for family in sorted({row["family"] for row in component_rows}):
        for component in sorted(
            {
                row["component"]
                for row in component_rows
                if row["family"] == family
            }
        ):
            selected = [
                row
                for row in component_rows
                if row["family"] == family
                and row["component"] == component
            ]
            role_counts = Counter(
                role
                for row in selected
                for role in str(row["roles"]).split(" | ")
                if role
            )
            family_rows.append(
                {
                    "family": family,
                    "component": component,
                    "seed_count": len(selected),
                    "multifunctional_count": sum(
                        int(row["multifunctional"]) for row in selected
                    ),
                    "overloop_destabilizer_count": sum(
                        int(row["overloop_destabilizer"])
                        for row in selected
                    ),
                    "role_prevalence": "; ".join(
                        f"{role}:{count}/{len(selected)}"
                        for role, count in sorted(role_counts.items())
                    ),
                    "max_skip_drop_median": _median(
                        [
                            _float(row["max_skip_endpoint_drop"])
                            for row in selected
                        ]
                    ),
                    "skip_drop_range_median": _median(
                        [_float(row["skip_drop_range"]) for row in selected]
                    ),
                    "max_repeat_drop_median": _median(
                        [
                            _float(row["max_repeat_endpoint_drop"])
                            for row in selected
                        ]
                    ),
                    "semantic_shift_sites_median": _median(
                        [
                            _float(row["semantic_shift_site_count"])
                            for row in selected
                        ]
                    ),
                    "max_overloop_skip_drop_median": _median(
                        [
                            _float(row["max_overloop_skip_drop"])
                            for row in selected
                        ]
                    ),
                    "max_overloop_skip_improvement_median": _median(
                        [
                            _float(
                                row["max_overloop_skip_improvement"]
                            )
                            for row in selected
                        ]
                    ),
                }
            )

    _write_csv(args.out_dir / "run_compression_rows.csv", run_rows)
    _write_csv(args.out_dir / "component_multifunction_rows.csv", component_rows)
    _write_csv(args.out_dir / "family_component_summary.csv", family_rows)
    _plot_trajectories(
        run_rows, args.out_dir / "decoded_trajectory_by_seed.png"
    )
    _plot_component_effects(
        family_rows, args.out_dir / "component_skip_repeat_effects.png"
    )
    (args.out_dir / "aggregate_summary.json").write_text(
        json.dumps(
            {
                "matched_run_count": len(shared),
                "seed_comparison": (
                    "paired by seed with shared initialization and "
                    "training-batch stream"
                    if args.paired_seeds
                    else "unpaired family distributions; trainer uses "
                    "seed + 1009 * loops"
                ),
                "run_count_by_family": Counter(
                    row["family"] for row in run_rows
                ),
                "trajectory_counts_by_family": {
                    family: Counter(
                        row["trajectory_subtype"]
                        for row in run_rows
                        if row["family"] == family
                    )
                    for family in sorted(
                        {row["family"] for row in run_rows}
                    )
                },
                "earliest_endpoint_loop_median_by_family": {
                    family: _median(
                        [
                            _float(row["earliest_endpoint_loop_acc80"])
                            for row in run_rows
                            if row["family"] == family
                            and int(row["earliest_endpoint_loop_acc80"]) > 0
                        ]
                    )
                    for family in sorted(
                        {row["family"] for row in run_rows}
                    )
                },
                "multifunctional_count_by_family": {
                    family: sum(
                        int(row["multifunctional"])
                        for row in component_rows
                        if row["family"] == family
                    )
                    for family in sorted(
                        {row["family"] for row in component_rows}
                    )
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
