from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _float(value: str | None, default: float = float("nan")) -> float:
    if value in (None, ""):
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _run_parts(name: str) -> tuple[str, int | None]:
    match = re.fullmatch(r"(D\d+_L\d+)_seed(\d+)", name)
    if match is None:
        return name, None
    return match.group(1), int(match.group(2))


def _lookup(
    rows: list[dict[str, str]],
    **criteria: str,
) -> dict[str, str] | None:
    for row in rows:
        if all(row.get(key) == value for key, value in criteria.items()):
            return row
    return None


def _progression(
    rows: list[dict[str, str]],
    *,
    site: str,
    branch: str,
) -> dict[str, Any]:
    before_stage = "input" if branch == "attention" else "post_attention"
    after_stage = "post_attention" if branch == "attention" else "post_mlp"
    before = _lookup(rows, site=site, stage=before_stage)
    after = _lookup(rows, site=site, stage=after_stage)
    if before is None or after is None:
        return {
            "best_position_before": None,
            "best_position_after": None,
            "best_accuracy_before": float("nan"),
            "best_accuracy_after": float("nan"),
            "endpoint_margin_delta": float("nan"),
        }
    return {
        "best_position_before": int(before["best_path_position"]),
        "best_position_after": int(after["best_path_position"]),
        "best_accuracy_before": _float(before["best_path_accuracy"]),
        "best_accuracy_after": _float(after["best_path_accuracy"]),
        "endpoint_margin_delta": _float(after["endpoint_margin"])
        - _float(before["endpoint_margin"]),
    }


def _position_metrics(
    rows: list[dict[str, str]],
    *,
    site: str,
    component: str,
) -> dict[str, Any]:
    selected = [
        row
        for row in rows
        if row["site"] == site and row["component"] == component
    ]
    if not selected:
        return {
            "max_position_group": None,
            "max_position_accuracy_drop": float("nan"),
            "max_position_margin_drop": float("nan"),
            "answer_accuracy_drop": float("nan"),
            "answer_margin_drop": float("nan"),
        }
    strongest = max(
        selected,
        key=lambda row: (
            _float(row["accuracy_drop"]),
            _float(row["margin_drop"]),
        ),
    )
    answer = next(
        (row for row in selected if row["position_group"] == "answer"),
        None,
    )
    return {
        "max_position_group": strongest["position_group"],
        "max_position_accuracy_drop": _float(strongest["accuracy_drop"]),
        "max_position_margin_drop": _float(strongest["margin_drop"]),
        "answer_accuracy_drop": (
            _float(answer["accuracy_drop"])
            if answer is not None
            else float("nan")
        ),
        "answer_margin_drop": (
            _float(answer["margin_drop"])
            if answer is not None
            else float("nan")
        ),
    }


def _patch_metric(
    rows: list[dict[str, str]],
    *,
    site: str,
    pair_type: str,
    role_probe: str,
) -> tuple[float, float]:
    row = _lookup(
        rows,
        site=site,
        pair_type=pair_type,
        role_probe=role_probe,
    )
    if row is None:
        return float("nan"), float("nan")
    return _float(row["recovery"]), _float(row["specific_recovery"])


def _neuron_metrics(
    rows: list[dict[str, str]],
    *,
    site: str,
    top_k: int,
) -> dict[str, float]:
    result: dict[str, float] = {}
    for condition in ("all", f"top{top_k}", f"random{top_k}", "complement"):
        row = _lookup(rows, site=site, condition=condition)
        result[f"neuron_{condition}_recovery"] = (
            _float(row["recovery"]) if row is not None else float("nan")
        )
    result["neuron_top_specificity"] = (
        result[f"neuron_top{top_k}_recovery"]
        - result[f"neuron_random{top_k}_recovery"]
    )
    result["neuron_top_unique_effect"] = (
        result["neuron_all_recovery"]
        - result["neuron_complement_recovery"]
    )
    return result


def _transition_metrics(
    rows: list[dict[str, str]],
    *,
    site: str,
    branch: str,
) -> dict[str, Any]:
    def drop(role: str) -> tuple[float, float]:
        row = _lookup(rows, site=site, role_probe=role)
        if row is None:
            return float("nan"), float("nan")
        return _float(row["accuracy_drop"]), _float(row["margin_drop"])

    if branch == "attention":
        output_accuracy, output_margin = drop("attention_output_answer")
        current_accuracy, current_margin = drop("pattern_current_edge")
        random_accuracy, random_margin = drop("pattern_random_edge")
        current_head_rows = [
            row
            for row in rows
            if row["site"] == site
            and row["role_probe"] == "head_pattern_current_edge"
        ]
        top_head = (
            max(
                current_head_rows,
                key=lambda row: _float(row["accuracy_drop"]),
            )
            if current_head_rows
            else None
        )
        top_head_random = (
            _lookup(
                rows,
                site=site,
                role_probe="head_pattern_random_edge",
                component=top_head["component"],
            )
            if top_head is not None
            else None
        )
        return {
            "transition_output_accuracy_drop": output_accuracy,
            "transition_output_margin_drop": output_margin,
            "transition_current_edge_accuracy_drop": current_accuracy,
            "transition_random_edge_accuracy_drop": random_accuracy,
            "transition_edge_accuracy_specificity": (
                current_accuracy - random_accuracy
            ),
            "transition_edge_margin_specificity": current_margin - random_margin,
            "transition_top_current_head": (
                top_head["component"] if top_head is not None else ""
            ),
            "transition_top_head_current_drop": (
                _float(top_head["accuracy_drop"])
                if top_head is not None
                else float("nan")
            ),
            "transition_top_head_random_drop": (
                _float(top_head_random["accuracy_drop"])
                if top_head_random is not None
                else float("nan")
            ),
        }
    output_accuracy, output_margin = drop("mlp_output_answer")
    return {
        "transition_output_accuracy_drop": output_accuracy,
        "transition_output_margin_drop": output_margin,
        "transition_current_edge_accuracy_drop": float("nan"),
        "transition_random_edge_accuracy_drop": float("nan"),
        "transition_edge_accuracy_specificity": float("nan"),
        "transition_edge_margin_specificity": float("nan"),
        "transition_top_current_head": "",
        "transition_top_head_current_drop": float("nan"),
        "transition_top_head_random_drop": float("nan"),
    }


def _finite_gt(value: float, threshold: float) -> bool:
    return math.isfinite(value) and value > threshold


def _localized_transition_executor(
    rows: list[dict[str, str]],
) -> dict[str, Any] | None:
    sites = sorted(
        {
            row["site"]
            for row in rows
            if re.fullmatch(r"L\d+\.B2", row["site"])
        },
        key=lambda label: tuple(int(item) for item in re.findall(r"\d+", label)),
    )
    candidates: list[dict[str, Any]] = []
    for site in sites:
        metrics = _transition_metrics(rows, site=site, branch="attention")
        edge_specific = (
            _finite_gt(
                float(metrics["transition_edge_accuracy_specificity"]),
                0.10,
            )
            or _finite_gt(
                float(metrics["transition_edge_margin_specificity"]),
                1.0,
            )
        )
        output_necessary = (
            _finite_gt(
                float(metrics["transition_output_accuracy_drop"]),
                0.10,
            )
            or _finite_gt(
                float(metrics["transition_output_margin_drop"]),
                1.0,
            )
        )
        if edge_specific and output_necessary:
            candidates.append({"site": site, "block": 2, **metrics})
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda row: (
            float(row["transition_edge_accuracy_specificity"]),
            float(row["transition_edge_margin_specificity"]),
        ),
    )


def _classify(row: dict[str, Any]) -> tuple[str, str, str]:
    branch = row["branch"]
    causal_localization = (
        _finite_gt(row["answer_accuracy_drop"], 0.03)
        or _finite_gt(row["answer_margin_drop"], 0.5)
        or _finite_gt(row["max_position_accuracy_drop"], 0.03)
        or _finite_gt(row["max_position_margin_drop"], 0.5)
    )
    if branch == "attention":
        edge_specific = (
            _finite_gt(row["transition_edge_accuracy_specificity"], 0.10)
            or _finite_gt(row["transition_edge_margin_specificity"], 1.0)
        )
        transition_output = (
            _finite_gt(row["transition_output_accuracy_drop"], 0.10)
            or _finite_gt(row["transition_output_margin_drop"], 1.0)
        )
        query_routing = _finite_gt(
            row["query_q_specific_recovery"], 0.10
        )
        graph_content = _finite_gt(
            max(
                row["graph_k_specific_recovery"],
                row["graph_v_specific_recovery"],
            ),
            0.10,
        )
        if edge_specific and transition_output:
            return (
                "successor edge lookup/executor",
                "A",
                "held-out transition fails on current edge but not random edge",
            )
        if query_routing and graph_content and causal_localization:
            return (
                "query-conditioned graph lookup",
                "A",
                "Q and graph K/V patches plus position ablation agree",
            )
        if query_routing:
            return (
                "answer-state router",
                "B",
                "query-specific Q patch is causal; lookup specificity incomplete",
            )
        if graph_content:
            return (
                "graph-content reader",
                "B",
                "graph K/V patch is causal; query routing incomplete",
            )
        if transition_output:
            return (
                "transition-supporting attention",
                "B",
                "attention output is necessary for held-out next step",
            )
        if causal_localization:
            return (
                "localized attention update",
                "B",
                "position-specific ablation is causal; algorithm unresolved",
            )
        return (
            "distributed/redundant attention support",
            "C",
            "no strong isolated causal effect at tested granularity",
        )

    transition_output = (
        _finite_gt(row["transition_output_accuracy_drop"], 0.10)
        or _finite_gt(row["transition_output_margin_drop"], 1.0)
    )
    query_transform = _finite_gt(
        row["query_mlp_specific_recovery"], 0.10
    )
    neuron_specific = _finite_gt(row["neuron_top_specificity"], 0.02)
    if transition_output and neuron_specific:
        return (
            "successor-state nonlinear transform",
            "A",
            "held-out next-step necessity plus neuron random control",
        )
    if transition_output:
        return (
            "transition-supporting nonlinear transform",
            "B",
            "MLP output is necessary for held-out next step",
        )
    if query_transform and neuron_specific and causal_localization:
        return (
            "query-specific answer-state transform",
            "A",
            "hidden patch, neuron control, and position ablation agree",
        )
    if query_transform:
        return (
            "answer-state nonlinear transform",
            "B",
            "query-specific hidden activation patch is causal",
        )
    if causal_localization:
        label = (
            "graph-token feature transform"
            if row["max_position_group"]
            in {"edge_marker", "source", "destination"}
            else "localized nonlinear update"
        )
        return (
            label,
            "B",
            "position-specific MLP ablation is causal; semantic variable unresolved",
        )
    return (
        "distributed/redundant nonlinear support",
        "C",
        "no strong isolated causal effect at tested granularity",
    )


def aggregate_run(run_dir: Path, *, top_k: int) -> list[dict[str, Any]]:
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    name = summary["name"]
    family, seed = _run_parts(name)
    progression_rows = _read_csv(run_dir / "branch_progression_rows.csv")
    position_rows = _read_csv(run_dir / "position_ablation_rows.csv")
    patch_rows = _read_csv(run_dir / "internal_patching_rows.csv")
    neuron_rows = _read_csv(run_dir / "mlp_neuron_rows.csv")
    transition_rows = _read_csv(run_dir / "transition_function_rows.csv")
    sites = sorted(
        {row["site"] for row in progression_rows},
        key=lambda label: tuple(int(item) for item in re.findall(r"\d+", label)),
    )
    output: list[dict[str, Any]] = []
    for site in sites:
        loop_index, block_index = (
            int(item) for item in re.findall(r"\d+", site)
        )
        for branch, component in (
            ("attention", "attention_out"),
            ("mlp", "mlp_out"),
        ):
            row: dict[str, Any] = {
                "run": name,
                "family": family,
                "seed": seed,
                "site": site,
                "loop": loop_index,
                "block": block_index,
                "branch": branch,
                "baseline_accuracy": summary["baseline"]["endpoint_accuracy"],
                "baseline_margin": summary["baseline"]["endpoint_margin"],
                "transition_status": summary["functional_tests"]["transition"][
                    "status"
                ],
                **_progression(
                    progression_rows, site=site, branch=branch
                ),
                **_position_metrics(
                    position_rows, site=site, component=component
                ),
                **_transition_metrics(
                    transition_rows, site=site, branch=branch
                ),
            }
            query_context, query_context_specific = _patch_metric(
                patch_rows,
                site=site,
                pair_type="query",
                role_probe=(
                    "context_answer"
                    if branch == "attention"
                    else "mlp_hidden_answer"
                ),
            )
            row["query_component_recovery"] = query_context
            row["query_component_specific_recovery"] = (
                query_context_specific
            )
            if branch == "attention":
                _, row["query_q_specific_recovery"] = _patch_metric(
                    patch_rows,
                    site=site,
                    pair_type="query",
                    role_probe="q_answer",
                )
                _, row["graph_k_specific_recovery"] = _patch_metric(
                    patch_rows,
                    site=site,
                    pair_type="graph",
                    role_probe="k_graph",
                )
                _, row["graph_v_specific_recovery"] = _patch_metric(
                    patch_rows,
                    site=site,
                    pair_type="graph",
                    role_probe="v_graph",
                )
                row["query_mlp_specific_recovery"] = float("nan")
                row.update(
                    {
                        "neuron_all_recovery": float("nan"),
                        f"neuron_top{top_k}_recovery": float("nan"),
                        f"neuron_random{top_k}_recovery": float("nan"),
                        "neuron_complement_recovery": float("nan"),
                        "neuron_top_specificity": float("nan"),
                        "neuron_top_unique_effect": float("nan"),
                    }
                )
            else:
                row["query_q_specific_recovery"] = float("nan")
                row["graph_k_specific_recovery"] = float("nan")
                row["graph_v_specific_recovery"] = float("nan")
                row["query_mlp_specific_recovery"] = (
                    query_context_specific
                )
                row.update(
                    _neuron_metrics(
                        neuron_rows, site=site, top_k=top_k
                    )
                )
            role, grade, rationale = _classify(row)
            row["candidate_function"] = role
            row["evidence_grade"] = grade
            row["evidence_rationale"] = rationale
            output.append(row)
    return output


def _plot_role_counts(rows: list[dict[str, Any]], path: Path) -> None:
    families = sorted({row["family"] for row in rows})
    branches = ("attention", "mlp")
    figure, axes = plt.subplots(
        len(branches),
        len(families),
        figsize=(5.2 * len(families), 7.2),
        squeeze=False,
    )
    for branch_index, branch in enumerate(branches):
        for family_index, family in enumerate(families):
            axis = axes[branch_index, family_index]
            counts = Counter(
                row["candidate_function"]
                for row in rows
                if row["branch"] == branch and row["family"] == family
            )
            labels = list(counts)
            values = [counts[label] for label in labels]
            axis.barh(range(len(labels)), values, color="#4c78a8")
            axis.set_yticks(range(len(labels)), labels, fontsize=8)
            axis.invert_yaxis()
            axis.set_title(f"{family} · {branch}")
            seed_count = len(
                {
                    row["seed"]
                    for row in rows
                    if row["family"] == family
                }
            )
            axis.set_xlabel(f"effective sites across {seed_count} seeds")
            axis.grid(axis="x", alpha=0.25)
    figure.suptitle("Fine-grained functional labels (candidate, causal grade)")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _bootstrap_median_interval(
    values: list[float],
    *,
    seed: int = 20260725,
    draws: int = 5000,
) -> tuple[float, float, float]:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return float("nan"), float("nan"), float("nan")
    generator = random.Random(seed)
    medians = sorted(
        statistics.median(
            generator.choices(finite, k=len(finite))
        )
        for _ in range(draws)
    )
    lower = medians[int(0.025 * (draws - 1))]
    upper = medians[int(0.975 * (draws - 1))]
    return statistics.median(finite), lower, upper


def _wilson_interval(
    successes: int,
    total: int,
    *,
    z: float = 1.959963984540054,
) -> tuple[float, float, float]:
    if total < 1:
        return float("nan"), float("nan"), float("nan")
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    half_width = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z * z / (4 * total * total)
        )
        / denominator
    )
    return proportion, center - half_width, center + half_width


def _seed_stability_rows(
    rows: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    summary_by_name = {summary["name"]: summary for summary in summaries}
    output: list[dict[str, Any]] = []
    for run in sorted({row["run"] for row in rows}):
        run_rows = [row for row in rows if row["run"] == run]
        first_attention = next(
            row
            for row in run_rows
            if row["site"] == "L1.B1" and row["branch"] == "attention"
        )
        first_mlp = next(
            row
            for row in run_rows
            if row["site"] == "L1.B1" and row["branch"] == "mlp"
        )
        second_attention = [
            row
            for row in run_rows
            if row["block"] == 2 and row["branch"] == "attention"
        ]
        second_mlp = [
            row
            for row in run_rows
            if row["block"] == 2 and row["branch"] == "mlp"
        ]
        strongest_q = max(
            second_attention,
            key=lambda row: float(row["query_q_specific_recovery"]),
        )
        strongest_mlp = max(
            second_mlp,
            key=lambda row: float(row["query_mlp_specific_recovery"]),
        )
        strongest_neurons = max(
            second_mlp,
            key=lambda row: float(row["neuron_top_specificity"]),
        )
        run_summary = summary_by_name[run]
        transition = run_summary["functional_tests"]["transition"]
        passed = transition["status"] == "passed_heldout_transition_gate"
        executor = run_summary.get("_localized_transition_executor")
        output.append(
            {
                "run": run,
                "family": first_attention["family"],
                "seed": first_attention["seed"],
                "checkpoint_step": run_summary["checkpoint_step"],
                "loss_mode": run_summary["loss_mode"],
                "trained_loops": run_summary["trained_loops"],
                "physical_blocks": run_summary["physical_blocks"],
                "baseline_endpoint_accuracy": float(
                    run_summary["baseline"]["endpoint_accuracy"]
                ),
                "baseline_endpoint_margin": float(
                    run_summary["baseline"]["endpoint_margin"]
                ),
                "l1_b1_graph_v_specific_recovery": float(
                    first_attention["graph_v_specific_recovery"]
                ),
                "l1_b1_graph_v_gt_0_10": int(
                    float(first_attention["graph_v_specific_recovery"]) > 0.10
                ),
                "l1_b1_mlp_strongest_group": first_mlp[
                    "max_position_group"
                ],
                "l1_b1_mlp_destination": int(
                    first_mlp["max_position_group"] == "destination"
                ),
                "l1_b1_mlp_margin_drop": float(
                    first_mlp["max_position_margin_drop"]
                ),
                "b2_q_max_site": strongest_q["site"],
                "b2_q_max_specific_recovery": float(
                    strongest_q["query_q_specific_recovery"]
                ),
                "b2_q_gt_0_10": int(
                    float(strongest_q["query_q_specific_recovery"]) > 0.10
                ),
                "b2_mlp_max_site": strongest_mlp["site"],
                "b2_mlp_max_specific_recovery": float(
                    strongest_mlp["query_mlp_specific_recovery"]
                ),
                "b2_mlp_gt_0_10": int(
                    float(strongest_mlp["query_mlp_specific_recovery"]) > 0.10
                ),
                "b2_neuron_max_site": strongest_neurons["site"],
                "b2_neuron_top_specificity": float(
                    strongest_neurons["neuron_top_specificity"]
                ),
                "b2_neuron_specificity_gt_0_02": int(
                    float(strongest_neurons["neuron_top_specificity"]) > 0.02
                ),
                "transition_status": transition["status"],
                "transition_passed": int(passed),
                "transition_executor_localized": int(executor is not None),
                "transition_heldout_accuracy": (
                    float(transition["heldout_next_accuracy"])
                    if passed
                    else float("nan")
                ),
                "transition_executor_site": (
                    executor["site"] if executor is not None else ""
                ),
                "transition_output_accuracy_drop": (
                    float(executor["transition_output_accuracy_drop"])
                    if executor is not None
                    else float("nan")
                ),
                "transition_current_edge_accuracy_drop": (
                    float(executor["transition_current_edge_accuracy_drop"])
                    if executor is not None
                    else float("nan")
                ),
                "transition_random_edge_accuracy_drop": (
                    float(executor["transition_random_edge_accuracy_drop"])
                    if executor is not None
                    else float("nan")
                ),
                "transition_top_head": (
                    executor["transition_top_current_head"]
                    if executor is not None
                    else ""
                ),
            }
        )
    return output


def _family_stability_rows(
    run_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    continuous = (
        "l1_b1_graph_v_specific_recovery",
        "l1_b1_mlp_margin_drop",
        "b2_q_max_specific_recovery",
        "b2_mlp_max_specific_recovery",
        "b2_neuron_top_specificity",
        "transition_heldout_accuracy",
        "transition_output_accuracy_drop",
        "transition_current_edge_accuracy_drop",
        "transition_random_edge_accuracy_drop",
    )
    binary = (
        "l1_b1_graph_v_gt_0_10",
        "l1_b1_mlp_destination",
        "b2_q_gt_0_10",
        "b2_mlp_gt_0_10",
        "b2_neuron_specificity_gt_0_02",
        "transition_passed",
        "transition_executor_localized",
    )
    for family in sorted({row["family"] for row in run_rows}):
        selected = [row for row in run_rows if row["family"] == family]
        row: dict[str, Any] = {
            "family": family,
            "seed_count": len(selected),
        }
        for key in continuous:
            median, lower, upper = _bootstrap_median_interval(
                [float(item[key]) for item in selected],
                seed=20260725 + sum(ord(char) for char in family + key),
            )
            row[f"{key}_median"] = median
            row[f"{key}_bootstrap95_low"] = lower
            row[f"{key}_bootstrap95_high"] = upper
        for key in binary:
            successes = sum(int(item[key]) for item in selected)
            rate, lower, upper = _wilson_interval(successes, len(selected))
            row[f"{key}_count"] = successes
            row[f"{key}_rate"] = rate
            row[f"{key}_wilson95_low"] = lower
            row[f"{key}_wilson95_high"] = upper
        output.append(row)
    return output


def _write_report(
    *,
    rows: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
    path: Path,
    top_k: int,
) -> None:
    runs = sorted({row["run"] for row in rows})
    run_count = len(runs)
    seed_stability = _seed_stability_rows(rows, summaries)
    family_stability = {
        row["family"]: row
        for row in _family_stability_rows(seed_stability)
    }
    first_block_graph_values: list[float] = []
    first_block_mlp_groups: list[str] = []
    first_block_mlp_margins: list[float] = []
    second_block_q_maxima: list[float] = []
    second_block_mlp_maxima: list[float] = []
    second_block_neuron_maxima: list[float] = []
    for run in runs:
        run_rows = [row for row in rows if row["run"] == run]
        first_attention = next(
            row
            for row in run_rows
            if row["site"] == "L1.B1" and row["branch"] == "attention"
        )
        first_mlp = next(
            row
            for row in run_rows
            if row["site"] == "L1.B1" and row["branch"] == "mlp"
        )
        second_attention = [
            row
            for row in run_rows
            if row["block"] == 2 and row["branch"] == "attention"
        ]
        second_mlp = [
            row
            for row in run_rows
            if row["block"] == 2 and row["branch"] == "mlp"
        ]
        first_block_graph_values.append(
            float(first_attention["graph_v_specific_recovery"])
        )
        first_block_mlp_groups.append(first_mlp["max_position_group"])
        first_block_mlp_margins.append(
            float(first_mlp["max_position_margin_drop"])
        )
        second_block_q_maxima.append(
            max(
                float(row["query_q_specific_recovery"])
                for row in second_attention
            )
        )
        second_block_mlp_maxima.append(
            max(
                float(row["query_mlp_specific_recovery"])
                for row in second_mlp
            )
        )
        second_block_neuron_maxima.append(
            max(float(row["neuron_top_specificity"]) for row in second_mlp)
        )

    summary_by_name = {summary["name"]: summary for summary in summaries}
    transition_executors = [
        {"run": run, **summary["_localized_transition_executor"]}
        for run, summary in summary_by_name.items()
        if summary.get("_localized_transition_executor") is not None
    ]

    endpoint_control_rows: list[tuple[str, int, float, float]] = []
    d6_l8_runs = sorted(
        {
            row["run"]
            for row in rows
            if row["family"] == "D6_L8"
        }
    )
    for run in d6_l8_runs:
        for loop in (7, 8):
            attention = next(
                row
                for row in rows
                if row["run"] == run
                and row["site"] == f"L{loop}.B2"
                and row["branch"] == "attention"
            )
            mlp = next(
                row
                for row in rows
                if row["run"] == run
                and row["site"] == f"L{loop}.B2"
                and row["branch"] == "mlp"
            )
            endpoint_control_rows.append(
                (
                    run,
                    loop,
                    float(attention["query_q_specific_recovery"]),
                    float(mlp["query_mlp_specific_recovery"]),
                )
            )

    lines = [
        "# Graph 置换复合：Attention/MLP 细粒度功能图谱",
        "",
        "## 先说结论",
        "",
        f"在 {run_count} 个同配方 checkpoint 上，最稳定的物理分工发生在 **Block 2：attention 路由 answer state，MLP 对答案态做非线性写回或终点维持**。Block 1 通常建立图/token 特征，但具体由 graph-value 读取还是某个 token 位置的局部变换承担，存在明显 seed 与架构差异。通过跨图 state transplant 只说明内部状态可迁移；还需当前边相对随机边的特异消融，才能把 attention 定位为 successor executor。本实验有 {len(transition_executors)} 个 checkpoint 同时满足这两层标准。",
        "",
        f"- **Block 1 attention（首次调用）通常读图内容**：L1.B1 的 graph-V 特异恢复中位数为 {statistics.median(first_block_graph_values):.3f}，{run_count} 个 checkpoint 中 {sum(value > 0.10 for value in first_block_graph_values)}/{run_count} 超过 0.10；未通过者不能仅凭位置消融继续命名为 graph reader。",
        f"- **Block 1 MLP 做早期 token 特征变换**：L1.B1 MLP 的最敏感位置在 {sum(group == 'destination' for group in first_block_mlp_groups)}/{run_count} checkpoint 是 destination token；其余 seed 转向 answer 或 edge-marker，因此“destination-centered edge builder”只是多数模式，不是统一 circuit。最强位置消融的终点 margin drop 中位数为 {statistics.median(first_block_mlp_margins):.2f}。",
        f"- **Block 2 attention 稳定路由 answer state**：每个 checkpoint 内最强的 B2 Q-answer 特异恢复中位数为 {statistics.median(second_block_q_maxima):.3f}，{sum(value > 0.10 for value in second_block_q_maxima)}/{run_count} 超过 0.10；只有下表通过 edge-specific 控制的子集可进一步称作查边 executor。",
        f"- **Block 2 MLP 写回/整形 answer state**：每个 checkpoint 内最强的 query-specific hidden patch 恢复中位数为 {statistics.median(second_block_mlp_maxima):.3f}；top-{top_k} 减同尺寸 random-{top_k} 的恢复优势中位数为 {statistics.median(second_block_neuron_maxima):.3f}，{sum(value > 0.0 for value in second_block_neuron_maxima)}/{run_count} 为正。",
        "",
        "## 真正的 successor executor",
        "",
        "| checkpoint | held-out 下一步准确率 | executor site | attention-out 消融 drop | 当前边 drop | 随机边 drop | 主导 head（当前/随机 drop） |",
        "|---|---:|---|---:|---:|---:|---|",
    ]
    for row in sorted(transition_executors, key=lambda item: item["run"]):
        heldout = summary_by_name[row["run"]]["functional_tests"][
            "transition"
        ]["heldout_next_accuracy"]
        lines.append(
            f"| {row['run']} | {heldout:.3f} | {row['site']} | "
            f"{float(row['transition_output_accuracy_drop']):.3f} | "
            f"{float(row['transition_current_edge_accuracy_drop']):.3f} | "
            f"{float(row['transition_random_edge_accuracy_drop']):.3f} | "
            f"{row['transition_top_current_head']} "
            f"({float(row['transition_top_head_current_drop']):.3f}/"
            f"{float(row['transition_top_head_random_drop']):.3f}) |"
        )
    lines.extend(
        [
            "",
            f"{len(transition_executors)} 个同时通过 state-transplant 和 edge-specific 定位的 executor 都落在 **Block 2 attention**。当前边与随机边的差异及主导 head 见上表；稳定性结论应落在 Block 2 的功能，而不是固定 head 编号。",
            "",
            "## 6–8 的终点控制",
            "",
            f"D6–L8 在完成 f⁶ 后还有 loop 7–8。这里检查全部 {len(d6_l8_runs)} 个 seed：若 B2 Q-answer patch 在这两轮接近 0、而 B2 MLP hidden patch 仍保留 query-specific 恢复，则支持额外两轮关闭查边 attention、由 MLP 维持/整形终点态；具体数值如下。",
            "",
            "| checkpoint | loop | B2 Q 特异恢复 | B2 MLP 特异恢复 |",
            "|---|---:|---:|---:|",
        ]
    )
    for run, loop, q_value, mlp_value in endpoint_control_rows:
        lines.append(
            f"| {run} | {loop} | {q_value:.3f} | {mlp_value:.3f} |"
        )
    lines.extend(
        [
            "",
            "## 结论口径",
            "",
            "这里的“功能”不是由 attention map 或 probe 直接命名。A 级要求至少两个相互独立的因果证据，B 级有单类因果干预，C 级只说明在当前粒度下分布式、冗余或尚未解析。所有模型仍是 final-only loss；每个 loop 复用 2 个物理 Transformer block。",
            "",
            "top-neuron 编号只在各 checkpoint 内成立；这里验证的是稀疏子集相对随机子集的因果优势，不声称不同 seed 共享同一组 neuron ID。",
            "",
            "## 跨 seed 总览",
            "",
            "| family | B1 graph-V 恢复中位数 [bootstrap 95%] | B1-MLP destination | B2-Q 最大恢复中位数 [bootstrap 95%] | B2-MLP 最大恢复中位数 [bootstrap 95%] | top-neuron 优势中位数 [bootstrap 95%] | state-transplant gate | edge-specific executor |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for family in sorted({row["family"] for row in rows}):
        stability = family_stability[family]
        lines.append(
            f"| {family} | "
            f"{stability['l1_b1_graph_v_specific_recovery_median']:.3f} "
            f"[{stability['l1_b1_graph_v_specific_recovery_bootstrap95_low']:.3f}, "
            f"{stability['l1_b1_graph_v_specific_recovery_bootstrap95_high']:.3f}] | "
            f"{stability['l1_b1_mlp_destination_count']}/{stability['seed_count']} | "
            f"{stability['b2_q_max_specific_recovery_median']:.3f} "
            f"[{stability['b2_q_max_specific_recovery_bootstrap95_low']:.3f}, "
            f"{stability['b2_q_max_specific_recovery_bootstrap95_high']:.3f}] | "
            f"{stability['b2_mlp_max_specific_recovery_median']:.3f} "
            f"[{stability['b2_mlp_max_specific_recovery_bootstrap95_low']:.3f}, "
            f"{stability['b2_mlp_max_specific_recovery_bootstrap95_high']:.3f}] | "
            f"{stability['b2_neuron_top_specificity_median']:.3f} "
            f"[{stability['b2_neuron_top_specificity_bootstrap95_low']:.3f}, "
            f"{stability['b2_neuron_top_specificity_bootstrap95_high']:.3f}] | "
            f"{stability['transition_passed_count']}/{stability['seed_count']} | "
            f"{stability['transition_executor_localized_count']}/{stability['seed_count']} |"
        )
    lines.append("")
    for family in sorted({row["family"] for row in rows}):
        selected = [row for row in rows if row["family"] == family]
        stability = family_stability[family]
        transition_runs = [
            summary["name"]
            for summary in summaries
            if summary["name"].startswith(family)
            and summary["functional_tests"]["transition"]["status"]
            == "passed_heldout_transition_gate"
        ]
        executor_runs = [
            summary["name"]
            for summary in summaries
            if summary["name"].startswith(family)
            and summary.get("_localized_transition_executor") is not None
        ]
        grade_counts = Counter(row["evidence_grade"] for row in selected)
        function_counts = Counter(row["candidate_function"] for row in selected)
        top_functions = "; ".join(
            f"{label}: {count}"
            for label, count in function_counts.most_common(5)
        )
        lines.extend(
            [
                f"- **{family}**：可做 held-out 跨图下一步迁移的运行为 "
                + (", ".join(transition_runs) if transition_runs else "无")
                + (
                    f"；通过率 {stability['transition_passed_count']}/"
                    f"{stability['seed_count']} = "
                    f"{stability['transition_passed_rate']:.3f}"
                    f"（Wilson 95% CI "
                    f"{stability['transition_passed_wilson95_low']:.3f}–"
                    f"{stability['transition_passed_wilson95_high']:.3f}）"
                )
                + "；其中可进一步定位到 edge-specific Block 2 executor 的运行为 "
                + (", ".join(executor_runs) if executor_runs else "无")
                + (
                    f"（{stability['transition_executor_localized_count']}/"
                    f"{stability['seed_count']}，Wilson 95% CI "
                    f"{stability['transition_executor_localized_wilson95_low']:.3f}–"
                    f"{stability['transition_executor_localized_wilson95_high']:.3f}）"
                )
                + f"；证据等级 A/B/C = {grade_counts['A']}/{grade_counts['B']}/{grade_counts['C']}。主要候选功能：{top_functions}。",
            ]
        )
    lines.extend(
        [
            "",
            "## 读表方法",
            "",
            "- `position_ablation` 回答这个 branch 在 answer、source、destination、start、depth 等位置的更新是否必要。",
            "- `query patch` 使用同一张图、不同起点，区分 answer-state routing；`graph patch` 使用同一起点、不同图，区分 K/V 中的图内容。",
            f"- MLP neuron 以独立校准 batch 选 top-{top_k}，再在 held-out batch 与同尺寸随机 neuron、complement 比较。",
            "- `current edge vs random edge` 只在 held-out 跨图下一步准确率达到 0.40 后运行，因此可用于判断 successor lookup，而不是只看终点答案。",
            "",
            "逐 loop、逐 block 的完整数值和候选功能见 `functional_site_roles.csv`；自动标签只是索引，论文式结论应回到对应的 patch/ablation 数值。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate fine-grained graph-path functional circuits."
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    run_dirs = sorted(
        path.parent
        for path in args.input_dir.rglob("summary.json")
        if path.parent != args.input_dir
    )
    if not run_dirs:
        raise FileNotFoundError("no run summary.json files found")
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        rows.extend(aggregate_run(run_dir, top_k=args.top_k))
        summary = json.loads(
            (run_dir / "summary.json").read_text(encoding="utf-8")
        )
        if (
            summary["functional_tests"]["transition"]["status"]
            == "passed_heldout_transition_gate"
        ):
            summary["_localized_transition_executor"] = (
                _localized_transition_executor(
                    _read_csv(run_dir / "transition_function_rows.csv")
                )
            )
        else:
            summary["_localized_transition_executor"] = None
        summaries.append(summary)
    seed_stability_rows = _seed_stability_rows(rows, summaries)
    family_stability_rows = _family_stability_rows(seed_stability_rows)
    _write_csv(args.out_dir / "functional_site_roles.csv", rows)
    _write_csv(
        args.out_dir / "seed_stability_run_rows.csv",
        seed_stability_rows,
    )
    _write_csv(
        args.out_dir / "seed_stability_family_summary.csv",
        family_stability_rows,
    )
    _plot_role_counts(rows, args.out_dir / "functional_role_counts.png")
    _write_report(
        rows=rows,
        summaries=summaries,
        path=args.out_dir / "GRAPH_PATH_FUNCTIONAL_CIRCUIT_REPORT_CN.md",
        top_k=args.top_k,
    )
    (args.out_dir / "aggregate_summary.json").write_text(
        json.dumps(
            {
                "run_count": len(run_dirs),
                "seed_count_by_family": {
                    row["family"]: row["seed_count"]
                    for row in family_stability_rows
                },
                "effective_branch_site_count": len(rows),
                "evidence_grade_counts": Counter(
                    row["evidence_grade"] for row in rows
                ),
                "candidate_function_counts": Counter(
                    row["candidate_function"] for row in rows
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
