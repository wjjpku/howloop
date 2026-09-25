from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from numbers import Real
from pathlib import Path
from typing import Any


def _metric_value(
    scorecards: dict[int, dict[str, float | None]],
    *,
    seed: int,
    metric: str,
) -> float:
    value = scorecards[seed].get(metric)
    if value is None or isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"missing metric {metric!r} for seed {seed}")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"metric {metric!r} for seed {seed} must be finite")
    return value


def aggregate_transition(
    before: dict[int, dict[str, float | None]],
    after: dict[int, dict[str, float | None]],
    *,
    metric: str,
    strong_effect: float,
) -> dict[str, Any]:
    """Aggregate a mechanism change without breaking seed pairing.

    The three-seed expansion gate is deliberately conservative: every paired
    delta must have the same nonzero sign, and at least two absolute deltas
    must meet ``strong_effect``. Inputs with different seed sets are rejected
    instead of silently averaging different runs.
    """

    if not metric:
        raise ValueError("metric must be nonempty")
    if not math.isfinite(strong_effect) or strong_effect < 0:
        raise ValueError("strong_effect must be finite and nonnegative")
    before_seeds = set(before)
    after_seeds = set(after)
    if before_seeds != after_seeds:
        missing_after = sorted(before_seeds - after_seeds)
        missing_before = sorted(after_seeds - before_seeds)
        raise ValueError(
            "unpaired seeds: "
            f"missing after={missing_after}, missing before={missing_before}"
        )
    if not before_seeds:
        raise ValueError("at least one paired seed is required")

    seeds = sorted(before_seeds)
    paired_deltas: dict[int, float] = {}
    for seed in seeds:
        before_value = _metric_value(before, seed=seed, metric=metric)
        after_value = _metric_value(after, seed=seed, metric=metric)
        paired_deltas[seed] = after_value - before_value

    deltas = list(paired_deltas.values())
    all_positive = all(delta > 0 for delta in deltas)
    all_negative = all(delta < 0 for delta in deltas)
    sign_agreement = all_positive or all_negative
    direction = "positive" if all_positive else "negative" if all_negative else "mixed"
    strong_effect_count = sum(abs(delta) >= strong_effect for delta in deltas)

    return {
        "metric": metric,
        "strong_effect_threshold": strong_effect,
        "seeds": seeds,
        "paired_deltas": paired_deltas,
        "mean_delta": statistics.fmean(deltas),
        "std_delta": statistics.pstdev(deltas),
        "sign_agreement": sign_agreement,
        "direction": direction,
        "strong_effect_count": strong_effect_count,
        "expand_to_five_seeds": (
            len(seeds) == 3 and sign_agreement and strong_effect_count >= 2
        ),
    }


def _canonical_run_name(name: str, mode: str, seed: int) -> str:
    if re.fullmatch(r"(?:final|transition)_N\d+_D\d+_d\d+_L\d+_seed\d+", name):
        return f"base_{mode}_seed{seed}"
    return name


def _run_parts(run_name: str, target_mode: str) -> tuple[str, str, str, int]:
    match = re.search(r"_seed(\d+)$", run_name)
    if match is None:
        raise ValueError(f"run name does not end in _seedN: {run_name}")
    seed = int(match.group(1))
    family = run_name[: match.start()]
    if family.startswith("base_"):
        source_mode = family.removeprefix("base_")
        stage = "base"
    elif family.startswith("continue_"):
        source_mode = family.removeprefix("continue_")
        stage = "continuation"
    elif "_to_" in family:
        source_mode, parsed_target = family.split("_to_", 1)
        if parsed_target != target_mode:
            raise ValueError(
                f"target mode mismatch for {run_name}: {parsed_target} != {target_mode}"
            )
        stage = "switch"
    else:
        source_mode = target_mode
        stage = "base"
    return family, source_mode, stage, seed


def _load_behavior_summaries(root: Path) -> dict[str, dict[str, Any]]:
    runs: dict[str, dict[str, Any]] = {}
    for path in root.rglob("summary.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(payload, dict) or "final_metrics" not in payload:
            continue
        mode = str(payload["mode"])
        seed = int(payload["seed"])
        raw_name = Path(payload["run_dir"]).name
        name = _canonical_run_name(raw_name, mode, seed)
        if name in runs:
            raise ValueError(f"duplicate behavior summary for {name}")
        runs[name] = payload
    return runs


def _load_temporal_summaries(root: Path) -> dict[str, dict[str, Any]]:
    models: dict[str, dict[str, Any]] = {}
    for path in root.glob("analysis/*/temporal/combined_summary.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for name, summary in payload["models"].items():
            if name in models:
                raise ValueError(f"duplicate temporal summary for {name}")
            models[name] = summary
    return models


def _load_diagnostic_scorecards(root: Path) -> dict[str, dict[str, Any]]:
    cards: dict[str, dict[str, Any]] = {}
    for path in root.glob("analysis/*/diagnostics/*/scorecard.json"):
        name = path.parent.name
        if name in cards:
            raise ValueError(f"duplicate diagnostic scorecard for {name}")
        cards[name] = json.loads(path.read_text(encoding="utf-8"))
    return cards


def _load_component_family_cosines(root: Path) -> dict[str, float]:
    output: dict[str, float] = {}
    for path in root.glob("analysis/*/components/family_summary.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for family, summary in payload["families"].items():
            if family in output:
                raise ValueError(f"duplicate component family summary for {family}")
            value = float(summary["mean_loop_profile_cosine"])
            if math.isfinite(value):
                output[family] = value
    return output


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path.name}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate_result_tree(*, root: Path, out_dir: Path) -> dict[str, Any]:
    behaviors = _load_behavior_summaries(root)
    temporal = _load_temporal_summaries(root)
    diagnostics = _load_diagnostic_scorecards(root)
    component_cosines = _load_component_family_cosines(root)
    if len(behaviors) < 30:
        raise ValueError(f"expected at least 30 completed runs, found {len(behaviors)}")

    rows_by_name: dict[str, dict[str, Any]] = {}
    for name, summary in sorted(behaviors.items()):
        metrics = summary["final_metrics"]
        family, source_mode, stage, seed = _run_parts(name, str(summary["mode"]))
        temporal_summary = temporal.get(name)
        algorithm_score: float | None = None
        receiver_sensitivity: float | None = None
        if temporal_summary is not None:
            cross_time = temporal_summary["cross_time"]
            by_delta = [float(value) for value in cross_time["mean_transplant_acc_by_delta"]]
            algorithm_score = statistics.fmean(by_delta[1:])
            receiver_sensitivity = statistics.fmean(
                float(value)
                for value in cross_time["mean_receiver_sensitivity_by_delta"][1:]
            )
        diagnostic = diagnostics.get(name, {})
        row = {
            "run_name": name,
            "family": family,
            "seed": seed,
            "stage": stage,
            "source_mode": source_mode,
            "target_mode": summary["mode"],
            "start_step": int(summary["start_step"]),
            "stop_step": int(summary["steps"]),
            "final_accuracy": float(metrics["final_accuracy"]),
            "mean_loop_accuracy": statistics.fmean(metrics["loop_accuracy"]),
            "mean_portable_one_step_accuracy": statistics.fmean(
                metrics["portable_one_step_accuracy"]
            ),
            "algorithm_cross_time_score": algorithm_score,
            "receiver_loop_sensitivity": receiver_sensitivity,
            "receiver_rule_context_gain": diagnostic.get("receiver_rule_context_gain"),
            "attractor_refinement": diagnostic.get("attractor_refinement"),
            "component_profile_cosine_family": component_cosines.get(family),
        }
        rows_by_name[name] = row

    transition_rows: list[dict[str, Any]] = []
    for name, after in sorted(rows_by_name.items()):
        if after["stage"] == "base":
            continue
        before_name = f"base_{after['source_mode']}_seed{after['seed']}"
        before = rows_by_name.get(before_name)
        if before is None:
            raise ValueError(f"missing paired source run {before_name} for {name}")
        transition_rows.append(
            {
                "transition": after["family"],
                "seed": after["seed"],
                "before_run": before_name,
                "after_run": name,
                "before_final_accuracy": before["final_accuracy"],
                "after_final_accuracy": after["final_accuracy"],
                "delta_final_accuracy": after["final_accuracy"] - before["final_accuracy"],
                "before_portable_accuracy": before["mean_portable_one_step_accuracy"],
                "after_portable_accuracy": after["mean_portable_one_step_accuracy"],
                "delta_portable_accuracy": (
                    after["mean_portable_one_step_accuracy"]
                    - before["mean_portable_one_step_accuracy"]
                ),
                "before_algorithm_score": before["algorithm_cross_time_score"],
                "after_algorithm_score": after["algorithm_cross_time_score"],
                "delta_algorithm_score": (
                    None
                    if before["algorithm_cross_time_score"] is None
                    or after["algorithm_cross_time_score"] is None
                    else after["algorithm_cross_time_score"]
                    - before["algorithm_cross_time_score"]
                ),
            }
        )

    def family_cards(family: str) -> dict[int, dict[str, float | None]]:
        return {
            int(row["seed"]): row
            for row in rows_by_name.values()
            if row["family"] == family and int(row["seed"]) in {0, 1, 2}
        }

    gates = {
        "final_vs_transition_portable_accuracy": aggregate_transition(
            family_cards("base_final"),
            family_cards("base_transition"),
            metric="mean_portable_one_step_accuracy",
            strong_effect=0.5,
        ),
        "final_vs_transition_algorithm_score": aggregate_transition(
            family_cards("base_final"),
            family_cards("base_transition"),
            metric="algorithm_cross_time_score",
            strong_effect=0.5,
        ),
        "final_to_transition_conversion": aggregate_transition(
            family_cards("base_final"),
            family_cards("final_to_transition"),
            metric="algorithm_cross_time_score",
            strong_effect=0.2,
        ),
        "transition_to_final_retention_scores": {
            seed: card["algorithm_cross_time_score"]
            for seed, card in family_cards("transition_to_final").items()
        },
        "portable_10k_scores": {
            seed: card["algorithm_cross_time_score"]
            for seed, card in family_cards("continue_portable").items()
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    run_rows = list(rows_by_name.values())
    (out_dir / "run_scorecards.json").write_text(
        json.dumps({"runs": run_rows}, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    _write_csv(out_dir / "run_scorecards.csv", run_rows)
    _write_csv(out_dir / "mechanism_transition.csv", transition_rows)
    (out_dir / "three_seed_gate.json").write_text(
        json.dumps(gates, indent=2, allow_nan=False), encoding="utf-8"
    )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    family_values: dict[str, list[float]] = {}
    for row in run_rows:
        value = row["algorithm_cross_time_score"]
        if value is not None:
            family_values.setdefault(str(row["family"]), []).append(float(value))
    ordered = sorted(family_values)
    means = [statistics.fmean(family_values[family]) for family in ordered]
    fig, axis = plt.subplots(figsize=(max(8.0, len(ordered) * 0.8), 4.8))
    axis.bar(range(len(ordered)), means, color="#3b82f6")
    axis.axhline(0.125, color="#64748b", linestyle="--", linewidth=1, label="chance")
    axis.set_xticks(range(len(ordered)), ordered, rotation=40, ha="right")
    axis.set_ylim(0.0, 1.05)
    axis.set_ylabel("cross-time algorithm score")
    axis.set_title("Graph reuse mechanism by training path")
    axis.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out_dir / "mechanism_transition.png", dpi=180)
    plt.close(fig)
    return {"run_count": len(run_rows), "transition_count": len(transition_rows), **gates}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate graph reuse runs and transitions.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = aggregate_result_tree(root=args.root, out_dir=args.out_dir)
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
