from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
from pathlib import Path
from typing import Any


def _selection_score(early: float, middle: float, late: float) -> float:
    return (
        0.15 * early
        + 0.25 * middle
        + 0.60 * late
        - 2.0 * max(0.0, 0.98 - early)
        - 1.0 * max(0.0, 0.95 - middle)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--phase-summary", type=Path, required=True)
    parser.add_argument("--screen-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--python-bin", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument(
        "--evaluation-seeds",
        type=int,
        nargs="+",
        default=(402004, 403004),
    )
    parser.add_argument("--evaluation-batches", type=int, default=8)
    parser.add_argument("--physical-gpu", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ranking = json.loads(
        (args.screen_root / "ranking.json").read_text(encoding="utf-8")
    )
    candidates = list(ranking[: args.top_k])
    included_names = {item["name"] for item in candidates}
    required_names = {"baseline_no_finetune"}
    required_names.update(
        next(
            item["name"]
            for item in ranking
            if item["task_parameterization"] == parameterization
        )
        for parameterization in (
            "lora",
            "identity_low_rank",
            "scalar_low_rank",
            "diagonal_low_rank",
            "weight_low_rank",
        )
        if any(
            item["task_parameterization"] == parameterization
            for item in ranking
        )
    )
    candidates.extend(
        item
        for item in ranking
        if item["name"] in required_names
        and item["name"] not in included_names
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.log_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for candidate_index, candidate in enumerate(candidates, start=1):
        name = candidate["name"]
        artifact = args.screen_root / name / "unit_j_maps.pt"
        for seed in args.evaluation_seeds:
            run_name = f"{name}_eval{seed}"
            out_dir = args.output_root / run_name
            summary_path = out_dir / "summary.json"
            if not summary_path.exists():
                command = [
                    str(args.python_bin),
                    "-m",
                    "reasoning_loop.graph_path_telomere_unit_j",
                    "--checkpoint",
                    str(args.checkpoint),
                    "--phase-summary",
                    str(args.phase_summary),
                    "--out-dir",
                    str(out_dir),
                    "--device",
                    "cuda",
                    "--frozen-model-loss-placement",
                    "final CE at loop 8 plus intermediate CE on p_min(2t,D), t=1..7",
                    "--initial-map-artifact",
                    str(artifact),
                    "--initial-map-label",
                    "task",
                    "--calibration-batch-size",
                    "16",
                    "--calibration-batches",
                    "1",
                    "--dagger-rounds",
                    "0",
                    "--power-rounds",
                    "0",
                    "--task-rounds",
                    "0",
                    "--rollout-horizons",
                    "32",
                    "48",
                    "64",
                    "--policies",
                    "unit_every",
                    "--train-start-ages",
                    "8",
                    "--position-group",
                    "all",
                    "--operating-age",
                    "3",
                    "--direct-operating-target-age",
                    "3",
                    "--evaluation-batch-size",
                    "128",
                    "--evaluation-batches",
                    str(args.evaluation_batches),
                    "--continuation-loops",
                    "64",
                    "--calibration-seed",
                    "402001",
                    "--evaluation-seed",
                    str(seed),
                    "--physical-gpu",
                    str(args.physical_gpu),
                    "--declared-peak-gib",
                    "3",
                    "--reserve-gib",
                    "16",
                ]
                log_path = args.log_root / f"{run_name}.log"
                print(
                    f"START {candidate_index}/{len(candidates)} "
                    f"{run_name}",
                    flush=True,
                )
                with log_path.open("w", encoding="utf-8") as log:
                    result = subprocess.run(
                        command,
                        cwd=args.code_dir,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=False,
                    )
                if result.returncode != 0:
                    raise RuntimeError(f"validation failed: {run_name}")
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            curve = summary["curves"]["task_unit_every"][
                "accuracy_nonendpoint"
            ]
            early = float(curve["auc_1_24"])
            middle = float(curve["auc_25_48"])
            late = float(curve["auc_49_64"])
            rows.append(
                {
                    "name": name,
                    "evaluation_seed": seed,
                    "auc_1_24": early,
                    "auc_25_48": middle,
                    "auc_49_64": late,
                    "cycle64": float(curve["values"][63]),
                    "selection_score": _selection_score(
                        early,
                        middle,
                        late,
                    ),
                    "screen_rank": candidate_index,
                    "screen_score": candidate["selection_score"],
                    "screen_late": candidate["auc_49_64"],
                }
            )
    aggregate = []
    for candidate in candidates:
        name = candidate["name"]
        group = [row for row in rows if row["name"] == name]
        record: dict[str, Any] = {
            "name": name,
            "screen_rank": next(
                row["screen_rank"] for row in group
            ),
            "screen_score": candidate["selection_score"],
            "screen_late": candidate["auc_49_64"],
            "evaluations": group,
        }
        for metric in (
            "auc_1_24",
            "auc_25_48",
            "auc_49_64",
            "cycle64",
            "selection_score",
        ):
            values = [row[metric] for row in group]
            record[f"{metric}_mean"] = statistics.mean(values)
            record[f"{metric}_std"] = statistics.pstdev(values)
            record[f"{metric}_min"] = min(values)
        aggregate.append(record)
    aggregate.sort(
        key=lambda item: (
            item["selection_score_mean"],
            item["selection_score_min"],
        ),
        reverse=True,
    )
    payload = {
        "status": "complete",
        "top_k": args.top_k,
        "evaluated_candidates": len(candidates),
        "required_candidates": sorted(required_names),
        "evaluation_seeds": args.evaluation_seeds,
        "evaluation_graphs_per_seed": 128 * args.evaluation_batches,
        "rows": rows,
        "ranking": aggregate,
    }
    path = args.output_root / "validation_ranking.json"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    for item in aggregate:
        print(
            item["name"],
            f"late={item['auc_49_64_mean']:.6f}"
            f"+-{item['auc_49_64_std']:.6f}",
            f"score={item['selection_score_mean']:.6f}",
        )


if __name__ == "__main__":
    main()
