from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


def _config(
    name: str,
    **overrides: Any,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "name": name,
        "task_rounds": 6,
        "task_learning_rate": 3e-6,
        "task_parameterization": "full",
        "task_rank": 256,
        "task_time_weighting": "uniform",
        "task_weight_decay": 0.0,
        "task_grad_clip": 1.0,
        "task_detach_interval": 0,
        "task_freeze_bias": False,
        "rollout_horizons": [32, 48, 64],
    }
    result.update(overrides)
    return result


def default_screen_configs() -> list[dict[str, Any]]:
    configs = [_config("baseline_no_finetune", task_rounds=0)]
    configs.extend(
        [
            _config("full_base"),
            _config("full_lr1e6", task_learning_rate=1e-6),
            _config("full_lr1e5", task_learning_rate=1e-5),
            _config("full_time_linear", task_time_weighting="linear"),
            _config("full_time_quadratic", task_time_weighting="quadratic"),
            _config("full_time_latehalf", task_time_weighting="late_half"),
            _config("full_time_terminal", task_time_weighting="terminal"),
            _config("full_detach8", task_detach_interval=8),
            _config("full_detach16", task_detach_interval=16),
            _config("full_wd1e5", task_weight_decay=1e-5),
            _config("full_wd1e4", task_weight_decay=1e-4),
            _config("full_clip03", task_grad_clip=0.3),
            _config("full_clip3", task_grad_clip=3.0),
            _config("full_h64_only", rollout_horizons=[64]),
            _config(
                "full_h16_32_48_64",
                rollout_horizons=[16, 32, 48, 64],
            ),
        ]
    )
    for rank in (8, 16, 32, 64, 128):
        configs.append(
            _config(
                f"lora_r{rank}",
                task_parameterization="lora",
                task_rank=rank,
            )
        )
    for rank in (16, 32, 64, 96, 128, 192):
        configs.append(
            _config(
                f"identity_lowrank_r{rank}",
                task_parameterization="identity_low_rank",
                task_rank=rank,
            )
        )
    for rank in (16, 32, 64, 96, 128):
        configs.append(
            _config(
                f"scalar_lowrank_r{rank}",
                task_parameterization="scalar_low_rank",
                task_rank=rank,
            )
        )
    for rank in (16, 32, 64, 96, 128):
        configs.append(
            _config(
                f"diagonal_lowrank_r{rank}",
                task_parameterization="diagonal_low_rank",
                task_rank=rank,
            )
        )
    for rank in (64, 128, 192):
        configs.append(
            _config(
                f"weight_lowrank_r{rank}",
                task_parameterization="weight_low_rank",
                task_rank=rank,
            )
        )
    return configs


def validate_pure_ce_config(config: dict[str, Any]) -> None:
    legacy_keys = {
        "task_state_loss_weight",
        "task_state_answer_weight",
    }
    present = sorted(legacy_keys.intersection(config))
    if present:
        raise ValueError(
            "hidden-state loss configuration is forbidden in the pure-CE "
            f"J trainer: {present}"
        )


def _metric(summary: dict[str, Any]) -> dict[str, float]:
    curve = summary["curves"]["task_unit_every"]["accuracy_nonendpoint"]
    early = float(curve["auc_1_24"])
    middle = float(curve["auc_25_48"])
    late = float(curve["auc_49_64"])
    score = (
        0.15 * early
        + 0.25 * middle
        + 0.60 * late
        - 2.0 * max(0.0, 0.98 - early)
        - 1.0 * max(0.0, 0.95 - middle)
    )
    return {
        "auc_1_24": early,
        "auc_25_48": middle,
        "auc_49_64": late,
        "cycle64": float(curve["values"][63]),
        "selection_score": score,
    }


def _write_progress(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--phase-summary", type=Path, required=True)
    parser.add_argument("--initial-j", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--python-bin", type=Path, required=True)
    parser.add_argument("--only", nargs="*")
    parser.add_argument("--max-configs", type=int)
    parser.add_argument(
        "--configs-json",
        type=Path,
        help="Optional expanded config list instead of the default screen grid.",
    )
    parser.add_argument("--task-seed", type=int, default=401003)
    parser.add_argument("--evaluation-seed", type=int, default=401004)
    parser.add_argument("--evaluation-batch-size", type=int, default=128)
    parser.add_argument("--evaluation-batches", type=int, default=4)
    parser.add_argument("--continuation-loops", type=int, default=64)
    parser.add_argument("--physical-gpu", type=int, default=2)
    parser.add_argument("--prelaunch-used-mib", type=int, default=4)
    parser.add_argument("--prelaunch-free-mib", type=int, default=81150)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configs = (
        json.loads(args.configs_json.read_text(encoding="utf-8"))
        if args.configs_json is not None
        else default_screen_configs()
    )
    for config in configs:
        validate_pure_ce_config(config)
    if args.only:
        requested = set(args.only)
        configs = [item for item in configs if item["name"] in requested]
        missing = requested - {item["name"] for item in configs}
        if missing:
            raise ValueError(f"unknown configs: {sorted(missing)}")
    if args.max_configs is not None:
        configs = configs[: args.max_configs]
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.log_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "expanded_configs.json").write_text(
        json.dumps(configs, indent=2) + "\n",
        encoding="utf-8",
    )
    progress_path = args.output_root / "progress.json"
    progress: list[dict[str, Any]] = []
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
    completed = {
        item["name"] for item in progress if item.get("status") == "complete"
    }

    for index, config in enumerate(configs, start=1):
        name = str(config["name"])
        out_dir = args.output_root / name
        summary_path = out_dir / "summary.json"
        if name in completed and summary_path.exists():
            print(f"SKIP {index}/{len(configs)} {name}", flush=True)
            continue
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
            str(args.initial_j),
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
            "--answer-weight",
            "28",
            "--identity-weight",
            "1",
            "--ridge",
            "0.01",
            "--task-batch-size",
            "32",
            "--task-batches-per-round",
            "9",
            "--task-rounds",
            str(config["task_rounds"]),
            "--task-learning-rate",
            str(config["task_learning_rate"]),
            "--task-parameterization",
            str(config["task_parameterization"]),
            "--task-rank",
            str(config["task_rank"]),
            "--task-time-weighting",
            str(config["task_time_weighting"]),
            "--task-weight-decay",
            str(config["task_weight_decay"]),
            "--task-grad-clip",
            str(config["task_grad_clip"]),
            "--task-detach-interval",
            str(config["task_detach_interval"]),
            "--rollout-horizons",
            *[str(value) for value in config["rollout_horizons"]],
            "--evaluation-batch-size",
            str(args.evaluation_batch_size),
            "--evaluation-batches",
            str(args.evaluation_batches),
            "--continuation-loops",
            str(args.continuation_loops),
            "--calibration-seed",
            "401001",
            "--task-seed",
            str(args.task_seed),
            "--evaluation-seed",
            str(args.evaluation_seed),
            "--physical-gpu",
            str(args.physical_gpu),
            "--prelaunch-used-mib",
            str(args.prelaunch_used_mib),
            "--prelaunch-free-mib",
            str(args.prelaunch_free_mib),
            "--declared-peak-gib",
            "3",
            "--reserve-gib",
            "16",
        ]
        if bool(config.get("task_freeze_bias", False)):
            command.append("--task-freeze-bias")
        log_path = args.log_root / f"{name}.log"
        print(f"START {index}/{len(configs)} {name}", flush=True)
        with log_path.open("w", encoding="utf-8") as log:
            result = subprocess.run(
                command,
                cwd=args.code_dir,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        row: dict[str, Any] = {
            **config,
            "status": "complete" if result.returncode == 0 else "failed",
            "returncode": result.returncode,
            "summary": str(summary_path),
            "log": str(log_path),
        }
        if result.returncode == 0 and summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            row.update(_metric(summary))
            row["observed_peak_gib"] = summary["gpu_runtime"][
                "observed_peak_gib"
            ]
            row["trainable_J_parameters"] = summary[
                "J_trainable_parameter_count"
            ]
        progress = [item for item in progress if item["name"] != name]
        progress.append(row)
        _write_progress(progress_path, progress)
        print(
            "DONE "
            f"{name} status={row['status']} "
            f"late={row.get('auc_49_64')} "
            f"score={row.get('selection_score')}",
            flush=True,
        )
        if result.returncode != 0:
            print(f"FAILED_LOG {log_path}", file=sys.stderr, flush=True)

    complete = [item for item in progress if item["status"] == "complete"]
    ranking = sorted(
        complete,
        key=lambda item: item["selection_score"],
        reverse=True,
    )
    _write_progress(args.output_root / "ranking.json", ranking)
    print(f"COMPLETE {len(complete)}/{len(configs)}", flush=True)


if __name__ == "__main__":
    main()
