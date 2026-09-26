"""Guarded, GPU-monitored m16 uniform continuation for one controller."""

import argparse
import json
import os
from pathlib import Path

from run_variant import monitored_run, sha256, write_json
from train_m20_uniform import (
    BACKBONE_SHA256,
    CONTROLLERS,
    PARENT_SHA256,
    SOURCE_SHA256,
    STEPS,
)


PYTHON = "/data/paperexperiment/.venvs/loopreasoner/bin/python"
BACKBONE = Path("/data/paperexperiment/kg_input_complexity_20260922/runs/e64r16/backbone/best.pt")
AFFINE_PARENT = Path("/data/paperexperiment/kg_input_complexity_20260922/affine_step1_3k_4to16/run/curriculum/dense/controller_m16.pt")
OTHER_ROOT = Path("/data/paperexperiment/kg_input_complexity_20260922/unit_step_three_controllers/runs")
PARALLEL_GPUS = [0, 1, 3, 6]


def parent_path(controller):
    if controller == "dense":
        return AFFINE_PARENT
    return OTHER_ROOT / controller / "curriculum" / controller / "controller_m16.pt"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller", choices=CONTROLLERS, required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--trainer", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.physical_gpu):
        raise RuntimeError("Physical GPU must be explicitly pinned")
    if args.physical_gpu not in PARALLEL_GPUS:
        raise RuntimeError("GPU outside recorded four-job campaign")
    if not str(args.run_root).startswith("/data/paperexperiment/"):
        raise ValueError("Run root must be under /data/paperexperiment")
    if not str(args.log_root).startswith("/data/paperexperiment/logs/"):
        raise ValueError("Log root must be under /data/paperexperiment/logs")
    if args.run_root.exists() or args.log_root.exists():
        raise FileExistsError("Fresh run and log roots are required")
    parent = parent_path(args.controller)
    if not args.trainer.is_file():
        raise FileNotFoundError(args.trainer)
    if sha256(BACKBONE) != BACKBONE_SHA256 or sha256(parent) != PARENT_SHA256[args.controller]:
        raise RuntimeError("Backbone or parent controller hash mismatch")

    args.run_root.mkdir(parents=True)
    args.log_root.mkdir(parents=True)
    manifest_path = args.run_root / "run_manifest.json"
    manifest = {
        "status": "starting", "controller": args.controller,
        "physical_gpu": args.physical_gpu,
        "user_authorized_parallel_jobs": 4,
        "selected_physical_gpus": PARALLEL_GPUS,
        "backbone_sha256": BACKBONE_SHA256,
        "parent_m16_sha256": PARENT_SHA256[args.controller],
        "source_sha256": SOURCE_SHA256,
        "trainer_sha256": sha256(args.trainer),
        "steps": STEPS,
        "sampling": "uniform_length_4_to_16_per_update",
        "ood_target_length": 20,
    }
    write_json(manifest_path, manifest)
    command = [
        PYTHON, "-u", str(args.trainer),
        "--controller", args.controller,
        "--backbone", str(BACKBONE),
        "--parent", str(parent),
        "--out-dir", str(args.run_root / "train"),
    ]
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(args.physical_gpu)
    try:
        monitored_run(args.controller, command, env, args.log_root,
                      args.physical_gpu, manifest_path, manifest)
        result_path = args.run_root / "train/train_result.json"
        result = json.loads(result_path.read_text())
        if result["status"] != "complete" or result["steps"] != STEPS:
            raise RuntimeError("Incomplete training result")
        if result["backbone_sha256"] != BACKBONE_SHA256:
            raise RuntimeError("Backbone mismatch")
        if result["parent_sha256"] != PARENT_SHA256[args.controller]:
            raise RuntimeError("Parent mismatch")
        if result["sampling"] != "uniform_length_4_to_16_per_update":
            raise RuntimeError("Sampling protocol mismatch")
        if sum(result["length_histogram"].values()) != STEPS:
            raise RuntimeError("Sampling histogram does not sum to updates")
        stage_hashes = {
            str(step): sha256(args.run_root / f"train/controller_m16_plus10k_step{step}.pt")
            for step in (5000, 10000)
        }
        manifest.update(
            status="complete", current_pid=None, current_command=None,
            result_sha256=sha256(result_path), stage_sha256=stage_hashes,
            elapsed_seconds=result["elapsed_seconds"],
        )
        write_json(manifest_path, manifest)
        print(f"COMPLETE {args.controller} m16 uniform 10k; OOD target n20", flush=True)
    except Exception as error:
        manifest.update(status="failed", failure=repr(error))
        write_json(manifest_path, manifest)
        raise


if __name__ == "__main__":
    main()
