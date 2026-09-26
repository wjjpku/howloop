"""Guard and monitor one 10k half-uniform/half-m16 WSD KG run."""

import argparse
import json
import os
from pathlib import Path

from run_variant import monitored_run, sha256, write_json
from run_m20_uniform import BACKBONE, parent_path
from train_m20_uniform import BACKBONE_SHA256, CONTROLLERS, PARENT_SHA256
from train_mixed_wsd_from_m16 import DECAY, PEAK_LR, STABLE, STEPS, WARMUP


PYTHON = "/data/paperexperiment/.venvs/loopreasoner/bin/python"
PARALLEL_GPUS = [0, 1, 3, 6]


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
    if not args.trainer.is_file():
        raise FileNotFoundError(args.trainer)
    parent = parent_path(args.controller)
    if sha256(BACKBONE) != BACKBONE_SHA256 or sha256(parent) != PARENT_SHA256[args.controller]:
        raise RuntimeError("Backbone or original m16 parent hash mismatch")

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
        "trainer_sha256": sha256(args.trainer),
        "steps": STEPS,
        "sampling": "half_uniform4to16_half_fixed16_per_update",
        "mode_updates": {"uniform_4_to_16": 5000, "fixed_16": 5000},
        "schedule": {"name": "wsd", "warmup": WARMUP, "stable": STABLE,
                     "decay": DECAY, "peak_lr": PEAK_LR, "final_lr": 0.0},
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
        if (result["status"] != "complete" or result["steps"] != STEPS
                or result["backbone_sha256"] != BACKBONE_SHA256
                or result["parent_m16_sha256"] != PARENT_SHA256[args.controller]
                or result["sampling"] != manifest["sampling"]
                or result["schedule"] != manifest["schedule"]
                or result["mode_histogram"] != manifest["mode_updates"]):
            raise RuntimeError("Mixed-WSD protocol or result mismatch")
        if sum(result["length_histogram"].values()) != STEPS:
            raise RuntimeError("Length histogram does not sum to updates")
        stage_hashes = {
            str(step): sha256(args.run_root / f"train/controller_m16_mixed_wsd_step{step}.pt")
            for step in (5000, 10000)
        }
        manifest.update(status="complete", current_pid=None,
                        current_command=None,
                        result_sha256=sha256(result_path),
                        stage_sha256=stage_hashes,
                        elapsed_seconds=result["elapsed_seconds"])
        write_json(manifest_path, manifest)
        print(f"COMPLETE {args.controller} mixed WSD 10k from m16", flush=True)
    except Exception as error:
        manifest.update(status="failed", failure=repr(error))
        write_json(manifest_path, manifest)
        raise


if __name__ == "__main__":
    main()
