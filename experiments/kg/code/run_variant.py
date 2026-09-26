"""Guarded, sequential backbone -> dense-J run for one KG input variant."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path


ORIGINAL_CODE = Path("/data/paperexperiment/kg-fj-nope-20260815-v1/code")
ORIGINAL_CLI = ORIGINAL_CODE / "experiments/kg_fj_length/cli.py"
ORIGINAL_TRAINING = ORIGINAL_CODE / "experiments/kg_fj_length/training.py"
CLI_SHA = "ab48f4dc49782deb14ce31d82d4201cbcadeb552a3bba08356341d1e1bc42edd"
TRAINING_SHA = "ac4a4bcfc772e5e118080dcd29ecbe99250e25440c3aefb8bafae0000823acbf"
SOURCE_COMMIT = "39d4a75386d1ebc1f052e15d411c61f747488668"
PYTHON = "/data/paperexperiment/.venvs/loopreasoner/bin/python"
RESERVE_MIB = 16384


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def gpu_metrics(physical_gpu: int) -> dict:
    output = subprocess.check_output([
        "nvidia-smi", "-i", str(physical_gpu),
        "--query-gpu=memory.free,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ], text=True).strip()
    free, used, utilization = (int(part.strip()) for part in output.split(","))
    return {"free_mib": free, "used_mib": used, "utilization": utilization}


def monitored_run(name: str, command: list[str], env: dict, log_dir: Path,
                  physical_gpu: int, manifest_path: Path, manifest: dict) -> None:
    log_path = log_dir / f"{name}.log"
    monitor_path = log_dir / f"{name}_gpu.jsonl"
    with log_path.open("x") as log, monitor_path.open("x") as monitor:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
        manifest.update(status=f"running_{name}", physical_gpu=physical_gpu,
                        current_pid=process.pid, current_command=command,
                        current_log=str(log_path))
        write_json(manifest_path, manifest)
        print(f"START {name} pid={process.pid} physical_gpu={physical_gpu}", flush=True)
        started = time.monotonic()
        while True:
            exit_code = process.poll()
            metrics = gpu_metrics(physical_gpu)
            monitor.write(json.dumps({"seconds": round(time.monotonic() - started, 1),
                                      "pid": process.pid, "physical_gpu": physical_gpu,
                                      **metrics}) + "\n")
            monitor.flush()
            if metrics["free_mib"] < RESERVE_MIB and exit_code is None:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                manifest.update(status=f"stopped_{name}_reserve", current_exit_code=process.returncode)
                write_json(manifest_path, manifest)
                raise RuntimeError(f"{name} stopped: GPU free memory crossed {RESERVE_MIB} MiB")
            if exit_code is not None:
                manifest.update(current_exit_code=exit_code)
                write_json(manifest_path, manifest)
                print(f"END {name} exit={exit_code}", flush=True)
                if exit_code != 0:
                    raise RuntimeError(f"{name} failed; see {log_path}")
                return
            time.sleep(30)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("e64r16", "e128r8"), required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--dense-wrapper", type=Path, required=True)
    args = parser.parse_args()
    entities, relations = {"e64r16": (64, 16), "e128r8": (128, 8)}[args.variant]
    if not 0 <= args.physical_gpu < 8:
        raise ValueError("Invalid physical GPU")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.physical_gpu):
        raise RuntimeError("Physical GPU must be explicitly pinned")
    if not str(args.run_root).startswith("/data/paperexperiment/"):
        raise ValueError("Run root must be under /data/paperexperiment")
    if not str(args.log_root).startswith("/data/paperexperiment/logs/"):
        raise ValueError("Log root must be under /data/paperexperiment/logs")
    if sha256(ORIGINAL_CLI) != CLI_SHA or sha256(ORIGINAL_TRAINING) != TRAINING_SHA:
        raise RuntimeError("Historical backbone source hash changed")
    if not args.dense_wrapper.is_file():
        raise FileNotFoundError(args.dense_wrapper)
    if args.run_root.exists() or args.log_root.exists():
        raise FileExistsError("Fresh run and log roots are required")
    args.run_root.mkdir(parents=True)
    args.log_root.mkdir(parents=True)
    manifest_path = args.run_root / "run_manifest.json"
    manifest = {
        "variant": args.variant, "entities": entities, "relations": relations,
        "physical_gpu": args.physical_gpu, "status": "starting",
        "backbone_seed": 101, "controller_seed": 20260922,
        "world_seed": 20260814, "source_commit": SOURCE_COMMIT,
        "original_cli_sha256": CLI_SHA, "original_training_sha256": TRAINING_SHA,
        "dense_wrapper_sha256": sha256(args.dense_wrapper),
    }
    write_json(manifest_path, manifest)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ORIGINAL_CODE)
    env["CUDA_VISIBLE_DEVICES"] = str(args.physical_gpu)
    backbone = args.run_root / "backbone"
    backbone_command = [
        PYTHON, "-u", "-m", "experiments.kg_fj_length.cli", "backbone",
        "--output-dir", str(backbone), "--device", "cuda:0",
        "--seed", "101", "--world-seed", "20260814",
        "--source-commit", SOURCE_COMMIT,
        "--entities", str(entities), "--relations", str(relations),
        "--d-model", "256", "--heads", "8", "--d-mlp", "1024",
        "--blocks", "2", "--position-encoding", "none",
        "--supervision-mode", "aligned_intermediate",
        "--steps", "80000", "--batch-size", "512", "--learning-rate", "3e-4",
        "--warmup-steps", "2000", "--stable-steps", "60000",
        "--decay-steps", "18000", "--eval-interval", "2000",
        "--selection-count", "512", "--test-count", "4096",
    ]
    try:
        monitored_run("backbone", backbone_command, env, args.log_root,
                      args.physical_gpu, manifest_path, manifest)
        summary = json.loads((backbone / "summary.json").read_text())
        accuracy = summary["selection_accuracy"]
        checkpoint = backbone / "best.pt"
        checkpoint_sha = sha256(checkpoint)
        if checkpoint_sha != summary["best_checkpoint_sha256"]:
            raise RuntimeError("Backbone checkpoint and summary hashes disagree")
        manifest.update(backbone_sha256=checkpoint_sha,
                        backbone_best_step=summary["best_step"],
                        backbone_selection_accuracy=accuracy)
        if not all(float(accuracy[str(length)]) >= .98 for length in (1, 2, 3)):
            manifest.update(status="backbone_gate_failed")
            write_json(manifest_path, manifest)
            print("BACKBONE_GATE_FAILED", accuracy, flush=True)
            return
        dense_command = [
            PYTHON, "-u", str(args.dense_wrapper),
            "--backbone", str(checkpoint), "--backbone-sha256", checkpoint_sha,
            "--entities", str(entities), "--relations", str(relations),
            "--seed", "20260922", "--out-root", str(args.run_root / "dense_curriculum"),
        ]
        monitored_run("dense_j", dense_command, env, args.log_root,
                      args.physical_gpu, manifest_path, manifest)
        result = args.run_root / "dense_curriculum/dense/results.json"
        payload = json.loads(result.read_text())
        if payload["controller"] != "dense" or payload["stages"] != list(range(4, 33, 4)):
            raise RuntimeError("Incomplete dense-J result schema")
        if len(payload["heatmap"]) != 8:
            raise RuntimeError("Incomplete 8-stage dense-J heatmap")
        stage_hashes = {str(stage): sha256(args.run_root / f"dense_curriculum/dense/controller_m{stage}.pt")
                        for stage in range(4, 33, 4)}
        manifest.update(status="complete", current_pid=None, current_command=None,
                        dense_result_sha256=sha256(result), dense_stage_sha256=stage_hashes)
        write_json(manifest_path, manifest)
        print("COMPLETE", args.variant, flush=True)
    except Exception as error:
        if manifest.get("status") not in {"backbone_gate_failed", "complete"}:
            manifest.update(status="failed", failure=repr(error))
            write_json(manifest_path, manifest)
        raise


if __name__ == "__main__":
    main()
