"""Monitor full 1..32 length-grid evaluation of one mixed-WSD controller."""

import argparse
import json
import os
from pathlib import Path

from run_variant import monitored_run, sha256, write_json
from train_m20_uniform import CONTROLLERS


PYTHON = "/data/paperexperiment/.venvs/loopreasoner/bin/python"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller", choices=CONTROLLERS, required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--evaluator", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.physical_gpu):
        raise RuntimeError("Physical GPU must be explicitly pinned")
    if not str(args.run_root).startswith("/data/paperexperiment/"):
        raise ValueError("Run root must be under /data/paperexperiment")
    if not str(args.log_root).startswith("/data/paperexperiment/logs/"):
        raise ValueError("Log root must be under /data/paperexperiment/logs")
    training = json.loads((args.run_root / "run_manifest.json").read_text())
    if training["status"] != "complete" or training["controller"] != args.controller:
        raise RuntimeError("Training incomplete or wrong controller")
    if (args.run_root / "analysis").exists():
        raise FileExistsError("Analysis directory already exists")
    if not args.evaluator.is_file():
        raise FileNotFoundError(args.evaluator)
    manifest_path = args.run_root / "evaluation_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(manifest_path)
    manifest = {
        "status": "starting", "controller": args.controller,
        "physical_gpu": args.physical_gpu,
        "training_manifest_sha256": sha256(args.run_root / "run_manifest.json"),
        "evaluator_sha256": sha256(args.evaluator),
    }
    write_json(manifest_path, manifest)
    command = [PYTHON, "-u", str(args.evaluator),
               "--controller", args.controller, "--run-root", str(args.run_root)]
    try:
        monitored_run(f"{args.controller}_eval", command, dict(os.environ),
                      args.log_root, args.physical_gpu, manifest_path, manifest)
        analysis = args.run_root / "analysis"
        grid = json.loads((analysis / "grid_replay.json").read_text())
        key = json.loads((analysis / "key_cells_1024_summary.json").read_text())
        if grid["status"] != "complete" or grid["archived_parent_cells_replayed"] != 29:
            raise RuntimeError("Parent replay incomplete")
        if (key["status"] != "complete" or len(key["cells"]) != 32
                or key["evaluated_lengths"] != list(range(1, 33))):
            raise RuntimeError("Independent 1..32 panel incomplete")
        if sha256(analysis / "key_cells_1024.jsonl") != key["per_example_sha256"]:
            raise RuntimeError("Per-example archive hash mismatch")
        manifest.update(
            status="complete", current_pid=None, current_command=None,
            grid_sha256=sha256(analysis / "grid_replay.json"),
            key_summary_sha256=sha256(analysis / "key_cells_1024_summary.json"),
            per_example_sha256=key["per_example_sha256"],
        )
        write_json(manifest_path, manifest)
        print(f"COMPLETE {args.controller} mixed-WSD length evaluation", flush=True)
    except Exception as error:
        manifest.update(status="failed", failure=repr(error))
        write_json(manifest_path, manifest)
        raise


if __name__ == "__main__":
    main()
