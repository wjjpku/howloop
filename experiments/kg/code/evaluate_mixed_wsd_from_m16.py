"""Matched full-length grid and independent 1024-example evaluation of mixed WSD."""

import argparse
import importlib.util
import json
import os
from pathlib import Path

import torch

from train_m20_uniform import (
    BACKBONE_SHA256,
    CONTROLLERS,
    PARENT_SHA256,
    SOURCE,
    SOURCE_SHA256,
    load_backbone,
    sha256,
)
from run_m20_uniform import BACKBONE, parent_path
from train_mixed_wsd_from_m16 import DECAY, PEAK_LR, STABLE, STEPS, WARMUP


REFERENCE_ROOT = Path("/data/paperexperiment/kg_input_complexity_20260922/ood20_uniform4to16_10k_20260922/runs")
TEST_LENGTHS = tuple(range(1, 33))
KEY_LENGTHS = tuple(range(1, 33))
N_GRID = 128
N_KEY = 1024


def make_controller(source, name, path, expected_sha, device):
    if sha256(path) != expected_sha:
        raise RuntimeError(f"Controller hash mismatch: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload["controller_type"] != name or payload["backbone_sha256"] != BACKBONE_SHA256:
        raise RuntimeError("Controller class or backbone mismatch")
    controller = source.make_controller(name).to(device).eval()
    controller.load_state_dict(payload["controller"], strict=True)
    return controller


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller", choices=CONTROLLERS, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in {str(i) for i in range(8)}:
        raise RuntimeError("Pin exactly one physical GPU")
    if sha256(SOURCE) != SOURCE_SHA256 or sha256(BACKBONE) != BACKBONE_SHA256:
        raise RuntimeError("Historical source or backbone hash mismatch")
    receipt = json.loads((args.run_root / "run_manifest.json").read_text())
    expected_schedule = {"name": "wsd", "warmup": WARMUP,
                         "stable": STABLE, "decay": DECAY,
                         "peak_lr": PEAK_LR, "final_lr": 0.0}
    if (receipt["status"] != "complete" or receipt["controller"] != args.controller
            or receipt["steps"] != STEPS
            or receipt["sampling"] != "half_uniform4to16_half_fixed16_per_update"
            or receipt["mode_updates"] != {"uniform_4_to_16": 5000, "fixed_16": 5000}
            or receipt["schedule"] != expected_schedule
            or receipt["parent_m16_sha256"] != PARENT_SHA256[args.controller]):
        raise RuntimeError("Mixed-WSD training protocol mismatch")
    result_path = args.run_root / "train/train_result.json"
    if sha256(result_path) != receipt["result_sha256"]:
        raise RuntimeError("Training result hash mismatch")
    result = json.loads(result_path.read_text())
    if result["mode_histogram"] != receipt["mode_updates"]:
        raise RuntimeError("Wrong 50/50 training mix")
    if set(result["length_histogram"]) != {str(n) for n in range(4, 17)}:
        raise RuntimeError("Training length range is not exactly 4..16")
    reference = REFERENCE_ROOT / args.controller
    reference_eval = json.loads((reference / "evaluation_manifest.json").read_text())
    if reference_eval["status"] != "complete":
        raise RuntimeError("Reference m16 evaluation incomplete")
    reference_grid_path = reference / "analysis/grid_replay.json"
    if sha256(reference_grid_path) != reference_eval["grid_sha256"]:
        raise RuntimeError("Reference grid hash mismatch")
    reference_grid = json.loads(reference_grid_path.read_text())
    reference_predictions_path = reference / "analysis/key_cells_1024.jsonl"
    if sha256(reference_predictions_path) != reference_eval["per_example_sha256"]:
        raise RuntimeError("Reference per-example archive hash mismatch")
    old_predictions = {(row["test_length"], row["example"]): row
                       for line in reference_predictions_path.open()
                       if (row := json.loads(line))}
    if len(old_predictions) != 8 * N_KEY:
        raise RuntimeError("Reference per-example count mismatch")

    spec = importlib.util.spec_from_file_location("kg_locked_mixed_wsd_eval", SOURCE)
    assert spec is not None and spec.loader is not None
    source = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(source)
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(.12, 0)
    device = torch.device("cuda")
    model, kg = load_backbone(source, BACKBONE, device)
    world = source.PermutationWorld.create(kg, source.WORLD_SEED)
    parent = make_controller(source, args.controller, parent_path(args.controller),
                             PARENT_SHA256[args.controller], device)
    final = make_controller(source, args.controller,
                            args.run_root / "train/controller_m16_mixed_wsd_step10000.pt",
                            receipt["stage_sha256"]["10000"], device)
    before_grid = [round(source.evaluate(model, parent, world, kg, n, device,
                                         N_GRID), 4) for n in TEST_LENGTHS]
    if before_grid[3:] != reference_grid["parent_m16_grid"]:
        raise RuntimeError("Original m16 checkpoint failed archived 29-cell replay")
    after_grid = [round(source.evaluate(model, final, world, kg, n, device,
                                        N_GRID), 4) for n in TEST_LENGTHS]
    analysis = args.run_root / "analysis"
    analysis.mkdir(exist_ok=False)
    grid_payload = {
        "status": "complete", "controller": args.controller,
        "backbone_sha256": BACKBONE_SHA256,
        "parent_sha256": PARENT_SHA256[args.controller],
        "final_sha256": receipt["stage_sha256"]["10000"],
        "sampling": "half_uniform4to16_half_fixed16_per_update",
        "updates": 10000,
        "train_length_cap": 16,
        "test_lengths": list(TEST_LENGTHS), "examples_per_cell": N_GRID,
        "parent_m16_grid": before_grid,
        "continued_m16_grid": after_grid,
        "archived_parent_cells_replayed": 29,
        "new_short_length_cells": 3,
    }
    (analysis / "grid_replay.json").write_text(json.dumps(grid_payload, indent=2) + "\n")
    print(f"{args.controller}: replayed archived 29/29 cells, evaluated all lengths 1..32", flush=True)

    rows = []
    with torch.no_grad(), (analysis / "key_cells_1024.jsonl").open("x") as output:
        for n in KEY_LENGTHS:
            generator = torch.Generator().manual_seed(1990000 + 7 * n)
            cpu_batch = source.sample_batch(kg, world, N_KEY, n, generator,
                                            frozenset())
            batch = cpu_batch.to(device)
            target = cpu_batch.target
            raw = model(batch.tokens, calls=n).argmax(-1).cpu()
            before = source.run_alternating(model, parent, batch.tokens,
                                            calls=n).argmax(-1).cpu()
            after = source.run_alternating(model, final, batch.tokens,
                                           calls=n).argmax(-1).cpu()
            before_ok, after_ok = before.eq(target), after.eq(target)
            cell = {
                "test_length": n, "n": N_KEY,
                "raw_hits": int(raw.eq(target).sum()),
                "before_hits": int(before_ok.sum()),
                "after_hits": int(after_ok.sum()),
                "after_only_hits": int((after_ok & ~before_ok).sum()),
                "before_only_hits": int((before_ok & ~after_ok).sum()),
            }
            rows.append(cell)
            for i in range(N_KEY):
                entry = {
                    "test_length": n, "example": i,
                    "start": int(cpu_batch.start[i]),
                    "relations": cpu_batch.relations[i].tolist(),
                    "target": int(target[i]),
                    "raw_prediction": int(raw[i]),
                    "before_prediction": int(before[i]),
                    "after_prediction": int(after[i]),
                }
                old = old_predictions.get((n, i))
                if old is not None:
                    for key in ("start", "relations", "target", "raw_prediction",
                                "before_prediction"):
                        if entry[key] != old[key]:
                            raise RuntimeError(f"Evaluation drift at n={n}, i={i}, {key}")
                output.write(json.dumps(entry) + "\n")
            print(json.dumps({"controller": args.controller, **cell}), flush=True)
    summary = {
        "status": "complete", "controller": args.controller,
        "backbone_sha256": BACKBONE_SHA256,
        "parent_sha256": PARENT_SHA256[args.controller],
        "final_sha256": receipt["stage_sha256"]["10000"],
        "evaluation_seed_rule": "1990000 + 7 * test_length",
        "evaluated_lengths": list(KEY_LENGTHS),
        "reference_overlap_lengths": sorted({n for n, _ in old_predictions}),
        "reference_per_example_sha256": reference_eval["per_example_sha256"],
        "per_example_sha256": sha256(analysis / "key_cells_1024.jsonl"),
        "cells": rows,
    }
    (analysis / "key_cells_1024_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
