"""Second 10k KG-controller continuation: uniform lengths 4..16, WSD LR."""

import argparse
import importlib.util
import json
import time
from pathlib import Path

import torch

from train_m20_uniform import (
    BACKBONE_SHA256,
    CONTROLLERS,
    SOURCE,
    SOURCE_SHA256,
    load_backbone,
    sha256,
)


STEPS = 10000
WARMUP = 500
STABLE = 8500
DECAY = 1000
PEAK_LR = 3e-5
BATCH = 128
MIN_LENGTH = 4
MAX_LENGTH = 16
DATA_SEED = 19201611
LENGTH_SEED = 19201612


def learning_rate(step):
    if not 1 <= step <= STEPS:
        raise ValueError(step)
    if step <= WARMUP:
        return PEAK_LR * step / WARMUP
    if step <= WARMUP + STABLE:
        return PEAK_LR
    return PEAK_LR * (STEPS - step) / DECAY


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller", choices=CONTROLLERS, required=True)
    parser.add_argument("--backbone", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--parent-sha256", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if sha256(SOURCE) != SOURCE_SHA256 or sha256(args.backbone) != BACKBONE_SHA256:
        raise RuntimeError("Historical source or backbone hash mismatch")
    if sha256(args.parent) != args.parent_sha256:
        raise RuntimeError("First 10k endpoint checkpoint hash mismatch")
    if args.out_dir.exists():
        raise FileExistsError(args.out_dir)

    spec = importlib.util.spec_from_file_location("locked_kg_second_wsd", SOURCE)
    assert spec is not None and spec.loader is not None
    source = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(source)
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(.12, 0)
    device = torch.device("cuda")
    model, kg = load_backbone(source, args.backbone, device)
    world = source.PermutationWorld.create(kg, source.WORLD_SEED)
    saved = torch.load(args.parent, map_location="cpu", weights_only=False)
    if (saved["controller_type"] != args.controller or saved["stage"] != 16
            or saved["training_step_in_stage"] != 10000
            or saved["sampling"] != "uniform_length_4_to_16_per_update"
            or saved["backbone_sha256"] != BACKBONE_SHA256):
        raise RuntimeError("Parent is not the completed first 10k endpoint")
    torch.manual_seed(20260923)
    ctrl = source.make_controller(args.controller).to(device)
    ctrl.load_state_dict(saved["controller"], strict=True)
    ctrl.train()
    optimizer = torch.optim.AdamW(ctrl.parameters(), lr=0.0,
                                  betas=(.9, .95), weight_decay=0)
    data_gen = torch.Generator().manual_seed(DATA_SEED)
    length_gen = torch.Generator().manual_seed(LENGTH_SEED)
    world_perms = world.permutations.to(device)
    updates = 2 if args.smoke else STEPS
    eval_every = 1 if args.smoke else 500
    histogram = {str(n): 0 for n in range(MIN_LENGTH, MAX_LENGTH + 1)}
    progress = []
    args.out_dir.mkdir(parents=True)
    began = time.monotonic()
    print(json.dumps({
        "controller": args.controller,
        "parent_sha256": args.parent_sha256,
        "backbone_sha256": BACKBONE_SHA256,
        "steps": updates,
        "schedule": {"name": "wsd", "warmup": WARMUP, "stable": STABLE,
                     "decay": DECAY, "peak_lr": PEAK_LR, "final_lr": 0.0},
        "sampling": "uniform_length_4_to_16_per_update",
        "data_seed": DATA_SEED,
        "length_seed": LENGTH_SEED,
    }, sort_keys=True), flush=True)

    for step in range(1, updates + 1):
        active = int(torch.randint(MIN_LENGTH, MAX_LENGTH + 1,
                                   (), generator=length_gen).item())
        histogram[str(active)] += 1
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(step)
        batch = source.sample_batch(kg, world, BATCH, active, data_gen,
                                    frozenset()).to(device)
        loss = source.truncated_unroll_loss(model, ctrl, batch, world_perms,
                                            active)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ctrl.parameters(), 1.0)
        optimizer.step()
        if step % eval_every == 0:
            ctrl.eval()
            panel = {str(n): round(source.evaluate(model, ctrl, world, kg,
                                                  n, device, 128), 4)
                     for n in (16, 18, 20, 21, 22, 24)}
            ctrl.train()
            item = {"step": step, "lr": learning_rate(step),
                    "loss": round(float(loss.item()), 6),
                    "accuracy": panel,
                    "seconds": round(time.monotonic() - began, 1)}
            progress.append(item)
            print(json.dumps(item), flush=True)
        if not args.smoke and step in (5000, 10000):
            checkpoint = {
                "controller": ctrl.state_dict(),
                "controller_type": args.controller,
                "stage": 16,
                "second_training_step": step,
                "parent_sha256": args.parent_sha256,
                "backbone_sha256": BACKBONE_SHA256,
                "world_seed": source.WORLD_SEED,
                "sampling": "uniform_length_4_to_16_per_update",
                "schedule": "wsd_500_8500_1000_peak_3e-5",
                "length_seed": LENGTH_SEED,
                "data_seed": DATA_SEED,
            }
            torch.save(checkpoint,
                       args.out_dir / f"controller_m16_plus20k_step{step}.pt")
    result = {
        "status": "complete",
        "controller": args.controller,
        "backbone_sha256": BACKBONE_SHA256,
        "parent_sha256": args.parent_sha256,
        "source_sha256": SOURCE_SHA256,
        "steps": updates,
        "batch": BATCH,
        "optimizer": {"name": "AdamW", "betas": [.9, .95],
                      "weight_decay": 0, "state_resumed": False},
        "schedule": {"name": "wsd", "warmup": WARMUP, "stable": STABLE,
                     "decay": DECAY, "peak_lr": PEAK_LR, "final_lr": 0.0},
        "sample_lengths": [MIN_LENGTH, MAX_LENGTH],
        "sampling": "uniform_length_4_to_16_per_update",
        "ood_target_length": 20,
        "length_seed": LENGTH_SEED,
        "data_seed": DATA_SEED,
        "length_histogram": histogram,
        "progress": progress,
        "elapsed_seconds": round(time.monotonic() - began, 1),
        "smoke": args.smoke,
    }
    (args.out_dir / "train_result.json").write_text(
        json.dumps(result, indent=2) + "\n")
    print(f"COMPLETE {args.controller} second 10k WSD", flush=True)


if __name__ == "__main__":
    main()
