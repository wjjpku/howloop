"""Retrain from m16: exactly half 4..16-uniform, half length-16 updates."""

import argparse
import importlib.util
import json
import time
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
from train_second_wsd import (
    BATCH,
    DECAY,
    PEAK_LR,
    STABLE,
    STEPS,
    WARMUP,
    learning_rate,
)


DATA_SEED = 19201621
LENGTH_SEED = 19201622
MODE_SEED = 19201623
UNIFORM_UPDATES = STEPS // 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller", choices=CONTROLLERS, required=True)
    parser.add_argument("--backbone", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if sha256(SOURCE) != SOURCE_SHA256 or sha256(args.backbone) != BACKBONE_SHA256:
        raise RuntimeError("Historical source or backbone hash mismatch")
    if sha256(args.parent) != PARENT_SHA256[args.controller]:
        raise RuntimeError("Original m16 parent hash mismatch")
    if args.out_dir.exists():
        raise FileExistsError(args.out_dir)

    spec = importlib.util.spec_from_file_location("locked_kg_mixed_wsd", SOURCE)
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
            or saved["backbone_sha256"] != BACKBONE_SHA256):
        raise RuntimeError("Parent is not the original matched m16 checkpoint")
    torch.manual_seed(20260924)
    ctrl = source.make_controller(args.controller).to(device)
    ctrl.load_state_dict(saved["controller"], strict=True)
    ctrl.train()
    optimizer = torch.optim.AdamW(ctrl.parameters(), lr=0.0,
                                  betas=(.9, .95), weight_decay=0)
    data_gen = torch.Generator().manual_seed(DATA_SEED)
    length_gen = torch.Generator().manual_seed(LENGTH_SEED)
    mode_gen = torch.Generator().manual_seed(MODE_SEED)
    # Exactly 5,000 of 10,000 optimizer updates use each arm, shuffled once.
    uniform_by_step = torch.randperm(STEPS, generator=mode_gen) < UNIFORM_UPDATES
    world_perms = world.permutations.to(device)
    updates = 2 if args.smoke else STEPS
    eval_every = 1 if args.smoke else 500
    length_histogram = {str(n): 0 for n in range(4, 17)}
    mode_histogram = {"uniform_4_to_16": 0, "fixed_16": 0}
    progress = []
    args.out_dir.mkdir(parents=True)
    began = time.monotonic()
    print(json.dumps({
        "controller": args.controller,
        "parent_m16_sha256": PARENT_SHA256[args.controller],
        "backbone_sha256": BACKBONE_SHA256,
        "steps": updates,
        "mix": "exactly 5000 uniform-4..16 updates and 5000 fixed-16 updates",
        "schedule": {"name": "wsd", "warmup": WARMUP, "stable": STABLE,
                     "decay": DECAY, "peak_lr": PEAK_LR, "final_lr": 0.0},
        "data_seed": DATA_SEED,
        "length_seed": LENGTH_SEED,
        "mode_seed": MODE_SEED,
    }, sort_keys=True), flush=True)

    for step in range(1, updates + 1):
        if bool(uniform_by_step[step - 1]):
            mode = "uniform_4_to_16"
            active = int(torch.randint(4, 17, (), generator=length_gen).item())
        else:
            mode = "fixed_16"
            active = 16
        mode_histogram[mode] += 1
        length_histogram[str(active)] += 1
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
                "mixed_training_step": step,
                "parent_m16_sha256": PARENT_SHA256[args.controller],
                "backbone_sha256": BACKBONE_SHA256,
                "world_seed": source.WORLD_SEED,
                "sampling": "half_uniform4to16_half_fixed16_per_update",
                "schedule": "wsd_500_8500_1000_peak_3e-5",
                "length_seed": LENGTH_SEED,
                "mode_seed": MODE_SEED,
                "data_seed": DATA_SEED,
            }
            torch.save(checkpoint,
                       args.out_dir / f"controller_m16_mixed_wsd_step{step}.pt")
    result = {
        "status": "complete",
        "controller": args.controller,
        "backbone_sha256": BACKBONE_SHA256,
        "parent_m16_sha256": PARENT_SHA256[args.controller],
        "source_sha256": SOURCE_SHA256,
        "steps": updates,
        "batch": BATCH,
        "optimizer": {"name": "AdamW", "betas": [.9, .95],
                      "weight_decay": 0, "state_resumed": False},
        "schedule": {"name": "wsd", "warmup": WARMUP, "stable": STABLE,
                     "decay": DECAY, "peak_lr": PEAK_LR, "final_lr": 0.0},
        "sampling": "half_uniform4to16_half_fixed16_per_update",
        "ood_target_length": 20,
        "data_seed": DATA_SEED,
        "length_seed": LENGTH_SEED,
        "mode_seed": MODE_SEED,
        "mode_histogram": mode_histogram,
        "length_histogram": length_histogram,
        "progress": progress,
        "elapsed_seconds": round(time.monotonic() - began, 1),
        "smoke": args.smoke,
    }
    (args.out_dir / "train_result.json").write_text(
        json.dumps(result, indent=2) + "\n")
    print(f"COMPLETE {args.controller} mixed WSD 10k from m16", flush=True)


if __name__ == "__main__":
    main()
