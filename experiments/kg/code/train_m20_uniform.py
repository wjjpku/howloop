"""Continue one verified m16 KG controller for 10k uniform 4–16 updates."""

import argparse
import importlib.util
import json
import time
from pathlib import Path

import torch

from train_other_controllers_variant import (
    BACKBONE_SHA256,
    EXPECTED_MODEL_CONFIG,
    SOURCE,
    SOURCE_SHA256,
    sha256,
)


CONTROLLERS = ("dense", "lora_r48", "mlp_256", "attention")
PARENT_SHA256 = {
    "dense": "23ae8dc795ec16d9b20fbcc45bc53cfeeed361daa1744070a2a2a09ff024dcc0",
    "lora_r48": "421291ab5c8ee3984229129259a349e08f9fbc30a15502c6a028c954ac972314",
    "mlp_256": "eb1b538ee2c235613fafc7563eeea3ff67d033d5f0978dc4040e1241083f8d86",
    "attention": "dcd7a7d307865f3ead3a037d805ef4d006ffaf757a23412f3180ecc6a8b131f6",
}
STEPS = 10000
MIN_LENGTH = 4
MAX_LENGTH = 16
EVAL_TARGET_LENGTH = 20
BATCH = 128
LR = 1e-4
DATA_SEED = 19201601
LENGTH_SEED = 19201602


def load_backbone(source, checkpoint_path, device):
    kg_original = source.KGLengthConfig(entity_count=64, relation_count=16, max_length=6)
    model_config = source.ModelConfig(**EXPECTED_MODEL_CONFIG)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("schema") != "kg-fj-executor-checkpoint-v1":
        raise RuntimeError("Unexpected backbone schema")
    if checkpoint.get("kg_config") != vars(kg_original):
        raise RuntimeError("KG dimensions mismatch")
    if checkpoint.get("model_config") != EXPECTED_MODEL_CONFIG:
        raise RuntimeError("Backbone architecture mismatch")
    if checkpoint.get("world_seed") != source.WORLD_SEED:
        raise RuntimeError("World seed mismatch")
    if checkpoint.get("train_config", {}).get("supervision_mode") != "aligned_intermediate":
        raise RuntimeError("Backbone supervision mismatch")
    model = source.LoopedCompositionTransformer(kg_original, model_config)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model = model.to(device).eval().requires_grad_(False)
    kg = source.KGLengthConfig(entity_count=64, relation_count=16, max_length=32)
    model.kg_config = kg
    return model, kg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller", choices=CONTROLLERS, required=True)
    parser.add_argument("--backbone", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if sha256(SOURCE) != SOURCE_SHA256:
        raise RuntimeError("Historical trainer hash mismatch")
    if sha256(args.backbone) != BACKBONE_SHA256:
        raise RuntimeError("Backbone hash mismatch")
    if sha256(args.parent) != PARENT_SHA256[args.controller]:
        raise RuntimeError("Parent m16 controller hash mismatch")
    if args.out_dir.exists():
        raise FileExistsError(args.out_dir)

    spec = importlib.util.spec_from_file_location("locked_kg_m20_uniform", SOURCE)
    assert spec is not None and spec.loader is not None
    source = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(source)
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(.12, 0)
    device = torch.device("cuda")
    model, kg = load_backbone(source, args.backbone, device)
    world = source.PermutationWorld.create(kg, source.WORLD_SEED)
    saved = torch.load(args.parent, map_location="cpu", weights_only=False)
    if saved["controller_type"] != args.controller or saved["stage"] != 16:
        raise RuntimeError("Parent checkpoint type or stage mismatch")
    if saved["backbone_sha256"] != BACKBONE_SHA256:
        raise RuntimeError("Parent backbone mismatch")
    torch.manual_seed(20260922)
    ctrl = source.make_controller(args.controller).to(device)
    ctrl.load_state_dict(saved["controller"], strict=True)
    ctrl.train()
    optimizer = torch.optim.AdamW(ctrl.parameters(), lr=LR, betas=(.9, .95), weight_decay=0)
    data_gen = torch.Generator().manual_seed(DATA_SEED)
    length_gen = torch.Generator().manual_seed(LENGTH_SEED)
    world_perms = world.permutations.to(device)
    total_steps = 2 if args.smoke else STEPS
    eval_every = 1 if args.smoke else 500
    histogram = {str(n): 0 for n in range(MIN_LENGTH, MAX_LENGTH + 1)}
    progress = []
    args.out_dir.mkdir(parents=True)
    began = time.monotonic()
    print(json.dumps({
        "controller": args.controller,
        "backbone_sha256": BACKBONE_SHA256,
        "parent_sha256": PARENT_SHA256[args.controller],
        "source_sha256": SOURCE_SHA256,
        "steps": total_steps,
        "sample_lengths": [MIN_LENGTH, MAX_LENGTH],
        "ood_target_length": EVAL_TARGET_LENGTH,
        "sampling": "one i.i.d. uniform length per optimizer update",
        "data_seed": DATA_SEED, "length_seed": LENGTH_SEED,
        "smoke": args.smoke,
    }, sort_keys=True), flush=True)

    for step in range(1, total_steps + 1):
        active = int(torch.randint(MIN_LENGTH, MAX_LENGTH + 1, (), generator=length_gen).item())
        histogram[str(active)] += 1
        batch = source.sample_batch(kg, world, BATCH, active, data_gen, frozenset()).to(device)
        loss = source.truncated_unroll_loss(model, ctrl, batch, world_perms, active)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ctrl.parameters(), 1.0)
        optimizer.step()
        if step % eval_every == 0:
            ctrl.eval()
            panel = {str(n): round(source.evaluate(model, ctrl, world, kg, n, device, 128), 4)
                     for n in (16, 18, 20, 21, 22, 24)}
            ctrl.train()
            item = {"step": step, "loss": round(float(loss.item()), 6),
                    "accuracy": panel, "seconds": round(time.monotonic() - began, 1)}
            progress.append(item)
            print(json.dumps(item), flush=True)
        if not args.smoke and step in (5000, 10000):
            checkpoint = {
                "controller": ctrl.state_dict(), "controller_type": args.controller,
                "stage": 16, "parent_stage": 16, "training_step_in_stage": step,
                "ood_target_length": EVAL_TARGET_LENGTH,
                "parent_sha256": PARENT_SHA256[args.controller],
                "backbone_sha256": BACKBONE_SHA256,
                "world_seed": source.WORLD_SEED,
                "sampling": "uniform_length_4_to_16_per_update",
                "length_seed": LENGTH_SEED, "data_seed": DATA_SEED,
            }
            torch.save(checkpoint, args.out_dir / f"controller_m16_plus10k_step{step}.pt")
    result = {
        "status": "complete", "controller": args.controller,
        "backbone_sha256": BACKBONE_SHA256,
        "parent_sha256": PARENT_SHA256[args.controller],
        "source_sha256": SOURCE_SHA256,
        "steps": total_steps, "batch": BATCH, "learning_rate": LR,
        "sample_lengths": [MIN_LENGTH, MAX_LENGTH],
        "sampling": "uniform_length_4_to_16_per_update",
        "ood_target_length": EVAL_TARGET_LENGTH,
        "length_seed": LENGTH_SEED, "data_seed": DATA_SEED,
        "length_histogram": histogram,
        "progress": progress,
        "elapsed_seconds": round(time.monotonic() - began, 1),
        "smoke": args.smoke,
    }
    (args.out_dir / "train_result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"COMPLETE {args.controller} m16 uniform continuation {total_steps}", flush=True)


if __name__ == "__main__":
    main()
