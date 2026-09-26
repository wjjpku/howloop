"""Locked 64x16 dense-J source with unit-length stages and 100% early stop."""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

from train_dense_variant import EXPECTED_MODEL_CONFIG, SOURCE, SOURCE_SHA256, sha256


BACKBONE_SHA256 = "c72a7703b5908ebb7bcdec6d1f95968ac0fcb8ef23fda8125cd50dcdb59f37fa"
STAGES = list(range(4, 17))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--smoke", action="store_true",
                        help="Two stages × two updates: integration only, no performance claim")
    args = parser.parse_args()
    if sha256(SOURCE) != SOURCE_SHA256:
        raise RuntimeError("Historical controller source hash mismatch")
    if sha256(args.backbone) != BACKBONE_SHA256:
        raise RuntimeError("64x16 backbone SHA mismatch")

    spec = importlib.util.spec_from_file_location("locked_kg_unit_curriculum", SOURCE)
    assert spec is not None and spec.loader is not None
    historical = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(historical)
    historical.BACKBONE = args.backbone
    historical.BACKBONE_SHA = BACKBONE_SHA256

    def load_64x16_backbone(device):
        kg_original = historical.KGLengthConfig(
            entity_count=64, relation_count=16, max_length=6
        )
        model_config = historical.ModelConfig(**EXPECTED_MODEL_CONFIG)
        checkpoint = historical.torch.load(
            args.backbone, map_location="cpu", weights_only=False
        )
        if checkpoint.get("schema") != "kg-fj-executor-checkpoint-v1":
            raise RuntimeError("Unexpected backbone schema")
        if checkpoint.get("kg_config") != {
            "entity_count": 64, "relation_count": 16, "max_length": 6
        }:
            raise RuntimeError("Backbone KG dimensions mismatch")
        if checkpoint.get("model_config") != EXPECTED_MODEL_CONFIG:
            raise RuntimeError("Backbone model configuration mismatch")
        if checkpoint.get("world_seed") != historical.WORLD_SEED:
            raise RuntimeError("Backbone world seed mismatch")
        if checkpoint.get("train_config", {}).get("supervision_mode") != "aligned_intermediate":
            raise RuntimeError("Backbone supervision mismatch")
        model = historical.LoopedCompositionTransformer(kg_original, model_config)
        model.load_state_dict(checkpoint["model_state"], strict=True)
        model = model.to(device).eval().requires_grad_(False)
        model.kg_config = historical.KGLengthConfig(
            entity_count=64, relation_count=16, max_length=32
        )
        return model, model.kg_config

    historical.load_backbone = load_64x16_backbone
    historical.STAGES = STAGES
    historical.MAX_STEPS = 3000
    historical.EARLY_STOP = 1.0
    if args.smoke:
        historical.STAGES = [4, 5]
        historical.MAX_STEPS = 2
        historical.EVAL_EVERY = 1
        historical.BATCH = 16
        historical.EVAL_N = 16
    print(json.dumps({
        "original_source_sha256": SOURCE_SHA256,
        "backbone_sha256": BACKBONE_SHA256,
        "entities": 64, "relations": 16,
        "controller": "dense", "controller_seed": args.seed,
        "stages": historical.STAGES, "steps_per_stage": historical.MAX_STEPS,
        "early_stop_threshold": historical.EARLY_STOP,
        "smoke": args.smoke,
    }, sort_keys=True), flush=True)
    sys.argv = [
        str(SOURCE), "--controller", "dense",
        "--out-root", str(args.out_root),
        "--seed", str(args.seed), "--save-checkpoints",
    ]
    historical.main()


if __name__ == "__main__":
    main()
