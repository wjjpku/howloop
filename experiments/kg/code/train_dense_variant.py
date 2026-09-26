"""Run the unchanged KG dense-J curriculum on a newly trained input variant.

The historical curriculum source hard-codes the original 128x16 backbone.
This wrapper verifies its source hash, substitutes only the new backbone and
its KG vocabulary dimensions, then calls the original training main().
"""

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path


SOURCE = Path(__file__).with_name('train_5k.py')
SOURCE_SHA256 = "5f38bb12f70fd615004965f8af37ad1c39ddf007d078094301fc0a6f178585e6"
EXPECTED_MODEL_CONFIG = {
    "d_model": 256,
    "n_heads": 8,
    "d_mlp": 1024,
    "physical_blocks": 2,
    "dropout": 0.0,
    "position_encoding": "none",
    "input_injection": False,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", type=Path, required=True)
    parser.add_argument("--backbone-sha256", required=True)
    parser.add_argument("--entities", type=int, required=True)
    parser.add_argument("--relations", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true", help="Tiny integration check, not an experiment")
    args = parser.parse_args()
    if (args.entities, args.relations) not in {(64, 16), (128, 8)}:
        raise ValueError("Only the two registered input variants are allowed")
    if sha256(SOURCE) != SOURCE_SHA256:
        raise RuntimeError("Historical dense-J training source changed")
    if sha256(args.backbone) != args.backbone_sha256:
        raise RuntimeError("Backbone checkpoint hash mismatch")

    spec = importlib.util.spec_from_file_location("locked_kg_curriculum", SOURCE)
    assert spec is not None and spec.loader is not None
    historical = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(historical)
    historical.BACKBONE = args.backbone
    historical.BACKBONE_SHA = args.backbone_sha256

    def load_variant_backbone(device):
        kg_orig = historical.KGLengthConfig(
            entity_count=args.entities, relation_count=args.relations, max_length=6
        )
        model_config = historical.ModelConfig(**EXPECTED_MODEL_CONFIG)
        checkpoint = historical.torch.load(
            args.backbone, map_location="cpu", weights_only=False
        )
        if checkpoint.get("schema") != "kg-fj-executor-checkpoint-v1":
            raise RuntimeError("Unexpected backbone checkpoint schema")
        if checkpoint.get("kg_config") != {
            "entity_count": args.entities,
            "relation_count": args.relations,
            "max_length": 6,
        }:
            raise RuntimeError("Checkpoint KG dimensions do not match request")
        if checkpoint.get("model_config") != EXPECTED_MODEL_CONFIG:
            raise RuntimeError("Checkpoint architecture is not the registered NoPE executor")
        if checkpoint.get("world_seed") != historical.WORLD_SEED:
            raise RuntimeError("Checkpoint world seed differs from the original")
        if checkpoint.get("train_config", {}).get("supervision_mode") != "aligned_intermediate":
            raise RuntimeError("Backbone supervision differs from the original")
        model = historical.LoopedCompositionTransformer(kg_orig, model_config)
        model.load_state_dict(checkpoint["model_state"], strict=True)
        model = model.to(device).eval().requires_grad_(False)
        model.kg_config = historical.KGLengthConfig(
            entity_count=args.entities, relation_count=args.relations, max_length=32
        )
        return model, model.kg_config

    historical.load_backbone = load_variant_backbone
    if args.smoke:
        historical.STAGES = [4]
        historical.MAX_STEPS = 2
        historical.EVAL_EVERY = 1
        historical.BATCH = 16
        historical.EVAL_N = 16
        historical.EARLY_STOP = 2.0
    print(json.dumps({
        "original_source_sha256": SOURCE_SHA256,
        "backbone_sha256": args.backbone_sha256,
        "entities": args.entities,
        "relations": args.relations,
        "seed": args.seed,
        "controller": "dense",
        "smoke": args.smoke,
        "only_change": "backbone path and KG vocabulary dimensions",
    }, sort_keys=True), flush=True)
    sys.argv = [
        str(SOURCE), "--controller", "dense", "--out-root", str(args.out_root),
        "--seed", str(args.seed), "--save-checkpoints",
    ]
    historical.main()


if __name__ == "__main__":
    main()
