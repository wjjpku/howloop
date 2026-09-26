"""Run the locked KG curriculum on the 64x16 backbone for three other J classes."""

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path


SOURCE = Path(__file__).with_name('train_5k.py')
SOURCE_SHA256 = "5f38bb12f70fd615004965f8af37ad1c39ddf007d078094301fc0a6f178585e6"
BACKBONE_SHA256 = "c72a7703b5908ebb7bcdec6d1f95968ac0fcb8ef23fda8125cd50dcdb59f37fa"
EXPECTED_MODEL_CONFIG = {
    "d_model": 256, "n_heads": 8, "d_mlp": 1024, "physical_blocks": 2,
    "dropout": 0.0, "position_encoding": "none", "input_injection": False,
}
CONTROLLERS = ("lora_r48", "mlp_256", "attention")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller", choices=CONTROLLERS, required=True)
    parser.add_argument("--backbone", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if sha256(SOURCE) != SOURCE_SHA256:
        raise RuntimeError("Historical curriculum source hash mismatch")
    if sha256(args.backbone) != BACKBONE_SHA256:
        raise RuntimeError("64x16 backbone hash mismatch")

    spec = importlib.util.spec_from_file_location("locked_kg_curriculum", SOURCE)
    assert spec is not None and spec.loader is not None
    historical = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(historical)
    historical.BACKBONE = args.backbone
    historical.BACKBONE_SHA = BACKBONE_SHA256

    def load_variant_backbone(device):
        kg_orig = historical.KGLengthConfig(
            entity_count=64, relation_count=16, max_length=6
        )
        model_config = historical.ModelConfig(**EXPECTED_MODEL_CONFIG)
        checkpoint = historical.torch.load(
            args.backbone, map_location="cpu", weights_only=False
        )
        if checkpoint.get("schema") != "kg-fj-executor-checkpoint-v1":
            raise RuntimeError("Unexpected backbone checkpoint schema")
        if checkpoint.get("kg_config") != {
            "entity_count": 64, "relation_count": 16, "max_length": 6
        }:
            raise RuntimeError("Backbone KG dimensions differ from 64x16")
        if checkpoint.get("model_config") != EXPECTED_MODEL_CONFIG:
            raise RuntimeError("Backbone architecture differs")
        if checkpoint.get("world_seed") != historical.WORLD_SEED:
            raise RuntimeError("Backbone world seed differs")
        if checkpoint.get("train_config", {}).get("supervision_mode") != "aligned_intermediate":
            raise RuntimeError("Backbone supervision differs")
        model = historical.LoopedCompositionTransformer(kg_orig, model_config)
        model.load_state_dict(checkpoint["model_state"], strict=True)
        model = model.to(device).eval().requires_grad_(False)
        model.kg_config = historical.KGLengthConfig(
            entity_count=64, relation_count=16, max_length=32
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
        "backbone_sha256": BACKBONE_SHA256,
        "entities": 64, "relations": 16,
        "controller": args.controller, "seed": args.seed,
        "smoke": args.smoke,
        "only_changes": ["backbone path", "KG vocabulary dimensions", "controller class"],
    }, sort_keys=True), flush=True)
    sys.argv = [
        str(SOURCE), "--controller", args.controller,
        "--out-root", str(args.out_root),
        "--seed", str(args.seed), "--save-checkpoints",
    ]
    historical.main()


if __name__ == "__main__":
    main()
