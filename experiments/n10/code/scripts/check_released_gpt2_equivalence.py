#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from reasoning_loop.paper_length_telomere import (
    PAPER_TASKS,
    PaperLoopedTransformer,
    PaperModelConfig,
    generate_paper_batch,
    selected_state,
)


def load_released_model(official_src: Path) -> type[torch.nn.Module]:
    # The repository vendors an old Transformers snapshot.  These shims only
    # satisfy import-time checks removed from modern dependencies; they do not
    # alter GPT-2 model code or its forward pass.
    import huggingface_hub

    original_version = importlib.metadata.version

    def compatible_version(name: str) -> str:
        if name == "sacremoses":
            return "0.0.53"
        return original_version(name)

    importlib.metadata.version = compatible_version
    if not hasattr(huggingface_hub, "HfFolder"):
        huggingface_hub.HfFolder = type("HfFolder", (), {})
    if not hasattr(huggingface_hub, "Repository"):
        huggingface_hub.Repository = type("Repository", (), {})
    sys.path.insert(0, str(official_src))
    from models import GeneralTransformerModel

    return GeneralTransformerModel


def copy_released_weights(
    released: torch.nn.Module, candidate: PaperLoopedTransformer
) -> list[tuple[str, torch.nn.Parameter, torch.nn.Parameter, bool]]:
    pairs: list[tuple[str, torch.nn.Parameter, torch.nn.Parameter, bool]] = []

    def copy_pair(
        name: str,
        target: torch.nn.Parameter,
        source: torch.nn.Parameter,
        transpose: bool = False,
    ) -> None:
        value = source.detach().T if transpose else source.detach()
        with torch.no_grad():
            target.copy_(value)
        pairs.append((name, target, source, transpose))

    copy_pair("read_in.weight", candidate.read_in.weight, released._read_in.weight)
    copy_pair("read_in.bias", candidate.read_in.bias, released._read_in.bias)
    for index, (source, target) in enumerate(
        zip(released._backbone.h, candidate.layers, strict=True)
    ):
        prefix = f"layers.{index}"
        copy_pair(
            f"{prefix}.attention_norm.weight",
            target.attention_norm.weight,
            source.ln_1.weight,
        )
        copy_pair(
            f"{prefix}.attention_norm.bias",
            target.attention_norm.bias,
            source.ln_1.bias,
        )
        copy_pair(
            f"{prefix}.attention.qkv.weight",
            target.attention.qkv.weight,
            source.attn.c_attn.weight,
            transpose=True,
        )
        copy_pair(
            f"{prefix}.attention.qkv.bias",
            target.attention.qkv.bias,
            source.attn.c_attn.bias,
        )
        copy_pair(
            f"{prefix}.attention.output.weight",
            target.attention.output.weight,
            source.attn.c_proj.weight,
            transpose=True,
        )
        copy_pair(
            f"{prefix}.attention.output.bias",
            target.attention.output.bias,
            source.attn.c_proj.bias,
        )
        copy_pair(
            f"{prefix}.mlp_norm.weight",
            target.mlp_norm.weight,
            source.ln_2.weight,
        )
        copy_pair(
            f"{prefix}.mlp_norm.bias",
            target.mlp_norm.bias,
            source.ln_2.bias,
        )
        copy_pair(
            f"{prefix}.mlp.0.weight",
            target.mlp[0].weight,
            source.mlp.c_fc.weight,
            transpose=True,
        )
        copy_pair(
            f"{prefix}.mlp.0.bias",
            target.mlp[0].bias,
            source.mlp.c_fc.bias,
        )
        copy_pair(
            f"{prefix}.mlp.2.weight",
            target.mlp[2].weight,
            source.mlp.c_proj.weight,
            transpose=True,
        )
        copy_pair(
            f"{prefix}.mlp.2.bias",
            target.mlp[2].bias,
            source.mlp.c_proj.bias,
        )
    copy_pair(
        "final_norm.weight",
        candidate.final_norm.weight,
        released._backbone.ln_f.weight,
    )
    copy_pair(
        "final_norm.bias",
        candidate.final_norm.bias,
        released._backbone.ln_f.bias,
    )
    copy_pair("read_out.weight", candidate.read_out.weight, released._read_out.weight)
    copy_pair("read_out.bias", candidate.read_out.bias, released._read_out.bias)
    return pairs


def maximum_gradient_differences(
    pairs: list[tuple[str, torch.nn.Parameter, torch.nn.Parameter, bool]]
) -> dict[str, float]:
    result: dict[str, float] = {}
    for name, candidate, released, transpose in pairs:
        if candidate.grad is None or released.grad is None:
            raise RuntimeError(f"missing gradient for mapped parameter {name}")
        released_gradient = released.grad.T if transpose else released.grad
        result[name] = float(
            (candidate.grad - released_gradient).abs().max().detach()
        )
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    torch.manual_seed(args.seed)
    GeneralTransformerModel = load_released_model(args.official_src)
    released = GeneralTransformerModel(
        6,
        64,
        n_embd=args.d_model,
        n_layer=args.layers,
        n_head=args.heads,
        linear_embedding=True,
    ).train()
    candidate = PaperLoopedTransformer(
        PaperModelConfig(
            d_model=args.d_model,
            n_heads=args.heads,
            d_mlp=4 * args.d_model,
            block_layers=args.layers,
        )
    ).train()
    mapped = copy_released_weights(released, candidate)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed + 1)
    batch = generate_paper_batch(
        PAPER_TASKS["parity"],
        batch_size=args.batch_size,
        min_length=1,
        max_length=args.maximum_length,
        generator=generator,
    )
    horizon = int(batch.target_steps.max())
    released_trajectory = released.looped_forward(batch.inputs, horizon=horizon)
    candidate_states = candidate.states(batch.inputs, steps=horizon)
    candidate_trajectory = [candidate.decode(state) for state in candidate_states]
    per_step_max_abs = [
        float((left - right).abs().max().detach())
        for left, right in zip(
            released_trajectory, candidate_trajectory, strict=True
        )
    ]
    released_selected = torch.stack(
        [
            released_trajectory[int(length) - 1][row]
            for row, length in enumerate(batch.target_steps)
        ]
    )
    candidate_selected = candidate.decode(
        selected_state(candidate_states, batch.target_steps)
    )
    released_loss = F.cross_entropy(
        released_selected[batch.answer_mask], batch.targets[batch.answer_mask]
    )
    candidate_loss = F.cross_entropy(
        candidate_selected[batch.answer_mask], batch.targets[batch.answer_mask]
    )
    released_loss.backward()
    candidate_loss.backward()
    gradient_differences = maximum_gradient_differences(mapped)
    commit = subprocess.check_output(
        ["git", "-C", str(args.official_src.parent), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    summary = {
        "status": "complete",
        "official_commit": commit,
        "configuration": {
            "d_model": args.d_model,
            "heads": args.heads,
            "layers": args.layers,
            "batch_size": args.batch_size,
            "maximum_length": args.maximum_length,
            "horizon": horizon,
        },
        "per_step_max_abs_logit_difference": per_step_max_abs,
        "maximum_logit_difference": max(per_step_max_abs),
        "selected_logit_difference": float(
            (released_selected - candidate_selected).abs().max().detach()
        ),
        "released_loss": float(released_loss.detach()),
        "candidate_loss": float(candidate_loss.detach()),
        "loss_difference": abs(
            float(released_loss.detach()) - float(candidate_loss.detach())
        ),
        "maximum_gradient_difference": max(gradient_differences.values()),
        "gradient_differences": gradient_differences,
        "thresholds": {
            "maximum_logit_difference": args.logit_tolerance,
            "loss_difference": args.loss_tolerance,
            "maximum_gradient_difference": args.gradient_tolerance,
        },
    }
    summary["passed"] = bool(
        summary["maximum_logit_difference"] <= args.logit_tolerance
        and summary["loss_difference"] <= args.loss_tolerance
        and summary["maximum_gradient_difference"] <= args.gradient_tolerance
    )
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.out.with_suffix(args.out.suffix + ".tmp")
        temporary.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, args.out)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-src", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--seed", type=int, default=270001)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--maximum-length", type=int, default=7)
    parser.add_argument("--logit-tolerance", type=float, default=2e-5)
    parser.add_argument("--loss-tolerance", type=float, default=2e-6)
    parser.add_argument("--gradient-tolerance", type=float, default=5e-5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run(args)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
