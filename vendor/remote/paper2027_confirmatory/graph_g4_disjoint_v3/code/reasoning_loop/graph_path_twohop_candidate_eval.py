from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from reasoning_loop.graph_path_depth_circuit import load_checkpoint
from reasoning_loop.graph_path_loop import pick_device
from reasoning_loop.graph_path_twohop_reprogram_j import (
    DenseAffineJ,
    _permutation_digest,
    build_all_start_age_dataset,
    evaluate_closure,
    evaluate_strict_conditions,
    strict_unseen_permutations,
    validate_d8l8_onehop_config,
    write_summary_atomic,
)


def select_candidate(
    candidates: Mapping[str, Mapping[str, Any]],
    *,
    preloop_weight: float,
) -> tuple[str, Mapping[str, Any]]:
    eligible = [
        (name, payload)
        for name, payload in candidates.items()
        if bool(payload["spec"]["execute"])
        and float(payload["spec"]["preloop_weight"]) == preloop_weight
    ]
    if not eligible:
        raise ValueError(f"no execution candidate has preloop weight {preloop_weight}")
    return max(
        eligible,
        key=lambda item: float(
            item[1]["validation"]["distinct_post_two_accuracy"]
        ),
    )


def run_evaluation(
    *,
    checkpoint: Path,
    controllers_path: Path,
    out_path: Path,
    device_name: str,
    eval_permutations: int,
    eval_seed: int,
    batch_size: int,
    preloop_weight: float,
    closure_cycles: int,
) -> dict[str, Any]:
    device = pick_device(device_name)
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.30, device=torch.cuda.current_device())
        torch.cuda.reset_peak_memory_stats(device)
    model, cfg, checkpoint_payload = load_checkpoint(checkpoint, device)
    validate_d8l8_onehop_config(cfg)
    model.requires_grad_(False)
    training_summary = json.loads(
        (controllers_path.parent / "summary.json").read_text(encoding="utf-8")
    )
    payload = torch.load(controllers_path, map_location=device, weights_only=False)
    name, selected = select_candidate(
        payload["candidates"], preloop_weight=preloop_weight
    )
    controller = DenseAffineJ(cfg.d_model).to(device)
    controller.load_state_dict(
        {key: value.to(device) for key, value in selected["state_dict"].items()}
    )
    controller.eval()
    formal_eval = strict_unseen_permutations(
        cfg.node_count, set(), count=eval_permutations, seed=eval_seed
    )
    formal_hash = _permutation_digest(formal_eval)
    if formal_hash != training_summary["strict_eval_permutation_sha256"]:
        raise ValueError("formal evaluation permutation hash mismatch")
    dataset = build_all_start_age_dataset(
        model=model,
        cfg=cfg,
        permutations=formal_eval,
        device=device,
        collection_batch_size=batch_size,
    )
    per_age_rows, condition_summary = evaluate_strict_conditions(
        model=model,
        exec_controller=controller,
        direct_controller=controller,
        dataset=dataset,
        batch_size=batch_size,
        shuffle_seed=eval_seed + int(training_summary["controller_seed"]),
    )
    closure_rows = evaluate_closure(
        model=model,
        cfg=cfg,
        controller=controller,
        permutations=formal_eval,
        device=device,
        batch_size=batch_size,
        cycles=closure_cycles,
    )
    result = {
        "status": "complete",
        "checkpoint_step": int(checkpoint_payload.get("step", -1)),
        "checkpoint_sha256": training_summary["checkpoint_sha256"],
        "controller_seed": int(training_summary["controller_seed"]),
        "candidate": name,
        "spec": dict(selected["spec"]),
        "validation": dict(selected["validation"]),
        "strict_eval_permutations": eval_permutations,
        "strict_eval_permutation_sha256": formal_hash,
        "per_age_rows": per_age_rows,
        "condition_summary": condition_summary,
        "closure_rows": closure_rows,
        "peak_cuda_memory_gib": (
            float(torch.cuda.max_memory_allocated(device) / 2**30)
            if device.type == "cuda"
            else 0.0
        ),
    }
    write_summary_atomic(out_path, result)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--controllers", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--eval-permutations", type=int, default=512)
    parser.add_argument("--eval-seed", type=int, default=8_609_003)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--preloop-weight", type=float, default=0.0)
    parser.add_argument("--closure-cycles", type=int, default=4)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result = run_evaluation(
        checkpoint=args.checkpoint,
        controllers_path=args.controllers,
        out_path=args.out,
        device_name=args.device,
        eval_permutations=args.eval_permutations,
        eval_seed=args.eval_seed,
        batch_size=args.batch_size,
        preloop_weight=args.preloop_weight,
        closure_cycles=args.closure_cycles,
    )
    selected = next(
        row
        for row in result["condition_summary"]
        if row["condition"] == "exec_F_after_J"
    )
    print(json.dumps({"candidate": result["candidate"], **selected}, indent=2))


if __name__ == "__main__":
    main()
