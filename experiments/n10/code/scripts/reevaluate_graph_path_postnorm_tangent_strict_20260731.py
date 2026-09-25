from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from reasoning_loop.graph_path_depth_circuit import load_checkpoint
from reasoning_loop.graph_path_loop import pick_device
from reasoning_loop.graph_path_postnorm_tangent_rejuvenator import (
    _make_rejuvenator,
    evaluate_repeated_rejuvenation,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Strict collision-controlled reevaluation of a saved rejuvenator."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--batches", type=int, default=4)
    parser.add_argument("--cycles", type=int, default=200)
    parser.add_argument("--seed", type=int, default=9732)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = pick_device(args.device)
    model, cfg, _ = load_checkpoint(args.checkpoint, device)
    payload = torch.load(args.artifact, map_location=device, weights_only=False)
    summary = payload["summary"]
    rejuvenator = _make_rejuvenator(
        model,
        transform=summary["transform"],
    ).to(device)
    rejuvenator.load_state_dict(payload["state_dict"])
    rejuvenator.eval()
    rows = evaluate_repeated_rejuvenation(
        model=model,
        cfg=cfg,
        rejuvenator=rejuvenator,
        position_mode=summary["position_mode"],
        pair_mode=summary["pair_mode"],
        batch_size=args.batch_size,
        batches=args.batches,
        cycles=args.cycles,
        device=device,
        seed=args.seed,
        conditions=("learned",),
    )
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(
        json.dumps(
            {
                "artifact": str(args.artifact),
                "out_csv": str(args.out_csv),
                "rows": len(rows),
                "strict_cycle_1": rows[0]["strict_novel_target_accuracy"],
                "strict_cycle_final": rows[-1][
                    "strict_novel_target_accuracy"
                ],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
