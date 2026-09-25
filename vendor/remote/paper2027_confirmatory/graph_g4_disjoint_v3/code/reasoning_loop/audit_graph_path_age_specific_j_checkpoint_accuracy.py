"""Compare saved age-specific J checkpoints on fixed graph/trajectory accuracy."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from reasoning_loop.graph_path_depth_circuit import load_checkpoint
from reasoning_loop.graph_path_loop import pick_device, set_seed
from reasoning_loop.train_graph_path_age_specific_j_bank import (
    AgeSpecificJBank,
    evaluate_random_trajectories,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--phase-summary", type=Path, required=True)
    parser.add_argument("--bank-artifacts", type=Path, nargs="+", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=820001)
    parser.add_argument("--trajectories", type=int, default=56)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--fixed-start-age", type=int, choices=range(1, 9))
    parser.add_argument("--cuda-memory-fraction", type=float, default=0.045)
    parser.add_argument("--write-trajectory-details", action="store_true")
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(args.cuda_memory_fraction)
    set_seed(args.seed)
    model, cfg, _ = load_checkpoint(args.checkpoint, device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    phase_payload = json.loads(args.phase_summary.read_text(encoding="utf-8"))
    phase_positions = [
        int(value)
        for value in phase_payload["trajectory_positions_including_initial"]
    ]
    positions = tuple(range(cfg.seq_len))
    specs = (
        dict(split="single_J", max_total=1, max_run=1, minimum=1,
             required=1, mandatory=True, offset=1),
        dict(split="focused_mixture_a", max_total=5, max_run=5, minimum=None,
             required=0, mandatory=True, offset=2),
        dict(split="focused_mixture_b", max_total=5, max_run=5, minimum=None,
             required=0, mandatory=True, offset=3),
        dict(split="hard_boundary_T05_R5", max_total=5, max_run=5, minimum=5,
             required=5, mandatory=False, offset=4),
    )
    summary_rows: list[dict[str, object]] = []
    detail_rows: list[dict[str, object]] = []
    for artifact_index, artifact in enumerate(args.bank_artifacts):
        payload = torch.load(artifact, map_location="cpu", weights_only=False)
        architecture = payload.get("map_architecture", "diagonal_lora")
        bank = AgeSpecificJBank(
            dimension=cfg.d_model,
            rank=int(payload.get("rank", cfg.d_model)),
            stage_rank=int(payload.get("stage_rank", payload.get("rank", cfg.d_model))),
            map_architecture=architecture,
        ).to(device)
        bank.load_state_dict(payload["state_dict"])
        bank = bank.frozen()
        label = artifact.parent.name + "/" + artifact.stem
        for spec in specs:
            trajectories, summary = evaluate_random_trajectories(
                model=model,
                cfg=cfg,
                bank=bank,
                phase_positions=phase_positions,
                positions=positions,
                device=device,
                trajectories=args.trajectories,
                batch_size=args.batch_size,
                max_extra_backs=None,
                max_total_backs=spec["max_total"],
                max_consecutive_backs=spec["max_run"],
                minimum_total_backs=spec["minimum"],
                required_consecutive_backs=spec["required"],
                schedule_mandatory_j=spec["mandatory"],
                seed=args.seed + spec["offset"],
                split=spec["split"],
                conditions=("learned",),
                fixed_start_age=args.fixed_start_age,
            )
            if args.write_trajectory_details:
                for trajectory in trajectories:
                    trajectory.update(
                        artifact_index=artifact_index,
                        artifact_label=label,
                        artifact=str(artifact),
                        completed_stages=int(payload.get("completed_stages", -1)),
                        evaluation_start_age=args.fixed_start_age,
                    )
                    detail_rows.append(trajectory)
            row = dict(summary[0])
            row.update(
                artifact_index=artifact_index,
                artifact_label=label,
                artifact=str(artifact),
                completed_stages=int(payload.get("completed_stages", -1)),
                evaluation_start_age=args.fixed_start_age,
            )
            summary_rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
        del bank
    write_csv(args.out_dir / "checkpoint_accuracy.csv", summary_rows)
    if args.write_trajectory_details:
        write_csv(args.out_dir / "trajectory_accuracy.csv", detail_rows)
    (args.out_dir / "summary.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "metric": "accuracy only",
                "trajectories_per_split": args.trajectories,
                "examples_per_trajectory": args.batch_size,
                "evaluation_start_age": args.fixed_start_age,
                "evaluation_seed": args.seed,
                "trajectory_detail_rows": len(detail_rows),
                "rows": summary_rows,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
