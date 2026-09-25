from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the P3 graph-path transition-consistency experiment across seeds: "
            "transition training, final-only finetune from the transition checkpoint, "
            "overloop evaluation, and aggregate reporting."
        )
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--node-count", type=int, default=8)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--d-mlp", type=int, default=1024)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--loops", type=int, default=6)
    parser.add_argument("--transition-steps", type=int, default=20000)
    parser.add_argument("--final-steps", type=int, default=8000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=2048)
    parser.add_argument("--eval-batches", type=int, default=32)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--transition-weight", type=float, default=1.0)
    parser.add_argument("--first-anchor-weight", type=float, default=1.0)
    parser.add_argument("--final-anchor-weight", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overloop-max-loop", type=int, default=16)
    parser.add_argument("--overloop-positions", type=int, default=16)
    parser.add_argument("--overloop-batches", type=int, default=96)
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands and write the run manifest without executing training.",
    )
    return parser.parse_args()


def run_dir_name(args: argparse.Namespace, *, loss_mode: str, seed: int, suffix: str = "") -> str:
    name = (
        f"stepgraph_{loss_mode}_N{args.node_count}_D{args.max_depth}_d{args.d_model}_"
        f"B{args.n_layers}_L{args.loops}_seed{seed}"
    )
    return f"{name}_{suffix}" if suffix else name


def command_base(args: argparse.Namespace, *, seed: int, loss_mode: str, steps: int) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "reasoning_loop.graph_path_stepwise",
        "--node-count",
        str(args.node_count),
        "--max-depth",
        str(args.max_depth),
        "--d-model",
        str(args.d_model),
        "--n-heads",
        str(args.n_heads),
        "--d-mlp",
        str(args.d_mlp),
        "--n-layers",
        str(args.n_layers),
        "--loops",
        str(args.loops),
        "--loss-mode",
        loss_mode,
        "--steps",
        str(steps),
        "--batch-size",
        str(args.batch_size),
        "--eval-batch-size",
        str(args.eval_batch_size),
        "--eval-batches",
        str(args.eval_batches),
        "--eval-every",
        str(args.eval_every),
        "--print-every",
        str(args.eval_every),
        "--lr",
        str(args.lr),
        "--weight-decay",
        str(args.weight_decay),
        "--warmup-steps",
        str(args.warmup_steps),
        "--seed",
        str(seed),
        "--device",
        args.device,
        "--out-dir",
        str(args.out_dir),
        "--transition-weight",
        str(args.transition_weight),
        "--first-anchor-weight",
        str(args.first_anchor_weight),
        "--final-anchor-weight",
        str(args.final_anchor_weight),
    ]
    cmd.append("--amp" if args.amp else "--no-amp")
    if args.force:
        cmd.append("--force")
    return cmd


def run_command(cmd: list[str], *, cwd: Path, dry_run: bool) -> None:
    printable = " ".join(cmd)
    print(f"$ {printable}", flush=True)
    if dry_run:
        return
    subprocess.run(cmd, cwd=cwd, check=True)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_training(seed: int, label: str, run_dir: Path) -> dict[str, Any]:
    summary = load_json(run_dir / "summary.json")
    final = summary["final_metrics"]
    return {
        "seed": seed,
        "model": label,
        "run_dir": str(run_dir),
        "best_step": summary["best_step"],
        "trained_step_mean_acc": final["trained_step_mean_acc"],
        "final_target_acc": final["final_target_acc"],
        "rolling_acc_by_loop": final["rolling_acc_by_loop"],
        "best_position_by_loop": final["best_position_by_loop"],
    }


def aggregate_overloop_rows(seed: int, label: str, overloop_dir: Path) -> list[dict[str, Any]]:
    path = overloop_dir / "stepwise_overloop_loop_metrics.csv"
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["model"] != label:
                continue
            rows.append(
                {
                    "seed": seed,
                    "model": label,
                    "loop": int(row["loop"]),
                    "rolling_acc": float(row["rolling_acc"]),
                    "trained_final_acc": float(row["trained_final_acc"]),
                    "best_position": int(row["best_position"]),
                    "best_acc": float(row["best_acc"]),
                    "mean_entropy": float(row["mean_entropy"]),
                    "mean_top1_margin": float(row["mean_top1_margin"]),
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    train_rows: list[dict[str, Any]] = []
    overloop_rows: list[dict[str, Any]] = []

    for seed in args.seeds:
        transition_name = run_dir_name(args, loss_mode="transition", seed=seed)
        transition_dir = args.out_dir / transition_name
        transition_final = transition_dir / "final.pt"

        if args.skip_existing and (transition_dir / "summary.json").exists() and transition_final.exists():
            print(f"skip existing transition seed={seed}: {transition_dir}", flush=True)
        else:
            cmd = command_base(args, seed=seed, loss_mode="transition", steps=args.transition_steps)
            run_command(cmd, cwd=repo_root, dry_run=args.dry_run)
        manifest.append({"seed": seed, "stage": "transition", "run_dir": str(transition_dir)})

        final_name = run_dir_name(args, loss_mode="final", seed=seed, suffix="from_transition")
        final_dir = args.out_dir / final_name
        final_final = final_dir / "final.pt"

        if args.skip_existing and (final_dir / "summary.json").exists() and final_final.exists():
            print(f"skip existing final-ft seed={seed}: {final_dir}", flush=True)
        else:
            cmd = command_base(args, seed=seed, loss_mode="final", steps=args.final_steps)
            cmd.extend(["--init-checkpoint", str(transition_final), "--run-suffix", "from_transition"])
            run_command(cmd, cwd=repo_root, dry_run=args.dry_run)
        manifest.append({"seed": seed, "stage": "final_from_transition", "run_dir": str(final_dir)})

        overloop_dir = args.out_dir / f"overloop_seed{seed}"
        if args.skip_existing and (overloop_dir / "stepwise_overloop_summary.json").exists():
            print(f"skip existing overloop seed={seed}: {overloop_dir}", flush=True)
        else:
            cmd = [
                sys.executable,
                "-m",
                "reasoning_loop.graph_path_stepwise_overloop",
                "--run",
                f"transition_seed{seed}={transition_final}",
                "--run",
                f"transition_final_ft_seed{seed}={final_final}",
                "--out-dir",
                str(overloop_dir),
                "--max-eval-loop",
                str(args.overloop_max_loop),
                "--path-positions",
                str(args.overloop_positions),
                "--batch-size",
                str(args.eval_batch_size),
                "--batches",
                str(args.overloop_batches),
                "--seed",
                str(2026070900 + seed),
                "--device",
                args.device,
            ]
            run_command(cmd, cwd=repo_root, dry_run=args.dry_run)
        manifest.append({"seed": seed, "stage": "overloop", "run_dir": str(overloop_dir)})

        if not args.dry_run and (transition_dir / "summary.json").exists():
            train_rows.append(summarize_training(seed, f"transition_seed{seed}", transition_dir))
        if not args.dry_run and (final_dir / "summary.json").exists():
            train_rows.append(summarize_training(seed, f"transition_final_ft_seed{seed}", final_dir))
        if not args.dry_run and overloop_dir.exists():
            overloop_rows.extend(aggregate_overloop_rows(seed, f"transition_seed{seed}", overloop_dir))
            overloop_rows.extend(aggregate_overloop_rows(seed, f"transition_final_ft_seed{seed}", overloop_dir))

    (args.out_dir / "p3_multiseed_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    if train_rows:
        flat_train_rows = [
            {
                **{k: v for k, v in row.items() if k not in {"rolling_acc_by_loop", "best_position_by_loop"}},
                **{
                    f"rolling_acc_loop_{idx + 1}": value
                    for idx, value in enumerate(row["rolling_acc_by_loop"])
                },
                **{
                    f"best_position_loop_{idx + 1}": value
                    for idx, value in enumerate(row["best_position_by_loop"])
                },
            }
            for row in train_rows
        ]
        write_csv(args.out_dir / "p3_multiseed_training_summary.csv", flat_train_rows)
    if overloop_rows:
        write_csv(args.out_dir / "p3_multiseed_overloop_summary.csv", overloop_rows)


if __name__ == "__main__":
    main()
