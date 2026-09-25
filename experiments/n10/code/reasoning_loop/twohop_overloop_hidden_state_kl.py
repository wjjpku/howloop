from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from reasoning_loop.twohop_hidden_state_kl import (
    ROLE_NAMES,
    _write_long_csv,
    aggregate_seed_metrics,
    analyze_models_for_seed,
    evaluate_models_by_depth,
    fit_shared_pca,
    load_model,
    make_plots,
)
from reasoning_loop.twohop_margin_stage_kl import aggregate_behavior


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate S6 and P2x3 hidden-state trajectories to effective "
            "depth 12. S6 repeats its complete trained six-block stack as an "
            "explicit non-shared stack-cycle control."
        )
    )
    parser.add_argument("--suite-dir", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=["S6", "P2x3"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--active-depth", type=int, default=12)
    parser.add_argument("--examples", type=int, default=8192)
    parser.add_argument("--pca-examples", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--pca-dim", type=int, default=64)
    parser.add_argument("--cov-shrinkage", type=float, default=0.05)
    parser.add_argument("--cov-jitter", type=float, default=1e-6)
    parser.add_argument("--question-seed-base", type=int, default=81000)
    parser.add_argument("--sample-question-count", type=int, default=16)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--cuda-memory-fraction", type=float, default=0.08)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def pick_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def model_schedule(
    model_name: str,
    *,
    trained_depth: int,
    period: int,
    active_depth: int,
) -> list[int]:
    if model_name == "S6":
        return [depth % trained_depth for depth in range(active_depth)]
    return [depth % period for depth in range(active_depth)]


def plot_behavior(
    out_dir: Path,
    *,
    behavior: Mapping[str, Mapping[str, np.ndarray]],
    model_names: Sequence[str],
    active_depth: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    depths = np.arange(1, active_depth + 1)
    colors = {"S6": "#4c78a8", "P2x3": "#e45756"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for name in model_names:
        color = colors.get(name)
        accuracy = behavior[name]["depth_accuracy_mean"]
        accuracy_std = behavior[name]["depth_accuracy_std"]
        margin = behavior[name]["depth_margin_mean"]
        margin_std = behavior[name]["depth_margin_std"]
        axes[0].plot(depths, accuracy, marker="o", label=name, color=color)
        axes[0].fill_between(
            depths,
            accuracy - accuracy_std,
            accuracy + accuracy_std,
            alpha=0.16,
            color=color,
        )
        axes[1].plot(depths, margin, marker="o", label=name, color=color)
        axes[1].fill_between(
            depths,
            margin - margin_std,
            margin + margin_std,
            alpha=0.16,
            color=color,
        )
    for axis in axes:
        axis.axvline(6, linestyle="--", color="black", alpha=0.5)
        axis.set_xlabel("effective depth")
        axis.grid(alpha=0.25)
        axis.legend()
    axes[0].set_ylabel("accuracy")
    axes[1].set_ylabel("answer margin")
    fig.suptitle(
        "Behavior beyond the trained readout horizon "
        "(dashed line: trained depth 6)"
    )
    fig.savefig(out_dir / "overloop_behavior_depth_1_12.png", dpi=220)
    plt.close(fig)


def write_report(
    out_dir: Path,
    *,
    args: argparse.Namespace,
    aggregate: Mapping[str, Mapping[str, np.ndarray]],
    behavior: Mapping[str, Mapping[str, np.ndarray]],
    pca_explained: Mapping[int, float],
    schedules: Mapping[str, Sequence[int]],
) -> None:
    query_role = ROLE_NAMES.index("query")
    lines = [
        "# S6 versus P2x3: hidden-state overloop to effective depth 12",
        "",
        "Both models are the original trained-depth-6 checkpoints. P2x3 natively "
        "continues its P0/P1 recurrence. S6 has no native layer beyond D6, so "
        "D7–D12 explicitly repeat the full trained stack P0…P5 as a stack-cycle "
        "control. This is not a 12-layer standard Transformer trained from scratch.",
        "",
        "Within each seed, PCA is shared by both models and all D0…D12 states. "
        "The two primary post-block state matrices and both update matrices are "
        "exactly 12×12. Red means lower KL / greater distributional similarity; "
        "blue means higher KL / lower similarity.",
        "",
        f"- seeds: {', '.join(str(seed) for seed in args.seeds)}",
        f"- examples per model/seed: {args.examples}",
        f"- PCA dimension: {args.pca_dim}",
        f"- mean retained variance: {np.mean(list(pca_explained.values())):.4f}",
        "",
        "## Behavior and trajectory summary",
        "",
        "| model | D6 acc | D12 acc | D6 margin | D12 margin | "
        "peak-margin depth | KL(query D6,D12) | KL(D11,D12) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in args.models:
        accuracy = behavior[name]["depth_accuracy_mean"]
        margin = behavior[name]["depth_margin_mean"]
        query_matrix = aggregate[name]["role_depth_pairwise_mean"][query_role]
        transition = aggregate[name]["role_transition_mean"][:, query_role]
        peak_depth = int(np.argmax(margin) + 1)
        lines.append(
            f"| {name} | {accuracy[5]:.5f} | {accuracy[11]:.5f} | "
            f"{margin[5]:.3f} | {margin[11]:.3f} | {peak_depth} | "
            f"{query_matrix[6, 12]:.3f} | {transition[11]:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Parameter schedules",
            "",
        ]
    )
    for name in args.models:
        lines.append(
            f"- `{name}`: "
            + ", ".join(f"D{i + 1}=P{p}" for i, p in enumerate(schedules[name]))
        )
    lines.extend(
        [
            "",
            "The KL heatmaps characterize distributional dynamics. They do not "
            "by themselves prove an attractor or a causal finite-state circuit.",
            "",
            "## Files",
            "",
            "- `query_depth_pairwise_kl_postblock.png`: primary 12×12 "
            "post-block query-state heatmaps.",
            "- `query_delta_step_pairwise_kl.png`: 12×12 query-update heatmaps.",
            "- `query_depth_pairwise_kl.png`: supplementary 13×13 matrices "
            "including the D0 embedding state.",
            "- `overloop_behavior_depth_1_12.png`: accuracy and margin curves.",
            "- `aggregate_metrics.npz`: raw aggregate arrays.",
            "- `per_seed/`: per-seed PCA, KL, and behavior arrays.",
            "",
        ]
    )
    (out_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.models != ["S6", "P2x3"]:
        raise ValueError("this experiment requires --models S6 P2x3")
    if args.active_depth != 12:
        raise ValueError("this experiment is pre-registered at active depth 12")
    if args.examples < 2 or args.pca_examples < 2 or args.batch_size < 1:
        raise ValueError("invalid example or batch size")
    if args.out_dir.exists() and any(args.out_dir.iterdir()) and not args.force:
        raise FileExistsError(f"{args.out_dir} is non-empty; pass --force")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    per_seed_dir = args.out_dir / "per_seed"
    per_seed_dir.mkdir(parents=True, exist_ok=True)

    suite_summary = json.loads(
        (args.suite_dir / "summary.json").read_text(encoding="utf-8")
    )
    run_lookup = {
        (str(run["model"]), int(run["seed"])): run
        for run in suite_summary["runs"]
    }
    device = pick_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(
            args.cuda_memory_fraction,
            device=torch.cuda.current_device(),
        )
        torch.cuda.reset_peak_memory_stats(device)

    all_seed_metrics: dict[int, dict[str, dict[str, np.ndarray]]] = {}
    all_seed_behavior: dict[int, dict[str, dict[str, np.ndarray]]] = {}
    pca_explained: dict[int, float] = {}
    sampled_questions: dict[str, list[dict[str, Any]]] = {}
    schedules: dict[str, list[int]] = {}

    for seed in args.seeds:
        print(f"=== overloop hidden-state KL seed={seed} ===", flush=True)
        models = {
            name: load_model(
                args.suite_dir,
                run_lookup[(name, seed)],
                device,
            )
            for name in args.models
        }
        for name, model in models.items():
            schedules[name] = model_schedule(
                name,
                trained_depth=model.cfg.total_depth,
                period=model.cfg.period,
                active_depth=args.active_depth,
            )
        pca_seed = args.question_seed_base + 1000 * seed
        pca_mean, pca_basis, explained = fit_shared_pca(
            models,
            examples=args.pca_examples,
            batch_size=args.batch_size,
            seed=pca_seed,
            pca_dim=args.pca_dim,
            device=device,
            active_depth=args.active_depth,
            cycle_standard_beyond_depth=True,
        )
        metrics, questions = analyze_models_for_seed(
            models,
            examples=args.examples,
            batch_size=args.batch_size,
            seed=pca_seed + 1,
            pca_mean=pca_mean,
            pca_basis=pca_basis,
            shrinkage=args.cov_shrinkage,
            jitter=args.cov_jitter,
            device=device,
            sample_question_count=args.sample_question_count,
            active_depth=args.active_depth,
            cycle_standard_beyond_depth=True,
        )
        behavior = evaluate_models_by_depth(
            models,
            examples=args.examples,
            batch_size=args.batch_size,
            seed=pca_seed + 1,
            device=device,
            active_depth=args.active_depth,
            cycle_standard_beyond_depth=True,
        )
        all_seed_metrics[seed] = metrics
        all_seed_behavior[seed] = behavior
        pca_explained[seed] = float(explained)
        sampled_questions[str(seed)] = questions

        payload: dict[str, np.ndarray] = {
            "pca_mean": pca_mean.cpu().numpy(),
            "pca_basis": pca_basis.cpu().numpy(),
            "pca_explained_variance": np.array(explained),
        }
        for name in args.models:
            for metric_name, value in metrics[name].items():
                payload[f"{name}__{metric_name}"] = value
            for metric_name, value in behavior[name].items():
                payload[f"{name}__{metric_name}"] = value
        np.savez_compressed(per_seed_dir / f"seed_{seed}.npz", **payload)
        del models, metrics, behavior
        if device.type == "cuda":
            torch.cuda.empty_cache()

    aggregate = aggregate_seed_metrics(all_seed_metrics, args.models)
    behavior_aggregate = aggregate_behavior(all_seed_behavior, args.models)
    payload = {
        f"{name}__{metric_name}": value
        for name in args.models
        for metric_name, value in {
            **aggregate[name],
            **behavior_aggregate[name],
        }.items()
    }
    np.savez_compressed(args.out_dir / "aggregate_metrics.npz", **payload)
    (args.out_dir / "sampled_questions.json").write_text(
        json.dumps(sampled_questions, indent=2),
        encoding="utf-8",
    )
    _write_long_csv(args.out_dir, aggregate)
    make_plots(args.out_dir, aggregate, args.models, schedules)
    plot_behavior(
        args.out_dir,
        behavior=behavior_aggregate,
        model_names=args.models,
        active_depth=args.active_depth,
    )
    write_report(
        args.out_dir,
        args=args,
        aggregate=aggregate,
        behavior=behavior_aggregate,
        pca_explained=pca_explained,
        schedules=schedules,
    )
    manifest: dict[str, Any] = {
        "experiment": "trained-depth-6 models evaluated to depth 12",
        "standard_control": "S6 repeats its complete P0..P5 stack",
        "periodic_schedule": "P2x3 continues alternating P0/P1",
        "seeds": args.seeds,
        "pca_explained_variance_by_seed": pca_explained,
        "parameter_schedule_by_model": schedules,
        "heatmap_transform": "log1p",
        "heatmap_colormap": (
            "coolwarm_r: red=lower KL/more similar; "
            "blue=higher KL/less similar"
        ),
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "caveat": (
            "S6 beyond D6 is an explicit stack-cycle control; KL is "
            "distributional evidence, not causal proof."
        ),
    }
    if device.type == "cuda":
        manifest["observed_peak_allocated_mib"] = (
            torch.cuda.max_memory_allocated(device) / (1024**2)
        )
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(f"wrote overloop outputs under {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
