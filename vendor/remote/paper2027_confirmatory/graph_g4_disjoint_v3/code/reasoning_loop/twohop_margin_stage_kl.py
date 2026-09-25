from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from reasoning_loop.twohop_hidden_state_kl import (
    ROLE_NAMES,
    _operator_signature,
    _plot_heatmap,
    aggregate_seed_metrics,
    analyze_models_for_seed,
    evaluate_models_by_depth,
    fit_shared_pca,
)
from reasoning_loop.twohop_in_context import (
    TwoHopConfig,
    TwoHopTransformer,
    build_twohop_model,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Track P2x3 query-state KL geometry across six post-solution "
            "training checkpoints with increasing answer margin."
        )
    )
    parser.add_argument("--suite-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument(
        "--stage-steps",
        type=int,
        nargs="+",
        default=[2000, 3000, 4500, 6000, 7500, 10000],
    )
    parser.add_argument("--examples", type=int, default=8192)
    parser.add_argument("--pca-examples", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--pca-dim", type=int, default=64)
    parser.add_argument("--cov-shrinkage", type=float, default=0.05)
    parser.add_argument("--cov-jitter", type=float, default=1e-6)
    parser.add_argument("--question-seed-base", type=int, default=71000)
    parser.add_argument("--sample-question-count", type=int, default=16)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--cuda-memory-fraction", type=float, default=0.08)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def stage_name(step: int) -> str:
    return f"step_{step:06d}"


def pick_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def load_stage_model(
    checkpoint_path: Path,
    device: torch.device,
) -> TwoHopTransformer:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    cfg = TwoHopConfig.from_dict(checkpoint["config"])
    if cfg.architecture != "periodic" or cfg.period != 2 or cfg.total_depth != 6:
        raise ValueError(
            f"{checkpoint_path} is not a trained-depth-6 P2x3 checkpoint"
        )
    model = build_twohop_model(
        cfg,
        seed=int(checkpoint.get("init_seed", checkpoint["seed"])),
        device=device,
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def checkpoint_path(suite_dir: Path, seed: int, step: int) -> Path:
    return (
        suite_dir
        / "runs"
        / f"P2x3_seed{seed}"
        / "checkpoints"
        / f"step_{step:06d}.pt"
    )


def aggregate_behavior(
    per_seed: Mapping[
        int,
        Mapping[str, Mapping[str, np.ndarray]],
    ],
    stage_names: Sequence[str],
) -> dict[str, dict[str, np.ndarray]]:
    result: dict[str, dict[str, np.ndarray]] = {}
    for name in stage_names:
        result[name] = {}
        for metric in ("depth_accuracy", "depth_margin", "depth_loss"):
            values = np.stack(
                [
                    per_seed[seed][name][metric]
                    for seed in sorted(per_seed)
                ]
            )
            result[name][f"{metric}_mean"] = values.mean(axis=0)
            result[name][f"{metric}_std"] = values.std(axis=0)
    return result


def _shared_vmax(matrices: Sequence[np.ndarray]) -> float:
    values = np.concatenate([np.log1p(matrix).ravel() for matrix in matrices])
    return max(float(np.quantile(values, 0.98)), 1e-8)


def plot_stage_heatmaps(
    out_dir: Path,
    *,
    steps: Sequence[int],
    aggregate: Mapping[str, Mapping[str, np.ndarray]],
    behavior: Mapping[str, Mapping[str, np.ndarray]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    query_role = ROLE_NAMES.index("query")
    names = [stage_name(step) for step in steps]

    state_matrices = [
        aggregate[name]["role_depth_pairwise_mean"][query_role][1:, 1:]
        for name in names
    ]
    delta_matrices = [
        aggregate[name]["delta_step_pairwise_mean"][query_role]
        for name in names
    ]
    for matrices, filename, labels, suptitle in (
        (
            state_matrices,
            "margin_stage_query_depth_kl_postblock.png",
            [f"D{depth}" for depth in range(1, 7)],
            "P2x3 query-state KL across margin stages",
        ),
        (
            delta_matrices,
            "margin_stage_query_delta_kl.png",
            [
                f"D{depth}→D{depth + 1}\nP{depth % 2}"
                for depth in range(6)
            ],
            "P2x3 query-update KL across margin stages",
        ),
    ):
        vmax = _shared_vmax(matrices)
        fig, axes = plt.subplots(
            2,
            3,
            figsize=(17, 10),
            constrained_layout=True,
        )
        image = None
        for index, (axis, step, name, matrix) in enumerate(
            zip(axes.flat, steps, names, matrices)
        ):
            margin = behavior[name]["depth_margin_mean"][-1]
            margin_std = behavior[name]["depth_margin_std"][-1]
            accuracy = behavior[name]["depth_accuracy_mean"][-1]
            image = _plot_heatmap(
                axis,
                matrix,
                title=(
                    f"stage {index + 1}: step {step}\n"
                    f"margin={margin:.2f}±{margin_std:.2f}, "
                    f"acc={accuracy:.3f}"
                ),
                xlabels=labels,
                ylabels=labels,
                vmax=vmax,
                annotate=True,
            )
        fig.colorbar(
            image,
            ax=axes,
            label="log(1 + symmetric Gaussian KL); red = more similar",
            shrink=0.82,
        )
        fig.suptitle(suptitle)
        fig.savefig(out_dir / filename, dpi=220)
        plt.close(fig)

    margins = np.array(
        [behavior[name]["depth_margin_mean"][-1] for name in names]
    )
    margin_std = np.array(
        [behavior[name]["depth_margin_std"][-1] for name in names]
    )
    accuracies = np.array(
        [behavior[name]["depth_accuracy_mean"][-1] for name in names]
    )
    accuracy_std = np.array(
        [behavior[name]["depth_accuracy_std"][-1] for name in names]
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    axes[0].errorbar(steps, margins, yerr=margin_std, marker="o", capsize=3)
    axes[0].set(xlabel="training step", ylabel="D6 answer margin")
    axes[0].grid(alpha=0.25)
    axes[1].errorbar(
        steps,
        accuracies,
        yerr=accuracy_std,
        marker="o",
        capsize=3,
    )
    axes[1].set(xlabel="training step", ylabel="D6 accuracy", ylim=(0.98, 1.001))
    axes[1].grid(alpha=0.25)
    fig.suptitle("Fixed-question behavior at the six margin stages")
    fig.savefig(out_dir / "margin_stage_behavior.png", dpi=220)
    plt.close(fig)


def write_report(
    out_dir: Path,
    *,
    args: argparse.Namespace,
    aggregate: Mapping[str, Mapping[str, np.ndarray]],
    behavior: Mapping[str, Mapping[str, np.ndarray]],
    pca_explained: Mapping[int, float],
) -> None:
    query_role = ROLE_NAMES.index("query")
    schedule = [0, 1, 0, 1, 0, 1]
    lines = [
        "# P2x3 hidden-state KL across six post-solution margin stages",
        "",
        "Every stage uses the same held-out questions. Within each seed, one PCA "
        "basis is fitted jointly across all six checkpoints; KL is computed "
        "inside that shared coordinate system and only scalar KL values are "
        "aggregated across seeds.",
        "",
        "Heatmaps use a shared color scale across all six panels. Red means "
        "lower KL / greater similarity; blue means higher KL / lower similarity.",
        "",
        f"- seeds: {', '.join(str(seed) for seed in args.seeds)}",
        f"- hidden-state examples per seed/stage: {args.examples}",
        f"- PCA examples per seed/stage: {args.pca_examples}",
        f"- PCA dimension: {args.pca_dim}",
        f"- mean retained variance: {np.mean(list(pca_explained.values())):.4f}",
        "",
        "## Fixed-question behavior",
        "",
        "| stage | step | D6 accuracy | D6 margin | retrieval-update KL | "
        "same/different block ratio |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for index, step in enumerate(args.stage_steps, start=1):
        name = stage_name(step)
        matrix = aggregate[name]["delta_step_pairwise_mean"][query_role]
        same, different = _operator_signature(matrix, schedule)
        ratio = same / different if different > 0 and math.isfinite(same) else math.nan
        lines.append(
            f"| {index} | {step} | "
            f"{behavior[name]['depth_accuracy_mean'][-1]:.5f} ± "
            f"{behavior[name]['depth_accuracy_std'][-1]:.5f} | "
            f"{behavior[name]['depth_margin_mean'][-1]:.3f} ± "
            f"{behavior[name]['depth_margin_std'][-1]:.3f} | "
            f"{matrix[1, 3]:.3f} | {ratio:.3f} |"
        )
    lines.extend(
        [
            "",
            "`retrieval-update KL` compares P1 at D1→D2 with P1 at D3→D4. "
            "The ratio compares all same-parameter query updates against all "
            "different-parameter updates.",
            "",
            "These are distributional trajectory measurements, not causal "
            "circuit sufficiency/necessity tests.",
            "",
            "## Files",
            "",
            "- `margin_stage_query_depth_kl_postblock.png`: primary six-panel "
            "query-state heatmap.",
            "- `margin_stage_query_delta_kl.png`: auxiliary shared-operator "
            "update heatmap.",
            "- `margin_stage_behavior.png`: accuracy and margin trajectory.",
            "- `aggregate_metrics.npz`: aggregate KL and behavior arrays.",
            "- `per_seed/`: per-seed PCA and KL outputs.",
            "",
        ]
    )
    (out_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if len(args.stage_steps) != 6:
        raise ValueError("exactly six stage steps are required")
    if sorted(set(args.stage_steps)) != list(args.stage_steps):
        raise ValueError("stage steps must be unique and strictly increasing")
    if args.examples < 2 or args.pca_examples < 2 or args.batch_size < 1:
        raise ValueError("invalid example or batch size")
    if args.out_dir.exists() and any(args.out_dir.iterdir()) and not args.force:
        raise FileExistsError(f"{args.out_dir} is non-empty; pass --force")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    per_seed_dir = args.out_dir / "per_seed"
    per_seed_dir.mkdir(parents=True, exist_ok=True)

    paths = [
        checkpoint_path(args.suite_dir, seed, step)
        for seed in args.seeds
        for step in args.stage_steps
    ]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "missing stage checkpoints:\n" + "\n".join(missing)
        )

    device = pick_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(
            args.cuda_memory_fraction,
            device=torch.cuda.current_device(),
        )
        torch.cuda.reset_peak_memory_stats(device)

    names = [stage_name(step) for step in args.stage_steps]
    all_seed_metrics: dict[int, dict[str, dict[str, np.ndarray]]] = {}
    all_seed_behavior: dict[int, dict[str, dict[str, np.ndarray]]] = {}
    pca_explained: dict[int, float] = {}
    sampled_questions: dict[str, list[dict[str, Any]]] = {}

    for seed in args.seeds:
        print(f"=== P2x3 margin-stage KL seed={seed} ===", flush=True)
        models = {
            stage_name(step): load_stage_model(
                checkpoint_path(args.suite_dir, seed, step),
                device,
            )
            for step in args.stage_steps
        }
        pca_seed = args.question_seed_base + 1000 * seed
        pca_mean, pca_basis, explained = fit_shared_pca(
            models,
            examples=args.pca_examples,
            batch_size=args.batch_size,
            seed=pca_seed,
            pca_dim=args.pca_dim,
            device=device,
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
        )
        behavior = evaluate_models_by_depth(
            models,
            examples=args.examples,
            batch_size=args.batch_size,
            seed=pca_seed + 1,
            device=device,
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
        for name in names:
            for metric_name, value in metrics[name].items():
                payload[f"{name}__{metric_name}"] = value
            for metric_name, value in behavior[name].items():
                payload[f"{name}__{metric_name}"] = value
        np.savez_compressed(per_seed_dir / f"seed_{seed}.npz", **payload)
        del models, metrics, behavior
        if device.type == "cuda":
            torch.cuda.empty_cache()

    aggregate = aggregate_seed_metrics(all_seed_metrics, names)
    behavior_aggregate = aggregate_behavior(all_seed_behavior, names)
    payload = {
        f"{name}__{metric_name}": value
        for name in names
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
    manifest = {
        "experiment": "P2x3 post-solution margin-stage hidden-state KL",
        "stage_steps": args.stage_steps,
        "seeds": args.seeds,
        "method": "per-seed PCA shared across all six checkpoints",
        "heatmap_transform": "log1p",
        "heatmap_colormap": (
            "coolwarm_r: red=lower KL/more similar; "
            "blue=higher KL/less similar"
        ),
        "pca_explained_variance_by_seed": pca_explained,
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "caveat": "KL is distributional evidence, not causal circuit proof.",
    }
    plot_stage_heatmaps(
        args.out_dir,
        steps=args.stage_steps,
        aggregate=aggregate,
        behavior=behavior_aggregate,
    )
    write_report(
        args.out_dir,
        args=args,
        aggregate=aggregate,
        behavior=behavior_aggregate,
        pca_explained=pca_explained,
    )
    if device.type == "cuda":
        manifest["observed_peak_allocated_mib"] = (
            torch.cuda.max_memory_allocated(device) / (1024**2)
        )
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(f"wrote margin-stage outputs under {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
