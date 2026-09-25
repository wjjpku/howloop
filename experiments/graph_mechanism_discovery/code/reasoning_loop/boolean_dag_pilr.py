from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn

from reasoning_loop.boolean_dag_data import (
    BooleanDAGBatch,
    BooleanDAGConfig,
    UNKNOWN,
    make_boolean_dag_batch,
    wavefront_targets,
)
from reasoning_loop.boolean_dag_model import (
    BooleanDAGBlock,
    BooleanDAGModelConfig,
    _BooleanDAGTransformerBase,
    count_parameters,
)
from reasoning_loop.graph_path_loop import cosine_lr, pick_device, set_seed


TASK_VERSION = "boolean_dag_pilr_v1"


@dataclass(frozen=True)
class PILRTrainConfig:
    objective: Literal["random_final", "pilr"] = "pilr"
    steps: int = 10_000
    batch_size: int = 512
    train_recurrences: int = 6
    eval_recurrences: int = 12
    eval_batch_size: int = 512
    eval_batches: int = 4
    eval_every: int = 500
    print_every: int = 500
    lr: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 500
    grad_clip: float = 1.0
    path_weight: float = 0.1
    monotonic_weight: float = 0.1
    variance_weight: float = 0.01
    initial_noise_std: float = 1.0
    seed: int = 0
    device: str = "auto"
    amp: bool = True
    out_dir: Path = Path("results/boolean_dag_pilr")
    run_name: str | None = None
    force: bool = False

    def __post_init__(self) -> None:
        if self.objective not in {"random_final", "pilr"}:
            raise ValueError("objective must be random_final or pilr")
        if self.steps < 1 or self.batch_size < 2:
            raise ValueError("steps and batch_size must be positive")
        if self.train_recurrences < 1 or self.eval_recurrences < self.train_recurrences:
            raise ValueError("evaluation recurrence must cover the training recurrence")


class PathIndependentBooleanDAGTransformer(_BooleanDAGTransformerBase):
    """Recurrent core with immutable input injection and no timestep encoding."""

    def __init__(
        self,
        data_cfg: BooleanDAGConfig,
        model_cfg: BooleanDAGModelConfig,
    ) -> None:
        super().__init__(data_cfg, model_cfg)
        self.workspace_norm = nn.LayerNorm(model_cfg.d_model)
        self.input_norm = nn.LayerNorm(model_cfg.d_model)
        self.input_adapter = nn.Linear(2 * model_cfg.d_model, model_cfg.d_model, bias=False)
        self.block = BooleanDAGBlock(model_cfg)
        self.update_logit = nn.Parameter(torch.tensor(-0.5))

    def initial_workspace(
        self,
        encoded_input: torch.Tensor,
        *,
        noise_std: float = 0.0,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        workspace = torch.zeros_like(encoded_input)
        if noise_std:
            noise = torch.randn(
                encoded_input.shape,
                device=encoded_input.device,
                dtype=encoded_input.dtype,
                generator=generator,
            )
            workspace = workspace + noise_std * noise
        return workspace

    def apply_recurrence(
        self,
        workspace: torch.Tensor,
        encoded_input: torch.Tensor,
    ) -> torch.Tensor:
        injected = self.input_adapter(
            torch.cat(
                (self.workspace_norm(workspace), self.input_norm(encoded_input)),
                dim=-1,
            )
        )
        candidate = self.block(injected)
        rate = self.update_logit.sigmoid()
        return workspace + rate * (candidate - workspace)

    def forward_all(
        self,
        batch: BooleanDAGBatch,
        *,
        max_steps: int,
        noise_std: float = 0.0,
        generator: torch.Generator | None = None,
    ) -> dict[str, torch.Tensor]:
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        encoded_input = self.encode(batch)
        workspace = self.initial_workspace(
            encoded_input,
            noise_std=noise_std,
            generator=generator,
        )
        raw_states = []
        readout_states = []
        logits = []
        for _ in range(max_steps):
            workspace = self.apply_recurrence(workspace, encoded_input)
            step_logits, readout_state = self._readout(workspace)
            raw_states.append(workspace)
            readout_states.append(readout_state)
            logits.append(step_logits)
        return {
            "logits_by_step": torch.stack(logits, dim=1),
            "states_by_step": torch.stack(readout_states, dim=1),
            "raw_states_by_step": torch.stack(raw_states, dim=1),
        }


def _root_indices(batch: BooleanDAGBatch) -> torch.Tensor:
    return batch.root_mask.long().argmax(dim=1)


def root_logits_by_step(
    logits_by_step: torch.Tensor,
    batch: BooleanDAGBatch,
) -> torch.Tensor:
    root_indices = _root_indices(batch)
    return logits_by_step.gather(
        2,
        root_indices[:, None, None, None].expand(
            -1,
            logits_by_step.shape[1],
            1,
            logits_by_step.shape[-1],
        ),
    ).squeeze(2)


def gather_recurrence(values: torch.Tensor, recurrences: torch.Tensor) -> torch.Tensor:
    if values.shape[0] != recurrences.shape[0]:
        raise ValueError("values and recurrences must have matching batches")
    rows = torch.arange(values.shape[0], device=values.device)
    return values[rows, recurrences - 1]


def sample_recurrences(
    batch: BooleanDAGBatch,
    *,
    max_recurrences: int,
    generator: torch.Generator,
) -> torch.Tensor:
    minimum = batch.depths.clamp_max(max_recurrences)
    span = max_recurrences - minimum + 1
    offsets = torch.floor(
        torch.rand(batch.batch_size, device=batch.depths.device, generator=generator)
        * span
    ).long().clamp_max(span - 1)
    return minimum + offsets


def _per_example_root_ce(root_logits: torch.Tensor, batch: BooleanDAGBatch) -> torch.Tensor:
    return F.cross_entropy(root_logits, batch.root_values + 1, reduction="none")


def pilr_objective(
    clean: dict[str, torch.Tensor],
    alternate: dict[str, torch.Tensor] | None,
    batch: BooleanDAGBatch,
    recurrences: torch.Tensor,
    *,
    path_weight: float,
    monotonic_weight: float,
    variance_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    clean_root = root_logits_by_step(clean["logits_by_step"], batch)
    clean_selected = gather_recurrence(clean_root, recurrences)
    clean_ce = _per_example_root_ce(clean_selected, batch)
    task = clean_ce.mean()
    path = task.new_zeros(())
    variance = task.new_zeros(())

    selected_clean_state = gather_recurrence(clean["raw_states_by_step"], recurrences)
    if alternate is not None:
        alternate_root = root_logits_by_step(alternate["logits_by_step"], batch)
        alternate_selected = gather_recurrence(alternate_root, recurrences)
        alternate_ce = _per_example_root_ce(alternate_selected, batch)
        task = 0.5 * (clean_ce.mean() + alternate_ce.mean())
        selected_alternate_state = gather_recurrence(
            alternate["raw_states_by_step"], recurrences
        )
        clean_normalized = F.layer_norm(
            selected_clean_state,
            (selected_clean_state.shape[-1],),
        )
        alternate_normalized = F.layer_norm(
            selected_alternate_state,
            (selected_alternate_state.shape[-1],),
        )
        path = 1.0 - F.cosine_similarity(
            clean_normalized,
            alternate_normalized,
            dim=-1,
        ).mean()
        flattened = torch.cat((clean_normalized, alternate_normalized), dim=0).flatten(0, 1)
        variance = F.relu(0.5 - flattened.std(dim=0, unbiased=False)).mean()

    previous_recurrences = (recurrences - 1).clamp_min(1)
    previous_clean = gather_recurrence(clean_root, previous_recurrences)
    previous_ce = _per_example_root_ce(previous_clean, batch).detach()
    monotonic_candidates = F.relu(clean_ce - previous_ce)[recurrences.gt(1)]
    monotonic = (
        monotonic_candidates.mean()
        if monotonic_candidates.numel()
        else task.new_zeros(())
    )
    loss = (
        task
        + path_weight * path
        + monotonic_weight * monotonic
        + variance_weight * variance
    )
    return loss, {
        "task_ce": task.detach(),
        "path_loss": path.detach(),
        "monotonic_loss": monotonic.detach(),
        "variance_loss": variance.detach(),
    }


@torch.no_grad()
def evaluate_pilr(
    model: PathIndependentBooleanDAGTransformer,
    *,
    data_cfg: BooleanDAGConfig,
    max_steps: int,
    device: torch.device,
    batch_size: int,
    batches: int,
    amp: bool,
    noise_std: float,
) -> dict[str, Any]:
    if batch_size % data_cfg.eval_max_depth:
        raise ValueError("eval_batch_size must be divisible by eval_max_depth")
    model.eval()
    generator = torch.Generator(device=device).manual_seed(20_260_711)
    root_correct = torch.zeros(data_cfg.eval_max_depth, max_steps, device=device)
    root_count = torch.zeros_like(root_correct)
    state_correct = torch.zeros_like(root_correct)
    state_count = torch.zeros_like(root_correct)
    level_correct = torch.zeros(max_steps, data_cfg.eval_max_depth + 1, device=device)
    level_count = torch.zeros_like(level_correct)
    level_known = torch.zeros_like(level_correct)
    level_value_correct = torch.zeros_like(level_correct)
    path_cosine = torch.zeros(max_steps, device=device)
    path_disagreement = torch.zeros(max_steps, device=device)
    update_norm = torch.zeros(max_steps, device=device)
    path_batches = 0
    for _ in range(batches):
        depths = torch.arange(1, data_cfg.eval_max_depth + 1, device=device).repeat_interleave(
            batch_size // data_cfg.eval_max_depth
        )
        depths = depths[torch.randperm(batch_size, device=device, generator=generator)]
        batch = make_boolean_dag_batch(
            data_cfg,
            batch_size,
            device,
            depths=depths,
            generator=generator,
        )
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=amp and device.type == "cuda",
        ):
            clean = model.forward_all(batch, max_steps=max_steps)
            alternate = model.forward_all(
                batch,
                max_steps=max_steps,
                noise_std=noise_std,
                generator=generator,
            )
        root_clean = root_logits_by_step(clean["logits_by_step"], batch)
        root_alternate = root_logits_by_step(alternate["logits_by_step"], batch)
        predictions = root_clean.argmax(dim=-1)
        target = (batch.root_values + 1)[:, None]
        state_targets = wavefront_targets(batch, readouts=max_steps)
        state_predictions = clean["logits_by_step"].argmax(dim=-1)
        for depth in range(1, data_cfg.eval_max_depth + 1):
            mask = batch.depths.eq(depth)
            root_correct[depth - 1] += predictions[mask].eq(target[mask]).sum(dim=0)
            root_count[depth - 1] += mask.sum()
            state_correct[depth - 1] += state_predictions[mask].eq(
                state_targets[mask]
            ).sum(dim=(0, 2))
            state_count[depth - 1] += mask.sum() * batch.node_count
        for level in range(data_cfg.eval_max_depth + 1):
            mask = batch.levels.eq(level)
            level_correct[:, level] += (
                state_predictions.eq(state_targets) & mask[:, None, :]
            ).sum(dim=(0, 2))
            level_known[:, level] += (
                state_predictions.ne(UNKNOWN) & mask[:, None, :]
            ).sum(dim=(0, 2))
            level_value_correct[:, level] += (
                state_predictions.eq((batch.values + 1)[:, None, :])
                & mask[:, None, :]
            ).sum(dim=(0, 2))
            level_count[:, level] += mask.sum()
        clean_states = clean["raw_states_by_step"]
        alternate_states = alternate["raw_states_by_step"]
        path_cosine += F.cosine_similarity(
            F.layer_norm(clean_states, (clean_states.shape[-1],)),
            F.layer_norm(alternate_states, (alternate_states.shape[-1],)),
            dim=-1,
        ).mean(dim=(0, 2))
        path_disagreement += root_clean.argmax(dim=-1).ne(
            root_alternate.argmax(dim=-1)
        ).float().mean(dim=0)
        previous = torch.cat((torch.zeros_like(clean_states[:, :1]), clean_states[:, :-1]), dim=1)
        update_norm += (clean_states - previous).float().norm(dim=-1).mean(dim=(0, 2))
        path_batches += 1
    return {
        "root_accuracy_by_depth_step": (root_correct / root_count.clamp_min(1)).cpu().tolist(),
        "state_accuracy_by_depth_step": (state_correct / state_count.clamp_min(1)).cpu().tolist(),
        "wavefront_level_accuracy": (level_correct / level_count.clamp_min(1)).cpu().tolist(),
        "known_rate_by_step_level": (level_known / level_count.clamp_min(1)).cpu().tolist(),
        "value_accuracy_by_step_level": (
            level_value_correct / level_count.clamp_min(1)
        ).cpu().tolist(),
        "path_cosine_by_step": (path_cosine / path_batches).cpu().tolist(),
        "path_prediction_disagreement_by_step": (
            path_disagreement / path_batches
        ).cpu().tolist(),
        "update_norm_by_step": (update_norm / path_batches).cpu().tolist(),
        "update_rate": float(model.update_logit.sigmoid().cpu()),
    }


def _write_history(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot_heatmap(
    values: list[list[float]],
    *,
    path: Path,
    title: str,
    ylabel: str,
    ylabels: list[int] | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    image = ax.imshow(values, vmin=0.0, vmax=1.0, cmap="viridis", aspect="auto")
    ax.set(title=title, xlabel="recurrence", ylabel=ylabel)
    ax.set_xticks(range(len(values[0])), range(1, len(values[0]) + 1))
    labels = ylabels if ylabels is not None else list(range(1, len(values) + 1))
    ax.set_yticks(range(len(values)), labels)
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)


def save_evaluation_plots(metrics: dict[str, Any], out_dir: Path) -> None:
    _plot_heatmap(
        metrics["root_accuracy_by_depth_step"],
        path=out_dir / "root_accuracy_heatmap.png",
        title="Root accuracy: task depth x recurrence",
        ylabel="task depth",
    )
    _plot_heatmap(
        metrics["state_accuracy_by_depth_step"],
        path=out_dir / "state_accuracy_heatmap.png",
        title="Oracle wavefront state accuracy",
        ylabel="task depth",
    )
    known_by_level = list(map(list, zip(*metrics["known_rate_by_step_level"])))
    _plot_heatmap(
        known_by_level,
        path=out_dir / "known_rate_level_heatmap.png",
        title="Spontaneous computation frontier: known rate",
        ylabel="node level",
        ylabels=list(range(len(known_by_level))),
    )
    value_by_level = list(map(list, zip(*metrics["value_accuracy_by_step_level"])))
    _plot_heatmap(
        value_by_level,
        path=out_dir / "value_accuracy_level_heatmap.png",
        title="True value availability by node level",
        ylabel="node level",
        ylabels=list(range(len(value_by_level))),
    )
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    steps = range(1, len(metrics["path_cosine_by_step"]) + 1)
    axes[0].plot(steps, metrics["path_cosine_by_step"])
    axes[0].set(title="Path alignment", xlabel="recurrence", ylabel="cosine", ylim=(-0.05, 1.05))
    axes[1].plot(steps, metrics["path_prediction_disagreement_by_step"])
    axes[1].set(title="Prediction disagreement", xlabel="recurrence", ylabel="fraction", ylim=(-0.01, 1.01))
    axes[2].plot(steps, metrics["update_norm_by_step"])
    axes[2].set(title="Recurrent update norm", xlabel="recurrence", ylabel="norm")
    fig.tight_layout()
    fig.savefig(out_dir / "path_independence_curves.png", dpi=180, facecolor="white")
    plt.close(fig)


def train_pilr(
    *,
    data_cfg: BooleanDAGConfig,
    model_cfg: BooleanDAGModelConfig,
    train_cfg: PILRTrainConfig,
) -> Path:
    run_name = train_cfg.run_name or f"pilr_{train_cfg.objective}_seed{train_cfg.seed}"
    run_dir = train_cfg.out_dir / run_name
    if run_dir.exists() and any(run_dir.iterdir()) and not train_cfg.force:
        raise FileExistsError(f"{run_dir} exists; pass --force to overwrite")
    run_dir.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(run_dir).free < 256 * 1024**2:
        raise RuntimeError(f"insufficient free space for checkpointing in {run_dir}")
    set_seed(train_cfg.seed)
    device = pick_device(train_cfg.device)
    model = PathIndependentBooleanDAGTransformer(data_cfg, model_cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_cfg.lr, betas=(0.9, 0.95), weight_decay=train_cfg.weight_decay
    )
    train_generator = torch.Generator(device=device).manual_seed(60_711_000 + train_cfg.seed)
    history: list[dict[str, Any]] = []
    last_metrics: dict[str, Any] = {}
    start = time.time()
    for step in range(1, train_cfg.steps + 1):
        model.train()
        lr = cosine_lr(
            step - 1,
            base_lr=train_cfg.lr,
            total_steps=train_cfg.steps,
            warmup_steps=train_cfg.warmup_steps,
        )
        for group in optimizer.param_groups:
            group["lr"] = lr
        batch = make_boolean_dag_batch(
            data_cfg,
            train_cfg.batch_size,
            device,
            generator=train_generator,
        )
        recurrences = sample_recurrences(
            batch,
            max_recurrences=train_cfg.train_recurrences,
            generator=train_generator,
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=train_cfg.amp and device.type == "cuda",
        ):
            clean = model.forward_all(batch, max_steps=train_cfg.train_recurrences)
            alternate = None
            if train_cfg.objective == "pilr":
                alternate = model.forward_all(
                    batch,
                    max_steps=train_cfg.train_recurrences,
                    noise_std=train_cfg.initial_noise_std,
                    generator=train_generator,
                )
            loss, pieces = pilr_objective(
                clean,
                alternate,
                batch,
                recurrences,
                path_weight=train_cfg.path_weight if alternate is not None else 0.0,
                monotonic_weight=train_cfg.monotonic_weight if alternate is not None else 0.0,
                variance_weight=train_cfg.variance_weight if alternate is not None else 0.0,
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        optimizer.step()
        if step == 1 or step % train_cfg.eval_every == 0 or step == train_cfg.steps:
            last_metrics = evaluate_pilr(
                model,
                data_cfg=data_cfg,
                max_steps=train_cfg.eval_recurrences,
                device=device,
                batch_size=train_cfg.eval_batch_size,
                batches=train_cfg.eval_batches,
                amp=train_cfg.amp,
                noise_std=train_cfg.initial_noise_std,
            )
            depth4 = last_metrics["root_accuracy_by_depth_step"][3]
            depth8 = last_metrics["root_accuracy_by_depth_step"][7]
            final_step = train_cfg.eval_recurrences - 1
            final_loop = train_cfg.eval_recurrences
            row = {
                "step": step,
                "lr": lr,
                "loss": float(loss.detach().cpu()),
                **{name: float(value.cpu()) for name, value in pieces.items()},
                "depth4_acc_loop4": depth4[3],
                "depth8_acc_loop8": depth8[7],
                f"depth8_acc_loop{final_loop}": depth8[final_step],
                f"path_cosine_loop{final_loop}": last_metrics["path_cosine_by_step"][final_step],
                f"path_disagreement_loop{final_loop}": last_metrics[
                    "path_prediction_disagreement_by_step"
                ][final_step],
                "elapsed_sec": time.time() - start,
            }
            history.append(row)
            _write_history(run_dir / "history.csv", history)
            print(
                f"[{train_cfg.objective} step={step:05d}] loss={row['loss']:.4f} "
                f"D4L4={row['depth4_acc_loop4']:.3f} D8L8={row['depth8_acc_loop8']:.3f} "
                f"PI{final_loop}={row[f'path_cosine_loop{final_loop}']:.3f}",
                flush=True,
            )
    checkpoint = {
        "task_version": TASK_VERSION,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "data_config": asdict(data_cfg),
        "model_config": asdict(model_cfg),
        "train_config": {**asdict(train_cfg), "out_dir": str(train_cfg.out_dir)},
        "parameter_count": count_parameters(model),
        "metrics": last_metrics,
    }
    temporary = run_dir / "final.pt.tmp"
    torch.save(checkpoint, temporary)
    os.replace(temporary, run_dir / "final.pt")
    summary = {
        "task_version": TASK_VERSION,
        "objective": train_cfg.objective,
        "seed": train_cfg.seed,
        "steps": train_cfg.steps,
        "parameter_count": checkpoint["parameter_count"],
        "data_config": asdict(data_cfg),
        "model_config": asdict(model_cfg),
        "train_config": checkpoint["train_config"],
        "metrics": last_metrics,
        "elapsed_sec": time.time() - start,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    save_evaluation_plots(last_metrics, run_dir)
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train path-independent latent refinement on Boolean DAGs.")
    parser.add_argument("--objective", choices=["random_final", "pilr"], default="pilr")
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--train-recurrences", type=int, default=6)
    parser.add_argument("--eval-recurrences", type=int, default=12)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--eval-batches", type=int, default=4)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--d-mlp", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--path-weight", type=float, default=0.1)
    parser.add_argument("--monotonic-weight", type=float, default=0.1)
    parser.add_argument("--variance-weight", type=float, default=0.01)
    parser.add_argument("--initial-noise-std", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--out-dir", type=Path, default=Path("results/boolean_dag_pilr"))
    parser.add_argument("--run-name", type=str)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_pilr(
        data_cfg=BooleanDAGConfig(),
        model_cfg=BooleanDAGModelConfig(
            d_model=args.d_model,
            n_heads=args.n_heads,
            d_mlp=args.d_mlp,
            steps=args.train_recurrences,
        ),
        train_cfg=PILRTrainConfig(
            objective=args.objective,
            steps=args.steps,
            batch_size=args.batch_size,
            train_recurrences=args.train_recurrences,
            eval_recurrences=args.eval_recurrences,
            eval_batch_size=args.eval_batch_size,
            eval_batches=args.eval_batches,
            eval_every=args.eval_every,
            lr=args.lr,
            path_weight=args.path_weight,
            monotonic_weight=args.monotonic_weight,
            variance_weight=args.variance_weight,
            initial_noise_std=args.initial_noise_std,
            seed=args.seed,
            device=args.device,
            amp=args.amp,
            out_dir=args.out_dir,
            run_name=args.run_name,
            force=args.force,
        ),
    )


if __name__ == "__main__":
    main()
