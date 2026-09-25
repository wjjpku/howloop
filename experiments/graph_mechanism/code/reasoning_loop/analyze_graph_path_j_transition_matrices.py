"""Analyse a shared loop-boundary J against native per-loop transitions.

This is deliberately a geometry/localisation analysis, not a causal circuit
claim.  It exposes (1) the actual diagonal + low-rank parameters of a trained
J, (2) locally fitted native maps h_t -> h_{t+1}, and (3) how the *same* J is
activated by states from different loop ages.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from reasoning_loop.graph_path_depth_circuit import fixed_depth_batch, load_checkpoint
from reasoning_loop.graph_path_functional_circuit import explicit_depth_position_groups
from reasoning_loop.graph_path_loop import pick_device, set_seed
from reasoning_loop.graph_path_telomere_task_lora_j import (
    DiagonalIdentityLoRAJ,
    load_task_lora_modules,
)
from reasoning_loop.graph_path_temporal_intervention import apply_shared_stack


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot a canonical loop-boundary J and compare native local maps."
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--operator-artifact", type=Path, required=True)
    parser.add_argument("--operator-label", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--calibration-examples", type=int, default=1024)
    parser.add_argument("--heldout-examples", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument(
        "--parameters-only",
        action="store_true",
        help="render the learned J only; no backbone states or local maps are used",
    )
    parser.add_argument(
        "--forward-stage-cards-only",
        action="store_true",
        help="fit only h_t→h_(t+1) maps and render their six parameter-style cards",
    )
    return parser.parse_args(argv)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def cache_natural_states(
    *,
    model: torch.nn.Module,
    cfg: Any,
    device: torch.device,
    examples: int,
    batch_size: int,
) -> list[torch.Tensor]:
    if examples % batch_size:
        raise ValueError("examples must be divisible by batch_size")
    per_age: list[list[torch.Tensor]] = [[] for _ in range(cfg.max_loops + 1)]
    for _ in range(examples // batch_size):
        tokens, _, _, _ = fixed_depth_batch(
            cfg, batch_size, device, path_positions=cfg.max_depth
        )
        state = model.token_embed(tokens) + model.pos_embed.unsqueeze(0)
        per_age[0].append(state.float())
        for loop_index in range(cfg.max_loops):
            state = apply_shared_stack(model, state, loop_index=loop_index)
            per_age[loop_index + 1].append(state.float())
    return [torch.cat(chunks, dim=0) for chunks in per_age]


def _flat(value: torch.Tensor) -> torch.Tensor:
    return value.reshape(-1, value.shape[-1]).float()


@torch.no_grad()
def fit_affine(
    source: torch.Tensor, target: torch.Tensor, ridge: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit target ≈ source @ weight + bias, with a scale-aware ridge."""
    x, y = _flat(source), _flat(target)
    x_mean, y_mean = x.mean(dim=0), y.mean(dim=0)
    xc, yc = x - x_mean, y - y_mean
    covariance = xc.transpose(0, 1) @ xc / x.shape[0]
    cross = xc.transpose(0, 1) @ yc / x.shape[0]
    scale = torch.diagonal(covariance).mean().clamp_min(1e-8)
    eye = torch.eye(x.shape[1], device=x.device, dtype=x.dtype)
    weight = torch.linalg.solve(covariance + ridge * scale * eye, cross)
    bias = y_mean - x_mean @ weight
    return weight, bias


@torch.no_grad()
def affine_metrics(
    source: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> dict[str, float]:
    x, y = _flat(source), _flat(target)
    prediction = x @ weight + bias
    residual = prediction - y
    mse = residual.square().mean()
    variance = (y - y.mean(dim=0)).square().mean().clamp_min(1e-12)
    cosine = torch.nn.functional.cosine_similarity(prediction, y, dim=-1).mean()
    return {
        "relative_rmse": float(residual.norm() / y.norm().clamp_min(1e-12)),
        "r2": float(1.0 - mse / variance),
        "mean_cosine": float(cosine),
    }


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    value = (left.reshape(-1) @ right.reshape(-1)) / (
        left.norm() * right.norm()
    ).clamp_min(1e-12)
    return float(value.clamp(-1.0, 1.0))


def _norm_mean(value: torch.Tensor) -> float:
    return float(value.float().norm(dim=-1).mean())


def _symmetric_bound(values: list[torch.Tensor], quantile: float = 0.995) -> float:
    flat = torch.cat([value.detach().abs().reshape(-1).cpu() for value in values])
    return max(float(torch.quantile(flat, quantile)), 1e-8)


def _heatmap(
    axis: plt.Axes,
    value: torch.Tensor,
    *,
    title: str,
    bound: float,
    xlabel: str = "input dimension",
    ylabel: str = "output dimension",
) -> Any:
    image = axis.imshow(
        value.detach().float().cpu().numpy().transpose(),
        cmap="RdBu_r",
        vmin=-bound,
        vmax=bound,
        aspect="auto",
        interpolation="nearest",
    )
    axis.set_title(title, fontsize=9)
    axis.set_xlabel(xlabel, fontsize=8)
    axis.set_ylabel(ylabel, fontsize=8)
    return image


def _save_parameter_figure(
    *,
    out_dir: Path,
    diagonal: torch.Tensor,
    low_rank: torch.Tensor,
    full_update: torch.Tensor,
    bias: torch.Tensor,
) -> None:
    bound = _symmetric_bound([low_rank, full_update])
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), constrained_layout=True)
    _heatmap(axes[0, 0], full_update, title="J − I (full update)", bound=bound)
    _heatmap(axes[0, 1], low_rank, title="A B (low-rank update)", bound=bound)
    axes[0, 2].plot(diagonal.detach().cpu().numpy(), lw=0.8)
    axes[0, 2].axhline(0.0, color="black", lw=0.6)
    axes[0, 2].set_title("diagonal_scale − 1")
    axes[0, 2].set_xlabel("hidden dimension")
    axes[0, 2].set_ylabel("coefficient")
    axes[0, 3].plot(bias.detach().cpu().numpy(), lw=0.8)
    axes[0, 3].axhline(0.0, color="black", lw=0.6)
    axes[0, 3].set_title("bias")
    axes[0, 3].set_xlabel("hidden dimension")
    axes[0, 3].set_ylabel("coefficient")
    arrays = {
        "diagonal_scale − 1": diagonal,
        "AB entries": low_rank,
        "J − I entries": full_update,
        "bias": bias,
    }
    for (label, value), axis in zip(arrays.items(), axes[1], strict=True):
        axis.hist(value.detach().float().reshape(-1).cpu().numpy(), bins=80, color="#4477AA")
        axis.set_title(label)
        axis.set_xlabel("value")
        axis.set_ylabel("count")
    fig.suptitle("Canonical shared J parameters: D8L8 seed0", fontsize=13)
    fig.savefig(out_dir / "01_canonical_J_parameters.png", dpi=220)
    plt.close(fig)


def _save_transition_figure(
    *,
    out_dir: Path,
    forward: dict[str, tuple[torch.Tensor, torch.Tensor]],
    reverse: dict[str, tuple[torch.Tensor, torch.Tensor]],
    full_update: torch.Tensor,
) -> None:
    updates = [weight - torch.eye(weight.shape[0], device=weight.device) for weight, _ in forward.values()]
    updates += [weight - torch.eye(weight.shape[0], device=weight.device) for weight, _ in reverse.values()]
    bound = _symmetric_bound(updates + [full_update])
    fig, axes = plt.subplots(4, 3, figsize=(13, 16), constrained_layout=True)
    ordered = list(forward.items()) + list(reverse.items())
    for axis, (label, (weight, bias)) in zip(axes.flat, ordered, strict=False):
        update = weight - torch.eye(weight.shape[0], device=weight.device)
        _heatmap(axis, update, title=f"{label}: M − I", bound=bound)
        axis.text(
            0.02,
            0.02,
            f"||b||={bias.norm():.2f}",
            transform=axis.transAxes,
            fontsize=7,
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
        )
    for axis in axes.flat[len(ordered) :]:
        axis.axis("off")
    fig.suptitle("Locally fitted native transitions (all token positions pooled)", fontsize=13)
    fig.savefig(out_dir / "02_native_transition_heatmaps.png", dpi=220)
    plt.close(fig)


def _save_exact_inverse_figure(
    *,
    out_dir: Path,
    exact_inverse: dict[str, tuple[torch.Tensor, torch.Tensor]],
) -> None:
    updates = [
        weight - torch.eye(weight.shape[0], device=weight.device)
        for weight, _ in exact_inverse.values()
    ]
    bound = _symmetric_bound(updates)
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.5), constrained_layout=True)
    for axis, (label, (weight, fitted_bias)) in zip(
        axes.flat, exact_inverse.items(), strict=True
    ):
        update = weight - torch.eye(weight.shape[0], device=weight.device)
        _heatmap(axis, update, title=f"{label}: F⁻¹ − I", bound=bound)
        axis.text(
            0.02,
            0.02,
            f"||b||={fitted_bias.norm():.2f}",
            transform=axis.transAxes,
            fontsize=7,
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
        )
    fig.suptitle("Algebraic inverses of fitted forward maps (not re-fitted R)", fontsize=13)
    fig.savefig(out_dir / "05_exact_forward_inverses.png", dpi=220)
    plt.close(fig)


def _safe_name(label: str) -> str:
    return (
        label.replace("→", "_to_")
        .replace("⁻¹", "_inverse")
        .replace(" ", "_")
        .replace("/", "_")
    )


def _save_stage_cards(
    *,
    out_dir: Path,
    maps: dict[str, tuple[torch.Tensor, torch.Tensor]],
    family: str,
) -> None:
    """Save one parameter-style card per effective loop-stage map."""
    stage_dir = out_dir / "stage_cards" / family
    stage_dir.mkdir(parents=True, exist_ok=True)
    for label, (weight, fitted_bias) in maps.items():
        identity = torch.eye(weight.shape[0], device=weight.device, dtype=weight.dtype)
        update = weight - identity
        diagonal = torch.diagonal(update)
        off_diagonal = update - torch.diag(diagonal)
        bound = _symmetric_bound([update, off_diagonal])
        fig, axes = plt.subplots(2, 4, figsize=(18, 8), constrained_layout=True)
        _heatmap(
            axes[0, 0],
            update,
            title="M − I (full update)",
            bound=bound,
        )
        _heatmap(
            axes[0, 1],
            off_diagonal,
            title="off-diagonal part (cross-dimension mixing)",
            bound=bound,
        )
        axes[0, 2].plot(diagonal.detach().cpu().numpy(), lw=0.8)
        axes[0, 2].axhline(0.0, color="black", lw=0.6)
        axes[0, 2].set_title("diagonal(M) − 1")
        axes[0, 2].set_xlabel("hidden dimension")
        axes[0, 2].set_ylabel("coefficient")
        axes[0, 3].plot(fitted_bias.detach().cpu().numpy(), lw=0.8)
        axes[0, 3].axhline(0.0, color="black", lw=0.6)
        axes[0, 3].set_title("affine bias")
        axes[0, 3].set_xlabel("hidden dimension")
        axes[0, 3].set_ylabel("coefficient")
        distributions = {
            "diagonal(M) − 1": diagonal,
            "off-diagonal entries": off_diagonal[~torch.eye(
                off_diagonal.shape[0], device=off_diagonal.device, dtype=torch.bool
            )],
            "M − I entries": update,
            "affine bias": fitted_bias,
        }
        for (title, value), axis in zip(distributions.items(), axes[1], strict=True):
            axis.hist(value.detach().float().reshape(-1).cpu().numpy(), bins=80, color="#4477AA")
            axis.set_title(title)
            axis.set_xlabel("value")
            axis.set_ylabel("count")
        fig.suptitle(f"{family}: {label} (same-position local affine map)", fontsize=13)
        fig.savefig(stage_dir / f"{_safe_name(label)}.png", dpi=220)
        plt.close(fig)


def _save_relation_figure(
    *,
    out_dir: Path,
    matrices: dict[str, torch.Tensor],
    singular_values: torch.Tensor,
) -> list[dict[str, Any]]:
    labels = list(matrices)
    updates = [matrices[label] for label in labels]
    similarity = np.array([[_cosine(left, right) for right in updates] for left in updates])
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
    image = axes[0].imshow(similarity, cmap="RdBu_r", vmin=-1.0, vmax=1.0)
    axes[0].set_xticks(range(len(labels)), labels, rotation=75, ha="right", fontsize=7)
    axes[0].set_yticks(range(len(labels)), labels, fontsize=7)
    axes[0].set_title("cosine of update matrices (visual reference; not an inverse test)")
    fig.colorbar(image, ax=axes[0], fraction=0.046)
    axes[1].semilogy(
        np.arange(1, singular_values.numel() + 1),
        singular_values.detach().cpu().numpy(),
        marker=".",
        lw=1,
    )
    axes[1].set_title("singular values of J low-rank term AB")
    axes[1].set_xlabel("mode index")
    axes[1].set_ylabel("singular value")
    axes[1].grid(alpha=0.25)
    fig.savefig(out_dir / "03_matrix_relations.png", dpi=220)
    plt.close(fig)
    return [
        {
            "left": labels[i],
            "right": labels[j],
            "update_cosine": float(similarity[i, j]),
        }
        for i in range(len(labels))
        for j in range(len(labels))
    ]


@torch.no_grad()
def _loop_effect_rows(
    *,
    states: list[torch.Tensor],
    diagonal_update: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    bias: torch.Tensor,
    groups: dict[str, tuple[int, ...]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for age, state in enumerate(states[1:], start=1):
        diagonal = state * diagonal_update
        low_rank = (state @ A) @ B
        bias_term = bias.view(1, 1, -1).expand_as(state)
        total = diagonal + low_rank + bias_term
        for group, positions in groups.items():
            selected = list(positions)
            source = state[:, selected, :]
            for component, value in {
                "state": source,
                "diagonal": diagonal[:, selected, :],
                "low_rank": low_rank[:, selected, :],
                "bias": bias_term[:, selected, :],
                "total_J_minus_identity": total[:, selected, :],
            }.items():
                rows.append(
                    {
                        "age": age,
                        "group": group,
                        "component": component,
                        "mean_token_l2": _norm_mean(value),
                    }
                )
    return rows


@torch.no_grad()
def _mode_activations(
    *,
    states: list[torch.Tensor],
    low_rank: torch.Tensor,
) -> tuple[list[dict[str, Any]], torch.Tensor]:
    U, singular, _ = torch.linalg.svd(low_rank, full_matrices=False)
    active = singular > singular.max().clamp_min(1e-12) * 1e-6
    mode_count = int(active.sum())
    U, singular = U[:, :mode_count], singular[:mode_count]
    all_energies = []
    rows: list[dict[str, Any]] = []
    for age, state in enumerate(states[1:], start=1):
        coordinates = state @ U
        energies = coordinates.abs().mean(dim=(0, 1)) * singular
        all_energies.append(energies)
        for mode, (energy, value) in enumerate(zip(energies, singular, strict=True), start=1):
            rows.append(
                {
                    "age": age,
                    "mode": mode,
                    "singular_value": float(value),
                    "mean_abs_output_coordinate": float(energy),
                }
            )
    return rows, torch.stack(all_energies)


def _save_loop_effect_figure(
    *,
    out_dir: Path,
    effect_rows: list[dict[str, Any]],
    mode_energy: torch.Tensor,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
    group = "answer"
    components = ["diagonal", "low_rank", "bias", "total_J_minus_identity"]
    for component in components:
        selected = [
            row["mean_token_l2"]
            for row in effect_rows
            if row["group"] == group and row["component"] == component
        ]
        axes[0].plot(range(1, len(selected) + 1), selected, marker="o", label=component)
    axes[0].set_title("Actual action of the same J at the answer token")
    axes[0].set_xlabel("native loop age of J input")
    axes[0].set_ylabel("mean token L2 norm")
    axes[0].set_xticks(range(1, mode_energy.shape[0] + 1))
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    image = axes[1].imshow(
        np.log10(mode_energy.detach().cpu().numpy() + 1e-8),
        cmap="magma",
        aspect="auto",
        interpolation="nearest",
    )
    axes[1].set_title("J low-rank mode activity by native loop age")
    axes[1].set_xlabel("SVD mode of AB (ordered by singular value)")
    axes[1].set_ylabel("native loop age of J input")
    axes[1].set_yticks(range(mode_energy.shape[0]), range(1, mode_energy.shape[0] + 1))
    fig.colorbar(image, ax=axes[1], label="log10 mean |input coord| × singular value")
    fig.savefig(out_dir / "04_loop_conditioned_J_action.png", dpi=220)
    plt.close(fig)


def _report(
    *,
    out_dir: Path,
    args: argparse.Namespace,
    payload: dict[str, Any],
    transition_rows: list[dict[str, Any]],
    inverse_rows: list[dict[str, Any]],
    similarity_rows: list[dict[str, Any]],
    full_update: torch.Tensor,
    low_rank: torch.Tensor,
) -> None:
    forward = [row for row in transition_rows if row["direction"] == "forward"]
    reverse = [row for row in transition_rows if row["direction"] == "reverse"]
    forward_table = "\n".join(
        f"| {row['label']} | {row['update_frobenius']:.3f} | {row['bias_l2']:.3f} | {row['relative_rmse']:.4f} | {row['r2']:.4f} |"
        for row in forward
    )
    reverse_table = "\n".join(
        f"| {row['label']} | {row['update_frobenius']:.3f} | {row['bias_l2']:.3f} | {row['relative_rmse']:.4f} | {row['r2']:.4f} |"
        for row in reverse
    )
    inverse_table = "\n".join(
        f"| {row['label']} | {row['condition_number']:.2f} | {row['inverse_relative_rmse']:.4f} | {row['refit_relative_rmse']:.4f} | {row['inverse_refit_output_relative_difference']:.4f} | {row['refit_after_forward_relative_rmse']:.4f} | {row['forward_after_refit_relative_rmse']:.4f} |"
        for row in inverse_rows
    )
    text = f"""# D8L8 seed0: shared J and native loop-transition matrices

## What is being compared

The trained controller is one *shared* loop-boundary map, reused after every
two-block loop: `J(h) = h*D + (hA)B + b`.  It is not eight separately trained
matrices.  To make the user's `h2→h3`, `h3→h4`, … question concrete, this run
fits separate local affine maps to the frozen model's **native no-J** states:
`M_t→t+1(h) = h W_t + b_t`.  All 29 token positions are pooled when fitting a
map, so these maps share J's no-token-mixing architecture.

## Scope boundary

These are state-geometry and parameter-localisation measurements.  A heatmap,
matrix similarity, or regression fit does **not** establish that a particular
attention head or MLP neuron causally implements the corresponding part.  The
next circuit test should intervene on modes selected here and measure final
readout recovery against matched random modes.

## Controller

- artifact label: `{args.operator_label}`
- parameterisation: `{payload['modules'][args.operator_label]['parameterization']}`
- dimension/rank: `{payload['modules'][args.operator_label]['dimension']}` / `{payload['modules'][args.operator_label]['rank']}`
- `||J-I||_F = {full_update.norm():.4f}`, `||AB||_F = {low_rank.norm():.4f}`
- calibration / held-out examples: {args.calibration_examples} / {args.heldout_examples}
- ridge: {args.ridge:g}

## Held-out local transition fits

| native map | ||M-I||_F | ||b|| | rel. RMSE | R² |
|---|---:|---:|---:|---:|
{forward_table}

The reverse regressions are separately fitted; they are comparison objects,
not algebraic inverses:

| reverse map | ||M-I||_F | ||b|| | rel. RMSE | R² |
|---|---:|---:|---:|---:|
{reverse_table}

## Exact inverse versus backward re-fit

For a fitted forward map `F(h)=hW+b`, the exact affine inverse is
`F⁻¹(z)=(z-b)W⁻¹`.  The table below compares it to the separately fitted
backward regression R.  `inverse/refit relative RMSE` are both evaluated on
held-out `h_(t+1)→h_t`; the next column is their output disagreement, relative
to the norm of h_t.  The last two columns are the direct composition tests
`R(F(h_t))≈h_t` and `F(R(h_(t+1)))≈h_(t+1)`.

| pair | cond(W) | exact inverse RMSE | re-fit R RMSE | inverse-vs-R diff | R∘F RMSE | F∘R RMSE |
|---|---:|---:|---:|---:|---:|---:|
{inverse_table}

## About matrix-update cosine

`matrix_similarity.csv` gives cosine similarity of `M-I` only as a visual
reference for whether update entries share a basis.  It must **not** be used
as an inverse test: even a perfect scalar inverse has opposite-signed updates
around identity.  The exact-inverse and composition table above is the valid
test of the F/R relation.

## Files

- `01_canonical_J_parameters.png`: the learned shared J's full heatmap,
  diagonal path, low-rank product and coefficient distributions.
- `02_native_transition_heatmaps.png`: fitted forward and reverse maps,
  plotted as `M-I` so the identity part does not hide differences.
- `03_matrix_relations.png`: pairwise update similarities and J's low-rank
  spectrum.
- `04_loop_conditioned_J_action.png`: which gauge-invariant SVD modes of the
  same low-rank update are actually activated at each native age.
- `05_exact_forward_inverses.png`: actual algebraic inverses of the fitted
  forward maps; these are distinct from the separately fitted backward maps.
- `stage_cards/`: one J-style parameter card per forward map, backward
  re-fit, and exact forward inverse.
"""
    (out_dir / "REPORT_CN.md").write_text(text, encoding="utf-8")


@torch.no_grad()
def main(args: argparse.Namespace) -> None:
    if args.calibration_examples % args.batch_size or args.heldout_examples % args.batch_size:
        raise ValueError("calibration and held-out example counts must divide batch-size")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)
    set_seed(args.seed)
    artifact_checkpoint, positions, modules, payload = load_task_lora_modules(
        args.operator_artifact, device=device
    )
    if args.operator_label not in modules:
        raise KeyError(f"missing operator label {args.operator_label}")
    operator = modules[args.operator_label]
    if not isinstance(operator, DiagonalIdentityLoRAJ):
        raise TypeError("this analysis expects a diagonal_low_rank J")

    dimension = int(operator.dimension)
    identity = torch.eye(dimension, device=device)
    diagonal_update = operator.diagonal_scale.float() - 1.0
    low_rank = operator.A.float() @ operator.B.float()
    full_update = torch.diag(diagonal_update) + low_rank
    bias = operator.bias.float()
    _save_parameter_figure(
        out_dir=args.out_dir,
        diagonal=diagonal_update,
        low_rank=low_rank,
        full_update=full_update,
        bias=bias,
    )
    if args.parameters_only:
        summary = {
            "operator_artifact": str(args.operator_artifact),
            "artifact_checkpoint": artifact_checkpoint,
            "operator_label": args.operator_label,
            "positions": list(positions),
            "parameter_only": True,
            "full_update_frobenius": float(full_update.norm()),
            "low_rank_frobenius": float(low_rank.norm()),
        }
        (args.out_dir / "summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        print(json.dumps(summary, indent=2))
        return
    if args.checkpoint is None:
        raise ValueError("--checkpoint is required unless --parameters-only is set")
    if str(args.checkpoint) != artifact_checkpoint:
        raise ValueError(f"artifact checkpoint mismatch: {artifact_checkpoint}")
    model, cfg, _ = load_checkpoint(args.checkpoint, device)
    model.eval()
    if positions != tuple(range(cfg.seq_len)):
        raise ValueError("expected a J that acts on every token position")
    calibration = cache_natural_states(
        model=model,
        cfg=cfg,
        device=device,
        examples=args.calibration_examples,
        batch_size=args.batch_size,
    )
    heldout = cache_natural_states(
        model=model,
        cfg=cfg,
        device=device,
        examples=args.heldout_examples,
        batch_size=args.batch_size,
    )

    forward: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    reverse: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    exact_inverse: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    transition_rows: list[dict[str, Any]] = []
    inverse_rows: list[dict[str, Any]] = []
    for age in range(2, cfg.max_loops):
        label = f"F h{age}→h{age + 1}"
        weight, fitted_bias = fit_affine(calibration[age], calibration[age + 1], args.ridge)
        metrics = affine_metrics(heldout[age], heldout[age + 1], weight, fitted_bias)
        metrics.update(
            {
                "update_frobenius": float((weight - identity).norm()),
                "bias_l2": float(fitted_bias.norm()),
            }
        )
        forward[label] = (weight, fitted_bias)
        transition_rows.append(
            {"label": label, "direction": "forward", "from_age": age, "to_age": age + 1, **metrics}
        )

    _save_stage_cards(out_dir=args.out_dir, maps=forward, family="forward")
    if args.forward_stage_cards_only:
        _write_csv(args.out_dir / "forward_stage_metrics.csv", transition_rows)
        summary = {
            "checkpoint": str(args.checkpoint),
            "operator_artifact": str(args.operator_artifact),
            "operator_label": args.operator_label,
            "seed": args.seed,
            "calibration_examples": args.calibration_examples,
            "heldout_examples": args.heldout_examples,
            "ridge": args.ridge,
            "scope": "forward h_t to h_(t+1) local maps only; no R or inverse fitted",
            "transition_rows": transition_rows,
        }
        (args.out_dir / "summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        print(json.dumps(summary, indent=2))
        return

    for age in range(2, cfg.max_loops):
        label = f"R h{age + 1}→h{age}"
        weight, fitted_bias = fit_affine(calibration[age + 1], calibration[age], args.ridge)
        metrics = affine_metrics(heldout[age + 1], heldout[age], weight, fitted_bias)
        metrics.update(
            {
                "update_frobenius": float((weight - identity).norm()),
                "bias_l2": float(fitted_bias.norm()),
            }
        )
        reverse[label] = (weight, fitted_bias)
        transition_rows.append(
            {"label": label, "direction": "reverse", "from_age": age + 1, "to_age": age, **metrics}
        )
        inverse_weight = torch.linalg.inv(forward[f"F h{age}→h{age + 1}"][0])
        inverse_bias = -forward[f"F h{age}→h{age + 1}"][1] @ inverse_weight
        inverse_label = f"F⁻¹ h{age + 1}→h{age}"
        exact_inverse[inverse_label] = (inverse_weight, inverse_bias)
        inverse_metrics = affine_metrics(
            heldout[age + 1], heldout[age], inverse_weight, inverse_bias
        )
        refit_metrics = affine_metrics(
            heldout[age + 1], heldout[age], weight, fitted_bias
        )
        exact_output = _flat(heldout[age + 1]) @ inverse_weight + inverse_bias
        refit_output = _flat(heldout[age + 1]) @ weight + fitted_bias
        source_flat = _flat(heldout[age])
        target_flat = _flat(heldout[age + 1])
        forward_weight, forward_bias = forward[f"F h{age}→h{age + 1}"]
        refit_after_forward = (source_flat @ forward_weight + forward_bias) @ weight + fitted_bias
        forward_after_refit = (target_flat @ weight + fitted_bias) @ forward_weight + forward_bias
        inverse_rows.append(
            {
                "label": f"h{age + 1}→h{age}",
                "from_age": age + 1,
                "to_age": age,
                "condition_number": float(torch.linalg.cond(forward[f"F h{age}→h{age + 1}"][0])),
                "inverse_relative_rmse": inverse_metrics["relative_rmse"],
                "refit_relative_rmse": refit_metrics["relative_rmse"],
                "inverse_refit_output_relative_difference": float(
                    (exact_output - refit_output).norm()
                    / _flat(heldout[age]).norm().clamp_min(1e-12)
                ),
                "inverse_weight_frobenius": float((inverse_weight - identity).norm()),
                "inverse_bias_l2": float(inverse_bias.norm()),
                "refit_after_forward_relative_rmse": float(
                    (refit_after_forward - source_flat).norm()
                    / source_flat.norm().clamp_min(1e-12)
                ),
                "forward_after_refit_relative_rmse": float(
                    (forward_after_refit - target_flat).norm()
                    / target_flat.norm().clamp_min(1e-12)
                ),
            }
        )

    _save_transition_figure(
        out_dir=args.out_dir,
        forward=forward,
        reverse=reverse,
        full_update=full_update,
    )
    _save_exact_inverse_figure(out_dir=args.out_dir, exact_inverse=exact_inverse)
    _save_stage_cards(out_dir=args.out_dir, maps=reverse, family="backward_refit")
    _save_stage_cards(out_dir=args.out_dir, maps=exact_inverse, family="exact_forward_inverse")
    singular_values = torch.linalg.svdvals(low_rank)
    matrices = {
        "J controller": full_update,
        **{label: weight - identity for label, (weight, _) in forward.items()},
        **{label: weight - identity for label, (weight, _) in reverse.items()},
        **{label: weight - identity for label, (weight, _) in exact_inverse.items()},
    }
    similarity_rows = _save_relation_figure(
        out_dir=args.out_dir, matrices=matrices, singular_values=singular_values
    )
    groups = explicit_depth_position_groups(cfg.node_count)
    effect_rows = _loop_effect_rows(
        states=heldout,
        diagonal_update=diagonal_update,
        A=operator.A.float(),
        B=operator.B.float(),
        bias=bias,
        groups=groups,
    )
    mode_rows, mode_energy = _mode_activations(states=heldout, low_rank=low_rank)
    _save_loop_effect_figure(
        out_dir=args.out_dir, effect_rows=effect_rows, mode_energy=mode_energy
    )

    _write_csv(args.out_dir / "transition_fit_metrics.csv", transition_rows)
    _write_csv(args.out_dir / "exact_inverse_comparison.csv", inverse_rows)
    _write_csv(args.out_dir / "matrix_similarity.csv", similarity_rows)
    _write_csv(args.out_dir / "loop_conditioned_J_effects.csv", effect_rows)
    _write_csv(args.out_dir / "lowrank_mode_activations.csv", mode_rows)
    summary = {
        "checkpoint": str(args.checkpoint),
        "operator_artifact": str(args.operator_artifact),
        "operator_label": args.operator_label,
        "positions": list(positions),
        "seed": args.seed,
        "calibration_examples": args.calibration_examples,
        "heldout_examples": args.heldout_examples,
        "ridge": args.ridge,
        "full_update_frobenius": float(full_update.norm()),
        "low_rank_frobenius": float(low_rank.norm()),
        "low_rank_numerical_rank": int((singular_values > singular_values.max() * 1e-6).sum()),
        "transition_rows": transition_rows,
        "inverse_rows": inverse_rows,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _report(
        out_dir=args.out_dir,
        args=args,
        payload=payload,
        transition_rows=transition_rows,
        inverse_rows=inverse_rows,
        similarity_rows=similarity_rows,
        full_update=full_update,
        low_rank=low_rank,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main(parse_args())
