from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from dataclasses import asdict
from pathlib import Path
from typing import Literal
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from reasoning_loop.graph_path_loop import (
    LoopedGraphPathTransformer,
    pick_device,
    set_seed,
)
from reasoning_loop.graph_path_stepwise import StepwiseGraphPathConfig, make_stepwise_batch
from reasoning_loop.graph_path_temporal_intervention import (
    cache_raw_states,
    logits_from_raw_state,
    replace_answer_state,
)


@dataclass(frozen=True)
class ComponentSite:
    block_index: int
    branch: Literal["attn", "mlp"]

    @property
    def label(self) -> str:
        return f"B{self.block_index + 1}.{self.branch}"


def apply_stack_instrumented(
    model: LoopedGraphPathTransformer,
    x: torch.Tensor,
    *,
    patch_site: ComponentSite | None = None,
    patch_answer: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[ComponentSite, torch.Tensor]]:
    if (patch_site is None) != (patch_answer is None):
        raise ValueError("patch_site and patch_answer must be provided together")
    if patch_site is not None and not 0 <= patch_site.block_index < len(model.blocks):
        raise ValueError("patch_site.block_index is outside the shared stack")
    cache: dict[ComponentSite, torch.Tensor] = {}
    for block_index, block in enumerate(model.blocks):
        attn_site = ComponentSite(block_index, "attn")
        attn_output = block.attn(block.ln_1(x))
        cache[attn_site] = attn_output[:, -1, :]
        if patch_site == attn_site:
            if patch_answer.shape != attn_output[:, -1, :].shape:
                raise ValueError("patch_answer has the wrong shape")
            attn_output = attn_output.clone()
            attn_output[:, -1, :] = patch_answer
        x = x + attn_output

        mlp_site = ComponentSite(block_index, "mlp")
        mlp_output = block.mlp(block.ln_2(x))
        cache[mlp_site] = mlp_output[:, -1, :]
        if patch_site == mlp_site:
            if patch_answer.shape != mlp_output[:, -1, :].shape:
                raise ValueError("patch_answer has the wrong shape")
            mlp_output = mlp_output.clone()
            mlp_output[:, -1, :] = patch_answer
        x = x + mlp_output
    return x, cache


def normalized_patch_score(
    clean_logit_difference: torch.Tensor,
    corrupt_logit_difference: torch.Tensor,
    patched_logit_difference: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    denominator = clean_logit_difference - corrupt_logit_difference
    score = (patched_logit_difference - corrupt_logit_difference) / denominator
    return torch.where(
        denominator.abs() > eps,
        score,
        torch.full_like(score, float("nan")),
    )


def _logit_difference(
    logits: torch.Tensor,
    target_a: torch.Tensor,
    target_b: torch.Tensor,
) -> torch.Tensor:
    logit_a = logits.gather(1, target_a[:, None]).squeeze(1)
    logit_b = logits.gather(1, target_b[:, None]).squeeze(1)
    return logit_a - logit_b


@torch.no_grad()
def component_patch_batch_metrics(
    *,
    model: LoopedGraphPathTransformer,
    donor_a_tokens: torch.Tensor,
    donor_a_targets: torch.Tensor,
    donor_b_tokens: torch.Tensor,
    donor_b_targets: torch.Tensor,
    receiver_tokens: torch.Tensor,
    receiver_successors: torch.Tensor,
    state_loops: list[int],
    behavior: Literal["transition", "endpoint"],
    final_target_position: int,
) -> dict[str, torch.Tensor | list[str]]:
    """Measure causal effects of each shared-stack branch on one update.

    A and B answer states are transplanted into the same receiver context. For
    each effective site, patch-in inserts A's branch output into B, while
    patch-out inserts B's branch output into A. Only pairs whose two clean runs
    predict their distinct intended targets are included in the score sums.
    """
    if behavior not in {"transition", "endpoint"}:
        raise ValueError("behavior must be 'transition' or 'endpoint'")
    if not state_loops or min(state_loops) < 1:
        raise ValueError("state_loops must contain 1-indexed positive values")
    if final_target_position < 1:
        raise ValueError("final_target_position must be >= 1")
    if donor_a_tokens.shape != donor_b_tokens.shape or donor_a_tokens.shape != receiver_tokens.shape:
        raise ValueError("donor and receiver token batches must have identical shapes")
    if donor_a_targets.shape != donor_b_targets.shape:
        raise ValueError("donor target batches must have identical shapes")
    if donor_a_targets.shape[0] != donor_a_tokens.shape[0]:
        raise ValueError("target and token batch sizes must match")
    if behavior == "transition" and max(state_loops) > donor_a_targets.shape[1]:
        raise ValueError("donor targets do not cover every requested state loop")
    if final_target_position > donor_a_targets.shape[1]:
        raise ValueError("final_target_position is outside donor targets")

    sites = [
        ComponentSite(block_index, branch)
        for block_index in range(len(model.blocks))
        for branch in ("attn", "mlp")
    ]
    device = donor_a_tokens.device
    shape = (len(state_loops), len(sites))
    metric_names = (
        "patch_in_score_sum",
        "patch_out_score_sum",
        "shuffled_score_sum",
        "patch_in_target_a_correct",
        "patch_in_target_b_correct",
        "patch_out_target_a_correct",
        "patch_out_target_b_correct",
        "shuffled_target_a_correct",
    )
    result: dict[str, torch.Tensor | list[str]] = {
        name: torch.zeros(shape, device=device) for name in metric_names
    }
    result["valid_count"] = torch.zeros(len(state_loops), device=device)
    result["candidate_count"] = torch.zeros(len(state_loops), device=device)
    result["clean_a_correct"] = torch.zeros(len(state_loops), device=device)
    result["clean_b_correct"] = torch.zeros(len(state_loops), device=device)
    result["site_labels"] = [site.label for site in sites]

    max_loop = max(state_loops)
    donor_a_states = cache_raw_states(model, donor_a_tokens, max_loop=max_loop)
    donor_b_states = cache_raw_states(model, donor_b_tokens, max_loop=max_loop)
    receiver_states = cache_raw_states(model, receiver_tokens, max_loop=max_loop)

    for loop_index, state_loop in enumerate(state_loops):
        receiver_state = receiver_states[state_loop - 1]
        state_a = replace_answer_state(receiver_state, donor_a_states[state_loop - 1])
        state_b = replace_answer_state(receiver_state, donor_b_states[state_loop - 1])
        clean_a, cache_a = apply_stack_instrumented(model, state_a)
        clean_b, cache_b = apply_stack_instrumented(model, state_b)
        logits_a = logits_from_raw_state(model, clean_a)
        logits_b = logits_from_raw_state(model, clean_b)

        if behavior == "transition":
            current_a = donor_a_targets[:, state_loop - 1]
            current_b = donor_b_targets[:, state_loop - 1]
            target_a = receiver_successors.gather(1, current_a[:, None]).squeeze(1)
            target_b = receiver_successors.gather(1, current_b[:, None]).squeeze(1)
        else:
            target_a = donor_a_targets[:, final_target_position - 1]
            target_b = donor_b_targets[:, final_target_position - 1]

        candidate = target_a.ne(target_b)
        clean_a_correct = logits_a.argmax(dim=-1).eq(target_a)
        clean_b_correct = logits_b.argmax(dim=-1).eq(target_b)
        valid = candidate & clean_a_correct & clean_b_correct
        result["candidate_count"][loop_index] = candidate.sum()
        result["valid_count"][loop_index] = valid.sum()
        result["clean_a_correct"][loop_index] = (candidate & clean_a_correct).sum()
        result["clean_b_correct"][loop_index] = (candidate & clean_b_correct).sum()
        clean_a_difference = _logit_difference(logits_a, target_a, target_b)
        clean_b_difference = _logit_difference(logits_b, target_a, target_b)

        for site_index, site in enumerate(sites):
            patch_in_state, _ = apply_stack_instrumented(
                model,
                state_b,
                patch_site=site,
                patch_answer=cache_a[site],
            )
            patch_out_state, _ = apply_stack_instrumented(
                model,
                state_a,
                patch_site=site,
                patch_answer=cache_b[site],
            )
            shuffled_state, _ = apply_stack_instrumented(
                model,
                state_b,
                patch_site=site,
                patch_answer=cache_a[site].roll(1, dims=0),
            )
            patch_in_logits = logits_from_raw_state(model, patch_in_state)
            patch_out_logits = logits_from_raw_state(model, patch_out_state)
            shuffled_logits = logits_from_raw_state(model, shuffled_state)

            patch_in_score = normalized_patch_score(
                clean_a_difference,
                clean_b_difference,
                _logit_difference(patch_in_logits, target_a, target_b),
            )
            patch_out_score = normalized_patch_score(
                clean_b_difference,
                clean_a_difference,
                _logit_difference(patch_out_logits, target_a, target_b),
            )
            shuffled_score = normalized_patch_score(
                clean_a_difference,
                clean_b_difference,
                _logit_difference(shuffled_logits, target_a, target_b),
            )
            for name, score in (
                ("patch_in_score_sum", patch_in_score),
                ("patch_out_score_sum", patch_out_score),
                ("shuffled_score_sum", shuffled_score),
            ):
                finite_valid = valid & score.isfinite()
                result[name][loop_index, site_index] = score[finite_valid].sum()

            result["patch_in_target_a_correct"][loop_index, site_index] = (
                patch_in_logits.argmax(dim=-1).eq(target_a) & valid
            ).sum()
            result["patch_in_target_b_correct"][loop_index, site_index] = (
                patch_in_logits.argmax(dim=-1).eq(target_b) & valid
            ).sum()
            result["patch_out_target_a_correct"][loop_index, site_index] = (
                patch_out_logits.argmax(dim=-1).eq(target_a) & valid
            ).sum()
            result["patch_out_target_b_correct"][loop_index, site_index] = (
                patch_out_logits.argmax(dim=-1).eq(target_b) & valid
            ).sum()
            result["shuffled_target_a_correct"][loop_index, site_index] = (
                shuffled_logits.argmax(dim=-1).eq(target_a) & valid
            ).sum()

    return result


def _safe_divide(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    denominator = denominator.to(numerator.dtype)
    while denominator.ndim < numerator.ndim:
        denominator = denominator.unsqueeze(-1)
    return torch.where(
        denominator > 0,
        numerator / denominator.clamp_min(1),
        torch.full_like(numerator, float("nan")),
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _save_component_heatmap(
    values: np.ndarray,
    *,
    row_labels: list[str],
    col_labels: list[str],
    title: str,
    colorbar_label: str,
    path: Path,
    score_map: bool,
) -> None:
    width = max(7.0, 1.1 * len(col_labels) + 2.5)
    height = max(4.5, 0.62 * len(row_labels) + 2.2)
    fig, ax = plt.subplots(figsize=(width, height))
    if score_map:
        finite = np.abs(values[np.isfinite(values)])
        limit = max(1.0, float(finite.max())) if finite.size else 1.0
        image = ax.imshow(values, aspect="auto", cmap="coolwarm", vmin=-limit, vmax=limit)
    else:
        image = ax.imshow(values, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(col_labels)), col_labels)
    ax.set_yticks(range(len(row_labels)), row_labels)
    ax.set_xlabel("shared-stack component")
    ax.set_ylabel("state before effective update")
    ax.set_title(title)
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            value = values[row, col]
            label = "-" if not np.isfinite(value) else f"{value:.2f}"
            ax.text(col, row, label, ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, label=colorbar_label)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _nanmean_std(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(values)
    count = finite.sum(axis=0)
    total = np.where(finite, values, 0.0).sum(axis=0)
    mean = np.divide(
        total,
        count,
        out=np.full_like(total, np.nan, dtype=float),
        where=count > 0,
    )
    squared_error = np.where(finite, (values - mean) ** 2, 0.0).sum(axis=0)
    variance = np.divide(
        squared_error,
        count,
        out=np.full_like(total, np.nan, dtype=float),
        where=count > 0,
    )
    return mean, np.sqrt(variance)


@torch.no_grad()
def analyze_component_checkpoint(
    *,
    name: str,
    checkpoint: Path,
    out_dir: Path,
    device: torch.device,
    state_loops: list[int],
    behavior: Literal["transition", "endpoint"],
    batch_size: int,
    batches: int,
    seed: int,
) -> dict[str, Any]:
    if batch_size < 1 or batches < 1:
        raise ValueError("batch_size and batches must be >= 1")
    if behavior not in {"transition", "endpoint"}:
        raise ValueError("behavior must be 'transition' or 'endpoint'")
    if not state_loops or min(state_loops) < 1:
        raise ValueError("state_loops must contain 1-indexed positive values")

    set_seed(seed)
    checkpoint_data = torch.load(
        checkpoint, map_location=device, weights_only=False
    )
    cfg = StepwiseGraphPathConfig(**checkpoint_data["config"])
    model = LoopedGraphPathTransformer(cfg).to(device)
    model.load_state_dict(checkpoint_data["model"])
    model.eval()

    sums: dict[str, torch.Tensor] = {}
    site_labels: list[str] | None = None
    path_positions = max(cfg.max_depth, max(state_loops))
    for _ in range(batches):
        donor_a_tokens, donor_a_targets, _, _ = make_stepwise_batch(
            cfg, batch_size, device, path_positions=path_positions
        )
        donor_b_tokens, donor_b_targets, _, _ = make_stepwise_batch(
            cfg, batch_size, device, path_positions=path_positions
        )
        receiver_tokens, _, receiver_successors, _ = make_stepwise_batch(
            cfg, batch_size, device, path_positions=path_positions
        )
        batch_metrics = component_patch_batch_metrics(
            model=model,
            donor_a_tokens=donor_a_tokens,
            donor_a_targets=donor_a_targets,
            donor_b_tokens=donor_b_tokens,
            donor_b_targets=donor_b_targets,
            receiver_tokens=receiver_tokens,
            receiver_successors=receiver_successors,
            state_loops=state_loops,
            behavior=behavior,
            final_target_position=cfg.max_depth,
        )
        if site_labels is None:
            site_labels = list(batch_metrics["site_labels"])
        for key, value in batch_metrics.items():
            if isinstance(value, torch.Tensor):
                if key not in sums:
                    sums[key] = torch.zeros_like(value)
                sums[key] += value

    assert site_labels is not None
    sums = {key: value.cpu() for key, value in sums.items()}
    valid_count = sums["valid_count"]
    candidate_count = sums["candidate_count"]
    patch_in_score = _safe_divide(sums["patch_in_score_sum"], valid_count)
    patch_out_score = _safe_divide(sums["patch_out_score_sum"], valid_count)
    shuffled_score = _safe_divide(sums["shuffled_score_sum"], valid_count)
    patch_in_target_a_accuracy = _safe_divide(sums["patch_in_target_a_correct"], valid_count)
    patch_out_target_b_accuracy = _safe_divide(sums["patch_out_target_b_correct"], valid_count)
    shuffled_target_a_accuracy = _safe_divide(sums["shuffled_target_a_correct"], valid_count)
    clean_a_accuracy = _safe_divide(sums["clean_a_correct"], candidate_count)
    clean_b_accuracy = _safe_divide(sums["clean_b_correct"], candidate_count)
    valid_fraction = _safe_divide(valid_count, candidate_count)

    run_dir = out_dir / name
    run_dir.mkdir(parents=True, exist_ok=True)
    row_labels = [f"L{loop} -> L{loop + 1}" for loop in state_loops]
    for values, title, label, filename in (
        (
            patch_in_score,
            f"{name}: A component patched into B",
            "normalized A recovery",
            "patch_in_score.png",
        ),
        (
            patch_out_score,
            f"{name}: B component patched into A",
            "normalized B recovery",
            "patch_out_score.png",
        ),
        (
            shuffled_score,
            f"{name}: shuffled-A component patched into B",
            "normalized A recovery",
            "shuffled_control_score.png",
        ),
    ):
        _save_component_heatmap(
            values.numpy(),
            row_labels=row_labels,
            col_labels=site_labels,
            title=title,
            colorbar_label=label,
            path=run_dir / filename,
            score_map=True,
        )
    clean_values = torch.stack((clean_a_accuracy, clean_b_accuracy, valid_fraction), dim=1)
    _save_component_heatmap(
        clean_values.numpy(),
        row_labels=row_labels,
        col_labels=["clean A", "clean B", "paired valid"],
        title=f"{name}: clean continuation baselines",
        colorbar_label="fraction of distinct-target pairs",
        path=run_dir / "clean_baseline_accuracy.png",
        score_map=False,
    )

    rows: list[dict[str, Any]] = []
    for loop_index, state_loop in enumerate(state_loops):
        for site_index, site_label in enumerate(site_labels):
            rows.append(
                {
                    "model": name,
                    "behavior": behavior,
                    "state_loop": state_loop,
                    "effective_update_loop": state_loop + 1,
                    "site": site_label,
                    "candidate_count": int(candidate_count[loop_index]),
                    "valid_count": int(valid_count[loop_index]),
                    "clean_a_accuracy": float(clean_a_accuracy[loop_index]),
                    "clean_b_accuracy": float(clean_b_accuracy[loop_index]),
                    "valid_fraction": float(valid_fraction[loop_index]),
                    "patch_in_score": float(patch_in_score[loop_index, site_index]),
                    "patch_out_score": float(patch_out_score[loop_index, site_index]),
                    "shuffled_score": float(shuffled_score[loop_index, site_index]),
                    "shuffled_target_a_accuracy": float(
                        shuffled_target_a_accuracy[loop_index, site_index]
                    ),
                    "patch_in_target_a_accuracy": float(
                        patch_in_target_a_accuracy[loop_index, site_index]
                    ),
                    "patch_out_target_b_accuracy": float(
                        patch_out_target_b_accuracy[loop_index, site_index]
                    ),
                }
            )
    _write_csv(run_dir / "component_patch_rows.csv", rows)

    summary = {
        "name": name,
        "checkpoint": str(checkpoint),
        "checkpoint_step": checkpoint_data.get("step"),
        "loss_mode": checkpoint_data.get("loss_mode", ""),
        "behavior": behavior,
        "config": asdict(cfg),
        "seed": seed,
        "examples": batch_size * batches,
        "state_loops": state_loops,
        "effective_update_loops": [loop + 1 for loop in state_loops],
        "site_labels": site_labels,
        "candidate_count": candidate_count.tolist(),
        "valid_count": valid_count.tolist(),
        "clean_a_accuracy": clean_a_accuracy.tolist(),
        "clean_b_accuracy": clean_b_accuracy.tolist(),
        "valid_fraction": valid_fraction.tolist(),
        "patch_in_score": patch_in_score.tolist(),
        "patch_out_score": patch_out_score.tolist(),
        "shuffled_score": shuffled_score.tolist(),
        "shuffled_target_a_accuracy": shuffled_target_a_accuracy.tolist(),
        "patch_in_target_a_accuracy": patch_in_target_a_accuracy.tolist(),
        "patch_out_target_b_accuracy": patch_out_target_b_accuracy.tolist(),
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def aggregate_component_summaries(
    summaries: dict[str, dict[str, Any]],
    *,
    out_dir: Path,
) -> dict[str, Any]:
    """Aggregate checkpoint summaries whose names end in ``_seedN``."""
    grouped: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for name, summary in summaries.items():
        match = re.fullmatch(r"(.+)_seed(\d+)", name)
        if match is None:
            continue
        grouped.setdefault(match.group(1), []).append((int(match.group(2)), summary))
    if not grouped:
        raise ValueError("no NAME_seedN summaries were provided")

    out_dir.mkdir(parents=True, exist_ok=True)
    family_output: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    array_keys = (
        "patch_in_score",
        "patch_out_score",
        "shuffled_score",
        "patch_in_target_a_accuracy",
        "patch_out_target_b_accuracy",
        "shuffled_target_a_accuracy",
    )
    for family, seeded_summaries in sorted(grouped.items()):
        seeded_summaries.sort(key=lambda item: item[0])
        seeds = [seed for seed, _ in seeded_summaries]
        members = [summary for _, summary in seeded_summaries]
        reference = members[0]
        state_loops = reference["state_loops"]
        site_labels = reference["site_labels"]
        for member in members[1:]:
            if member["state_loops"] != state_loops or member["site_labels"] != site_labels:
                raise ValueError(f"family {family!r} has incompatible axes")

        stats: dict[str, dict[str, list[Any]]] = {}
        arrays: dict[str, np.ndarray] = {}
        for key in array_keys:
            arrays[key] = np.asarray([member[key] for member in members], dtype=float)
            mean, std = _nanmean_std(arrays[key])
            stats[key] = {
                "mean": mean.tolist(),
                "std": std.tolist(),
            }
        causal_specificity = arrays["patch_in_score"] - arrays["shuffled_score"]
        target_accuracy_gain = (
            arrays["patch_in_target_a_accuracy"] - arrays["shuffled_target_a_accuracy"]
        )
        specificity_mean, specificity_std = _nanmean_std(causal_specificity)
        accuracy_gain_mean, accuracy_gain_std = _nanmean_std(target_accuracy_gain)
        stats["causal_specificity"] = {
            "mean": specificity_mean.tolist(),
            "std": specificity_std.tolist(),
        }
        stats["target_accuracy_gain"] = {
            "mean": accuracy_gain_mean.tolist(),
            "std": accuracy_gain_std.tolist(),
        }

        specificity_mean = np.asarray(stats["causal_specificity"]["mean"])
        active_sites: list[str | None] = []
        for profile in specificity_mean:
            if np.isfinite(profile).any():
                active_sites.append(site_labels[int(np.nanargmax(profile))])
            else:
                active_sites.append(None)
        nonnull_active = [site for site in active_sites if site is not None]
        agreement = 0.0
        if nonnull_active:
            agreement = max(nonnull_active.count(site) for site in set(nonnull_active)) / len(
                nonnull_active
            )

        cosine_values: list[float] = []
        finite_profiles = np.nan_to_num(specificity_mean, nan=0.0)
        for first in range(len(state_loops)):
            for second in range(first + 1, len(state_loops)):
                denominator = np.linalg.norm(finite_profiles[first]) * np.linalg.norm(
                    finite_profiles[second]
                )
                if denominator > 0:
                    cosine_values.append(
                        float(np.dot(finite_profiles[first], finite_profiles[second]) / denominator)
                    )

        family_output[family] = {
            "behavior": reference["behavior"],
            "seeds": seeds,
            "seed_count": len(seeds),
            "state_loops": state_loops,
            "effective_update_loops": reference["effective_update_loops"],
            "site_labels": site_labels,
            "active_site_by_loop": active_sites,
            "active_site_agreement_fraction": agreement,
            "mean_loop_profile_cosine": (
                float(np.mean(cosine_values)) if cosine_values else float("nan")
            ),
            "valid_count_mean": np.mean(
                np.asarray([member["valid_count"] for member in members]), axis=0
            ).tolist(),
            "valid_fraction_mean": np.mean(
                np.asarray([member["valid_fraction"] for member in members]), axis=0
            ).tolist(),
            "metrics": stats,
        }

        row_labels = [f"L{loop} -> L{loop + 1}" for loop in state_loops]
        for values, title, label, filename in (
            (
                np.asarray(stats["patch_in_score"]["mean"]),
                f"{family}: patch-in score, mean across seeds",
                "normalized A recovery",
                f"{family}_patch_in_mean.png",
            ),
            (
                specificity_mean,
                f"{family}: patch-in minus shuffled, mean across seeds",
                "causal specificity",
                f"{family}_causal_specificity_mean.png",
            ),
            (
                np.asarray(stats["target_accuracy_gain"]["mean"]),
                f"{family}: target accuracy gain over shuffled",
                "accuracy gain",
                f"{family}_target_accuracy_gain_mean.png",
            ),
        ):
            _save_component_heatmap(
                values,
                row_labels=row_labels,
                col_labels=site_labels,
                title=title,
                colorbar_label=label,
                path=out_dir / filename,
                score_map=True,
            )

        for loop_index, state_loop in enumerate(state_loops):
            for site_index, site_label in enumerate(site_labels):
                row: dict[str, Any] = {
                    "family": family,
                    "behavior": reference["behavior"],
                    "seed_count": len(seeds),
                    "state_loop": state_loop,
                    "effective_update_loop": state_loop + 1,
                    "site": site_label,
                }
                for key in (*array_keys, "causal_specificity", "target_accuracy_gain"):
                    row[f"{key}_mean"] = stats[key]["mean"][loop_index][site_index]
                    row[f"{key}_std"] = stats[key]["std"][loop_index][site_index]
                rows.append(row)

    output = {"families": family_output}
    (out_dir / "family_summary.json").write_text(
        json.dumps(output, indent=2),
        encoding="utf-8",
    )
    _write_csv(out_dir / "family_component_rows.csv", rows)
    return output


def _parse_run_spec(text: str) -> tuple[str, Literal["transition", "endpoint"], Path]:
    if "=" not in text:
        raise ValueError(f"run spec must be NAME:BEHAVIOR=/path/to/checkpoint.pt, got {text!r}")
    descriptor, path = text.split("=", 1)
    if ":" not in descriptor:
        raise ValueError("run descriptor must contain :transition or :endpoint")
    name, behavior = descriptor.rsplit(":", 1)
    if not name or behavior not in {"transition", "endpoint"}:
        raise ValueError(f"invalid run descriptor: {descriptor!r}")
    return name, behavior, Path(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Causally localize graph-path updates to shared attention/MLP branches."
    )
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="NAME:transition=/path.pt or NAME:endpoint=/path.pt",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--state-loops", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--batches", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)
    summaries: dict[str, Any] = {}
    for name, behavior, checkpoint in [_parse_run_spec(text) for text in args.run]:
        summaries[name] = analyze_component_checkpoint(
            name=name,
            checkpoint=checkpoint,
            out_dir=args.out_dir,
            device=device,
            state_loops=args.state_loops,
            behavior=behavior,
            batch_size=args.batch_size,
            batches=args.batches,
            seed=args.seed,
        )
        print(f"done {name}", flush=True)
    (args.out_dir / "combined_summary.json").write_text(
        json.dumps({"models": summaries}, indent=2),
        encoding="utf-8",
    )
    if all(re.fullmatch(r".+_seed\d+", name) for name in summaries):
        aggregate_component_summaries(summaries, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
