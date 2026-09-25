"""Continue an age-specific J bank with CE-only path-equivalence curricula.

Every graph starts from the natural H1 residual.  A word over F (+1) and J
(-1) is constrained to H1..H8 and ends at H8.  All words with the same number
of rollback calls have the same task semantics: they advance the graph by the
same number of F calls and return to the trained readout phase.  Training uses
only final graph cross entropy; no hidden-state target is used.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as torch_functional

from reasoning_loop.graph_path_depth_circuit import fixed_depth_batch, load_checkpoint
from reasoning_loop.graph_path_loop import pick_device, set_seed
from reasoning_loop.graph_path_telomere_overloop import advance_nodes
from reasoning_loop.graph_path_temporal_intervention import logits_from_raw_state
from reasoning_loop.train_graph_path_age_specific_j_bank import (
    MAX_AGE,
    MIN_AGE,
    ROLLBACK_SOURCE_AGES,
    AgeSpecificJBank,
    _learning_rate_scale,
)


@dataclass(frozen=True)
class ActionSemantics:
    start_age: int
    end_age: int
    forward_count: int
    back_count: int
    age_path: tuple[int, ...]
    rollback_sources: tuple[int, ...]
    max_rollback_run: int


@dataclass(frozen=True)
class EquivalenceStage:
    name: str
    steps: int
    mode: str
    back_counts: tuple[int, ...]
    learning_rate: float
    data_seed: int


DEFAULT_STAGES = (
    EquivalenceStage("local_single", 280, "single", (1,), 5e-6, 828101),
    EquivalenceStage("commuting_squares", 420, "commute", (1,), 5e-6, 828201),
    EquivalenceStage("equivalent_k2", 700, "equivalent", (2,), 4e-6, 828301),
    EquivalenceStage("equivalent_k2_to_k5", 1400, "equivalent", (2, 3, 4, 5), 3e-6, 828401),
    EquivalenceStage(
        "equivalent_k3_to_k12",
        2200,
        "equivalent",
        tuple(range(3, 13)),
        2e-6,
        828501,
    ),
)


def _inverse_reuse_back_counts(minimum: int, maximum: int) -> tuple[int, ...]:
    """Return a deterministic p(k) proportional to 1/k curriculum cycle."""
    if not 1 <= minimum <= maximum:
        raise ValueError("invalid inverse-reuse range")
    return tuple(
        back_count
        for back_count in range(minimum, maximum + 1)
        for _ in range(max(1, round(maximum / back_count)))
    )


EXTENSION_STAGES = (
    EquivalenceStage(
        "equivalent_inverse_k3_to_k16",
        3000,
        "equivalent",
        _inverse_reuse_back_counts(3, 16),
        1e-6,
        830101,
    ),
    EquivalenceStage(
        "equivalent_inverse_k3_to_k24",
        5000,
        "equivalent",
        _inverse_reuse_back_counts(3, 24),
        7e-7,
        830201,
    ),
)


AVAILABLE_STAGES = DEFAULT_STAGES + EXTENSION_STAGES


def action_semantics(
    actions: Iterable[int],
    *,
    start_age: int = 1,
) -> ActionSemantics:
    if not MIN_AGE <= start_age <= MAX_AGE:
        raise ValueError("start age must lie in H1..H8")
    age = start_age
    age_path = [age]
    rollback_sources: list[int] = []
    forwards = backs = rollback_run = maximum_run = 0
    for action in actions:
        if action == 1:
            forwards += 1
            rollback_run = 0
        elif action == -1:
            backs += 1
            rollback_sources.append(age)
            rollback_run += 1
            maximum_run = max(maximum_run, rollback_run)
        else:
            raise ValueError("actions must be +1 (F) or -1 (J)")
        age += action
        if not MIN_AGE <= age <= MAX_AGE:
            raise ValueError("action word leaves H1..H8")
        age_path.append(age)
    return ActionSemantics(
        start_age=start_age,
        end_age=age,
        forward_count=forwards,
        back_count=backs,
        age_path=tuple(age_path),
        rollback_sources=tuple(rollback_sources),
        max_rollback_run=maximum_run,
    )


def single_rollback_word_from_h1(source_age: int) -> tuple[int, ...]:
    if source_age not in ROLLBACK_SOURCE_AGES:
        raise ValueError("source age must lie in H2..H8")
    actions = (
        (1,) * (source_age - 1)
        + (-1,)
        + (1,) * (MAX_AGE - source_age + 1)
    )
    semantics = action_semantics(actions)
    if semantics.end_age != MAX_AGE or semantics.rollback_sources != (source_age,):
        raise RuntimeError("single-J word construction failed")
    return actions


def commuting_pair_from_h1(anchor_age: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return J_a,F and F,J_(a+1), both completed from H1 to H8."""

    if not 2 <= anchor_age <= 7:
        raise ValueError("commuting-square anchor must lie in H2..H7")
    prefix = (1,) * (anchor_age - 1)
    completion = (1,) * (MAX_AGE - anchor_age)
    left = prefix + (-1, 1) + completion
    right = prefix + (1, -1) + completion
    left_semantics = action_semantics(left)
    right_semantics = action_semantics(right)
    if (
        left_semantics.end_age != MAX_AGE
        or right_semantics.end_age != MAX_AGE
        or left_semantics.forward_count != right_semantics.forward_count
        or left_semantics.back_count != right_semantics.back_count
    ):
        raise RuntimeError("commuting-square words are not semantically equivalent")
    return left, right


def _sample_bounded_word(
    *,
    rng: np.random.Generator,
    back_count: int,
    mandatory_source_age: int | None,
) -> tuple[int, ...]:
    if back_count < 1:
        raise ValueError("back count must be positive")
    if (
        mandatory_source_age is not None
        and mandatory_source_age not in ROLLBACK_SOURCE_AGES
    ):
        raise ValueError("mandatory source age must lie in H2..H8")
    up_count = back_count + (MAX_AGE - MIN_AGE)

    @lru_cache(maxsize=None)
    def count(age: int, ups: int, downs: int, seen: bool) -> int:
        if ups == 0 and downs == 0:
            return int(
                age == MAX_AGE
                and (mandatory_source_age is None or seen)
            )
        total = 0
        if ups and age < MAX_AGE:
            total += count(age + 1, ups - 1, downs, seen)
        if downs and age > MIN_AGE:
            total += count(
                age - 1,
                ups,
                downs - 1,
                seen or age == mandatory_source_age,
            )
        return total

    if count(MIN_AGE, up_count, back_count, False) == 0:
        raise RuntimeError("no bounded word satisfies the requested stratum")
    age, ups, downs, seen = MIN_AGE, up_count, back_count, False
    actions: list[int] = []
    while ups or downs:
        choices: list[tuple[int, int, int, int, int, bool]] = []
        if ups and age < MAX_AGE:
            ways = count(age + 1, ups - 1, downs, seen)
            if ways:
                choices.append((1, ways, age + 1, ups - 1, downs, seen))
        if downs and age > MIN_AGE:
            next_seen = seen or age == mandatory_source_age
            ways = count(age - 1, ups, downs - 1, next_seen)
            if ways:
                choices.append((-1, ways, age - 1, ups, downs - 1, next_seen))
        total = sum(choice[1] for choice in choices)
        if total < 2**63:
            draw = int(rng.integers(0, total))
        else:
            draw = min(total - 1, int(rng.random() * total))
        cumulative = 0
        selected = choices[-1]
        for choice in choices:
            cumulative += choice[1]
            if draw < cumulative:
                selected = choice
                break
        action, _, age, ups, downs, seen = selected
        actions.append(action)
    result = tuple(actions)
    semantics = action_semantics(result)
    if semantics.end_age != MAX_AGE or semantics.back_count != back_count:
        raise RuntimeError("bounded-word sampler returned invalid semantics")
    if (
        mandatory_source_age is not None
        and mandatory_source_age not in semantics.rollback_sources
    ):
        raise RuntimeError("bounded-word sampler missed mandatory J")
    return result


def sample_equivalent_word_pair(
    *,
    rng: np.random.Generator,
    back_count: int,
    mandatory_source_age: int | None = None,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    left = _sample_bounded_word(
        rng=rng,
        back_count=back_count,
        mandatory_source_age=mandatory_source_age,
    )
    for _ in range(128):
        right = _sample_bounded_word(
            rng=rng,
            back_count=back_count,
            mandatory_source_age=mandatory_source_age,
        )
        if right != left:
            break
    else:
        raise RuntimeError("could not sample two distinct equivalent words")
    left_semantics = action_semantics(left)
    right_semantics = action_semantics(right)
    if (
        left_semantics.end_age != right_semantics.end_age
        or left_semantics.forward_count != right_semantics.forward_count
        or left_semantics.back_count != right_semantics.back_count
    ):
        raise RuntimeError("sampled word pair is not semantically equivalent")
    return left, right


def _word_text(actions: Sequence[int]) -> str:
    return "".join("F" if action == 1 else "J" for action in actions)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _natural_h1(model, tokens: torch.Tensor) -> torch.Tensor:
    raw = model.token_embed(tokens) + model.pos_embed.unsqueeze(0)
    return model.apply_loop(raw, loop_index=0)


def _execute_word(
    *,
    model,
    bank: AgeSpecificJBank,
    initial_state: torch.Tensor,
    initial_current: torch.Tensor,
    successors: torch.Tensor,
    actions: Sequence[int],
    positions: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    state = initial_state
    current = initial_current
    logical_age = MIN_AGE
    for action in actions:
        if action == 1:
            state = model.apply_loop(state, loop_index=logical_age)
            current = advance_nodes(successors, current, steps=1)
            logical_age += 1
        else:
            state = bank.rollback(
                state,
                source_age=logical_age,
                positions=positions,
            )
            logical_age -= 1
        if not MIN_AGE <= logical_age <= MAX_AGE:
            raise RuntimeError("executor left H1..H8")
    if logical_age != MAX_AGE:
        raise RuntimeError("training word did not end at H8")
    return state, logits_from_raw_state(model, state), current


def _stage_specific_gradient_norms(bank: AgeSpecificJBank) -> dict[int, float]:
    norms: dict[int, float] = {}
    if bank.map_architecture == "shared_diagonal_stage_lora":
        for age in ROLLBACK_SOURCE_AGES:
            parameters = (bank.stage_A[str(age)], bank.stage_B[str(age)])
            square_sum = sum(
                parameter.grad.float().square().sum()
                for parameter in parameters
                if parameter.grad is not None
            )
            norms[age] = float(square_sum.sqrt()) if not isinstance(square_sum, int) else 0.0
        return norms
    for age in ROLLBACK_SOURCE_AGES:
        square_sum = sum(
            parameter.grad.float().square().sum()
            for parameter in bank.maps[str(age)].parameters()
            if parameter.grad is not None
        )
        norms[age] = float(square_sum.sqrt()) if not isinstance(square_sum, int) else 0.0
    return norms


def _balance_stage_specific_gradients(
    bank: AgeSpecificJBank,
    *,
    maximum_scale: float,
) -> tuple[dict[int, float], dict[int, float], dict[int, float]]:
    before = _stage_specific_gradient_norms(bank)
    positive = [value for value in before.values() if value > 0.0]
    if len(positive) != len(ROLLBACK_SOURCE_AGES):
        raise RuntimeError(f"not every stage received gradient: {before}")
    target = float(np.exp(np.mean(np.log(np.maximum(positive, 1e-30)))))
    scales: dict[int, float] = {}
    for age in ROLLBACK_SOURCE_AGES:
        scale = float(np.clip(target / before[age], 1.0 / maximum_scale, maximum_scale))
        scales[age] = scale
        parameters = (
            (bank.stage_A[str(age)], bank.stage_B[str(age)])
            if bank.map_architecture == "shared_diagonal_stage_lora"
            else tuple(bank.maps[str(age)].parameters())
        )
        for parameter in parameters:
            if parameter.grad is not None:
                parameter.grad.mul_(scale)
    return before, _stage_specific_gradient_norms(bank), scales


def _optimizer(
    bank: AgeSpecificJBank,
    *,
    learning_rate: float,
    diagonal_lr_multiplier: float,
) -> torch.optim.Optimizer:
    diagonal_parameters = []
    other_parameters = []
    for name, parameter in bank.named_parameters():
        if name.endswith("diagonal_scale"):
            diagonal_parameters.append(parameter)
        else:
            other_parameters.append(parameter)
    groups: list[dict[str, Any]] = [{"params": other_parameters}]
    if diagonal_parameters:
        groups.append(
            {
                "params": diagonal_parameters,
                "lr": learning_rate * diagonal_lr_multiplier,
            }
        )
    return torch.optim.AdamW(groups, lr=learning_rate, weight_decay=0.0)


def _candidate_score(
    cumulative: dict[int, int],
    words: Sequence[Sequence[int]],
) -> tuple[float, int]:
    after = dict(cumulative)
    for word in words:
        for age in action_semantics(word).rollback_sources:
            after[age] += 1
    target = sum(after.values()) / len(after)
    return (
        sum((after[age] - target) ** 2 for age in ROLLBACK_SOURCE_AGES),
        max(after.values()) - min(after.values()),
    )


def _balanced_equivalent_words(
    *,
    rng: np.random.Generator,
    back_count: int,
    cumulative_calls: dict[int, int],
    candidates_per_age: int,
) -> list[tuple[int, ...]]:
    selected: list[tuple[int, ...]] = []
    temporary_counts = dict(cumulative_calls)
    for mandatory in rng.permutation(ROLLBACK_SOURCE_AGES):
        candidates = [
            sample_equivalent_word_pair(
                rng=np.random.default_rng(int(rng.integers(0, 2**63 - 1))),
                back_count=back_count,
                mandatory_source_age=int(mandatory),
            )
            for _ in range(candidates_per_age)
        ]
        pair = min(candidates, key=lambda value: _candidate_score(temporary_counts, value))
        selected.extend(pair)
        for word in pair:
            for age in action_semantics(word).rollback_sources:
                temporary_counts[age] += 1
    return selected


def _training_words(
    *,
    stage: EquivalenceStage,
    stage_step: int,
    rng: np.random.Generator,
    cumulative_calls: dict[int, int],
    candidates_per_age: int,
) -> tuple[list[tuple[int, ...]], int]:
    if stage.mode == "single":
        return [single_rollback_word_from_h1(age) for age in ROLLBACK_SOURCE_AGES], 1
    if stage.mode == "commute":
        words: list[tuple[int, ...]] = []
        for anchor_age in range(2, 8):
            words.extend(commuting_pair_from_h1(anchor_age))
        # The six squares use endpoint stages once and interior stages twice.
        # Add one endpoint word each so every J is called exactly twice.
        words.extend((single_rollback_word_from_h1(2), single_rollback_word_from_h1(8)))
        return words, 1
    back_count = stage.back_counts[(stage_step - 1) % len(stage.back_counts)]
    return (
        _balanced_equivalent_words(
            rng=rng,
            back_count=back_count,
            cumulative_calls=cumulative_calls,
            candidates_per_age=candidates_per_age,
        ),
        back_count,
    )


def _save_bank(
    path: Path,
    *,
    source_payload: dict[str, Any],
    bank: AgeSpecificJBank,
    stages: Sequence[EquivalenceStage],
    completed_stages: int,
    args: argparse.Namespace,
) -> None:
    payload = {
        **{
            key: value
            for key, value in source_payload.items()
            if key != "state_dict"
        },
        "kind": "graph_path_age_specific_J_bank_path_equivalence_continuation",
        "parent_bank_artifact": str(args.bank_init_artifact),
        "training_loss": "final H8 graph CE only; no hidden-state loss and no KL",
        "training_start_age": 1,
        "curriculum": "path_equivalence",
        "path_equivalence_law": "F after J_a and J_(a+1) after F have equal task semantics",
        "gradient_balance": "per-stage branch gradients normalized toward their geometric mean",
        "gradient_balance_maximum_scale": args.gradient_balance_maximum_scale,
        "diagonal_lr_multiplier": args.diagonal_lr_multiplier,
        "stages": [asdict(stage) for stage in stages],
        "completed_stages": completed_stages,
        "state_dict": {
            key: value.detach().cpu()
            for key, value in bank.state_dict().items()
        },
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


@torch.no_grad()
def evaluate_fixed_suite(
    *,
    model,
    cfg,
    bank: AgeSpecificJBank,
    positions: tuple[int, ...],
    device: torch.device,
    seed: int,
    examples: int,
    batch_size: int,
    pair_count: int,
    path_generation_prefix_back_counts: tuple[int, ...] = (),
    long_back_counts: tuple[int, ...],
    label: str,
) -> list[dict[str, Any]]:
    if examples % batch_size:
        raise ValueError("evaluation examples must divide by batch size")
    set_seed(seed)
    specifications: list[tuple[str, str, tuple[int, ...], tuple[int, ...] | None]] = []
    for source_age in ROLLBACK_SOURCE_AGES:
        specifications.append(
            ("single_J", f"J{source_age - 1}", single_rollback_word_from_h1(source_age), None)
        )
    for anchor_age in range(2, 8):
        left, right = commuting_pair_from_h1(anchor_age)
        specifications.append(("commuting_square", f"H{anchor_age}_JF_vs_FJ", left, right))
    rng = np.random.default_rng(seed + 17)
    for back_count in path_generation_prefix_back_counts:
        for pair_index in range(pair_count):
            mandatory = ROLLBACK_SOURCE_AGES[pair_index % len(ROLLBACK_SOURCE_AGES)]
            sample_equivalent_word_pair(
                rng=rng,
                back_count=back_count,
                mandatory_source_age=mandatory,
            )
    for back_count in long_back_counts:
        for pair_index in range(pair_count):
            mandatory = ROLLBACK_SOURCE_AGES[pair_index % len(ROLLBACK_SOURCE_AGES)]
            left, right = sample_equivalent_word_pair(
                rng=rng,
                back_count=back_count,
                mandatory_source_age=mandatory,
            )
            specifications.append(
                (
                    "equivalent_words",
                    f"k{back_count}_pair{pair_index}_J{mandatory - 1}",
                    left,
                    right,
                )
            )
    accumulators: dict[tuple[str, str], dict[str, float]] = {}
    bank.eval()
    for batch_index in range(examples // batch_size):
        tokens, _, successors, start = fixed_depth_batch(
            cfg,
            batch_size,
            device,
            path_positions=cfg.max_depth,
        )
        h1 = _natural_h1(model, tokens)
        h1_current = advance_nodes(successors, start, steps=1)
        for family, name, left, right in specifications:
            _, left_logits, left_target = _execute_word(
                model=model,
                bank=bank,
                initial_state=h1,
                initial_current=h1_current,
                successors=successors,
                actions=left,
                positions=positions,
            )
            left_predictions = left_logits.argmax(-1)
            left_correct = float(left_predictions.eq(left_target).sum())
            slot = accumulators.setdefault(
                (family, name),
                {
                    "left_correct": 0.0,
                    "right_correct": 0.0,
                    "agreement": 0.0,
                    "count": 0.0,
                    "left_loss": 0.0,
                    "right_loss": 0.0,
                },
            )
            slot["left_correct"] += left_correct
            slot["left_loss"] += float(
                torch_functional.cross_entropy(left_logits.float(), left_target, reduction="sum")
            )
            if right is None:
                right_logits = left_logits
                right_target = left_target
            else:
                _, right_logits, right_target = _execute_word(
                    model=model,
                    bank=bank,
                    initial_state=h1,
                    initial_current=h1_current,
                    successors=successors,
                    actions=right,
                    positions=positions,
                )
                if not right_target.eq(left_target).all():
                    raise RuntimeError("equivalent paths produced different graph targets")
            right_predictions = right_logits.argmax(-1)
            slot["right_correct"] += float(right_predictions.eq(right_target).sum())
            slot["right_loss"] += float(
                torch_functional.cross_entropy(right_logits.float(), right_target, reduction="sum")
            )
            slot["agreement"] += float(left_predictions.eq(right_predictions).sum())
            slot["count"] += batch_size
    rows: list[dict[str, Any]] = []
    for family, name, left, right in specifications:
        slot = accumulators[(family, name)]
        count = slot["count"]
        left_semantics = action_semantics(left)
        rows.append(
            {
                "label": label,
                "family": family,
                "name": name,
                "back_count": left_semantics.back_count,
                "forward_count": left_semantics.forward_count,
                "left_accuracy": slot["left_correct"] / count,
                "right_accuracy": slot["right_correct"] / count,
                "prediction_agreement": slot["agreement"] / count,
                "left_ce": slot["left_loss"] / count,
                "right_ce": slot["right_loss"] / count,
                "left_word": _word_text(left),
                "right_word": _word_text(right) if right is not None else "",
                "examples": int(count),
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--phase-summary", type=Path, required=True)
    parser.add_argument("--bank-init-artifact", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=828001)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--stage-step-limit", type=int)
    parser.add_argument(
        "--stage-names",
        nargs="+",
        choices=tuple(stage.name for stage in AVAILABLE_STAGES),
        help="Run only the named curriculum stages, preserving their canonical order.",
    )
    parser.add_argument("--learning-rate-multiplier", type=float, default=1.0)
    parser.add_argument("--diagonal-lr-multiplier", type=float, default=0.1)
    parser.add_argument("--warmup-fraction", type=float, default=0.1)
    parser.add_argument("--warmup-start-factor", type=float, default=0.1)
    parser.add_argument("--decay-fraction", type=float, default=0.1)
    parser.add_argument("--decay-end-factor", type=float, default=0.1)
    parser.add_argument("--gradient-balance-maximum-scale", type=float, default=4.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--candidates-per-age", type=int, default=6)
    parser.add_argument("--eval-examples", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--eval-pairs", type=int, default=7)
    parser.add_argument(
        "--eval-back-counts",
        type=int,
        nargs="+",
        default=(2, 3, 5, 8, 12, 16, 24),
    )
    parser.add_argument("--skip-initial-eval", action="store_true")
    parser.add_argument("--cuda-memory-fraction", type=float, default=0.02)
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    if args.stage_step_limit is not None and args.stage_step_limit < 1:
        raise ValueError("stage step limit must be positive")
    if not 0.0 <= args.warmup_fraction <= 1.0:
        raise ValueError("warmup fraction must lie in [0,1]")
    if not 0.0 <= args.decay_fraction <= 1.0:
        raise ValueError("decay fraction must lie in [0,1]")
    if args.warmup_fraction + args.decay_fraction > 1.0:
        raise ValueError("warmup and decay fractions cannot overlap")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_dir / "run_manifest.json"
    manifest = {
        "status": "running",
        "pid": os.getpid(),
        "physical_cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "started_unix_time": time.time(),
        "command": "train_graph_path_j_path_equivalence",
        "checkpoint": str(args.checkpoint),
        "bank_init_artifact": str(args.bank_init_artifact),
        "out_dir": str(args.out_dir),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    device = pick_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(args.cuda_memory_fraction)
        torch.cuda.reset_peak_memory_stats()
    set_seed(args.seed)
    model, cfg, checkpoint_payload = load_checkpoint(args.checkpoint, device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if cfg.block_schedule != "all_blocks":
        raise ValueError("this audit-locked continuation expects all_blocks")
    phase_payload = json.loads(args.phase_summary.read_text(encoding="utf-8"))
    if phase_payload["trajectory_positions_including_initial"] != list(range(9)):
        raise ValueError("phase summary must identify natural H0..H8")
    source_payload = torch.load(
        args.bank_init_artifact,
        map_location="cpu",
        weights_only=False,
    )
    if source_payload.get("checkpoint") != str(args.checkpoint):
        raise ValueError("J bank and backbone checkpoint differ")
    bank = AgeSpecificJBank(
        dimension=cfg.d_model,
        rank=int(source_payload["rank"]),
        stage_rank=int(source_payload.get("stage_rank", source_payload["rank"])),
        map_architecture=str(source_payload["map_architecture"]),
    ).to(device)
    bank.load_state_dict(source_payload["state_dict"])
    positions = tuple(range(cfg.seq_len))
    selected_stage_names = (
        None if args.stage_names is None else set(args.stage_names)
    )
    stage_catalog = DEFAULT_STAGES if selected_stage_names is None else AVAILABLE_STAGES
    stages = tuple(
        EquivalenceStage(
            name=stage.name,
            steps=(
                min(stage.steps, args.stage_step_limit)
                if args.stage_step_limit is not None
                else stage.steps
            ),
            mode=stage.mode,
            back_counts=stage.back_counts,
            learning_rate=stage.learning_rate * args.learning_rate_multiplier,
            data_seed=stage.data_seed,
        )
        for stage in stage_catalog
        if selected_stage_names is None or stage.name in selected_stage_names
    )
    if not stages:
        raise ValueError("no curriculum stage was selected")

    evaluation_rows: list[dict[str, Any]] = []
    if not args.skip_initial_eval:
        evaluation_rows.extend(
            evaluate_fixed_suite(
                model=model,
                cfg=cfg,
                bank=bank,
                positions=positions,
                device=device,
                seed=args.seed + 900,
                examples=args.eval_examples,
                batch_size=args.eval_batch_size,
                pair_count=args.eval_pairs,
                long_back_counts=tuple(args.eval_back_counts),
                label="before",
            )
        )
        _write_csv(args.out_dir / "fixed_evaluation.csv", evaluation_rows)

    training_rows: list[dict[str, Any]] = []
    global_calls = {age: 0 for age in ROLLBACK_SOURCE_AGES}
    for stage_index, stage in enumerate(stages, start=1):
        optimizer = _optimizer(
            bank,
            learning_rate=stage.learning_rate,
            diagonal_lr_multiplier=args.diagonal_lr_multiplier,
        )
        target_lrs = [float(group["lr"]) for group in optimizer.param_groups]
        bank.train()
        stage_started = time.monotonic()
        for stage_step in range(1, stage.steps + 1):
            lr_scale = _learning_rate_scale(
                step=stage_step,
                total_steps=stage.steps,
                schedule="wsd",
                warmup_fraction=args.warmup_fraction,
                warmup_start_factor=args.warmup_start_factor,
                decay_fraction=args.decay_fraction,
                decay_end_factor=args.decay_end_factor,
            )
            for group, target_lr in zip(optimizer.param_groups, target_lrs, strict=True):
                group["lr"] = target_lr * lr_scale
            rng = np.random.default_rng(stage.data_seed + stage_step)
            words, back_count = _training_words(
                stage=stage,
                stage_step=stage_step,
                rng=rng,
                cumulative_calls=global_calls,
                candidates_per_age=args.candidates_per_age,
            )
            tokens, _, successors, start = fixed_depth_batch(
                cfg,
                args.batch_size,
                device,
                path_positions=cfg.max_depth,
            )
            with torch.no_grad():
                h1 = _natural_h1(model, tokens)
                h1_current = advance_nodes(successors, start, steps=1)
            optimizer.zero_grad(set_to_none=True)
            loss_sum = 0.0
            correct_sum = 0
            target_reference: torch.Tensor | None = None
            for word in words:
                state, logits, target = _execute_word(
                    model=model,
                    bank=bank,
                    initial_state=h1,
                    initial_current=h1_current,
                    successors=successors,
                    actions=word,
                    positions=positions,
                )
                del state
                if target_reference is None:
                    target_reference = target
                elif not target.eq(target_reference).all():
                    raise RuntimeError("same-stratum words produced different targets")
                loss = torch_functional.cross_entropy(logits.float(), target)
                (loss / len(words)).backward()
                loss_sum += float(loss.detach())
                correct_sum += int(logits.argmax(-1).eq(target).sum())
            before_grad, after_grad, grad_scales = _balance_stage_specific_gradients(
                bank,
                maximum_scale=args.gradient_balance_maximum_scale,
            )
            global_grad_norm = torch.nn.utils.clip_grad_norm_(
                bank.parameters(), args.grad_clip
            )
            optimizer.step()
            step_calls = {age: 0 for age in ROLLBACK_SOURCE_AGES}
            for word in words:
                for age in action_semantics(word).rollback_sources:
                    step_calls[age] += 1
                    global_calls[age] += 1
            row: dict[str, Any] = {
                "stage": stage.name,
                "stage_index": stage_index,
                "stage_step": stage_step,
                "mode": stage.mode,
                "back_count": back_count,
                "word_count": len(words),
                "batch_size_per_word": args.batch_size,
                "loss": loss_sum / len(words),
                "accuracy": correct_sum / (len(words) * args.batch_size),
                "learning_rate": optimizer.param_groups[0]["lr"],
                "lr_scale": lr_scale,
                "global_grad_norm": float(global_grad_norm),
                "call_spread_global": max(global_calls.values()) - min(global_calls.values()),
            }
            for age in ROLLBACK_SOURCE_AGES:
                user_j = age - 1
                row[f"J{user_j}_calls_step"] = step_calls[age]
                row[f"J{user_j}_calls_global"] = global_calls[age]
                row[f"J{user_j}_grad_before"] = before_grad[age]
                row[f"J{user_j}_grad_after"] = after_grad[age]
                row[f"J{user_j}_grad_scale"] = grad_scales[age]
            training_rows.append(row)
            if stage_step == 1 or stage_step % max(1, stage.steps // 10) == 0:
                print(
                    json.dumps(
                        {
                            "stage": stage.name,
                            "stage_step": stage_step,
                            "steps": stage.steps,
                            "loss": row["loss"],
                            "accuracy": row["accuracy"],
                            "back_count": back_count,
                            "global_calls": global_calls,
                            "global_grad_norm": row["global_grad_norm"],
                            "elapsed_seconds": time.monotonic() - stage_started,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
        _write_csv(args.out_dir / "training.csv", training_rows)
        _save_bank(
            args.out_dir / "age_specific_j_bank.pt",
            source_payload=source_payload,
            bank=bank,
            stages=stages,
            completed_stages=stage_index,
            args=args,
        )
        _save_bank(
            args.out_dir / f"age_specific_j_bank_after_{stage.name}.pt",
            source_payload=source_payload,
            bank=bank,
            stages=stages,
            completed_stages=stage_index,
            args=args,
        )
        evaluation_rows.extend(
            evaluate_fixed_suite(
                model=model,
                cfg=cfg,
                bank=bank,
                positions=positions,
                device=device,
                seed=args.seed + 900,
                examples=args.eval_examples,
                batch_size=args.eval_batch_size,
                pair_count=args.eval_pairs,
                long_back_counts=tuple(args.eval_back_counts),
                label=f"after_{stage.name}",
            )
        )
        _write_csv(args.out_dir / "fixed_evaluation.csv", evaluation_rows)

    summary = {
        "status": "complete",
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": checkpoint_payload.get("step"),
        "parent_bank_artifact": str(args.bank_init_artifact),
        "bank_architecture": bank.map_architecture,
        "rank": bank.rank,
        "stage_rank": bank.stage_rank,
        "parameter_count": bank.parameter_count,
        "training_loss": "final graph CE at semantic H8 only; hidden-state loss 0; KL 0",
        "initial_state": "natural H1 from raw tokens",
        "curriculum": [asdict(stage) for stage in stages],
        "global_J_calls_by_source_age": global_calls,
        "global_J_call_spread": max(global_calls.values()) - min(global_calls.values()),
        "gradient_balance": {
            "scope": "stage-specific A_i/B_i or per-map parameters",
            "target": "geometric mean per optimizer step",
            "maximum_scale": args.gradient_balance_maximum_scale,
        },
        "fixed_evaluation_rows": len(evaluation_rows),
        "peak_cuda_allocated_mib": (
            torch.cuda.max_memory_allocated() / 1024**2
            if device.type == "cuda"
            else None
        ),
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest.update(
        {
            "status": "complete",
            "completed_unix_time": time.time(),
            "peak_cuda_allocated_mib": summary["peak_cuda_allocated_mib"],
        }
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main(parse_args())
