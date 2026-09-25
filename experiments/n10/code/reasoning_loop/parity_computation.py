"""Pure utilities for locating and causally testing Parity computation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

import torch


ProbeTask = Literal["binary", "multiclass", "regression"]
COUNTERFACTUAL_NAMES = ("parity_flip", "count_shift", "pattern_control")


@dataclass(frozen=True)
class PrefixCounterfactualBatch:
    receiver: torch.Tensor
    parity_flip: torch.Tensor
    count_shift: torch.Tensor
    pattern_control: torch.Tensor
    valid: torch.Tensor
    changed_indices: torch.Tensor
    prefix_end: int

    def variant(self, name: str) -> torch.Tensor:
        if name not in COUNTERFACTUAL_NAMES:
            raise ValueError(f"unknown counterfactual family: {name}")
        return getattr(self, name)


@dataclass(frozen=True)
class RidgeProbe:
    task: ProbeTask
    weights: torch.Tensor
    target_mean: torch.Tensor
    feature_mean: torch.Tensor
    feature_scale: torch.Tensor
    penalty: float
    classes: torch.Tensor
    validation_score: float

    def decoder_rows(self) -> torch.Tensor:
        """Return decoder rows in the original representation coordinates."""

        raw = self.weights / self.feature_scale[:, None]
        return raw.T.contiguous()


def _choose_indices(candidates: torch.Tensor, count: int, generator: torch.Generator) -> torch.Tensor:
    if candidates.numel() < count:
        raise ValueError("not enough candidate indices")
    order = torch.randperm(candidates.numel(), generator=generator)
    return candidates[order[:count]]


def make_prefix_counterfactuals(
    bits: torch.Tensor,
    *,
    prefix_end: int,
    seed: int,
) -> PrefixCounterfactualBatch:
    """Create paired prefix interventions while preserving the registered factors.

    ``prefix_end`` is the current token and belongs to the prefix variable. All
    edits happen strictly before it, so the current bit and receiver suffix stay
    fixed.
    """

    if bits.ndim != 2 or bits.shape[1] < 2:
        raise ValueError("bits must have shape [batch, length>=2]")
    if bits.dtype == torch.bool:
        working = bits.long()
    else:
        working = bits.detach().to(dtype=torch.long)
    if not torch.all((working == 0) | (working == 1)):
        raise ValueError("bits must be binary")
    if not 1 <= prefix_end < working.shape[1]:
        raise ValueError("prefix_end must leave both an editable prefix and suffix boundary")

    generator = torch.Generator(device="cpu").manual_seed(seed)
    receiver = working.cpu().clone()
    variants = [receiver.clone() for _ in COUNTERFACTUAL_NAMES]
    valid = torch.zeros(receiver.shape[0], len(COUNTERFACTUAL_NAMES), dtype=torch.bool)
    changed = torch.full(
        (receiver.shape[0], len(COUNTERFACTUAL_NAMES), 2),
        -1,
        dtype=torch.long,
    )
    editable = torch.arange(prefix_end, dtype=torch.long)
    for row in range(receiver.shape[0]):
        row_bits = receiver[row]

        parity_index = _choose_indices(editable, 1, generator)
        variants[0][row, parity_index] = 1 - variants[0][row, parity_index]
        valid[row, 0] = True
        changed[row, 0, 0] = parity_index[0]

        zeros = editable[row_bits[:prefix_end] == 0]
        ones = editable[row_bits[:prefix_end] == 1]
        same_value_groups = [group for group in (zeros, ones) if group.numel() >= 2]
        if same_value_groups:
            group_index = int(
                torch.randint(len(same_value_groups), (1,), generator=generator)
            )
            chosen = _choose_indices(same_value_groups[group_index], 2, generator)
            variants[1][row, chosen] = 1 - variants[1][row, chosen]
            valid[row, 1] = True
            changed[row, 1] = chosen

        if zeros.numel() and ones.numel():
            zero = _choose_indices(zeros, 1, generator)
            one = _choose_indices(ones, 1, generator)
            chosen = torch.cat((zero, one))
            variants[2][row, chosen] = 1 - variants[2][row, chosen]
            valid[row, 2] = True
            changed[row, 2] = chosen

    output = PrefixCounterfactualBatch(
        receiver=receiver.to(bits.device),
        parity_flip=variants[0].to(bits.device),
        count_shift=variants[1].to(bits.device),
        pattern_control=variants[2].to(bits.device),
        valid=valid.to(bits.device),
        changed_indices=changed.to(bits.device),
        prefix_end=prefix_end,
    )
    validate_counterfactual_batch(output)
    return output


def validate_counterfactual_batch(batch: PrefixCounterfactualBatch) -> None:
    receiver = batch.receiver.long()
    if batch.valid.shape != (receiver.shape[0], 3):
        raise ValueError("valid mask must have shape [batch, 3]")
    if batch.changed_indices.shape != (receiver.shape[0], 3, 2):
        raise ValueError("changed_indices must have shape [batch, 3, 2]")
    prefix_stop = batch.prefix_end + 1
    for family_index, name in enumerate(COUNTERFACTUAL_NAMES):
        donor = batch.variant(name).long()
        if donor.shape != receiver.shape:
            raise ValueError(f"{name} shape does not match receiver")
        for row in torch.nonzero(batch.valid[:, family_index], as_tuple=False).flatten().tolist():
            changed = torch.nonzero(donor[row] != receiver[row], as_tuple=False).flatten()
            expected_changes = 1 if name == "parity_flip" else 2
            if changed.numel() != expected_changes:
                raise ValueError(f"{name} must change exactly {expected_changes} bits")
            if torch.any(changed >= batch.prefix_end):
                raise ValueError(f"{name} changed current bit or suffix")
            receiver_prefix_count = int(receiver[row, :prefix_stop].sum())
            donor_prefix_count = int(donor[row, :prefix_stop].sum())
            receiver_total_count = int(receiver[row].sum())
            donor_total_count = int(donor[row].sum())
            if name == "parity_flip":
                if donor_prefix_count % 2 == receiver_prefix_count % 2:
                    raise ValueError("parity_flip did not flip prefix parity")
                if donor_total_count % 2 == receiver_total_count % 2:
                    raise ValueError("parity_flip did not flip total parity")
            elif name == "count_shift":
                if abs(donor_prefix_count - receiver_prefix_count) != 2:
                    raise ValueError("count_shift must change prefix count by two")
                if donor_prefix_count % 2 != receiver_prefix_count % 2:
                    raise ValueError("count_shift changed prefix parity")
                if donor_total_count % 2 != receiver_total_count % 2:
                    raise ValueError("count_shift changed total parity")
            else:
                if donor_prefix_count != receiver_prefix_count:
                    raise ValueError("pattern_control changed prefix count")
                if donor_total_count != receiver_total_count:
                    raise ValueError("pattern_control changed total count")


def _standardize_fit(features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    values = features.detach().to(device="cpu", dtype=torch.float64)
    mean = values.mean(dim=0)
    scale = values.std(dim=0, unbiased=False).clamp_min(1e-8)
    return (values - mean) / scale, mean, scale


def _encode_targets(
    targets: torch.Tensor,
    task: ProbeTask,
    classes: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    values = targets.detach().to(device="cpu")
    if task == "regression":
        return values.to(torch.float64).reshape(-1, 1), torch.empty(0, dtype=torch.long)
    live_classes = torch.unique(values.long(), sorted=True) if classes is None else classes.long()
    if task == "binary":
        if live_classes.numel() != 2:
            raise ValueError("binary probe requires exactly two training classes")
        encoded = (values.long() == live_classes[1]).to(torch.float64).reshape(-1, 1)
        return encoded, live_classes
    if task == "multiclass":
        if live_classes.numel() < 2:
            raise ValueError("multiclass probe requires at least two training classes")
        matches = values.long()[:, None] == live_classes[None, :]
        if not matches.any(dim=1).all():
            raise ValueError("validation target contains an unseen class")
        return matches.to(torch.float64), live_classes
    raise ValueError(f"unsupported probe task: {task}")


def _fit_one_ridge(
    features: torch.Tensor,
    targets: torch.Tensor,
    penalty: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    target_mean = targets.mean(dim=0)
    centered = targets - target_mean
    gram = features.T @ features
    regularized = gram + float(penalty) * torch.eye(gram.shape[0], dtype=gram.dtype)
    weights = torch.linalg.pinv(regularized) @ features.T @ centered
    return weights, target_mean


def _predict_encoded(
    weights: torch.Tensor,
    target_mean: torch.Tensor,
    features: torch.Tensor,
    feature_mean: torch.Tensor,
    feature_scale: torch.Tensor,
) -> torch.Tensor:
    standardized = (features.detach().cpu().to(torch.float64) - feature_mean) / feature_scale
    return standardized @ weights + target_mean


def _decode_predictions(task: ProbeTask, values: torch.Tensor, classes: torch.Tensor) -> torch.Tensor:
    if task == "regression":
        return values[:, 0].to(torch.float32)
    if task == "binary":
        indices = (values[:, 0] >= 0.5).long()
        return classes[indices]
    return classes[values.argmax(dim=1)]


def fit_ridge(
    train_features: torch.Tensor,
    train_targets: torch.Tensor,
    validation_features: torch.Tensor,
    validation_targets: torch.Tensor,
    *,
    task: ProbeTask,
    penalties: Sequence[float] = (0.01, 0.1, 1.0, 10.0),
) -> RidgeProbe:
    if not penalties or any(value < 0 for value in penalties):
        raise ValueError("ridge penalties must be nonnegative and nonempty")
    train_x, mean, scale = _standardize_fit(train_features)
    train_y, classes = _encode_targets(train_targets, task)
    validation_y, _ = _encode_targets(validation_targets, task, classes)
    if train_x.shape[0] != train_y.shape[0]:
        raise ValueError("training features and targets have different sizes")
    if validation_features.shape[0] != validation_y.shape[0]:
        raise ValueError("validation features and targets have different sizes")
    best: tuple[float, float, torch.Tensor, torch.Tensor] | None = None
    for penalty in sorted(float(value) for value in penalties):
        weights, target_mean = _fit_one_ridge(train_x, train_y, penalty)
        encoded = _predict_encoded(
            weights, target_mean, validation_features, mean, scale
        )
        decoded = _decode_predictions(task, encoded, classes)
        score = score_probe(validation_targets, decoded, task=task)
        candidate = (score, -penalty, weights, target_mean)
        if best is None or candidate[:2] > best[:2]:
            best = candidate
    assert best is not None
    return RidgeProbe(
        task=task,
        weights=best[2].to(torch.float32),
        target_mean=best[3].to(torch.float32),
        feature_mean=mean.to(torch.float32),
        feature_scale=scale.to(torch.float32),
        penalty=-best[1],
        classes=classes,
        validation_score=best[0],
    )


def predict_ridge(probe: RidgeProbe, features: torch.Tensor) -> torch.Tensor:
    values = _predict_encoded(
        probe.weights.to(torch.float64),
        probe.target_mean.to(torch.float64),
        features,
        probe.feature_mean.to(torch.float64),
        probe.feature_scale.to(torch.float64),
    )
    return _decode_predictions(probe.task, values, probe.classes)


def score_probe(targets: torch.Tensor, predictions: torch.Tensor, *, task: ProbeTask) -> float:
    truth = targets.detach().cpu()
    predicted = predictions.detach().cpu()
    if truth.shape[0] != predicted.shape[0]:
        raise ValueError("targets and predictions have different sizes")
    if task == "regression":
        truth_float = truth.float().reshape(-1)
        predicted_float = predicted.float().reshape(-1)
        denominator = (truth_float - truth_float.mean()).square().sum()
        if float(denominator) <= 1e-12:
            raise ValueError("regression target variance is zero")
        return float(1.0 - (truth_float - predicted_float).square().sum() / denominator)
    classes = torch.unique(truth.long(), sorted=True)
    if task == "binary":
        recalls = []
        for value in classes.tolist():
            selected = truth.long() == value
            recalls.append(float((predicted.long()[selected] == value).float().mean()))
        return float(sum(recalls) / len(recalls))
    return float((truth.long() == predicted.long()).float().mean())


def permutation_baseline(
    train_features: torch.Tensor,
    train_targets: torch.Tensor,
    validation_features: torch.Tensor,
    validation_targets: torch.Tensor,
    *,
    task: ProbeTask,
    seeds: Sequence[int],
    penalties: Sequence[float] = (0.01, 0.1, 1.0, 10.0),
) -> float:
    scores = []
    targets = train_targets.detach().cpu()
    for seed in seeds:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        shuffled = targets[torch.randperm(targets.shape[0], generator=generator)]
        probe = fit_ridge(
            train_features,
            shuffled,
            validation_features,
            validation_targets,
            task=task,
            penalties=penalties,
        )
        scores.append(score_probe(validation_targets, predict_ridge(probe, validation_features), task=task))
    if not scores:
        raise ValueError("permutation baseline requires at least one seed")
    return float(torch.tensor(scores).median())


def _orthonormal_basis(
    rows: torch.Tensor,
    *,
    exclude: torch.Tensor | None = None,
    tolerance: float = 1e-6,
) -> torch.Tensor:
    if rows.ndim == 1:
        rows = rows[None, :]
    if rows.ndim != 2 or rows.numel() == 0:
        raise ValueError("decoder rows must be a nonempty matrix")
    columns = rows.detach().float().T
    if exclude is not None and exclude.numel():
        columns = columns - exclude @ (exclude.T @ columns)
    left, singular, _ = torch.linalg.svd(columns, full_matrices=False)
    if singular.numel() == 0 or float(singular.max()) <= 0:
        raise ValueError("variable decoder has numerical rank zero")
    rank = int((singular > tolerance * singular.max()).sum())
    if rank == 0:
        raise ValueError("variable decoder is contained in excluded span")
    return left[:, :rank]


def orthogonalized_variable_bases(
    decoder_rows: Mapping[str, torch.Tensor],
    *,
    tolerance: float = 1e-6,
) -> dict[str, torch.Tensor]:
    required = {"count", "count_mod4", "parity"}
    missing = sorted(required - set(decoder_rows))
    if missing:
        raise ValueError("missing decoder rows: " + ", ".join(missing))
    dimension = int(decoder_rows["parity"].shape[-1])
    current = (
        _orthonormal_basis(decoder_rows["current_bit"], tolerance=tolerance)
        if "current_bit" in decoder_rows
        else torch.empty(dimension, 0)
    )
    count_rows = torch.cat(
        (
            decoder_rows["count"].reshape(-1, current.shape[0]),
            decoder_rows["count_mod4"].reshape(-1, current.shape[0]),
        ),
        dim=0,
    )
    count = _orthonormal_basis(
        count_rows,
        exclude=current if current.numel() else None,
        tolerance=tolerance,
    )
    current_count = torch.cat((current, count), dim=1)
    parity = _orthonormal_basis(
        decoder_rows["parity"], exclude=current_count, tolerance=tolerance
    )
    current_parity = torch.cat((current, parity), dim=1)
    count_given_parity = _orthonormal_basis(
        count_rows, exclude=current_parity, tolerance=tolerance
    )
    return {
        "current_bit": current,
        "count": count,
        "parity": parity,
        "count_given_parity": count_given_parity,
    }


def patch_site_subspace(
    receiver: torch.Tensor,
    donor: torch.Tensor,
    basis: torch.Tensor,
    *,
    token_position: int,
) -> torch.Tensor:
    if receiver.shape != donor.shape or receiver.ndim != 3:
        raise ValueError("receiver and donor must have equal [batch, tokens, d_model] shapes")
    if not 0 <= token_position < receiver.shape[1]:
        raise ValueError("token position is out of range")
    live_basis = basis.to(device=receiver.device, dtype=torch.float32)
    if live_basis.ndim != 2 or live_basis.shape[0] != receiver.shape[2] or live_basis.shape[1] == 0:
        raise ValueError("basis has incompatible shape")
    output = receiver.clone()
    delta = donor[:, token_position].float() - receiver[:, token_position].float()
    output[:, token_position] = (
        receiver[:, token_position].float() + (delta @ live_basis) @ live_basis.T
    ).to(receiver.dtype)
    return output


def matched_random_patch(
    receiver: torch.Tensor,
    donor: torch.Tensor,
    target_basis: torch.Tensor,
    *,
    protected_basis: torch.Tensor | None,
    token_position: int,
    seed: int,
) -> torch.Tensor:
    dimension, rank = target_basis.shape
    exclusions = [target_basis.detach().float()]
    if protected_basis is not None and protected_basis.numel():
        exclusions.append(protected_basis.detach().float())
    exclude = torch.linalg.qr(torch.cat(exclusions, dim=1), mode="reduced").Q
    if dimension - exclude.shape[1] < rank:
        raise ValueError("not enough orthogonal dimensions for random control")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    candidate = torch.randn(dimension, rank, generator=generator)
    candidate -= exclude.cpu() @ (exclude.cpu().T @ candidate)
    random_basis = _orthonormal_basis(candidate.T)
    target_patch = patch_site_subspace(
        receiver, donor, target_basis, token_position=token_position
    )
    random_patch = patch_site_subspace(
        receiver, donor, random_basis, token_position=token_position
    )
    target_delta = target_patch[:, token_position].float() - receiver[:, token_position].float()
    random_delta = random_patch[:, token_position].float() - receiver[:, token_position].float()
    target_norm = torch.linalg.vector_norm(target_delta, dim=-1, keepdim=True)
    random_norm = torch.linalg.vector_norm(random_delta, dim=-1, keepdim=True)
    if bool((random_norm <= 1e-12).any()):
        raise ValueError("random control has zero projected donor displacement")
    output = receiver.clone()
    output[:, token_position] = (
        receiver[:, token_position].float() + random_delta * target_norm / random_norm
    ).to(receiver.dtype)
    return output


def normalized_recovery(
    patched: torch.Tensor | float,
    target: torch.Tensor | float,
    receiver: torch.Tensor | float,
    *,
    minimum_denominator: float = 1e-8,
) -> torch.Tensor:
    patched_tensor = torch.as_tensor(patched, dtype=torch.float32)
    target_tensor = torch.as_tensor(target, dtype=torch.float32)
    receiver_tensor = torch.as_tensor(receiver, dtype=torch.float32)
    denominator = target_tensor - receiver_tensor
    if bool((denominator.abs() < minimum_denominator).any()):
        raise ValueError("target/receiver recovery denominator is too small")
    return (patched_tensor - receiver_tensor) / denominator


def serializable_probe(probe: RidgeProbe) -> dict[str, Any]:
    return {
        "task": probe.task,
        "weights": probe.weights,
        "target_mean": probe.target_mean,
        "feature_mean": probe.feature_mean,
        "feature_scale": probe.feature_scale,
        "penalty": probe.penalty,
        "classes": probe.classes,
        "validation_score": probe.validation_score,
    }
