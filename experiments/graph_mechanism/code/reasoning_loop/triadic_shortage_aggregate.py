from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence


EXPLORATORY_SEEDS = (0, 1, 2)


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else float("nan")


def load_behavior_rows(run_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary_path in sorted(run_root.rglob("summary.json")):
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        heldout = payload.get("final_metrics", {}).get("heldout")
        if heldout is None:
            continue
        rows.append(
            {
                "run_name": payload.get("run_name", summary_path.parent.name),
                "summary_path": str(summary_path.resolve()),
                "condition": payload["condition"],
                "source_condition": payload.get("source_condition"),
                "architecture": payload["architecture"],
                "p": int(payload["p"]),
                "d_model": int(payload["d_model"]),
                "d_mlp": int(payload["d_mlp"]),
                "loops": int(payload["loops"]),
                "seed": int(payload["seed"]),
                "steps": int(payload["steps"]),
                "parameter_count": int(payload["parameter_count"]),
                "heldout_accuracy": float(heldout["final_accuracy"]),
                "heldout_loss": float(heldout["final_loss"]),
                "heldout_margin": float(heldout["final_margin"]),
                "per_loop_accuracy": heldout["per_loop_accuracy"],
            }
        )
    return rows


def _scratch_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("source_condition") in {None, ""}]


def select_compute_contrast(
    rows: list[dict[str, Any]],
    *,
    exploratory_seeds: tuple[int, ...] = EXPLORATORY_SEEDS,
    shallow_max: float = 0.80,
    deep_min: float = 0.99,
) -> dict[str, Any]:
    rows = _scratch_rows(rows)
    widths = sorted(
        {
            int(row["d_model"])
            for row in rows
            if row["condition"] == "full"
        }
    )
    qualified: list[int] = []
    details: dict[str, Any] = {}

    def find(width: int, architecture: str, loops: int, seed: int):
        matches = [
            row
            for row in rows
            if row["condition"] == "full"
            and int(row["d_model"]) == width
            and row["architecture"] == architecture
            and int(row["loops"]) == loops
            and int(row["seed"]) == seed
        ]
        if not matches:
            return None
        return max(matches, key=lambda row: int(row.get("steps", 0)))

    for width in widths:
        shallow = [find(width, "looped", 1, seed) for seed in exploratory_seeds]
        recurrent = [find(width, "looped", 6, seed) for seed in exploratory_seeds]
        unshared = [find(width, "unshared", 6, seed) for seed in exploratory_seeds]
        complete = all(row is not None for row in [*shallow, *recurrent, *unshared])
        shallow_acc = [float(row["heldout_accuracy"]) for row in shallow if row]
        recurrent_acc = [float(row["heldout_accuracy"]) for row in recurrent if row]
        unshared_acc = [float(row["heldout_accuracy"]) for row in unshared if row]
        passes = (
            complete
            and all(value < shallow_max for value in shallow_acc)
            and all(value >= deep_min for value in recurrent_acc)
            and all(value >= deep_min for value in unshared_acc)
        )
        details[str(width)] = {
            "complete": complete,
            "passes": passes,
            "l1_accuracy": shallow_acc,
            "l1x6_accuracy": recurrent_acc,
            "s6_accuracy": unshared_acc,
        }
        if passes:
            qualified.append(width)
    if not qualified:
        return {
            "status": "not_found",
            "qualified_widths": [],
            "seeds": list(exploratory_seeds),
            "thresholds": {"l1_max": shallow_max, "deep_min": deep_min},
            "details": details,
        }
    selected = max(qualified)
    return {
        "status": "selected",
        "d_model": selected,
        "qualified_widths": qualified,
        "seeds": list(exploratory_seeds),
        "thresholds": {"l1_max": shallow_max, "deep_min": deep_min},
        "selected_details": details[str(selected)],
        "details": details,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            serialized = {
                key: json.dumps(value) if isinstance(value, (list, dict)) else value
                for key, value in row.items()
            }
            writer.writerow(serialized)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")


def _seed_delta_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup: dict[tuple[int, int, str, int], float] = {}
    for row in _scratch_rows(rows):
        if row["condition"] != "full":
            continue
        key = (
            int(row["d_model"]),
            int(row["seed"]),
            str(row["architecture"]),
            int(row["loops"]),
        )
        lookup[key] = float(row["heldout_accuracy"])
    output = []
    for width, seed in sorted({(key[0], key[1]) for key in lookup}):
        l1 = lookup.get((width, seed, "looped", 1))
        recurrent = lookup.get((width, seed, "looped", 6))
        unshared = lookup.get((width, seed, "unshared", 6))
        output.append(
            {
                "d_model": width,
                "seed": seed,
                "l1_accuracy": l1,
                "l1x6_accuracy": recurrent,
                "s6_accuracy": unshared,
                "l1x6_minus_l1": None if l1 is None or recurrent is None else recurrent - l1,
                "s6_minus_l1": None if l1 is None or unshared is None else unshared - l1,
            }
        )
    return output


def _information_status(rows: list[dict[str, Any]], run_root: Path) -> dict[str, Any]:
    scratch = [
        row
        for row in _scratch_rows(rows)
        if str(row.get("run_name", "")).startswith(("info_", "confirm_"))
    ]
    by_condition: dict[str, list[float]] = defaultdict(list)
    by_condition_seeds: dict[str, list[int]] = defaultdict(list)
    for row in scratch:
        if row["architecture"] == "looped" and int(row["loops"]) == 6:
            by_condition[str(row["condition"])].append(float(row["heldout_accuracy"]))
            by_condition_seeds[str(row["condition"])].append(int(row["seed"]))
    scorecards = []
    for path in sorted(run_root.rglob("scorecard.json")):
        try:
            scorecards.append(json.loads(path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, KeyError):
            continue
    sequential_cards = [
        card
        for card in scorecards
        if card.get("condition") in {"sequential", "sequential_shuffled"}
        and card.get("reset_effect") is not None
        and Path(str(card.get("checkpoint", ""))).parent.name.startswith(
            ("info_", "confirm_")
        )
    ]
    behavior_pass = all(
        len(by_condition.get(condition, [])) >= 3
        and all(value >= 0.99 for value in by_condition[condition])
        for condition in ("sequential", "sequential_shuffled")
    )
    causal_pass = (
        len(sequential_cards) >= 3
        and all(float(card.get("reset_effect", 0.0)) >= 0.80 for card in sequential_cards)
        and all(
            float(card.get("hybrid_accuracy_mean", 0.0)) >= 0.90
            for card in sequential_cards
        )
        and all(
            float(card.get("max_pre_reveal_state_delta", 1.0)) == 0.0
            for card in sequential_cards
        )
    )
    status = "causal_pass" if behavior_pass and causal_pass else (
        "behavior_pass" if behavior_pass else "open"
    )
    return {
        "status": status,
        "behavior_pass": behavior_pass,
        "causal_pass": causal_pass,
        "condition_accuracy": dict(by_condition),
        "condition_seeds": dict(by_condition_seeds),
        "diagnostic_scorecards": len(sequential_cards),
        "order_sweep": {
            f"{card.get('condition')}_seed{card.get('seed')}": {
                "mean_accuracy": card.get("order_sweep_mean_accuracy"),
                "min_accuracy": card.get("order_sweep_min_accuracy"),
                "max_accuracy": card.get("order_sweep_max_accuracy"),
            }
            for card in sequential_cards
            if card.get("order_sweep_mean_accuracy") is not None
        },
    }


def _component_circuit_status(run_root: Path) -> dict[str, Any]:
    cards = []
    circuit_root = run_root / "analysis" / "circuits"
    for path in sorted(circuit_root.glob("*/scorecard.json")):
        try:
            card = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, KeyError):
            continue
        if card.get("condition") == "sequential":
            cards.append(card)
    seed_cards = {
        int(card["seed"]): card
        for card in cards
        if int(card.get("seed", -1)) in EXPLORATORY_SEEDS
    }
    passes = []
    for seed in EXPLORATORY_SEEDS:
        card = seed_cards.get(seed)
        if card is None:
            passes.append(False)
            continue
        parity = card.get("parity", {})
        passes.append(
            float(card.get("baseline_accuracy", 0.0)) >= 0.99
            and bool(card.get("all_operands_have_selected_circuit", False))
            and float(parity.get("max_logit_delta", 1.0)) <= 2e-4
            and float(parity.get("max_state_delta", 1.0)) <= 2e-4
        )
    return {
        "status": "causal_pass" if all(passes) else ("partial" if cards else "not_run"),
        "seeds": sorted(seed_cards),
        "seed_pass": dict(zip(EXPLORATORY_SEEDS, passes, strict=True)),
        "scorecards": len(cards),
        "evidence": [
            "head patch-in sufficiency",
            "complement-only patch-out degradation",
            "minimal subset search",
            "random donor controls",
            "attention and MLP loop ablations",
        ],
    }


def aggregate(run_root: Path, out_dir: Path) -> dict[str, Any]:
    rows = load_behavior_rows(run_root)
    selected = select_compute_contrast(rows)
    seed_deltas = _seed_delta_rows(rows)
    information = _information_status(rows, run_root)
    component_circuit = _component_circuit_status(run_root)
    claim_ledger = {
        "information_access_reuse": information,
        "computational_depth_reuse": {
            "status": selected["status"],
            "selected_contrast": selected,
        },
        "component_circuit": component_circuit,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "behavior_grid.csv", rows)
    _write_csv(out_dir / "seed_deltas.csv", seed_deltas)
    _write_json(out_dir / "selected_contrast.json", selected)
    _write_json(out_dir / "claim_ledger.json", claim_ledger)
    return {
        "run_count": len(rows),
        "selected_contrast": selected,
        "information_status": information,
        "claim_ledger": claim_ledger,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate triadic shortage runs.")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    print(json.dumps(aggregate(args.run_root, args.out_dir), indent=2), flush=True)


if __name__ == "__main__":
    main()
