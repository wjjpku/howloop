from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from reasoning_loop.paper_length_telomere import atomic_json_dump


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select a controller checkpoint using only in-distribution "
            "full-answer metrics."
        )
    )
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--lengths", type=int, nargs="+", required=True)
    parser.add_argument("--minimum-exact-match", type=float, default=0.99)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def _candidate(summary_path: Path, lengths: tuple[int, ...]) -> dict[str, Any]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    full_rows = {
        int(row["length"]): row
        for row in summary["rows"]
        if row["variant"] == "full"
    }
    missing = sorted(set(lengths) - set(full_rows))
    if missing:
        raise ValueError(f"{summary_path} lacks full rows for lengths {missing}")
    selected_rows = [full_rows[length] for length in lengths]
    controller_path = Path(summary["controller"])
    payload_name = controller_path.stem
    try:
        update = int(payload_name.rsplit("_", 1)[1])
    except (IndexError, ValueError) as error:
        raise ValueError(
            f"cannot recover optimizer update from {controller_path}"
        ) from error
    exact_matches = [float(row["exact_match"]) for row in selected_rows]
    cross_entropies = [
        float(row["answer_cross_entropy"]) for row in selected_rows
    ]
    return {
        "controller": str(controller_path),
        "evaluation": str(summary_path),
        "update": update,
        "minimum_exact_match": min(exact_matches),
        "mean_exact_match": sum(exact_matches) / len(exact_matches),
        "mean_answer_cross_entropy": (
            sum(cross_entropies) / len(cross_entropies)
        ),
    }


def select_snapshot(
    evaluation_root: Path,
    *,
    lengths: tuple[int, ...],
    minimum_exact_match: float,
) -> dict[str, Any]:
    if not lengths or any(length < 1 for length in lengths):
        raise ValueError("selection lengths must be positive")
    if not 0.0 <= minimum_exact_match <= 1.0:
        raise ValueError("minimum exact match must lie in [0, 1]")
    candidates = [
        _candidate(path, lengths)
        for path in sorted(evaluation_root.glob("controller_*/summary.json"))
    ]
    if not candidates:
        raise ValueError(f"no snapshot evaluations found under {evaluation_root}")
    eligible = [
        row
        for row in candidates
        if row["minimum_exact_match"] >= minimum_exact_match
    ]
    if eligible:
        chosen = min(
            eligible,
            key=lambda row: (row["mean_answer_cross_entropy"], row["update"]),
        )
        gate_passed = True
    else:
        chosen = min(
            candidates,
            key=lambda row: (
                -row["minimum_exact_match"],
                -row["mean_exact_match"],
                row["mean_answer_cross_entropy"],
                row["update"],
            ),
        )
        gate_passed = False
    return {
        "status": "complete",
        "selection_uses_ood_metrics": False,
        "selection_lengths": list(lengths),
        "minimum_exact_match_gate": minimum_exact_match,
        "gate_passed": gate_passed,
        "policy": (
            "among snapshots with per-length ID full-answer EM at or above "
            "the gate, minimize mean ID answer CE; if none pass, maximize "
            "minimum ID EM, then mean ID EM, then minimize ID CE"
        ),
        "selected": chosen,
        "candidates": sorted(candidates, key=lambda row: row["update"]),
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    payload = select_snapshot(
        args.evaluation_root,
        lengths=tuple(args.lengths),
        minimum_exact_match=args.minimum_exact_match,
    )
    atomic_json_dump(payload, args.out)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
