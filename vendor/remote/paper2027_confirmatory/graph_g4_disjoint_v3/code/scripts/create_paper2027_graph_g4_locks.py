#!/usr/bin/env python3
"""Create the two immutable, mutually disjoint G4 graph locks before training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from reasoning_loop.paper2027_graph_g4_protocol import (
    create_unique_lock,
    load_unique_lock,
    sha256,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--node-count", type=int, default=8)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--permutations", type=int, default=512)
    parser.add_argument("--selection-seed", type=int, default=2026081201)
    parser.add_argument("--test-seed", type=int, default=2026081202)
    args = parser.parse_args()

    lock_dir = args.root / "locks"
    selection_path = lock_dir / "selection_permutations_512.pt"
    final_path = lock_dir / "final_test_permutations_512.pt"
    selection = create_unique_lock(
        path=selection_path, node_count=args.node_count, max_depth=args.max_depth,
        permutations=args.permutations, seed=args.selection_seed, role="selection",
    )
    final = create_unique_lock(
        path=final_path, node_count=args.node_count, max_depth=args.max_depth,
        permutations=args.permutations, seed=args.test_seed, role="final_test",
        excluded_codes=selection["permutation_codes"],  # type: ignore[arg-type,index]
    )
    # Re-load both locks and prove their disjointness after serialization.
    selection = load_unique_lock(selection_path)
    final = load_unique_lock(final_path)
    overlap = set(selection["permutation_codes"].tolist()) & set(final["permutation_codes"].tolist())  # type: ignore[index]
    if overlap:
        raise RuntimeError("selection and final locks overlap")
    payload = {
        "protocol_id": "paper2027.graph.g4.disjoint_locks.v1",
        "node_count": args.node_count,
        "max_depth": args.max_depth,
        "selection_lock": str(selection_path),
        "selection_lock_sha256": sha256(selection_path),
        "final_test_lock": str(final_path),
        "final_test_lock_sha256": sha256(final_path),
        "selection_graphs": int(selection["permutations"]),
        "final_test_graphs": int(final["permutations"]),
        "overlap_graphs": len(overlap),
    }
    manifest = args.root / "manifests" / "locks.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
