from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
from typing import Callable

import numpy as np
import torch


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _haar_basis(dimension: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    matrix = torch.randn(
        dimension,
        dimension,
        generator=generator,
        dtype=torch.float64,
    )
    q, r = torch.linalg.qr(matrix)
    signs = torch.sign(torch.diagonal(r))
    signs[signs == 0] = 1
    return q * signs.unsqueeze(0)


def _selector(
    predicate: Callable[[np.ndarray], np.ndarray],
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    def select(real: np.ndarray, imag: np.ndarray) -> np.ndarray:
        return predicate(np.asarray(real) + 1j * np.asarray(imag))

    return select


def ordered_real_schur(
    weight: torch.Tensor,
    *,
    zero_threshold: float,
    one_threshold: float,
) -> tuple[torch.Tensor, dict[str, tuple[int, int]], np.ndarray]:
    """Return an orthogonal row-state basis ordered zero, one, middle.

    VectorAffine uses row vectors, z -> z @ W + b.  We therefore Schur
    decompose W.T so coordinates z @ Q evolve under the real quasi-triangular
    representation.
    """

    try:
        from scipy.linalg import schur
    except ImportError as error:
        raise RuntimeError(
            "building the Schur artifact requires scipy; evaluation does not"
        ) from error

    matrix = weight.detach().double().cpu().numpy().T
    t_zero, q_zero, zero_dim = schur(
        matrix,
        output="real",
        sort=_selector(lambda value: np.abs(value) < zero_threshold),
    )
    zero_dim = int(zero_dim)

    trailing = t_zero[zero_dim:, zero_dim:]
    t_one, q_one, one_dim = schur(
        trailing,
        output="real",
        sort=_selector(lambda value: np.abs(value - 1.0) < one_threshold),
    )
    one_dim = int(one_dim)

    rotation = np.eye(matrix.shape[0], dtype=np.float64)
    rotation[zero_dim:, zero_dim:] = q_one
    q = q_zero @ rotation
    t = rotation.T @ t_zero @ rotation
    reconstruction_error = np.linalg.norm(matrix - q @ t @ q.T) / np.linalg.norm(
        matrix
    )
    orthogonality_error = np.linalg.norm(q.T @ q - np.eye(q.shape[0]))
    if reconstruction_error > 1e-10 or orthogonality_error > 1e-10:
        raise RuntimeError(
            "unstable Schur construction: "
            f"reconstruction={reconstruction_error}, "
            f"orthogonality={orthogonality_error}"
        )

    eigenvalues = np.linalg.eigvals(matrix)
    expected_zero = int(np.sum(np.abs(eigenvalues) < zero_threshold))
    expected_one = int(np.sum(np.abs(eigenvalues - 1.0) < one_threshold))
    if zero_dim != expected_zero or one_dim != expected_one:
        raise RuntimeError(
            "Schur selection count mismatch: "
            f"zero {zero_dim}/{expected_zero}, one {one_dim}/{expected_one}"
        )

    dimension = matrix.shape[0]
    bands = {
        "zero": (0, zero_dim),
        "one": (zero_dim, zero_dim + one_dim),
        "middle": (zero_dim + one_dim, dimension),
    }
    return torch.from_numpy(q), bands, eigenvalues


def build_artifact(
    *,
    j_artifact: Path,
    j_label: str,
    out_path: Path,
    zero_threshold: float,
    one_threshold: float,
    random_seeds: tuple[int, ...],
) -> None:
    payload = torch.load(j_artifact, map_location="cpu", weights_only=False)
    if payload.get("kind") != "graph_path_telomere_unit_j":
        raise ValueError("unexpected unit-J artifact kind")
    item = payload["maps"][j_label]
    weight = item["weight"].double()
    q, bands, eigenvalues = ordered_real_schur(
        weight,
        zero_threshold=zero_threshold,
        one_threshold=one_threshold,
    )
    random_bases = {
        str(seed): _haar_basis(weight.shape[0], seed) for seed in random_seeds
    }
    band_payload = {
        label: {
            "start": start,
            "stop": stop,
            "dimension": stop - start,
        }
        for label, (start, stop) in bands.items()
    }
    result = {
        "kind": "graph_path_telomere_j_real_schur",
        "source_j_artifact": str(j_artifact),
        "source_j_sha256": _sha256(j_artifact),
        "source_j_label": j_label,
        "checkpoint": str(payload["checkpoint"]),
        "row_vector_convention": True,
        "decomposed_matrix": "weight.T",
        "zero_threshold": zero_threshold,
        "one_threshold": one_threshold,
        "basis": q.float(),
        "bands": band_payload,
        "random_bases": {
            seed: basis.float() for seed, basis in random_bases.items()
        },
        "random_seeds": list(random_seeds),
        "eigenvalues_real": torch.tensor(eigenvalues.real),
        "eigenvalues_imag": torch.tensor(eigenvalues.imag),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = out_path.with_suffix(out_path.suffix + ".tmp")
    torch.save(result, temporary)
    os.replace(temporary, out_path)

    print(
        {
            "out_path": str(out_path),
            "source_j_sha256": result["source_j_sha256"],
            "bands": band_payload,
            "random_seeds": list(random_seeds),
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--j-artifact", type=Path, required=True)
    parser.add_argument("--j-label", default="task")
    parser.add_argument("--out-path", type=Path, required=True)
    parser.add_argument("--zero-threshold", type=float, default=0.1)
    parser.add_argument("--one-threshold", type=float, default=0.1)
    parser.add_argument(
        "--random-seeds",
        type=int,
        nargs="+",
        default=(314159, 271828, 161803),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_artifact(
        j_artifact=args.j_artifact,
        j_label=args.j_label,
        out_path=args.out_path,
        zero_threshold=args.zero_threshold,
        one_threshold=args.one_threshold,
        random_seeds=tuple(args.random_seeds),
    )


if __name__ == "__main__":
    main()
