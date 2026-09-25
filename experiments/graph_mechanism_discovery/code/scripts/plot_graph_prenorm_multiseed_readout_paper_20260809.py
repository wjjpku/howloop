from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from reasoning_loop.graph_path_depth_circuit import fixed_depth_batch, load_checkpoint
from reasoning_loop.graph_path_loop import set_seed
from reasoning_loop.graph_path_temporal_intervention import (
    apply_shared_stack,
    logits_from_raw_state,
)


def checkpoint(root: Path, seed: int) -> Path:
    return (
        root
        / f"D8_L8_seed{seed}"
        / f"graphpath_N8_D8_d256_B2_L8_seed{seed}"
        / "best.pt"
    )


def eight_cycle_batch(
    *,
    batch_size: int,
    node_count: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample permutation graphs whose orbit from every start has length eight."""
    order = torch.rand(batch_size, node_count, device=device).argsort(dim=-1)
    successors = torch.empty_like(order)
    successors.scatter_(1, order, order.roll(shifts=-1, dims=1))
    start = torch.randint(0, node_count, (batch_size,), device=device)
    return successors, start


def orbit_nodes(
    successors: torch.Tensor,
    start: torch.Tensor,
) -> torch.Tensor:
    nodes = [start]
    current = start
    for _ in range(successors.shape[1] - 1):
        current = successors.gather(1, current[:, None]).squeeze(1)
        nodes.append(current)
    orbit = torch.stack(nodes, dim=1)
    if not torch.all(orbit.sort(dim=1).values.eq(
        torch.arange(successors.shape[1], device=successors.device)[None, :]
    )):
        raise RuntimeError("sampled graph is not a full cycle")
    return orbit


@torch.no_grad()
def evaluate_seed(
    checkpoint_path: Path,
    *,
    device: torch.device,
    batch_size: int,
    batches: int,
    loops: int,
    data_seed: int,
) -> dict[str, Any]:
    model, cfg, payload = load_checkpoint(checkpoint_path, device)
    if (cfg.node_count, cfg.max_depth, cfg.max_loops, cfg.n_layers) != (8, 8, 8, 2):
        raise ValueError(f"unexpected checkpoint config: {vars(cfg)}")

    probability_sum = np.zeros((loops + 1, cfg.node_count), dtype=np.float64)
    prediction_count = np.zeros((loops + 1, cfg.node_count), dtype=np.int64)
    examples = 0
    set_seed(data_seed)

    for _ in range(batches):
        successors, start = eight_cycle_batch(
            batch_size=batch_size,
            node_count=cfg.node_count,
            device=device,
        )
        tokens, _, _, _ = fixed_depth_batch(
            cfg,
            batch_size,
            device,
            path_positions=cfg.max_depth,
            successors=successors,
            start=start,
        )
        orbit = orbit_nodes(successors, start)
        inverse_orbit = torch.empty_like(orbit)
        inverse_orbit.scatter_(
            1,
            orbit,
            torch.arange(cfg.node_count, device=device)[None, :].expand_as(orbit),
        )

        state = model.token_embed(tokens) + model.pos_embed.unsqueeze(0)
        states = [state]
        for loop_index in range(loops):
            state = apply_shared_stack(model, state, loop_index=loop_index)
            states.append(state)

        for loop_index, selected_state in enumerate(states):
            logits = logits_from_raw_state(model, selected_state)
            probability = logits.softmax(dim=-1)
            orbit_probability = probability.gather(1, orbit)
            if not torch.allclose(
                orbit_probability.sum(dim=1),
                torch.ones(batch_size, device=device),
                atol=2e-5,
                rtol=2e-5,
            ):
                raise RuntimeError("orbit probability does not sum to one")
            probability_sum[loop_index] += (
                orbit_probability.sum(dim=0).float().cpu().numpy()
            )
            prediction_node = logits.argmax(dim=-1)
            prediction_position = inverse_orbit.gather(
                1, prediction_node[:, None]
            ).squeeze(1)
            prediction_count[loop_index] += np.bincount(
                prediction_position.cpu().numpy(), minlength=cfg.node_count
            )
        examples += batch_size

    probability = probability_sum / examples
    prediction_fraction = prediction_count / examples
    if not np.allclose(probability.sum(axis=1), 1.0, atol=2e-6):
        raise RuntimeError("mean probability rows do not sum to one")
    if not np.allclose(prediction_fraction.sum(axis=1), 1.0, atol=2e-6):
        raise RuntimeError("prediction fractions do not sum to one")

    return {
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": int(payload.get("step", -1)),
        "examples": examples,
        "loops": loops,
        "probability": probability,
        "prediction_fraction": prediction_fraction,
        "top_probability_position": probability.argmax(axis=1),
        "top_prediction_position": prediction_fraction.argmax(axis=1),
        "max_probability_row_sum_error": float(
            np.abs(probability.sum(axis=1) - 1.0).max()
        ),
        "post_final_hold_accuracy": float(prediction_fraction[9:, 0].mean()),
        "post_final_successor_accuracy": float(prediction_fraction[9:, 1].mean()),
        "post_final_other_rate": float(prediction_fraction[9:, 2:].sum(axis=1).mean()),
    }


def plot(results: dict[int, dict[str, Any]], output_path: Path) -> None:
    figure = plt.figure(figsize=(7.05, 5.4))
    grid = figure.add_gridspec(
        2,
        3,
        left=0.075,
        right=0.905,
        bottom=0.09,
        top=0.91,
        wspace=0.18,
        hspace=0.30,
    )
    heat_axes = [figure.add_subplot(grid[row, col]) for row in range(2) for col in range(3)]
    image = None
    for axis, seed in zip(heat_axes, sorted(results), strict=True):
        # Display the eight distinct nodes as task steps f^1,...,f^8.
        # On a directed 8-cycle, the endpoint node f^8 is the old f^0 class.
        values = np.asarray(results[seed]["probability"])[:, [1, 2, 3, 4, 5, 6, 7, 0]]
        image = axis.imshow(
            values,
            origin="upper",
            vmin=0.0,
            vmax=1.0,
            cmap="viridis",
            aspect="auto",
            interpolation="nearest",
        )
        maxima = values.argmax(axis=1)
        axis.plot(maxima, np.arange(values.shape[0]), "o", ms=2.1, color="white", mec="black", mew=0.25)
        axis.axhline(8.5, color="#ef4444", linestyle="--", linewidth=0.8)
        axis.set_title(f"seed {seed}", fontsize=9, pad=2)
        axis.set_xticks(range(8))
        axis.set_xticklabels([rf"$f^{p}$" for p in range(1, 9)], fontsize=6.8)
        axis.set_yticks([0, 2, 4, 6, 8, 10, 12, 14, 16])
        axis.tick_params(axis="y", labelsize=7)
        axis.set_xlabel("path target", fontsize=7.5, labelpad=1)
    for axis in heat_axes[::3]:
        axis.set_ylabel("loop boundary", fontsize=8)
    assert image is not None
    cbar_axis = figure.add_axes((0.92, 0.12, 0.014, 0.76))
    colorbar = figure.colorbar(image, cax=cbar_axis)
    colorbar.set_label("readout probability", fontsize=8)
    colorbar.ax.tick_params(labelsize=7)
    figure.text(
        0.075,
        0.975,
        "Intermediate readout schedules across backbones supervised only at loop 8",
        fontsize=9.5,
        fontweight="bold",
        va="top",
    )
    figure.savefig(output_path)
    plt.close(figure)


def write_outputs(results: dict[int, dict[str, Any]], out_dir: Path) -> None:
    rows: list[dict[str, Any]] = []
    for seed, result in results.items():
        probability = np.asarray(result["probability"])
        prediction = np.asarray(result["prediction_fraction"])
        for loop_index in range(probability.shape[0]):
            for position in range(probability.shape[1]):
                rows.append(
                    {
                        "seed": seed,
                        "loop": loop_index,
                        "orbit_position": position,
                        "mean_readout_probability": float(probability[loop_index, position]),
                        "top1_prediction_fraction": float(prediction[loop_index, position]),
                        "probability_row_sum": float(probability[loop_index].sum()),
                        "prediction_row_sum": float(prediction[loop_index].sum()),
                        "examples": int(result["examples"]),
                    }
                )
    with (out_dir / "multiseed_readout_rows.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "protocol": {
            "backbone": "Pre-Norm D8L8, two shared physical blocks, CE only after loop 8",
            "graph_slice": "uniformly sampled directed 8-cycles from the training graph family",
            "query_depth": 8,
            "evaluated_loops": 16,
            "position_classes": "the eight mutually exclusive orbit nodes f^0(start),...,f^7(start)",
            "seeds": sorted(results),
            "same_evaluation_stream_across_seeds": True,
        },
        "seeds": {
            str(seed): {
                key: (value.tolist() if isinstance(value, np.ndarray) else value)
                for key, value in result.items()
                if key not in {"probability", "prediction_fraction"}
            }
            for seed, result in results.items()
        },
        "aggregate_post_final": {
            "hold_accuracy_mean": float(np.mean([r["post_final_hold_accuracy"] for r in results.values()])),
            "hold_accuracy_min": float(np.min([r["post_final_hold_accuracy"] for r in results.values()])),
            "successor_accuracy_mean": float(np.mean([r["post_final_successor_accuracy"] for r in results.values()])),
            "successor_accuracy_max": float(np.max([r["post_final_successor_accuracy"] for r in results.values()])),
            "other_rate_mean": float(np.mean([r["post_final_other_rate"] for r in results.values()])),
            "max_probability_row_sum_error": float(max(r["max_probability_row_sum_error"] for r in results.values())),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def read_existing_rows(out_dir: Path) -> dict[int, dict[str, Any]]:
    """Reconstruct plot inputs from a completed evaluation without rerunning a model."""
    rows_path = out_dir / "multiseed_readout_rows.csv"
    grouped: dict[int, list[dict[str, str]]] = {}
    with rows_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            grouped.setdefault(int(row["seed"]), []).append(row)

    results: dict[int, dict[str, Any]] = {}
    for seed, rows in grouped.items():
        max_loop = max(int(row["loop"]) for row in rows)
        max_position = max(int(row["orbit_position"]) for row in rows)
        probability = np.zeros((max_loop + 1, max_position + 1), dtype=np.float64)
        prediction_fraction = np.zeros_like(probability)
        for row in rows:
            loop_index = int(row["loop"])
            position = int(row["orbit_position"])
            probability[loop_index, position] = float(row["mean_readout_probability"])
            prediction_fraction[loop_index, position] = float(row["top1_prediction_fraction"])
        results[seed] = {
            "probability": probability,
            "prediction_fraction": prediction_fraction,
        }
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--loops", type=int, default=16)
    parser.add_argument("--data-seed", type=int, default=8_092_206)
    parser.add_argument(
        "--plot-existing",
        action="store_true",
        help="replot multiseed_readout_rows.csv in out-dir without model evaluation",
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.plot_existing:
        plot(read_existing_rows(args.out_dir), args.out_dir / "multiseed_readout_and_terminal_hold.pdf")
        return
    if args.checkpoint_root is None:
        parser.error("--checkpoint-root is required unless --plot-existing is used")
    device = torch.device(args.device)
    results: dict[int, dict[str, Any]] = {}
    for seed in range(6):
        results[seed] = evaluate_seed(
            checkpoint(args.checkpoint_root, seed),
            device=device,
            batch_size=args.batch_size,
            batches=args.batches,
            loops=args.loops,
            data_seed=args.data_seed,
        )
        print(f"completed seed {seed}", flush=True)
    write_outputs(results, args.out_dir)
    plot(results, args.out_dir / "multiseed_readout_and_terminal_hold.pdf")


if __name__ == "__main__":
    main()
