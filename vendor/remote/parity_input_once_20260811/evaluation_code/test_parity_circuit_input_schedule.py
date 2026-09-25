import torch

from reasoning_loop.analyze_parity_circuit import (
    describe_head_ablation,
    describe_token_injection,
    run_states,
)
from reasoning_loop.paper_length_telomere import PaperLoopedTransformer, PaperModelConfig


class RecordingInputScheduleModel(PaperLoopedTransformer):
    def __init__(self, config: PaperModelConfig) -> None:
        super().__init__(config)
        self.requested_input_steps: list[int] = []

    def input_embeddings(
        self, inputs: torch.Tensor, *, step_index: int = 1
    ) -> torch.Tensor:
        self.requested_input_steps.append(step_index)
        return super().input_embeddings(inputs, step_index=step_index)


def test_parity_circuit_rollout_respects_input_once_schedule() -> None:
    model = RecordingInputScheduleModel(
        PaperModelConfig(
            vocab_size=6,
            d_model=8,
            n_heads=2,
            d_mlp=16,
            token_embedding_injection="initial_only",
        )
    ).eval()
    inputs = torch.nn.functional.one_hot(
        torch.tensor([[0, 1, 2], [1, 0, 2]]), num_classes=6
    ).float()
    run_states(model, inputs, steps=3)
    assert model.requested_input_steps == [1, 2, 3]


def test_parity_circuit_report_describes_checkpoint_input_schedule() -> None:
    assert describe_token_injection("initial_only") == "只在第一个 call 注入 token embedding"
    assert describe_token_injection("every_step") == "每个 call 都重新注入 token embedding"


def test_parity_circuit_report_does_not_hardcode_head_redundancy() -> None:
    redundant = describe_head_ablation(
        [{"ablated_head": 3, "accuracy": 1.0}], target_loop=20
    )
    necessary = describe_head_ablation(
        [{"ablated_head": 26, "accuracy": 0.59765625}], target_loop=20
    )
    assert "未发现强必要单 head" in redundant
    assert "head 26" in necessary
    assert "0.598" in necessary
