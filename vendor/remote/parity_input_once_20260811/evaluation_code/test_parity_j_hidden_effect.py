from pathlib import Path

import pytest

from reasoning_loop.analyze_parity_j_hidden_effect import (
    validate_controller_backbone,
)


def test_hidden_effect_accepts_the_requested_controller_backbone() -> None:
    checkpoint = Path("/data/experiment/backbones/parity_input_once_seed2/best.pt")
    validate_controller_backbone(str(checkpoint), checkpoint)


def test_hidden_effect_rejects_a_different_controller_backbone() -> None:
    with pytest.raises(ValueError, match="different backbone"):
        validate_controller_backbone(
            "/data/experiment/backbones/parity_input_once_seed1/best.pt",
            Path("/data/experiment/backbones/parity_input_once_seed2/best.pt"),
        )
