from abc import ABC, abstractmethod

import torch


class BaseMoeBackend(ABC):
    cpu_experts = False

    @abstractmethod
    def forward(
        self,
        hidden_states: torch.Tensor,
        w1: torch.Tensor | None,
        w2: torch.Tensor | None,
        gating_output: torch.Tensor,
        topk: int,
        renormalize: bool,
        activation: str,
        apply_router_weight_on_input: bool,
        layer_id: int = 0,
    ) -> torch.Tensor: ...
