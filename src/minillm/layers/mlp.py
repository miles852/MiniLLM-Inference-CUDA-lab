import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(
            self,
            hidden_size: int,
            intermediate_size: int
    ):
        super().__init__()

        if not isinstance(hidden_size, int) or isinstance(hidden_size, bool):
            raise TypeError("hidden_size must be an integer")

        if not isinstance(intermediate_size, int) or isinstance(intermediate_size, bool):
            raise TypeError("intermediate_size must be an integer")

        if hidden_size <= 0 or intermediate_size <= 0:
            raise ValueError("dimensions must be positive")

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        self.gate_proj = nn.Linear(
            hidden_size,
            intermediate_size,
            bias=False
        )

        self.up_proj = nn.Linear(
            hidden_size,
            intermediate_size,
            bias=False
        )

        self.down_proj = nn.Linear(
            intermediate_size,
            hidden_size,
            bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if not isinstance(x, torch.Tensor):
            raise TypeError("input must be a torch.Tensor")

        if x.ndim != 3:
            raise ValueError("input must have shape [B, S, H]")

        if x.shape[-1] != self.hidden_size:
            raise ValueError("input hidden dimension mismatch")

        if not torch.is_floating_point(x):
            raise TypeError("input must have a floating-point dtype")

        gate = self.gate_proj(x)
        up = self.up_proj(x)

        hidden = F.silu(gate) * up

        output = self.down_proj(hidden)

        return output