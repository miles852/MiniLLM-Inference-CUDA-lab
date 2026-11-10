import torch
import torch.nn as nn

class RMSNorm(nn.Module):

    def __init__(self, hidden_size: int , eps: float = 1e-6):
        super().__init__()

        if not isinstance(hidden_size, int) or isinstance(hidden_size, bool):
            raise TypeError("hidden_size must be an integer")

        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")

        if eps <= 0:
            raise ValueError("eps must be positive")

        self.hiddent_size = hidden_size
        self.eps = eps

        self.weight = nn.Parameter(
            torch.ones(hidden_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor :

        if not isinstance(x, torch.Tensor):
            raise TypeError("input must be a torch.Tensor")

        if x.ndim < 1:
            raise ValueError("input must have at least one dimension")

        if x.shape[-1] != self.hiddent_size:
            raise ValueError("last dimension must equal hidden dimension")

        if not torch.is_floating_point(x):
            raise TypeError("input must have a floating-point dtype")

        mean_square = x.pow(2).mean(
            dim=-1,
            keepdim=True
        )

        rms_inv  = torch.rsqrt(mean_square + self.eps)

        output = x * rms_inv * self.weight

        return output