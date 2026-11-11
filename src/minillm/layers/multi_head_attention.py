import torch
import torch.nn as nn

from minillm.layers.rope import RotaryEmbedding
from minillm.layers.attention import ScaleDotProductAttention

class MultiHeadAttention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int
    ):
        super().__init__()

        if not isinstance(hidden_size, int) or isinstance(hidden_size, bool):
            raise TypeError("hidden_size must be an integer")

        if not isinstance(num_heads, int) or isinstance(num_heads, bool):
            raise TypeError("num_heads must be an integer")

        if hidden_size <= 0 or num_heads <= 0:
            raise ValueError("dimensions must be positive")

        if hidden_size % num_heads != 0:
            raise ValueError(
                "hidden_size must be divisible by num_heads"
            )

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")

        self.q_proj = nn.Linear(
            hidden_size,
            hidden_size,
            bias=False
        )

        self.k_proj = nn.Linear(
                    hidden_size,
                    hidden_size,
                    bias=False
                )

        self.v_proj = nn.Linear(
                    hidden_size,
                    hidden_size,
                    bias=False
                )

        self.o_proj = nn.Linear(
                    hidden_size,
                    hidden_size,
                    bias=False
                )

        self.rope = RotaryEmbedding(head_dim=self.head_dim)

        self.attention = ScaleDotProductAttention(causal=True)

    def split_heads(
        self,
        x: torch.Tensor
    ) -> torch.Tensor :

        batch_size, seq_len, _ = x.shape

        x = x.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim
        )

        return x.transpose(1, 2)

    def merge_heads(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:

        batch_size, _, seq_len, _ = x.shape

        x = x.transpose(1, 2).contiguous()

        return x.view(
            batch_size,
            seq_len,
            self.hidden_size
        )

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:

        if not isinstance(x, torch.Tensor):
            raise TypeError("input must be a torch.Tensor")

        if x.ndim != 3:
            raise ValueError(
                "input must have shape [B, S, H]"
            )

        if x.shape[-1] != self.hidden_size:
            raise ValueError(
                "input hidden dimension mismatch"
            )

        if x.shape[1] == 0:
            raise ValueError("sequence length must be positive")

        if not torch.is_floating_point(x):
            raise TypeError(
                "input must have a floating-point dtype"
            )

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)

        q, k = self.rope(q, k)

        output = self.attention(q, k, v)

        output = self.merge_heads(output)

        output = self.o_proj(output)

        return output