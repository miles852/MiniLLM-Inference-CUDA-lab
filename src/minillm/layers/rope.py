import torch
import torch.nn as nn

class RotaryEmbedding(nn.Module):

    def __init__(
        self,
        head_dim: int,
        base: float = 10000.0
    ):
        super().__init__()

        if not isinstance(head_dim, int) or isinstance(head_dim, bool):
            raise TypeError("head_dim must be an integer")

        if head_dim <= 0 or head_dim % 2 != 0:
            raise ValueError("head_dim must be a positive even integer")

        if not isinstance(base, (int, float)) or isinstance(base, bool):
            raise TypeError("base must be a number")

        if base <= 1:
            raise ValueError("base must be greater than 1")

        self.head_dim = head_dim
        self.base = base

        inv_freq = 1.0 / (
            self.base ** (
                torch.arange(0, head_dim, 2, dtype=torch.float)
                / head_dim
            )
        )

        self.register_buffer(
            "inv_freq",
            inv_freq,
            persistent=False
        )

    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        half = self.head_dim // 2

        x1 = x[..., : half]
        x2 = x[..., half: ]

        return torch.cat((x2, x1), dim=-1)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:

        if not isinstance(q, torch.Tensor) or not isinstance(k, torch.Tensor):
            raise TypeError("q and k must be torch.Tensor")

        if q.ndim != 4 or k.ndim != 4:
            raise ValueError("q and k must have shape [B, N, S, D]")

        if q.shape != k.shape:
            raise ValueError("q and k must have the same shape")

        if q.shape[-1] != self.head_dim:
            raise ValueError("last dimension must equal head_dim")

        if q.device != k.device:
            raise ValueError("q and k must be on the same device")

        if q.dtype != k.dtype:
            raise TypeError("q and k must have the same dtype")

        if not torch.is_floating_point(q):
            raise TypeError("q and k must have floating-point dtype")

        seq_len = q.shape[-2]

        positions = torch.arange(
            seq_len,
            device=q.device,
            dtype=torch.float32
        )

        inv_freq = self.inv_freq.to(device=q.device)

        freqs = torch.outer(positions, inv_freq)

        emb = torch.cat((freqs, freqs), dim=-1)

        cos = emb.cos().to(dtype=q.dtype)[None, None, :, :]
        sin = emb.sin().to(dtype=q.dtype)[None, None, :, :]

        q_rot = q * cos + self.rotate_half(q) * sin
        k_rot = k * cos + self.rotate_half(k) * sin

        return q_rot, k_rot
