import math

import torch
import torch.nn as nn

class ScaleDotProductAttention(nn.Module):

    def __init__(
        self,
        causal: bool = True
    ):
        super().__init__()

        self.causal = causal

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor
    ) -> torch.Tensor:

        if not all(
            isinstance(x, torch.Tensor)
            for x in (q, k, v)
        ):
            raise TypeError("q, k and v must be torch.Tensor")

        if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
            raise ValueError(
                "q, k and v must have shape [B, N, S, D]"
            )

        if q.shape != k.shape or q.shape != v.shape:
            raise ValueError(
                "q, k and v must have the same shape"
            )

        if q.device != k.device or q.device != v.device:
            raise ValueError(
                "q, k and v must be on the same device"
            )

        if q.dtype != k.dtype or q.dtype != v.dtype:
            raise TypeError(
                "q, k and v must have the same dtype"
            )

        if not torch.is_floating_point(q):
            raise TypeError(
                "q, k and v must have floating-point dtype"
            )

        if q.shape[-2] == 0 or q.shape[-1] == 0:
            raise ValueError(
                "sequence length and head dimension must be positive"
            )

        scores = torch.matmul(
            q,
            k.transpose(-1, -2)
        )

        scale = 1.0 / math.sqrt(q.shape[-1])
        scores = scores * scale

        if self.causal:
            seq_len = q.shape[-2]

            mask = torch.triu(
                torch.ones(
                    seq_len,
                    seq_len,
                    device=q.device,
                    dtype=torch.bool
                ),
                diagonal=1
            )

            scores = scores.masked_fill(
                mask,
                float("-inf")
            )

        atten_weights = torch.softmax(
            scores,
            dim=-1
        )

        output = torch.matmul(atten_weights, v)

        return output