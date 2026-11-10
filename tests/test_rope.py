import pytest
import torch

from minillm.layers.rope import RotaryEmbedding


@pytest.fixture
def rope():
    return RotaryEmbedding(head_dim=8)


def test_rope_output_shape(rope):
    q = torch.randn(2, 4, 16, 8)
    k = torch.randn(2, 4, 16, 8)

    q_rot, k_rot = rope(q, k)

    assert q_rot.shape == q.shape
    assert k_rot.shape == k.shape


def test_rope_position_zero(rope):
    q = torch.randn(2, 4, 4, 8)
    k = torch.randn(2, 4, 4, 8)

    q_rot, k_rot = rope(q, k)

    assert torch.allclose(q_rot[:, :, 0, :], q[:, :, 0, :])
    assert torch.allclose(k_rot[:, :, 0, :], k[:, :, 0, :])


def test_rope_changes_later_positions(rope):
    q = torch.randn(2, 4, 4, 8)
    k = torch.randn(2, 4, 4, 8)

    q_rot, k_rot = rope(q, k)

    assert not torch.allclose(q_rot[:, :, 1:, :], q[:, :, 1:, :])
    assert not torch.allclose(k_rot[:, :, 1:, :], k[:, :, 1:, :])


def test_rope_gradient(rope):
    q = torch.randn(2, 4, 4, 8, requires_grad=True)
    k = torch.randn(2, 4, 4, 8, requires_grad=True)

    q_rot, k_rot = rope(q, k)

    loss = q_rot.sum() + k_rot.sum()
    loss.backward()

    assert q.grad is not None
    assert k.grad is not None


def test_invalid_head_dim():
    with pytest.raises(ValueError):
        RotaryEmbedding(head_dim=7)


def test_invalid_shape(rope):
    q = torch.randn(2, 4, 8)
    k = torch.randn(2, 4, 8)

    with pytest.raises(ValueError):
        rope(q, k)
