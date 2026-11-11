import pytest
import torch

from minillm.layers.attention import ScaleDotProductAttention

@pytest.fixture
def attention():
    return ScaleDotProductAttention(causal=True)

def test_attention_output_shape(attention):
    q = torch.randn(2, 4, 8, 16)
    k = torch.randn(2, 4, 8, 16)
    v = torch.randn(2, 4, 8, 16)

    output = attention(q, k, v)

    assert output.shape == (2, 4, 8, 16)

def test_causal_mask():
    attention = ScaleDotProductAttention(causal=True)

    q = torch.zeros(1, 1, 3, 1)
    k = torch.zeros(1, 1, 3, 1)

    v = torch.tensor(
        [[[[1.0], [3.0], [9.0]]]]
    )

    output = attention(q, k, v)

    expected = torch.tensor(
        [[[[1.0], [2.0], [13.0 / 3.0]]]]
    )

    assert torch.allclose(output, expected, atol=1e-6)


def test_non_causal_attention():
    attention = ScaleDotProductAttention(causal=False)

    q = torch.zeros(1, 1, 3, 1)
    k = torch.zeros(1, 1, 3, 1)

    v = torch.tensor(
        [[[[1.0], [3.0], [9.0]]]]
    )

    output = attention(q, k, v)

    expected = torch.full_like(output, 13.0 / 3.0)

    assert torch.allclose(output, expected, atol=1e-6)


def test_attention_gradient(attention):
    q = torch.randn(2, 2, 4, 8, requires_grad=True)
    k = torch.randn(2, 2, 4, 8, requires_grad=True)
    v = torch.randn(2, 2, 4, 8, requires_grad=True)

    output = attention(q, k, v)
    loss = output.sum()

    loss.backward()

    assert q.grad is not None
    assert k.grad is not None
    assert v.grad is not None


def test_invalid_input_shape(attention):
    q = torch.randn(2, 4, 8)
    k = torch.randn(2, 4, 8)
    v = torch.randn(2, 4, 8)

    with pytest.raises(ValueError):
        attention(q, k, v)


def test_mismatched_shapes(attention):
    q = torch.randn(2, 4, 8, 16)
    k = torch.randn(2, 4, 8, 16)
    v = torch.randn(2, 4, 8, 32)

    with pytest.raises(ValueError):
        attention(q, k, v)


def test_invalid_input_type(attention):
    with pytest.raises(TypeError):
        attention([[1, 2]], [[1, 2]], [[1, 2]])