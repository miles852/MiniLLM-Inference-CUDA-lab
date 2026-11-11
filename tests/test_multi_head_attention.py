
import pytest
import torch

from minillm.layers.multi_head_attention import MultiHeadAttention


@pytest.fixture
def mha():
    return MultiHeadAttention(
        hidden_size=32,
        num_heads=4
    )


def test_mha_output_shape(mha):
    x = torch.randn(2, 8, 32)

    output = mha(x)

    assert output.shape == (2, 8, 32)


def test_split_heads(mha):
    x = torch.randn(2, 8, 32)

    output = mha.split_heads(x)

    assert output.shape == (2, 4, 8, 8)


def test_merge_heads(mha):
    x = torch.randn(2, 4, 8, 8)

    output = mha.merge_heads(x)

    assert output.shape == (2, 8, 32)


def test_split_merge_round_trip(mha):
    x = torch.randn(2, 8, 32)

    result = mha.merge_heads(
        mha.split_heads(x)
    )

    assert torch.equal(result, x)


def test_mha_reference(mha):
    x = torch.randn(2, 8, 32)

    output = mha(x)

    batch_size, seq_len, _ = x.shape

    q = mha.split_heads(mha.q_proj(x))
    k = mha.split_heads(mha.k_proj(x))
    v = mha.split_heads(mha.v_proj(x))

    q, k = mha.rope(q, k)

    expected = mha.attention(q, k, v)

    expected = mha.merge_heads(expected)
    expected = mha.o_proj(expected)

    assert torch.allclose(
        output,
        expected,
        atol=1e-6
    )


def test_mha_gradient(mha):
    x = torch.randn(
        2, 8, 32,
        requires_grad=True
    )

    output = mha(x)
    loss = output.sum()

    loss.backward()

    assert x.grad is not None

    assert mha.q_proj.weight.grad is not None
    assert mha.k_proj.weight.grad is not None
    assert mha.v_proj.weight.grad is not None
    assert mha.o_proj.weight.grad is not None


def test_invalid_input_type(mha):
    with pytest.raises(TypeError):
        mha([[1, 2, 3]])


def test_invalid_input_shape(mha):
    x = torch.randn(2, 32)

    with pytest.raises(ValueError):
        mha(x)


def test_hidden_size_mismatch(mha):
    x = torch.randn(2, 8, 64)

    with pytest.raises(ValueError):
        mha(x)


def test_invalid_head_configuration():
    with pytest.raises(ValueError):
        MultiHeadAttention(
            hidden_size=30,
            num_heads=4
        )


def test_odd_head_dimension():
    with pytest.raises(ValueError):
        MultiHeadAttention(
            hidden_size=15,
            num_heads=3
        )
