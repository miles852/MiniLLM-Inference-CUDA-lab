import torch
import pytest

from minillm.layers.norm import RMSNorm
from minillm.embedding import TokenEmbedding

@pytest.fixture
def norm():
    return RMSNorm(hidden_size=8)

def test_rmsnorm_output_shape(norm):
    x = torch.randn(2, 4, 8)

    output = norm(x)

    assert output.shape == x.shape

def test_rmsnorm_reference(norm):
    x = torch.randn(2, 4, 8)

    output = norm(x)
    mean_square = x.pow(2).mean(
        dim=-1,
        keepdim=True
    )
    expected = x * torch.rsqrt(mean_square + norm.eps)
    expected = expected * norm.weight

    assert torch.allclose(output, expected, atol=1e-6)

def test_weight_initialization(norm):
    assert torch.equal(
        norm.weight,
        torch.ones(8)
    )

def test_rmsnorm_gradient(norm):
    x = torch.randn(2, 4, 8, requires_grad=True)

    output = norm(x)
    loss = output.sum()

    loss.backward()

    assert x.grad is not None
    assert norm.weight.grad is not None


def test_rmsnorm_invalid_hidden_size(norm):
    x = torch.randn(2, 4, 16)

    with pytest.raises(ValueError):
        norm(x)


def test_rmsnorm_invalid_input_type(norm):
    with pytest.raises(TypeError):
        norm([[1, 2, 3]])

def test_embedding_to_rmsnorm():
    embedding = TokenEmbedding(vocab_size=10, embedding_dim=8)

    norm = RMSNorm(hidden_size=8)

    input = torch.tensor(
        [[1, 2, 3],
         [4, 5, 6]],
         dtype=torch.long
    )
    x = embedding(input)
    y = norm(x)

    assert x.shape == (2, 3, 8)
    assert y.shape == (2, 3, 8)