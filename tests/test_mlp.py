import torch
import pytest
import torch.nn.functional as F

from minillm.layers.mlp import MLP
from minillm.layers.norm import RMSNorm
from minillm.embedding import TokenEmbedding

@pytest.fixture
def mlp():
    return MLP(hidden_size=8, intermediate_size=16)

def test_mlp_output_shape(mlp):
    x = torch.randn(2, 3, 8)

    output = mlp(x)

    assert output.shape == (2, 3, 8)

def test_mlp_inference(mlp):
    x = torch.randn(2, 3, 8)

    output = mlp(x)

    expected = mlp.down_proj(F.silu(mlp.gate_proj(x)) * mlp.up_proj(x))

    assert torch.allclose(output, expected, atol=1e-6)

def test_mlp_gradient(mlp):
    x = torch.randn(2, 3, 8, requires_grad=True)

    output = mlp(x)
    loss = output.sum()

    loss.backward()
    assert x.grad is not None
    assert mlp.down_proj is not None
    assert mlp.up_proj is not None
    assert mlp.gate_proj is not None

def test_mlp_invalid_input_type(mlp):
    with pytest.raises(TypeError):
        mlp([[1, 2, 3]])


def test_mlp_invalid_input_shape(mlp):
    x = torch.randn(2, 8)

    with pytest.raises(ValueError):
        mlp(x)


def test_mlp_hidden_size_mismatch(mlp):
    x = torch.randn(2, 4, 16)

    with pytest.raises(ValueError):
        mlp(x)

def test_mlp_projection_shapes(mlp):
    assert mlp.gate_proj.weight.shape == (16, 8)
    assert mlp.up_proj.weight.shape == (16, 8)
    assert mlp.down_proj.weight.shape == (8, 16)

def test_embedding_norm_mlp_pipeline():
    embedding = TokenEmbedding(vocab_size=10, embedding_dim=8)

    norm = RMSNorm(hidden_size=8)

    mlp = MLP(hidden_size=8, intermediate_size=16)

    input = torch.tensor(
        [[1, 2, 3],
         [4, 5, 6]],
         dtype=torch.long
    )

    x = embedding(input)
    y = norm(x)
    z = mlp(y)

    assert z.shape == (2, 3, 8)
