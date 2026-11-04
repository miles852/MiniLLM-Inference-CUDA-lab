import torch
import torch.nn as nn

def test_tensor_operations():
    x = torch.randn(2, 4, 768)
    assert x.shape == (2, 4, 768)

    linear = nn.Linear(768, 1024)
    y = linear(x)

    assert y.shape == (2, 4, 1024)

def test_embedding_lookup():
    embedding = nn.Embedding(10, 8)
    input_ids = torch.tensor(
        [[1, 2, 3],
         [4, 5, 6]]
    )

    output = embedding(input_ids)
    assert output.shape == (2, 3, 8)

    expected = embedding.weight[input_ids]
    assert torch.equal(expected, output)