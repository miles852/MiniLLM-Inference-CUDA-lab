import pytest
import torch

from minillm.embedding import TokenEmbedding
from minillm.tokenizer import SimpleTokenizer

@pytest.fixture
def embedding():
    return TokenEmbedding(vocab_size=10, embedding_dim=8)

def test_embedding_output_shape(embedding):
    input_ids = torch.tensor(
        [[1, 2, 3],
        [4, 5, 6]],
        dtype=torch.long
    )

    output = embedding(input_ids)
    assert output.shape == (2, 3, 8)

def test_embedding_lookup(embedding):
    input_ids = torch.tensor(
        [[1, 2, 3]],
        dtype=torch.long
    )

    expected = embedding.embedding.weight[input_ids]
    output = embedding(input_ids)

    assert torch.equal(expected, output)

def test_repeated_token_same_embedding(embedding):
    input_ids = torch.tensor(
        [[2, 2]],
        dtype=torch.long
    )

    out_put = embedding(input_ids)
    assert torch.equal(out_put[0, 0], out_put[0, 1])

def test_embedding_receives_gradients(embedding):
    input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)

    output = embedding(input_ids)
    loss = output.sum()

    loss.backward()

    grad = embedding.embedding.weight.grad

    assert grad is not None
    assert torch.count_nonzero(grad[1]) > 0


def test_invalid_input_type(embedding):
    with pytest.raises(TypeError):
        embedding([[1, 2, 3]])


def test_invalid_input_shape(embedding):
    input_ids = torch.tensor([1, 2, 3], dtype=torch.long)

    with pytest.raises(ValueError):
        embedding(input_ids)


def test_invalid_input_dtype(embedding):
    input_ids = torch.tensor([[1, 2, 3]], dtype=torch.float32)

    with pytest.raises(TypeError):
        embedding(input_ids)


def test_out_of_range_token_id(embedding):
    input_ids = torch.tensor([[100]], dtype=torch.long)

    with pytest.raises((IndexError, RuntimeError)):
        embedding(input_ids)

def test_embedding_tokenizer_pipeline():
    vocab = ["[UNK]", "I", "love", "CUDA", "Torch"]

    tokenizer = SimpleTokenizer(vocab)
    embedding = TokenEmbedding(vocab_size=len(vocab), embedding_dim=8)

    text = "I love CUDA"

    token_ids = tokenizer.encode(text)
    input_ids = torch.tensor(
        [token_ids],
        dtype=torch.long
    )

    output = embedding(input_ids)

    assert input_ids.shape == (1, 3)
    assert output.shape == (1, 3, 8)