import pytest

from minillm.tokenizer import SimpleTokenizer

@pytest.fixture
def tokenizer():
    vocab = ["[UNK]", "I", "love", "CUDA", "Pytorch"]
    return SimpleTokenizer(vocab)

def test_encode_known_tokens(tokenizer):
    result = tokenizer.encode("I love CUDA")

    assert result == [1, 2, 3]

def test_encode_unknown_tokens(tokenizer):
    result = tokenizer.encode("I love Python")

    assert result == [1, 2, 0]

def test_decode(tokenizer):
    result = tokenizer.decode([1, 2, 4])

    assert result == "I love Pytorch"

def test_empty_text(tokenizer):
    assert tokenizer.tokenize("") == []
    assert tokenizer.encode("") == []
    assert tokenizer.decode([]) == ""


def test_missing_unk_token():
    with pytest.raises(ValueError):
        SimpleTokenizer(["I", "love", "CUDA"])


def test_duplicate_tokens():
    with pytest.raises(ValueError):
        SimpleTokenizer(["[UNK]", "I", "love", "I"])


def test_invalid_token_id(tokenizer):
    with pytest.raises(ValueError):
        tokenizer.decode([1, 2, 100])


def test_invalid_text_type(tokenizer):
    with pytest.raises(TypeError):
        tokenizer.encode(123)


def test_invalid_token_id_type(tokenizer):
    with pytest.raises(TypeError):
        tokenizer.decode([1, "2", 3])