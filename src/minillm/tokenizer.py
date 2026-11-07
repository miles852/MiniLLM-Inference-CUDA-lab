class SimpleTokenizer:
    UNK_TOKEN = "[UNK]"

    def __init__(self, vocab):

        if not isinstance(vocab, (list, tuple)):
            raise TypeError("vocab must be a list or a tuple")

        if not all(isinstance(token, str) for token in vocab):
            raise TypeError("All vocabulary tokens must be strings")

        if self.UNK_TOKEN not in vocab:
            raise ValueError("vocab must contain UNK_TOKEN")

        if len(vocab) != len(set(vocab)):
            raise ValueError("vocab contains duplicate tokens")

        self.token_to_id = {
            token: idx
            for idx, token in enumerate(vocab)
        }

        self.id_to_token = {
            idx: token
            for idx, token in enumerate(vocab)
        }

        self.unk_id = self.token_to_id[self.UNK_TOKEN]

    def tokenize(self, text):

        if not isinstance(text, str):
            raise TypeError("text must be a string")

        return text.strip().split()


    def encode(self, text):

        tokens = self.tokenize(text)

        token_ids = [
            self.token_to_id.get(token, self.unk_id)
            for token in tokens
        ]

        return token_ids

    def decode(self, token_ids):

        if not isinstance(token_ids, (list, tuple)):
            raise TypeError("token_ids must be ad list or a tuple")

        tokens = []

        for token_id in token_ids:

            if not isinstance(token_id, int) or isinstance(token_id, bool):
                raise TypeError("token_id must be an integer")

            if token_id not in self.id_to_token:
                raise ValueError(f"invalid token ID: {token_id}")

            tokens.append(self.id_to_token[token_id])

        return " ".join(tokens)