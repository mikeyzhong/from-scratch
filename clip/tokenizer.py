import torch
import tiktoken


class CLIPTokenizer:
    def __init__(self, max_length: int = 77):
        self.enc = tiktoken.get_encoding("gpt2")
        self.max_length = max_length
        self.sot_token = self.enc.max_token_value + 1
        self.eot_token = self.enc.max_token_value + 2
        self.vocab_size = self.enc.max_token_value + 3

    def __call__(self, texts: list[str] | str) -> torch.LongTensor:
        if isinstance(texts, str):
            texts = [texts]

        batch = []
        for text in texts:
            tokens = [self.sot_token] + self.enc.encode(text.lower()) + [self.eot_token]
            tokens = tokens[:self.max_length]
            tokens = tokens + [0] * (self.max_length - len(tokens))
            batch.append(tokens)

        return torch.tensor(batch, dtype=torch.long)
