from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from smart_open import open


def main():
    with open("https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt", "r", encoding="utf-8") as f:
        text = f.read()
    foo = Foo()
    foo.train(text)


class Block(torch.nn.Module):
    def __init__(self, n_heads: int, n_embed: int, block_size: int, dropout: float):
        super().__init__()
        head_size = n_embed // n_heads
        self.sa = MultiHeadAttention(n_heads=n_heads, n_embed=n_embed, head_size=head_size, block_size=block_size, dropout=dropout)
        self.ffwd = FeedForward(n_embed=n_embed, dropout=dropout)
        self.ln1 = torch.nn.LayerNorm(n_embed)
        self.ln2 = torch.nn.LayerNorm(n_embed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.sa(self.ln1(x))  # Communication (between tokens)
        x = x + self.ffwd(self.ln2(x))  # Computation (think about information we now have)
        return x


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, n_heads: int, n_embed: int, head_size: int, block_size: int, dropout: float):
        super().__init__()
        self.heads = torch.nn.ModuleList([Head(n_embed, head_size, block_size, dropout) for _ in range(n_heads)])
        self.linear = torch.nn.Linear(n_embed, n_embed)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.cat([h(x) for h in self.heads], dim=-1)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class Head(torch.nn.Module):
    def __init__(self, n_embed: int, head_size: int, block_size: int, dropout: float):
        super().__init__()
        self.head_size = head_size
        self.block_size = block_size
        self.query = torch.nn.Linear(n_embed, head_size, bias=False)
        self.key = torch.nn.Linear(n_embed, head_size, bias=False)
        self.value = torch.nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))  # Not a param
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (shape: (batch_size, block_size, n_embed)): Embeddings for each token

        Returns:
            out (shape: (batch_size, block_size, n_embed))
        """
        q = self.query(x)  # (B, T, H)
        k = self.key(x)  # (B, T, H)
        v = self.value(x)
        """divide by sqrt headsize makes weights more evenly distributed. Takes variance from ~head_size to ~1.0.
        Shapes are (B, T, H) @ (B, H, T) -> (B, T, T)
        """
        weight = q @ k.transpose(-2, -1) * self.head_size ** -0.5
        weight = weight.masked_fill(self.tril[:self.block_size, :self.block_size] == 0, float("-inf"))
        weight = torch.nn.functional.softmax(weight, dim=-1)  # Trick that makes -inf -> zero. Normalize rest
        weight = self.dropout(weight)
        out = weight @ v  # (B, T, T) @ (B, T, H) -> (B, T, H)
        return out


class FeedForward(torch.nn.Module):
    def __init__(self, n_embed: int, dropout: float):
        super().__init__()
        self.sequential = torch.nn.Sequential(
            torch.nn.Linear(n_embed, 4 * n_embed),  # 4* b/c Transformer paper says more computation is better here
            torch.nn.ReLU(),
            torch.nn.Linear(4 * n_embed, n_embed),
            torch.nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sequential(x)


class NanoGPT(torch.nn.Module):
    def __init__(self, n_token: int, n_embed: int, block_size: int, n_heads: int, n_layer: int, dropout: float, device: str) -> None:
        super().__init__()
        self.device = device
        self.block_size = block_size
        self.tok_embed = torch.nn.Embedding(n_token, n_embed)  # (n_token, n_embed) transition matrix
        self.pos_embed = torch.nn.Embedding(block_size, n_embed)  # (block_size, n_embed) transition matrix
        self.blocks = torch.nn.Sequential(
            *[Block(n_heads=n_heads, n_embed=n_embed, block_size=block_size, dropout=dropout) for _ in range(n_layer)],
        )
        self.ln_f = torch.nn.LayerNorm(n_embed)
        self.lm_head = torch.nn.Linear(n_embed, n_token)  # language model head. from embedding space back to tokens

    def forward(
        self,
        tokens: torch.Tensor,
        targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            tokens (shape: (batch_size, block_size)): Tokens for each batch for each token in the
                block
            targets (shape: (batch_size, block_size)): Target for each corresponding cumulative
                token. Target `targets[i, j]` corresponds to `tokens[i, :j + 1]`

        Returns
            logits (shape: (batch_size, block_size, n_token)): Vector of logits for each batch for
                each cumulative input tokens
            loss (shape: (,)): Mean cross entropy loss accross batch and tokens in block
        """
        B, T = tokens.shape
        tok_embed = self.tok_embed(tokens)  # (batch_size, block_size, n_embed)
        pos_embed = self.pos_embed(torch.arange(T, device=self.device))  # (block_size, n_embed)
        x = tok_embed + pos_embed  # embedding space is now the sum of tok and pos embeddings
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (batch_size, block_size, n_token)
        if targets is None:
            return logits, None

        # cross_entropy expects shape (examples, channels). Move batch & block to single axis
        B, T, C = logits.shape  # batch, time, channel
        loss = torch.nn.functional.cross_entropy(logits.view(-1, C), targets.view(-1))
        return logits, loss

    def generate(self, tokens: torch.Tensor, k: int) -> torch.Tensor:
        """
        Args:
            tokens (shape: (batch_size, block_size))
            k: Number of new tokens to generate
        """
        for _ in range(k):
            tokens_cropped = tokens[:, -self.block_size:]  # use at most block_size tokens at inference time
            logits, _ = self(tokens_cropped)  # (batch_size, block_size, n_tokens)
            logits = logits[:, -1, :]  # get last token in block (what we want to predict)
            probs = torch.nn.functional.softmax(logits, dim=-1)  # (batch_size, n_tokens)
            token_next = torch.multinomial(probs, num_samples=1)  # sample 1 token for each batch
            # Concat predicted tokens to existing for next iteration (batch_size, k)
            tokens = torch.cat((tokens, token_next), dim=1)
        return tokens


@dataclass
class Foo():
    batch_size: int = 64
    block_size: int = 64
    n_embed: int = 128
    n_heads: int = 4  # Each head is 128 / 4 = 64 dimensional
    n_layer: int = 3
    dropout: float = 0.2
    max_step: int = 5000
    eval_steps: int = 200
    lr: float = 3e-4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    n_eval_batches: int = 200

    # batch_size: int = 64
    # block_size: int = 256
    # n_embed: int = 384
    # n_heads: int = 6  # Each head is 384 / 6 = 64 dimensional
    # n_layer: int = 6
    # dropout: float = 0.2
    # max_step: int = 5000
    # eval_steps: int = 200
    # lr: float = 3e-4
    # device: str = "cuda" if torch.cuda.is_available() else "cpu"
    # n_eval_batches: int = 200

    def train(self, text: str) -> None:
        # data
        tokenizer = Tokenizer(text)
        data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
        n = int(0.9 * len(data))
        train_data = data[:n]
        val_data = data[n:]

        # model
        self.model = NanoGPT(
            len(tokenizer),
            self.n_embed,
            self.block_size,
            self.n_heads,
            self.n_layer,
            self.dropout,
            device=self.device
        ).to(self.device)

        # train
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        for step in range(self.max_step):
            if step % self.eval_steps == 0:
                train_loss, val_loss = self.estimate_loss(train_data, val_data)
                print(f"step {step}: train_loss {train_loss:.4f}, val_loss {val_loss:.4f}")

                tokens = torch.zeros((1, self.block_size), dtype=torch.long).to(self.device)
                out = self.model.generate(tokens, 100)
                print(tokenizer.decode(out[0].tolist()))
            xb, yb = self.get_batch(train_data)
            logits, loss = self.model(xb.to(self.device), yb.to(self.device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        print(loss.item())

        tokens = torch.zeros((1, self.block_size), dtype=torch.long).to(self.device)
        out = self.model.generate(tokens, 10000)
        print(tokenizer.decode(out[0].tolist()))


    def get_batch(self, data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        idx = torch.randint(len(data) - self.block_size, (self.batch_size,))
        x = torch.stack([data[i:i + self.block_size] for i in idx])
        y = torch.stack([data[i + 1:i + self.block_size + 1] for i in idx])
        return x, y

    @torch.no_grad()
    def estimate_loss(self, train_data: torch.Tensor, val_data: torch.tensor) -> Tuple[float, float]:
        out = []
        self.model.eval()
        for data in [train_data, val_data]:
            losses = torch.zeros(self.n_eval_batches)
            for k in range(self.n_eval_batches):
                x, y = self.get_batch(data)
                _, loss = self.model(x, y)
                losses[k] = loss.item()
            out.append(losses.mean())
        self.model.train()
        return tuple(out)


class Tokenizer():
    def __init__(self, text: str) -> None:
        chars = sorted(list(set(text)))
        self.c2i = {c: i for i, c in enumerate(chars)}
        self.i2c = {i: c for i, c in enumerate(chars)}

    def __len__(self) -> int:
        return len(self.c2i)

    def encode(self, text: str) -> List[int]:
        return [self.c2i[c] for c in text]

    def decode(self, tokens: List[int]) -> str:
        return "".join([self.i2c[token] for token in tokens])


main()
