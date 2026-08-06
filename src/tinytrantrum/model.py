from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ModelConfig:
    vocabulary_size: int
    context_length: int = 256
    layers: int = 6
    heads: int = 6
    embedding_size: int = 384
    dropout: float = 0.2
    use_flash_attention: bool = True

    def __post_init__(self) -> None:
        if self.vocabulary_size <= 0:
            raise ValueError("vocabulary_size must be positive")
        if self.embedding_size % self.heads != 0:
            raise ValueError("embedding_size must be divisible by heads")


class Linear(nn.Module):
    def __init__(self, input_size: int, output_size: int, bias: bool = True) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        self.bias = nn.Parameter(torch.zeros(output_size)) if bias else None
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def forward(self, inputs: Tensor) -> Tensor:
        return F.linear(inputs, self.weight, self.bias)


class Embedding(nn.Module):
    def __init__(self, vocabulary_size: int, embedding_size: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(vocabulary_size, embedding_size))
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def forward(self, tokens: Tensor) -> Tensor:
        return self.weight[tokens]


class LayerNorm(nn.Module):
    def __init__(self, size: int, epsilon: float = 1e-5) -> None:
        super().__init__()
        self.epsilon = epsilon
        self.scale = nn.Parameter(torch.ones(size))
        self.shift = nn.Parameter(torch.zeros(size))

    def forward(self, inputs: Tensor) -> Tensor:
        mean = inputs.mean(dim=-1, keepdim=True)
        variance = (inputs - mean).pow(2).mean(dim=-1, keepdim=True)
        normalized = (inputs - mean) / torch.sqrt(variance + self.epsilon)
        return self.scale * normalized + self.shift


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.heads = config.heads
        self.head_size = config.embedding_size // config.heads
        self.query = Linear(config.embedding_size, config.embedding_size)
        self.key = Linear(config.embedding_size, config.embedding_size)
        self.value = Linear(config.embedding_size, config.embedding_size)
        self.output = Linear(config.embedding_size, config.embedding_size)
        self.attention_dropout = nn.Dropout(config.dropout)
        self.residual_dropout = nn.Dropout(config.dropout)
        self.use_flash_attention = config.use_flash_attention
        mask = torch.tril(torch.ones(config.context_length, config.context_length))
        self.register_buffer("causal_mask", mask.view(1, 1, config.context_length, config.context_length))

    def forward(self, inputs: Tensor, return_weights: bool = False) -> Tensor | tuple[Tensor, Tensor]:
        batch, length, channels = inputs.shape
        queries = self.query(inputs).view(batch, length, self.heads, self.head_size).transpose(1, 2)
        keys = self.key(inputs).view(batch, length, self.heads, self.head_size).transpose(1, 2)
        values = self.value(inputs).view(batch, length, self.heads, self.head_size).transpose(1, 2)
        weights = None
        if self.use_flash_attention and not return_weights and hasattr(F, "scaled_dot_product_attention"):
            attended = F.scaled_dot_product_attention(
                queries,
                keys,
                values,
                dropout_p=self.attention_dropout.p if self.training else 0.0,
                is_causal=True,
            )
        else:
            scores = queries @ keys.transpose(-2, -1) / math.sqrt(self.head_size)
            scores = scores.masked_fill(self.causal_mask[:, :, :length, :length] == 0, float("-inf"))
            weights = F.softmax(scores, dim=-1)
            attended = self.attention_dropout(weights) @ values
        attended = attended.transpose(1, 2).contiguous().view(batch, length, channels)
        output = self.residual_dropout(self.output(attended))
        return (output, weights) if return_weights else output


class FeedForward(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        hidden_size = 4 * config.embedding_size
        self.up = Linear(config.embedding_size, hidden_size)
        self.down = Linear(hidden_size, config.embedding_size)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.dropout(self.down(F.gelu(self.up(inputs))))


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.norm_attention = LayerNorm(config.embedding_size)
        self.attention = CausalSelfAttention(config)
        self.norm_feed_forward = LayerNorm(config.embedding_size)
        self.feed_forward = FeedForward(config)

    def forward(self, inputs: Tensor, return_weights: bool = False) -> Tensor | tuple[Tensor, Tensor]:
        attention_result = self.attention(self.norm_attention(inputs), return_weights=return_weights)
        if return_weights:
            attention_output, weights = attention_result
            inputs = inputs + attention_output
        else:
            inputs = inputs + attention_result
            weights = None
        output = inputs + self.feed_forward(self.norm_feed_forward(inputs))
        return (output, weights) if return_weights else output


class CharacterTransformer(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = Embedding(config.vocabulary_size, config.embedding_size)
        self.position_embedding = Embedding(config.context_length, config.embedding_size)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.layers)])
        self.final_norm = LayerNorm(config.embedding_size)
        self.lm_head = Linear(config.embedding_size, config.vocabulary_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        residual_std = 0.02 / math.sqrt(2 * config.layers)
        for block in self.blocks:
            nn.init.normal_(block.attention.output.weight, mean=0.0, std=residual_std)
            nn.init.normal_(block.feed_forward.down.weight, mean=0.0, std=residual_std)

    def forward(
        self,
        tokens: Tensor,
        targets: Tensor | None = None,
        return_attention: bool = False,
    ) -> tuple[Tensor, Tensor | None] | tuple[Tensor, Tensor | None, list[Tensor]]:
        _, length = tokens.shape
        if length > self.config.context_length:
            raise ValueError("Sequence length exceeds context_length")
        positions = torch.arange(length, device=tokens.device)
        hidden = self.dropout(self.token_embedding(tokens) + self.position_embedding(positions))
        attention_maps: list[Tensor] = []
        for block in self.blocks:
            block_result = block(hidden, return_weights=return_attention)
            if return_attention:
                hidden, weights = block_result
                attention_maps.append(weights)
            else:
                hidden = block_result
        logits = self.lm_head(self.final_norm(hidden))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return (logits, loss, attention_maps) if return_attention else (logits, loss)
