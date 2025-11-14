from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F


class Block(nn.Module):
    def _init_(self, config):
        super().__init_()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalselfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    block_size: int = 256  # 上下文长度/序列长度
    vocab_size: int = 65  # 词汇表大小
    n_layer: int = 6  # Transformer层数
    n_head: int = 6  # 注意力头数
    n_embd: int = 384  # 嵌入维度


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embd),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
