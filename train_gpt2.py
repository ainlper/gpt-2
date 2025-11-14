import math
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        # 确保嵌入维度能够被头数整除，以便均匀分割
        assert config.n_embd % config.n_head == 0
        # 将输入同时投影到Q、K、V三个矩阵，输出维度是3倍嵌入维度
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # 输出投影层，将多头注意力结果映射回原始维度
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # 创建下三角矩阵，确保只能看到左侧的token
        # torch.tril生成下三角矩阵，1表示允许关注，0表示禁止
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        # 获取输入张量的维度：批大小、序列长度、嵌入维度
        B, T, C = x.size()
        # 1. 通过线性变换同时计算Q、K、V，输出形状为(B, T, 3*C)
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        # 2. 重塑为多头格式
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        # 3. 计算注意力分数
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # 4. 应用因果掩码， 应用因果掩码：将掩码为0的位置填充为负无穷，这样softmax后概率为0
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        # 5. Softmax归一化
        att = F.softmax(att, dim=-1)
        # 6. 注意力加权求和
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        # 7. 重塑回原始格式
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        # 8. 输出投影
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 通过扩展再压缩的"瓶颈"结构来增加模型容量
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)  # 功能: 将输入维度从 n_embd 扩展到 4 * n_embd,作用: 增加模型的表示能力，为非线性变换提供更多维度
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)  # 功能: 将维度从 4 * n_embd 投影回 n_embd, 作用: 恢复原始维度，保持残差连接的兼容性

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    block_size: int = 1024  # 上下文长度/序列长度
    vocab_size: int = 50257  # 词汇表大小
    n_layer: int = 12  # Transformer层数
    n_head: int = 12  # 注意力头数
    n_embd: int = 768  # 嵌入维度


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

    def forward(self, idx, targets=None):
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=idx.device)  # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (t, n_embd)
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        return logits

    @classmethod
    def from_pretrained(cls, model_type):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)
        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2': dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
            'gpt2-large': dict(n_layer=36, n_head=20, n_embd=1280),  # 774M params
            'gpt2-xl': dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257  # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024  # always 1024 for GPT model checkpoints
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]  # discard this mask / buffer, not a param

        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]  # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]  # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model


num_return_sequences = 5
max_length = 30

'''enc = tiktoken.get_encoding('gpt2')
with open('input.txt', 'r') as f:
    text = f.read()
text = text[:1000]
tokens = enc.encode(text)
B, T = 4, 32
buf = torch.tensor(tokens[:B * T + 1])
x = buf[:-1].view(B, T)
y = buf[1:].view(B, T)'''

# model = GPT.from_pretrained('gpt2')
model = GPT(GPTConfig())
model.eval()
model.to('cuda')

enc = tiktoken.get_encoding('gpt2')
tokens = enc.encode("Hello,I'm a language model,")
tokens = torch.tensor(tokens, dtype=torch.long)  # (8,)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)  # (5,8
x = tokens.to('cuda')
torch.manual_seed(42)
while x.size(1) < max_length:
    with torch.no_grad():
        logits = model(x)
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=1)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        ix = torch.multinomial(topk_probs, 1)
        xcol = torch.gather(topk_indices, -1, ix)
        x = torch.cat((x, xcol), dim=1)
for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded)
