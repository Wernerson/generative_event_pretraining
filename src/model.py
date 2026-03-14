from torch import nn
import torch.nn.functional as F

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embed % config.n_head == 0
        self.fc_att = nn.Linear(config.n_embed, 3 * config.n_embed)
        self.fc_out = nn.Linear(config.n_embed, config.n_embed)
        self.n_head = config.n_head
        self.n_embed = config.n_embed

    def forward(self, x, is_causal):
        B, T, C = x.size()
        qkv = self.fc_att(x)
        q, k, v = qkv.split(self.n_embed, dim=-1)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)     # flash attention
        y = y.transpose(1, 2).contiguous().view(B, T, C)                # re-assemble all head outputs side by side
        y = self.fc_out(y)
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1    = nn.Linear(config.n_embed, 4 * config.n_embed)
        self.gelu   = nn.GELU()
        self.fc2    = nn.Linear(4 * config.n_embed, config.n_embed)

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        return x

class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int = 256):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
    
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1    = nn.LayerNorm(config.n_embed)
        self.atten  = CausalSelfAttention(config)
        self.ln2    = nn.LayerNorm(config.n_embed)
        # self.mlp    = MLP(config)
        self.mlp    = SwiGLU(config.n_embed, 4 * config.n_embed)

    def forward(self, x, is_causal=False):
        x = x + self.atten(self.ln1(x), is_causal=is_causal)
        x = x + self.mlp(self.ln2(x))
        return x

class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.event_encoder       = None
        self.image_encoder       = None
        self.transformer         = None
        self.decoder             = None


        
        