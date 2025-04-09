import math
import torch
from torch import nn
import torch.nn.functional as F

# helper functions

def default(val, default_val):
    return val if val is not None else default_val

def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor

# helper classes

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x):
        return x + self.fn(x)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
    def forward(self, x):
        return self.fn(self.norm(x))

# This wrapper is for cross-attention which requires two inputs.
class PreNormCross(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
    def forward(self, x1, x):
        # Normalize only x (the main input) and then call fn with x1 and normalized x
        return self.fn(x1, self.norm(x))

class GELU_(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

GELU = nn.GELU if hasattr(nn, 'GELU') else GELU_

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0., activation=None, glu=False):
        super().__init__()
        activation = default(activation, GELU)
        self.glu = glu
        self.w1 = nn.Linear(dim, dim * mult * (2 if glu else 1))
        self.act = activation()
        self.dropout = nn.Dropout(dropout)
        self.w2 = nn.Linear(dim * mult, dim)
    def forward(self, x, **kwargs):
        if not self.glu:
            x = self.w1(x)
            x = self.act(x)
        else:
            x, v = self.w1(x).chunk(2, dim=-1)
            x = self.act(x) * v
        x = self.dropout(x)
        x = self.w2(x)
        return x

# (Your existing LinformerCrossAttention definition remains unchanged)
class LinformerCrossAttention(nn.Module):
    def __init__(self, dim, seq_len, k=256, heads=8, dim_head=None, one_kv_head=False, share_kv=False, dropout=0.):
        super().__init__()
        assert (dim % heads) == 0, 'dimension must be divisible by the number of heads'
        self.seq_len = seq_len
        self.k = k
        self.heads = heads
        dim_head = default(dim_head, dim // heads)
        self.dim_head = dim_head
        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        kv_dim = dim_head if one_kv_head else (dim_head * heads)
        self.to_k = nn.Linear(dim, kv_dim, bias=False)
        self.proj_k = nn.Parameter(init_(torch.zeros(seq_len, k)))
        self.share_kv = share_kv
        if not share_kv:
            self.to_v = nn.Linear(dim, kv_dim, bias=False)
            self.proj_v = nn.Parameter(init_(torch.zeros(seq_len, k)))
        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(dim_head * heads, dim)
    def forward(self, x1, x, context=None, **kwargs):
        b, n, d, d_h, h, k = *x.shape, self.dim_head, self.heads, self.k
        kv_len = n if context is None else context.shape[1]
        assert kv_len == self.seq_len, f'the sequence length of the key/values must be {self.seq_len} - {kv_len} given'
        queries = self.to_q(x1)
        proj_seq_len = lambda args: torch.einsum('bnd,nk->bkd', *args)
        kv_input = x if context is None else context
        keys = self.to_k(kv_input)
        values = self.to_v(kv_input) if not self.share_kv else keys
        kv_projs = (self.proj_k, self.proj_v if not self.share_kv else self.proj_k)
        keys, values = map(proj_seq_len, zip((keys, values), kv_projs))
        queries = queries.reshape(b, 1, h, -1).transpose(1, 2)
        merge_key_values = lambda t: t.reshape(b, k, -1, d_h).transpose(1, 2).expand(-1, h, -1, -1)
        keys, values = map(merge_key_values, (keys, values))
        dots = torch.einsum('bhnd,bhkd->bhnk', queries, keys) * (d_h ** -0.5)
        attn = dots.softmax(dim=-1)
        attn = self.dropout(attn)
        out = torch.einsum('bhnk,bhkd->bhnd', attn, values)
        out = out.transpose(1, 2).reshape(b, 1, -1)
        return self.to_out(out)

# Now, define a custom transformer block for LinformerCA that handles the two different input requirements.
class LinformerCABlock(nn.Module):
    def __init__(self, dim, attn, ff):
        super().__init__()
        # Use PreNormCross for attention (which accepts x1, x)
        self.attn = PreNormCross(dim, attn)
        # Use standard PreNorm for feedforward (which takes one tensor)
        self.ff = PreNorm(dim, ff)
    def forward(self, x1, x):
        # Apply cross-attention: returns a tensor that we add to x
        attn_out = self.attn(x1, x)
        x = x + attn_out
        # Apply feedforward to x only
        ff_out = self.ff(x)
        x = x + ff_out
        return (x1, x)

# Finally, define LinformerCA using our custom block.
class LinformerCA(nn.Module):
    def __init__(self, dim, seq_len, depth, k=256, heads=8, dim_head=None, one_kv_head=False, share_kv=False, reversible=False, dropout=0.):
        super().__init__()
        layers = nn.ModuleList([])
        for _ in range(depth):
            attn = LinformerCrossAttention(dim, seq_len, k=k, heads=heads, dim_head=dim_head,
                                           one_kv_head=one_kv_head, share_kv=share_kv, dropout=dropout)
            ff = FeedForward(dim, dropout=dropout)
            layers.append(LinformerCABlock(dim, attn, ff))
        self.layers = layers

    def forward(self, x1, x):
        for layer in self.layers:
            x1, x = layer(x1, x)
        return x