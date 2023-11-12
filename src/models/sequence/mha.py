""" Wrapper around nn.MultiheadAttention to adhere to SequenceModule interface. """

import torch
import torch.nn.functional as F
from torch import nn
import hydra
from src.models.sequence.base import SequenceModule, TransposedModule
import src.models.nn.utils as U

from opt_einsum import contract as einsum
from einops import rearrange, repeat
from torch.nn.functional import scaled_dot_product_attention as sdpa
import copy

from local_attention.local_attention import LocalAttention, LocalFlashAttention
from local_attention.pykeops.local_attention import LocalAttention as LocalAttention_pykeops

try:
    import pykeops
    from pykeops.torch import LazyTensor

    pykeops.set_verbose(False)
    has_pykeops = True

except ImportError:
    has_pykeops = False


class RotaryEmbedding(nn.Module):
    """rope = RotaryEmbedding(64)
    q = torch.randn(1, 8, 1024, 64) # queries - (batch, heads, seq len, dimension of head)
    k = torch.randn(1, 8, 1024, 64) # keys

    # apply embeds to queries, keys after the heads have been split out but prior to dot products

    q = rope(q)
    k = rope(k)

    # then do attention with queries (q) and keys (k) as usual

    q_k = [q_k_0,...,q_k_d-1] is mapped to [...,cos(theta**(2mk/d))q_k_2m - sin(theta**(2mk/d))q_k_2m+1, cos(theta**(2mk/d))q_k_2m+1 + sin(theta**(2mk/d))q_k_2m,...]
    """

    def __init__(self, d, theta=10000):
        super().__init__()
        assert d % 2 == 0
        freqs = theta ** (-torch.arange(0, d, 2) / d)  # (d / 2)
        self.register_buffer('freqs', freqs)
        self.cache = dict()

    def get_freqs(self, pos, cache_key=None):
        if cache_key and cache_key in self.cache:
            return self.cache[cache_key]

        freqs = self.freqs  # (d/2)
        freqs = pos.to(freqs).view(-1, 1) * freqs  # (L d/2)

        cos, sin = freqs.cos(), freqs.sin()
        freqs = torch.stack((cos, -sin, sin, cos), dim=-1)  # (L d/2 4)
        freqs = rearrange(freqs, '... (r c) -> ... r c', c=2)  # (L d/2 2 2)

        if cache_key:
            self.cache[cache_key] = freqs

        return freqs  # (L d/2 2 2)

    def forward(self, x, seq_dim=-2):
        # x: (... L d)
        L = x.shape[seq_dim]
        freqs = self.get_freqs(torch.arange(L, device=x.device), L)  # (L d/2 2 2)
        x = rearrange(x, '... (d r) -> ... d r', r=2)  # (... L d/2 2)
        x = einsum('... r c, ... c -> ... r', freqs, x)  # (L d/2 2 2), (... L d/2 2)
        return rearrange(x, '... d r -> ... (d r)')


@TransposedModule
class MultiheadAttention(SequenceModule):
    """ Simple wrapper for MultiheadAttention """
    def __init__(self, d_model, n_heads, *args, causal=True, **kwargs):
        super().__init__()
        self.d_model = d_model
        self.d_output = d_model
        self.mha = nn.MultiheadAttention(d_model, n_heads, *args, batch_first=True, **kwargs)
        self.causal = causal

    def forward(self, src, attn_mask=None, key_padding_mask=None, state=None, **kwargs):
        """ state should represent a mask and key padding mask """
        if self.causal and attn_mask is None:
            attn_mask = torch.triu(torch.ones(src.size(-2), src.size(-2),
                                              dtype=torch.bool, device=src.device),
                                       diagonal=1)
        # attn_mask, key_padding_mask = state
        # Note that this returns None for the second argument
        y, _ = self.mha(src, src, src, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False)
        return y, None

    def step(self, x, state):
        # TODO proper cached inference
        # x: (B, D)
        y, z = self.mha(src, src, src, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False, **kwargs)

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

@TransposedModule
class MultiheadAttentionFlash(SequenceModule):
    """ Simple wrapper for MultiheadAttention using flash implementation of scaled dot product attention"""
    def __init__(self, d_model, n_heads, dropout=0.0, *args, bias=True, causal=True, rotary=False, **kwargs):
        "Take in model size and number of heads."
        super().__init__()
        # sequence model nessesary attributes
        self.d_model = d_model
        self.d_output = d_model

        assert d_model % n_heads == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // n_heads
        self.num_heads = n_heads
        self.causal = causal
        self.linears = clones(nn.Linear(d_model, d_model, bias=bias), 4)
        self.attn = sdpa
        self.dropout_p = dropout

        if rotary:
            self.rope = RotaryEmbedding(self.d_k)

    def forward(self, src, attn_mask=None, key_padding_mask=None, state=None, **kwargs):
        """
        src: (B, L, D)
        attn_mask: (B, L, L)
        """
        if key_padding_mask is not None:
            raise NotImplementedError("key padding Not implemented for now with module MultiHeadedAttentionFlash")
        if state is not None:
            raise NotImplementedError("state Not implemented for now with module MultiHeadedAttentionFlash")

        causal = self.causal if attn_mask is None else False
        nbatches = src.size(0)

        # 1) Do all the linear projections in batch from d_model => num_heads x d_k
        query, key, value = [
            lin(x).view(nbatches, -1, self.num_heads, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (src, src, src))
        ]  # (B, H, L, Dk)

        # 1.5) Add rotary positional embeddings if used
        if hasattr(self, 'rope'):
            query = self.rope(query)
            key = self.rope(key)

        # 2) Apply attention on all the projected vectors in batch.
        x = self.attn(query, key, value, attn_mask=attn_mask, dropout_p=self.dropout_p, is_causal=causal)  # (B, H, L, Dk)

        # 3) "Concat" using a view and apply a final linear.
        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(nbatches, -1, self.num_heads * self.d_k)
        )  # (B, L, D)
        del query
        del key
        del value
        return self.linears[-1](x), None  # None to match output signature of MultiheadAttention

    def step(self, x, state):
        raise NotImplementedError("Not implemented for now with module MultiHeadedAttentionFlash")


@TransposedModule
class MultiheadLocalAttention(SequenceModule):
    """
    Simple wrapper for MultiheadAttention with local attention using flash implementation of scaled dot product attention.

    Args:
        d_model (int): The input dimension of the model.
        n_heads (int): Number of attention heads.
        window_size (int): Window size for local attention.
        look_backward (int, optional): Number of positions to look backward in local attention. Defaults to 1.
        look_forward (int, optional): Number of positions to look forward in local attention. Defaults to 1.
        dropout (float, optional): Dropout probability. Defaults to 0.0.
        bias (bool, optional): If True, use bias in linear layers. Defaults to True.
        causal (bool, optional): If True, apply causal masking in attention. Defaults to True.

    Returns:
        torch.Tensor: The output of the multi-head local attention layer.
        None: Placeholder to match the output signature of MultiheadAttention.
    """
    def __init__(self, d_model, n_heads, window_size, look_backward=1, look_forward=1, dropout=0.0,
                 *args, bias=True, causal=True, rotary=False, use_pykeops=False, **kwargs):
        super().__init__()
        # sequence model nessesary attributes
        self.d_model = d_model
        self.d_output = d_model

        assert d_model % n_heads == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // n_heads
        self.num_heads = n_heads
        self.causal = causal
        self.use_pykeops = use_pykeops

        self.linears = clones(nn.Linear(d_model, d_model, bias=bias), 4)
        self.attn = LocalFlashAttention(window_size=window_size, look_backward=look_backward, look_forward=look_forward,
                                        causal=causal, dropout=dropout)
        if has_pykeops and use_pykeops:
            print('Init local attention module with pykeops')
            self.attn = LocalAttention_pykeops(window_size=window_size, causal=causal, look_backward=look_backward, look_forward=look_forward)

        self.dropout_p = dropout

        if rotary:
            self.rope = RotaryEmbedding(self.d_k)

    def forward(self, src, attn_mask=None, key_padding_mask=None, state=None, **kwargs):
        """
        src: (B, L, D)
        attn_mask: (B, L, L)
        """
        if key_padding_mask is not None:
            raise NotImplementedError("key padding Not implemented for now with module MultiHeadedAttentionFlash")
        if state is not None:
            raise NotImplementedError("state Not implemented for now with module MultiHeadedAttentionFlash")
        assert (attn_mask is None) or (self.causal is False), "Only one of attn_mask or self.causal can be used"

        causal = self.causal if attn_mask is None else False
        nbatches = src.size(0)

        # 1) Do all the linear projections in batch from d_model => num_heads x d_k
        query, key, value = [
            lin(x).view(nbatches, -1, self.num_heads, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (src, src, src))
        ]  # (B, H, L, Dk)

        # 1.5) Add rotary positional embeddings if used
        if hasattr(self, 'rope'):
            query = self.rope(query)
            key = self.rope(key)

        # 2) Apply attention on all the projected vectors in batch.

        # local attention expects dimension (B', L, D) and not (B, H, L, Dk)
        if not self.use_pykeops:
            query, key, value = map(lambda x: rearrange(x, 'b h l d -> (b h) l d'), (query, key, value))
            x = self.attn(query, key, value, mask=attn_mask)  # (B*H, L, Dk)
            x = rearrange(x, '(b h) l d -> b h l d', h=self.num_heads)
        # pykeops attention expects dimension (B, H, L, Dk)
        else:
            x = self.attn(query, key, value, input_mask=attn_mask)  # (B, H, L, Dk)

        # 3) "Concat" using a view and apply a final linear.
        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(nbatches, -1, self.num_heads * self.d_k)
        )  # (B, L, D)
        del query
        del key
        del value
        return self.linears[-1](x), None  # None to match output signature of MultiheadAttention

    def step(self, x, state):
        raise NotImplementedError("Not implemented for now with module MultiHeadedAttentionFlash")


class VitAttention(SequenceModule):
    """Copied from implementation for ViT: only used for ViT model

    This attention class makes several simplifying assumptions (commonly satisfied in vision
       applications):
    1. q = k = v
    2. No masks: no attention mask, no key padding mask
    3. Embed dimension = Input dimension, i.e. projection matrices are square.
    """

    @property
    def d_output(self):
        return self.dim

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.,
        # proj_drop=0.,
        packed_linear=True,
        linear_cfg=None,
        **kwargs,
    ):
        """packed_linear: whether to pack all 3 q_proj, k_proj, v_proj into 2 matrix.
        This option is to be compatible with T2T-ViT pretrained weights, where there's only one
        projection weight matrix.
        """
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5

        if linear_cfg is not None:
            packed_linear = False
        self.packed_linear = packed_linear
        if packed_linear:
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        else:
            if linear_cfg is None:
                linear_cfg = {'_target_': 'torch.nn.Linear'}
            self.q_proj = hydra.utils.instantiate(linear_cfg, dim, dim, bias=qkv_bias,
                                                  _recursive_=False)
            self.k_proj = hydra.utils.instantiate(linear_cfg, dim, dim, bias=qkv_bias,
                                                  _recursive_=False)
            self.v_proj = hydra.utils.instantiate(linear_cfg, dim, dim, bias=qkv_bias,
                                                  _recursive_=False)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)

        # Removing this dropout because we do this in SequenceResidualBlock
        # self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, state=None):
        B, N, C = x.shape
        if self.packed_linear:
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)
        else:
            q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
            q, k, v = [rearrange(x, 'b n (h d) -> b h n d', h=self.num_heads) for x in (q, k, v)]

        # attn = (q @ k.transpose(-2, -1) * self.scale)
        # Use `torch.baddbmm` (a bit more efficient w/ alpha param for scaling -- from Megatron-LM)
        bsz, num_heads, q_seq_len, dk = q.size()
        _, _, k_seq_len, _ = k.size()
        q = rearrange(q, 'b h t d -> (b h) t d')
        k = rearrange(k, 'b h s d -> (b h) d s')
        # Preallocate attn_weights for `baddbmm`
        attn = torch.empty(bsz * num_heads, q_seq_len, k_seq_len, dtype=q.dtype, device=q.device)
        attn = rearrange(torch.baddbmm(attn, q, k, beta=0, alpha=self.scale),
                         '(b h) t s -> b h t s', h = self.num_heads)

        attn = F.softmax(attn, dim=-1, dtype=v.dtype)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        # x = self.proj_drop(x)
        return x, None


