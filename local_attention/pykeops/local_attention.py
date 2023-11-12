"""
adapted from https://github.com/lucidrains/local-attention/blob/master/local_attention/local_attention.py
"""

import math
from operator import mul
from functools import reduce

import torch
from torch import nn
import torch.nn.functional as F

from opt_einsum import contract as einsum
from einops import rearrange, repeat

has_flash = False #hasattr(F, 'scaled_dot_product_attention')

try:
    import pykeops
    from pykeops.torch import LazyTensor
    pykeops.set_verbose(False)
    has_pykeops = True
    
except ImportError:
    has_pykeops = False


# helper functions

def default(value, d):
    return d if value is None else value

def to(t):
    return {'device': t.device, 'dtype': t.dtype}

def max_neg_value(tensor):
    return -torch.finfo(tensor.dtype).max

def merge_dims(ind_from, ind_to, tensor):
    shape = list(tensor.shape)
    arr_slice = slice(ind_from, ind_to + 1)
    shape[arr_slice] = [reduce(mul, shape[arr_slice])]
    return tensor.reshape(*shape)

def expand_dim(t, dim, k, unsqueeze=True):
    if unsqueeze:
        t = t.unsqueeze(dim)
    expand_shape = [-1] * len(t.shape)
    expand_shape[dim] = k
    return t.expand(*expand_shape)

def pad_to_multiple(tensor, multiple, dim=-1, value=0):
    if dim >= 0: 
        dim = dim - tensor.ndim
    seqlen = tensor.shape[dim]
    m = seqlen / multiple
    if m.is_integer():
        return tensor
    remainder = math.ceil(m) * multiple - seqlen
    pad_offset = (0,) * ((-1 - dim) * 2)
    return F.pad(tensor, (*pad_offset, 0, remainder), value=value)

def look_around(x, backward = 1, forward = 1, pad_value = -1, dim = 2):
    # x (b*h nw ws e)  / (1 nw ws)
    if not (backward+forward): 
        return x
    assert dim >= 0
    nw = x.shape[1]
    dims = (len(x.shape) - dim) * (0, 0)
    padded_x = F.pad(x, (*dims, backward, forward), value=pad_value)  # (b*h f+nw+b ws e)
    tensors = [padded_x[:, ind:(ind + nw)] for ind in range(forward + backward + 1)]
    return torch.cat(tensors, dim=dim)   # (b*h nw ws*(f+b+1) e)


# main class

class LocalAttention(nn.Module):
    def __init__(self, 
                 window_size, 
                 causal = False, 
                 look_backward = 1, 
                 look_forward = 1):
        
        super().__init__()
        look_forward = 0 if causal else look_forward
        self.window_size = window_size
        self.causal = causal
        self.look_backward = look_backward
        self.look_forward = look_forward
        
    def forward(self, q, k, v, input_mask=None):
        """ q, k, v: (b h t e)  out: (b h t e)
            input_mask: (b 1 1 t)
            attn temperature not applied
        """
        assert all([t is None or t.ndim == 4 for t in (q, k, v, input_mask)])
        assert q.shape[:-1] == k.shape[:-1] == v.shape[:-1], 'can only be used for self attn'
        
        shape = q.shape
        _, _, orig_t, e, device, dtype = *shape, q.device, q.dtype
        scale = e**-.5
        
        window_size, causal, look_backward, look_forward = self.window_size, self.causal, self.look_backward, self.look_forward
        
        if orig_t <= window_size:
            window_size, look_backward, look_forward = orig_t, 0, 0 
        
        merge_into_batch = lambda t: t.reshape(-1, *t.shape[2:])  
        q, k, v = map(merge_into_batch, (q, k, v))                  # (b*h orig_t e)

        q, k, v = (pad_to_multiple(t, window_size, dim=-2) for t in (q, k, v))  # (b*h t e)
        
        bh, t, e = q.shape
        assert (t % window_size) == 0, f'local attn: sequence length {t} must be divisible by chunk size {window_size}'
        windows = t // window_size   # nw, ws

        bucket_fn = lambda t: t.reshape(t.shape[0], windows, window_size, -1)
        bq, bk, bv = map(bucket_fn, (q, k, v))          # (b*h nw ws e)
        
        look_around_kwargs = {'backward': look_backward, 'forward': look_forward}
        bk, bv = (look_around(t, pad_value=0, **look_around_kwargs) for t in (bk, bv))  # (b*h nw 3*ws e)
        
        # mask
        bq_t = torch.arange(t, device=device)
        bq_t[orig_t:] = -1
        bq_t = bq_t.reshape(1, windows, window_size)                        # (1 nw ws)
        bq_k = look_around(bq_t, pad_value=-1, **look_around_kwargs)        # (1 nw 3*ws)
        
        mask = (bq_k == -1).unsqueeze(-2)                                   # (1 nw 1 3*ws)
        mask_value = max_neg_value(bq)
        
        if has_flash:
            is_causal = False
            if window_size == orig_t and causal:
                is_causal, mask = True, None
            elif causal:
                # caution: flash is_causal=True doesnt result in this mask
                mask = mask | (bq_t[:, :, :, None] < bq_k[:, :, None, :])   # (1 nw ws 3*ws)
            if mask is not None and mask.any():
                mask = mask * mask_value
            else:
                mask = None
            
            # internally applies temperature 
            out = F.scaled_dot_product_attention(bq, bk, bv, attn_mask=mask, is_causal=is_causal)
        
        elif not has_pykeops:
            if causal:
                mask = mask | (bq_t[:, :, :, None] < bq_k[:, :, None, :])   # (1 nw ws 3*ws)
        
            dots = einsum('bwie,bwje->bwij', bq * scale, bk)          # (b*h nw ws 3*ws)

            if mask.any():
                dots.masked_fill_(mask, mask_value)
            del mask

            attn = dots.float().softmax(dim=-1).to(dots)
            out = einsum('bwij,bwje->bwie', attn, bv)         # (b*h nw ws e)
        
        else:    
            bq = bq.unsqueeze(-2) * scale                     # [b*h nw l 1 e]  
            bk, bv = bk.unsqueeze(-3), bv.unsqueeze(-3)       # [b*h nw 1 l e]  
            
            bq, bk, bv = map(LazyTensor, (bq, bk, bv))
            dots = (bq | bk)                                  # [b*n nw l l]
            
            mask = LazyTensor(mask.float().unsqueeze(-1))
            
            if causal:
                bq_t = LazyTensor(bq_t.float()[:, :, :   , None, None]+.1)
                bq_k = LazyTensor(bq_k.float()[:, :, None, :   , None])
                mask += (-bq_t + bq_k).step()
            
            dots += mask*mask_value                       # **this op makes attn 40% slower!!!**
            out = dots.sumsoftmaxweight(bv, dim=len(dots.shape)-1)  
        
        out = out.reshape(out.shape[0], -1, e)            # (b*h t e)
        
        if t != orig_t:
            out = out[:, :orig_t]
        
        return out.reshape(*shape)                        # (b h t e)