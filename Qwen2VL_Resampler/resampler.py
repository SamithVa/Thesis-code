from einops import rearrange, repeat
from einops_exts import rearrange_many
from torch import einsum

import torch.nn as nn
import torch
import math

from typing import Any, Dict, List, Optional, Tuple, Union
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
import torch.nn.functional as F

logger = logging.get_logger(__name__)

#Resample model
from einops import rearrange, repeat
from einops_exts import rearrange_many
from torch import einsum
from merge import *


class FeedForward(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            norm_layer=None,
            bias=True,
            drop=0.,
            use_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.norm = nn.LayerNorm(in_features)
        self.fc1 = nn.Linear(in_features, hidden_features, bias=False)
        self.act = nn.GELU()
        
        self.fc2 =  nn.Linear(hidden_features, out_features, bias=False)
        self.scale = nn.Parameter(torch.ones(1))

        with torch.no_grad():
            nn.init.kaiming_uniform_(self.fc1.weight, a=math.sqrt(5))
            nn.init.zeros_(self.fc2.weight)

    def forward(self, x):
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x) 
        x = self.fc2(x)
        x = self.scale*x
        return x

# fc + layer_norm
class Block(nn.Module):
    def __init__(self, input_size,output_size):
        super().__init__()
        self.fc_1 = nn.Linear(input_size, output_size)
        self.norm = nn.LayerNorm(output_size)


    def forward(self, x):
        x = self.fc_1(x)
        x = self.norm(x)
        return x

class PerceiverAttention(nn.Module):
    def __init__(self, *, dim, dim_head=64, heads=8):
        super().__init__()

        self.scale = dim_head**-0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.layer_norm = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, latents): # x : key and values, latents : query
        x = self.layer_norm(x)
        latents = self.layer_norm(latents)

        k, v = self.to_kv(x).chunk(2, dim=-1)  # Compute keys and values only from x
        q = self.to_q(latents)

        b, n, _ = q.shape
        h = self.heads
        m = k.shape[1]
        h = self.heads

        q = q.reshape(b, n, h, -1).transpose(1, 2)  # (b, h, n, d)
        k = k.reshape(b, m, h, -1).transpose(1, 2)  # (b, h, m, d)
        v = v.reshape(b, m, h, -1).transpose(1, 2)  # (b, h, m, d)

        q = q * self.scale
        
        # Compute attention
        # import pdb 
        # pdb.set_trace()
        sim = torch.einsum("bhid, bhjd->bhij", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = torch.einsum("bhij,bhjd->bhid", attn, v)
        out = out.transpose(1, 2).contiguous().view(b, n, -1)
        return self.to_out(out)

class PerceiverSdpaAttention(nn.Module):
    def __init__(self, *, dim, dim_head=64, heads=8):
        super().__init__()

        self.heads = heads
        inner_dim = dim_head * heads

        self.norm_media = nn.LayerNorm(dim)
        self.norm_latents = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, latents):  # x : key and values, latents : query
        x = self.norm_media(x)
        latents = self.norm_latents(latents)

        h = self.heads
        q = self.to_q(latents)
        kv_input = torch.cat((x, latents), dim=-2)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)
        
        # Rearrange to multi-head format
        q = rearrange(q, "b n (h d) -> b h n d", h=h)
        k = rearrange(k, "b n (h d) -> b h n d", h=h)
        v = rearrange(v, "b n (h d) -> b h n d", h=h)
        
        # Use PyTorch's optimized SDPA
        out = F.scaled_dot_product_attention(q, k, v)
        
        # Merge heads back
        out = rearrange(out, "b h n d -> b n (h d)", h=h)
        return self.to_out(out)

class PerceiverResampler(nn.Module):
    def __init__(
        self,
        *,
        in_dim=1024, 
        out_dim=4096,
        depth=1,
        dim_head=128,
        heads=8,
        # visual_tokens_num=512,
        # ff_mult=4,
    ):
        super().__init__()

        # self.downsample = nn.Linear(out_dim,in_dim,bias=False)
        self.linear = nn.Linear(in_dim, out_dim, bias=False)

        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        # PerceiverAttention(dim=in_dim, dim_head=dim_head, heads=heads),
                        PerceiverAttention(dim=in_dim, heads=heads),
                        FeedForward(in_features=in_dim, hidden_features=in_dim,out_features=out_dim),
                    ]
                )
            )

    def forward(self, x,r=0):
        
        bsz, seq_len, emb_dim = x.shape
        latents = self_soft_matching(x, r)[0]  # x [bsz, seq_len, in_dim]  # [bsz, r, in_dim]
        # down_x = self.linear(x) # [bsz, seq, out_dim]
        # down_latent = self.linear(latents)  # [bsz, r, out_dim]
        # for attn, ff in self.layers: # cross attention
        #     down_latent = attn(down_x, down_latent)  # [bsz, r, out_dim], q: latent | key, value: down_x
        #     latents = ff(down_latent) + latents #
        output = torch.ones_like(x)
        output[:, :r, :] = latents.clone() # [bsz, r, out_dim]
        return output

class PerceiverSdpaResampler(nn.Module):
    def __init__(
        self,
        *,
        in_dim=1024, 
        out_dim=4096,
        depth=1,
        dim_head=128,
        heads=8,
        # visual_tokens_num=512,
        # ff_mult=4,
    ):
        super().__init__()

        # self.downsample = nn.Linear(out_dim,in_dim,bias=False)
        self.linear = nn.Linear(in_dim, out_dim, bias=False)

        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        # PerceiverAttention(dim=in_dim, dim_head=dim_head, heads=heads),
                        PerceiverSdpaAttention(dim=in_dim, heads=heads),
                        FeedForward(in_features=in_dim, hidden_features=in_dim,out_features=out_dim),
                    ]
                )
            )

    def forward(self, x,r=0):
        """
        Args:
            x: Input tensor of shape [bsz, seq_len, in_dim]
            r: Reduced sequence length (default: 512)
        Returns:
            output: Tensor of shape [bsz, seq_len, out_dim] with cross-attention results in the first `r` tokens
        """
        bsz, seq_len, dim = x.shape
        if r > seq_len:
            raise ValueError(f"r ({r}) cannot be greater than the sequence length ({seq_len})")

        latents, _ = self_soft_matching(x, r)  # [bsz, r, in_dim]
        down_x = self.linear(x) # [bsz, seq, out_dim]
        down_latent = self.linear(latents)  # [bsz, r, out_dim]
        for attn, ff in self.layers: # cross attention
            down_latent = attn(down_x, down_latent)  # [bsz, r, out_dim], q: latent | key, value: down_x
            latents = ff(down_latent) + latents #
        output = torch.ones_like(x, dtype=x.dtype, device=x.device)
        output[:, :r, :] = latents.clone() # [bsz, r, out_dim]
        return output


from torch.nn.init import trunc_normal_

class Resampler(nn.Module):
    """
    A 2D perceiver-resampler network with one cross attention layers by
        (grid_size**2) learnable queries and 2d sincos pos_emb
    Outputs:
        A tensor with the shape of (grid_size**2, embed_dim)
    """

    def __init__(
            self,
            grid_size,
            embed_dim,
            num_heads,
            kv_dim=None,
            norm_layer=nn.LayerNorm
    ):
        super().__init__()
        self.num_queries = grid_size ** 2 # 14 ** 2 = 256
        self.embed_dim = embed_dim 
        self.num_heads = num_heads

        # self.pos_embed = nn.Parameter(
        #     torch.from_numpy(get_2d_sincos_pos_embed(embed_dim, grid_size)).float()
        # ).requires_grad_(False)

        self.query = nn.Parameter(torch.zeros(self.num_queries, embed_dim))
        trunc_normal_(self.query, std=.02)

        if kv_dim is not None and kv_dim != embed_dim:
            self.kv_proj = nn.Linear(kv_dim, embed_dim, bias=False)
        else:
            self.kv_proj = nn.Identity()

        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.ln_q = norm_layer(embed_dim)
        self.ln_kv = norm_layer(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, attn_mask=None):

        # pos_embed = get_abs_pos(self.pos_embed, x.size(1))

        x = self.kv_proj(x)
        x = self.ln_kv(x).permute(1, 0, 2) # [seq_length, batch_size, embed_dim]

        N = x.shape[1] # batch_size
        q = self.ln_q(self.query) # normalize queries
        out = self.attn(
            self._repeat(q, N), # repeat q, for N head
            x,
            x,
            attn_mask=attn_mask)[0]
        return out.permute(1, 0, 2)

    def _repeat(self, query, N: int):
        return query.unsqueeze(1).repeat(1, N, 1)

import time 
if __name__ == "__main__":
    bsz, seq_len, emd_dim = 1, 5, 5
    out_dim = 5
    device = 'cuda'

    # verify perceiver and sdpa_perceiver output the same result 
    # start_time = time.time()
    # perceiver = PerceiverResampler(in_dim=emd_dim, out_dim=out_dim).cuda()
    # naive_time = time.time() - start_time

    # start_time = time.time()
    # sdpa_perceiver = PerceiverSdpaResampler(in_dim=emd_dim, out_dim=out_dim).cuda()
    # sdpa_time = time.time() - start_time
    # x = torch.randn(bsz, seq_len, emd_dim, device='cuda')
    # output = perceiver(x, r=512)
    # output2 = sdpa_perceiver(x, r=512)
    # print(output.shape, output2.shape)
    # same = torch.allclose(output, output2, atol=1e-6)
    # print("Are the outputs the same?", same)
    # print(f'naive time : {naive_time}, sdpa : {sdpa_time}')
    inputs = torch.randint(low=-47, high=90, size=[bsz, seq_len, emd_dim]).to(device, dtype=torch.bfloat16)
    print(inputs)
    perceiver = PerceiverResampler(in_dim=emd_dim, out_dim=out_dim).to(device, dtype=torch.bfloat16)
    output_tokens = perceiver(inputs, r=2)
    print(output_tokens)

