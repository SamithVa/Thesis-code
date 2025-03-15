
import torch.nn as nn
import torch
import math

from transformers.utils import (
    logging,
)
import torch.nn.functional as F

logger = logging.get_logger(__name__)

#Resample model
from einops import rearrange, repeat
from einops_exts import rearrange_many
from torch import einsum
from merge_2d import *

# fc + layer_norm
"""
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
"""


class FeedForward(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
    ):
        super().__init__()

        self.norm = nn.LayerNorm(in_features)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        
        self.fc2 =  nn.Linear(hidden_features, out_features)
        self.scale = nn.Parameter(torch.ones(1))

        # with torch.no_grad():
        #     nn.init.kaiming_uniform_(self.fc1.weight, a=math.sqrt(5))
        #     nn.init.zeros_(self.fc2.weight)

    def forward(self, x):
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x) 
        x = self.fc2(x)
        x = self.scale*x
        return x



class PerceiverSdpaAttention(nn.Module):
    def __init__(self, in_dim, dim_head=64, heads=8):
        super().__init__()

        self.heads = heads
        intermediate_dim = dim_head * heads

        self.norm = nn.LayerNorm(in_dim)

        self.to_q = nn.Linear(in_dim, intermediate_dim)
        self.to_kv = nn.Linear(in_dim, intermediate_dim * 2)
        self.to_out = nn.Linear(intermediate_dim, in_dim)

    def forward(self, x, latents):  # x : key and values, latents : query
        # import pdb 
        # pdb.set_trace()
        x = self.norm(x)
        latents = self.norm(latents)

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


class PerceiverSdpaResampler(nn.Module):
    def __init__(
        self,
        in_dim=1024, 
        out_dim=4096,
        depth=1,
        dim_head=128,
        heads=8,
    ):
        super().__init__()

        # self.downsample = nn.Linear(out_dim,in_dim,bias=False)
        self.linear = nn.Linear(in_dim, out_dim)

        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PerceiverSdpaAttention(in_dim=in_dim, dim_head=dim_head, heads=heads),
                        FeedForward(in_features=in_dim, hidden_features=in_dim,out_features=out_dim),
                    ]
                )
            )

    def forward(self, x,r=0):
        """
        Args:
            x: Input tensor of shape [seq_len, in_dim]
            r: Reduced sequence length (default: 512)
        Returns:
            output: Tensor of shape [bsz, seq_len, out_dim] with cross-attention results in the first `r` tokens
        """
        
        seq_len, emb_dim = x.shape
        if r > seq_len:
            raise ValueError(f"r ({r}) cannot be greater than the sequence length ({seq_len})")

        latents, selected_mask = self_soft_matching(x, r)  # [bsz, r, in_dim]
        down_x = self.linear(x) # [bsz, seq, out_dim]
        down_latent = self.linear(latents)  # [bsz, r, out_dim]
        for attn, ff in self.layers: # cross attention
            # import pdb
            # pdb.set_trace()
            down_latent = attn(down_x, down_latent)  # [bsz, r, out_dim], q: latent | key, value: down_x
            latents = ff(down_latent) + latents #
        output = torch.empty([seq_len,emb_dim], dtype=x.dtype, device=x.device)
        # print("selected_mask", selected_mask)
        output[selected_mask[0], :] = latents.clone() # [bsz, r, out_dim]
        return output, selected_mask

import time 
if __name__ == "__main__":
    bsz, seq_len, emd_dim = 1, 5, 5
    out_dim = 5
    device = 'cuda'

    
    inputs = torch.rand(size=[bsz, seq_len, emd_dim]).to(device, dtype=torch.bfloat16)
    print(inputs)
    perceiver = PerceiverSdpaResampler(in_dim=emd_dim, out_dim=out_dim).to(device, dtype=torch.bfloat16)
    output_tokens, selected_mask = perceiver(inputs, r=2)
    print("output", output_tokens)
    print("selected_mask", selected_mask)

