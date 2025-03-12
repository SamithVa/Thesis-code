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

# From Qwen2VL


# Copied from transformers.models.llama.modeling_llama.rotate_half
# def rotate_half(x):
#     """Rotates half the hidden dims of the input."""
#     x1 = x[..., : x.shape[-1] // 2]
#     x2 = x[..., x.shape[-1] // 2 :]
#     return torch.cat((-x2, x1), dim=-1)


# def apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section, unsqueeze_dim=1):
#     """Applies Rotary Position Embedding with Multimodal Sections to the query and key tensors (https://qwenlm.github.io/blog/qwen2-vl/).

#     Explanation:
#         Multimodal 3D rotary position embedding is an extension to 1D rotary position embedding. The input embedding
#         sequence contains vision (images / videos) embedding and text embedding or just contains text embedding. For
#         vision embedding part, we apply rotary position embedding on temporal, height and width dimension seperately.
#         Here we split the channel dimension to 3 chunks for the temporal, height and width rotary position embedding.
#         For text embedding part, we just apply 1D rotary position embedding. The three rotary position index (temporal,
#         height and width) of text embedding is always the same, so the text embedding rotary position embedding has no
#         difference with modern LLMs.

#     Args:
#         q (`torch.Tensor`): The query tensor.
#         k (`torch.Tensor`): The key tensor.
#         cos (`torch.Tensor`): The cosine part of the rotary embedding.
#         sin (`torch.Tensor`): The sine part of the rotary embedding.
#         position_ids (`torch.Tensor`):
#             The position indices of the tokens corresponding to the query and key tensors. For example, this can be
#             used to pass offsetted position ids when working with a KV-cache.
#         mrope_section(`List(int)`):
#             Multimodal rope section is for channel dimension of temporal, height and width in rope calculation.
#         unsqueeze_dim (`int`, *optional*, defaults to 1):
#             The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
#             sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
#             that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
#             k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
#             cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
#             the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
#     Returns:
#         `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
#     """
#     mrope_section = mrope_section * 2
#     cos = torch.cat([m[i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))], dim=-1).unsqueeze(
#         unsqueeze_dim
#     )
#     sin = torch.cat([m[i % 3] for i, m in enumerate(sin.split(mrope_section, dim=-1))], dim=-1).unsqueeze(
#         unsqueeze_dim
#     )

#     q_embed = (q * cos) + (rotate_half(q) * sin)
#     k_embed = (k * cos) + (rotate_half(k) * sin)
#     return q_embed, k_embed


# def apply_rotary_pos_emb_vision(
#     q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
# ) -> Tuple[torch.Tensor, torch.Tensor]:
#     orig_q_dtype = q.dtype
#     orig_k_dtype = k.dtype
#     q, k = q.float(), k.float()
#     cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
#     q_embed = (q * cos) + (rotate_half(q) * sin)
#     k_embed = (k * cos) + (rotate_half(k) * sin)
#     q_embed = q_embed.to(orig_q_dtype)
#     k_embed = k_embed.to(orig_k_dtype)
#     return q_embed, k_embed

# class VisionSdpaAttention(nn.Module):
#     def __init__(self, dim: int, num_heads: int = 16) -> None:
#         super().__init__()
#         self.num_heads = num_heads
#         self.qkv = nn.Linear(dim, dim * 3, bias=True)
#         self.proj = nn.Linear(dim, dim)

#     def forward(
#         self,
#         hidden_states: torch.Tensor,
#         cu_seqlens: torch.Tensor,
#         rotary_pos_emb: Optional[torch.Tensor] = None,
#         position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
#     ) -> torch.Tensor:
#         seq_length = hidden_states.shape[0]
#         q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
#         if position_embeddings is None:
#             logger.warning_once(
#                 "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
#                 "through `rotary_pos_emb` (2D tensor of RoPE theta values), to using externally computed "
#                 "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.54 `rotary_pos_emb` will be "
#                 "removed and `position_embeddings` will be mandatory."
#             )
#             emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
#             cos = emb.cos()
#             sin = emb.sin()
#         else:
#             cos, sin = position_embeddings
#         q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)

#         attention_mask = torch.zeros([1, seq_length, seq_length], device=q.device, dtype=torch.bool)
#         for i in range(1, len(cu_seqlens)):
#             attention_mask[..., cu_seqlens[i - 1] : cu_seqlens[i], cu_seqlens[i - 1] : cu_seqlens[i]] = True
#         q = q.transpose(0, 1)
#         k = k.transpose(0, 1)
#         v = v.transpose(0, 1)
#         attn_output = F.scaled_dot_product_attention(q, k, v, attention_mask, dropout_p=0.0)
#         attn_output = attn_output.transpose(0, 1)
#         attn_output = attn_output.reshape(seq_length, -1)
#         attn_output = self.proj(attn_output)
#         return attn_output


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

        self.norm_media = nn.LayerNorm(dim)
        self.norm_latents = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, latents): # x : key and values, latents : query
        x = self.norm_media(x)
        latents = self.norm_latents(latents)

        h = self.heads
        q = self.to_q(latents)
        kv_input = torch.cat((x, latents), dim=-2)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)
        q, k, v = rearrange_many((q, k, v), "b n (h d) -> b h n d", h=h)
        q = q * self.scale
        # attention
        sim = einsum("... i d, ... j d  -> ... i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = einsum("... i j, ... j d -> ... i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)", h=h)
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
        merge = self_soft_matching(x, r)  # x [bsz, seq_len, in_dim]
        latents = merge(x)  # [bsz, r, in_dim]
        down_x = self.linear(x) # [bsz, seq, out_dim]
        down_latent = self.linear(latents)  # [bsz, r, out_dim]
        for attn, ff in self.layers: # cross attention
            down_latent = attn(down_x, down_latent)  # [bsz, r, out_dim], q: latent | key, value: down_x
            latents = ff(down_latent) + latents #
        
        return latents

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
        latents, _ = self_soft_matching(x, r)  # [bsz, r, in_dim]
        down_x = self.linear(x) # [bsz, seq, out_dim]
        down_latent = self.linear(latents)  # [bsz, r, out_dim]
        for attn, ff in self.layers: # cross attention
            down_latent = attn(down_x, down_latent)  # [bsz, r, out_dim], q: latent | key, value: down_x
            latents = ff(down_latent) + latents #
        output = torch.zeros_like(x, dtype=x.dtype)
        output[:, :r, :] = latents[:, :r, :] # [bsz, r, out_dim]
        return output

import time 
if __name__ == "__main__":
    bsz, seq_len, emd_dim = 2, 5, 5
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

    inputs = torch.rand([bsz, seq_len, emd_dim]).to(device, dtype=torch.bfloat16)
    print(inputs)
    sdpa_perceiver = PerceiverSdpaResampler(in_dim=emd_dim, out_dim=out_dim).to(device, dtype=torch.bfloat16)
    output_tokens = sdpa_perceiver(inputs, r=2)
    print(output_tokens)

