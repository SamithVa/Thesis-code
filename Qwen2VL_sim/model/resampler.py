import torch
import torch.nn as nn
import torch.nn.functional as F
from .merge_2d import *

class FeedForward(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks """
    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()
        self.norm = nn.LayerNorm(in_features)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x) 
        x = self.fc2(x)
        return x * self.scale

class PerceiverSdpaAttention(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.to_q = nn.Linear(in_dim, in_dim)
        self.to_kv = nn.Linear(in_dim, in_dim * 2)
        self.to_out = nn.Linear(in_dim, in_dim)

    def forward(self, q, kv):  # q: [r, hidden_size], kv: [seq_length, hidden_size]
        q = self.norm(q)
        kv = self.norm(kv)
        
        k, v = self.to_kv(kv).chunk(2, dim=-1)
        
        # Scaled dot-product attention
        attn_output = F.scaled_dot_product_attention(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)).squeeze(0)
        
        return self.to_out(attn_output)

class PerceiverSdpaResampler(nn.Module):
    def __init__(self, in_dim=1024, out_dim=4096, depth=1):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.layers = nn.ModuleList([
            nn.ModuleList([
                PerceiverSdpaAttention(in_dim=in_dim),
                FeedForward(in_features=in_dim, hidden_features=in_dim, out_features=out_dim),
            ]) for _ in range(depth)
        ])

    def forward(self, x, r=0):
        seq_len, emb_dim = x.shape
        if r > seq_len:
            raise ValueError(f"r ({r}) cannot be greater than the sequence length ({seq_len})")
        
        latents, selected_mask = self_soft_matching_2d(x, r)
        down_x = self.linear(x)
        down_latent = self.linear(latents)
        
        for attn, ff in self.layers:
            down_latent = attn(down_latent, down_x)  # q: latents, kv: x
            latents = ff(down_latent) + latents
        
        output = torch.empty([seq_len, emb_dim], dtype=x.dtype, device=x.device)
        output[selected_mask, :] = latents.clone()
        return output, selected_mask

# Test the function
if __name__ == "__main__":
    seq_len, emd_dim, out_dim = 10, 5, 5
    device = 'cuda'
    inputs = torch.rand(size=[seq_len, emd_dim]).to(device, dtype=torch.bfloat16)
    perceiver = PerceiverSdpaResampler(in_dim=emd_dim, out_dim=out_dim).to(device, dtype=torch.bfloat16)
    output_tokens, selected_mask = perceiver(inputs, r=5)
    print("Output Tokens:", output_tokens)
    print("Selected Mask:", selected_mask)
