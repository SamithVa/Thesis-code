import torch
from torch.nn import LayerNorm

if __name__ == '__main__':
    batch, seq_len, emb_dim = 2, 128, 64
    x = torch.randn(batch, seq_len, emb_dim)
    layer_norm = LayerNorm(emb_dim)
    print(layer_norm(x).shape)
