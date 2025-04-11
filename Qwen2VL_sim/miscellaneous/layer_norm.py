import torch
from torch.nn import LayerNorm

if __name__ == '__main__':
    w = torch.empty(3, 5)
    torch.nn.init.ones_(w)
    print(w)
    # batch, seq_len, emb_dim = 1, 10, 10
    # x = torch.randn([batch, seq_len, emb_dim], device='cuda', dtype=torch.bfloat16)
    # x[:, 2, :] = 1.0
    # layer_norm = LayerNorm(emb_dim, device='cuda', dtype=torch.bfloat16)
    # output = layer_norm(x)
    # print(output)
