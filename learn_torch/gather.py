import torch 
bsz, seq_len, hidden_dim = 1, 5, 10

x = torch.randint(0, 50, size=(bsz, seq_len, hidden_dim))
print(x)
index = torch.randperm(seq_len).reshape(1, seq_len, 1)
print(index)

result = x.gather(dim=1, index=index.expand(bsz, seq_len, hidden_dim))
print(result)