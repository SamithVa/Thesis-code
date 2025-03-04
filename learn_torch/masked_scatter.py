import torch

self = torch.tensor([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])
mask = torch.tensor([[0, 0, 0, 1, 1], [1, 1, 0, 1, 1]], dtype=torch.bool)
source = torch.tensor([[5, 1, 2, 3, 4], [5, 6, 7, 8, 9]])

result = self.masked_scatter(mask, source)
print(result)
"""
tensor([[0, 0, 0, 5, 1],
        [2, 3, 0, 4, 5]])
"""