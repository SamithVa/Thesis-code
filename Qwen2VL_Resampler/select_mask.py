import torch


input_ids = torch.randint(1, 1000, [1, 512])

# let assurme <vision_start> = 128
vision_start = 128

indices = (input_ids == vision_start).nonzero(as_tuple=True)
print(indices) 