import torch 

# x = torch.arange(12).reshape(3, 4)
# print(x)
# print(x.T)
# y = x.view(4, 3)
# print(y)

x = torch.randn(2,1,4)
y = x.view(2,-1)
print(y)