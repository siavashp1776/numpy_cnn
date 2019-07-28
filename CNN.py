import torch
import torch.nn.functional as F

filters = torch.randn(8,4,3,3)
inputs = torch.randn(1,4,5,5)
res = F.conv2d(inputs, filters, padding=1)

print(res)
print(2)

print('new lines added to CNN.py')
print('whoa')