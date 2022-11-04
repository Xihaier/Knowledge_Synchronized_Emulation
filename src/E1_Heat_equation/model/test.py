import torch
import operator
import torch.nn as nn
import torch.nn.functional as F

from functools import reduce
from torchsummary import summary


class PhysicsLayer(nn.Module):
    def __init__(self, device):
        super(PhysicsLayer, self).__init__()
        self.physics_filter = torch.FloatTensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]]).to(device).unsqueeze(0).unsqueeze(0)
        self.p2d = (1, 1, 1, 1)

    def forward(self, x):
        for idx in range(x.shape[1]):
            temp = F.pad(x[:,idx,:,:], self.p2d, 'constant', 0).unsqueeze(1)       
            temp = F.conv2d(temp, self.physics_filter, stride=1, padding=0, bias=None)
            if idx == 0:
                out = temp
            else:
                out = torch.cat((out, temp), 1)
        return out

device = 'cpu'
model = PhysicsLayer(device)
x = torch.rand(10, 3, 64, 64)
y = model(x)
print('y shape ', y.shape)


