'''
=====
Distributed by: Computational Science Initiative, Brookhaven National Laboratory (MIT Liscense)
- Associated publication:
url: 
doi: 
github: 
=====
'''
import torch
import operator
import torch.nn as nn
import torch.nn.functional as F

from functools import reduce
from model.torchsummary import summary


class ResNetBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding):
        super(ResNetBlock, self).__init__()
        residual = x
        
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        return torch.cat([x, out], 1)


class Resblock(nn.Module):
    def __init__(self, input_channels, hidden_dim, dropout_rate, res = True):
        super(Resblock, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(input_channels, hidden_dim, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate)
        ) 
        self.layer2 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate)
        ) 
        self.res = res
        
    def forward(self, x):
        out = self.layer1(x)
        if self.res:
            out = self.layer2(out) + x
        else:
            out = self.layer2(out)
        return out



class EncodingBlock(nn.Module):
    def __init__(self, in_planes, mid_planes, out_planes):
        super(EncodingBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, mid_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(mid_planes, out_planes, kernel_size=3, stride=2, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(self.relu1(self.bn1(x)))
        out = self.conv2(self.relu2(self.bn2(out)))
        return out

    
class TransitionBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(TransitionBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        
    def forward(self, x):
        out = self.conv1(self.relu1(self.bn1(x)))
        return out    

    
class UpsamplingNearest2d(nn.Module):
    def __init__(self, scale_factor=2.):
        super().__init__()
        self.scale_factor = scale_factor
    
    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode='nearest', recompute_scale_factor=True)


class DecodingBlock(nn.Module):
    def __init__(self, in_planes, mid_planes, out_planes, scale_factor):
        super(DecodingBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, mid_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.upsample = UpsamplingNearest2d(scale_factor)
        self.conv2 = nn.Conv2d(mid_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(self.relu1(self.bn1(x)))
        out = self.conv2(self.upsample(self.relu2(self.bn2(out))))
        return out


class DenseBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, growth_rate, block, kernel_size, stride, padding):
        super(DenseBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, growth_rate, nb_layers, kernel_size, stride, padding)

    def _make_layer(self, block, in_planes, growth_rate, nb_layers, kernel_size, stride, padding):
        layers = []
        for i in range(nb_layers):
            layers.append(block(in_planes+i*growth_rate, growth_rate, kernel_size, stride, padding))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class LastLayer(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(LastLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        return self.conv1(self.relu(self.bn1(x)))


class DenseNet(nn.Module):
    def __init__(self, nic):
        super(DenseNet, self).__init__()        
        self.conv_init = nn.Conv2d(nic, 48, kernel_size=3, stride=1, padding=1, bias=False)
        self.block1 = DenseBlock(3, 48, 16, BasicBlock, kernel_size=3, stride=1, padding=1)
        self.encoder1 = EncodingBlock(96, 48, 48)
        self.block2 = DenseBlock(3, 48, 16, BasicBlock, kernel_size=3, stride=1, padding=1)
        self.decoder1 = DecodingBlock(96, 48, 48, scale_factor=2)
        self.block3 = DenseBlock(3, 48, 16, BasicBlock, kernel_size=3, stride=1, padding=1)
        self.decoder2 = DecodingBlock(96, 48, 48, scale_factor=1)
        self.conv_last = LastLayer(48, 1)
        self._count_params()

    def forward(self, x):
        out = self.encoder1(self.block1(self.conv_init(x)))
        out = self.decoder2(self.block3(self.decoder1(self.block2(out))))
        out = self.conv_last(out)
        return out

    def _count_params(self):
        c = 0
        for p in self.parameters():
            c += reduce(operator.mul, list(p.size()))

        print('-'*37)
        print('Model summary')  
        print('Total params: %.2fM' % (c/1000000.0))
        print('Total params: %.2fk' % (c/1000.0))
        print('-'*37+'\n')
        
    def reset_parameters(self, verbose=False):
        for module in self.modules():
            if isinstance(module, self.__class__):
                continue
            if 'reset_parameters' in dir(module):
                if callable(module.reset_parameters):
                    module.reset_parameters()
                    if verbose:
                        print("Reset parameters in {}".format(module))


if __name__ == '__main__':
    model = DenseNet()
    summary(model, (1, 64, 64), device='cpu')


