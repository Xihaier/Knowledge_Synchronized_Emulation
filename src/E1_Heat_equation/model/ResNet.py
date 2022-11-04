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


class Resblock(nn.Module):
    def __init__(self, num_filters, dropout_rate):
        super(Resblock, self).__init__()
        self.layer1 = nn.Sequential(
            nn.BatchNorm2d(num_filters),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Dropout(dropout_rate)
        )

        self.layer2 = nn.Sequential(
            nn.BatchNorm2d(num_filters),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Dropout(dropout_rate)
        )
        
    def forward(self, x):
        residual = x
        out = self.layer2(self.layer1(x))
        return out + residual


class LastLayer(nn.Module):
    def __init__(self, num_filters, output_channels):
        super(LastLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(num_filters, output_channels, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        return self.conv1(self.relu(self.bn1(x)))


class ResNet(nn.Module):
    def __init__(self, args):
        super(ResNet, self).__init__()
        self.input_layer = nn.Conv2d(args.input_channels, args.num_filters, kernel_size=3, stride=1, padding=1, bias=False)
        layers = [Resblock(args.num_filters, args.dropout_rate) for _ in range(args.num_layers)]
        self.middle_layer = nn.Sequential(*layers)
        self.output_layer = LastLayer(args.num_filters, args.output_channels)
        self._count_params()

    def forward(self, x):
        out = self.input_layer(x)
        out = self.middle_layer(out)
        out = self.output_layer(out)
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


class PhyResNet(nn.Module):
    def __init__(self, args):
        super(PhyResNet, self).__init__()
        self.physics_layer = PhysicsLayer(args.device)
        self.input_layer = nn.Conv2d(args.input_channels, args.num_filters, kernel_size=3, stride=1, padding=1, bias=False)
        layers = [Resblock(args.num_filters, args.dropout_rate) for _ in range(args.num_layers)]
        self.middle_layer = nn.Sequential(*layers)
        self.output_layer = LastLayer(args.num_filters, args.output_channels)
        self._count_params()

    def forward(self, x):
        out = self.physics_layer(x)
        out = self.input_layer(out)
        out = self.middle_layer(out)
        out = self.output_layer(out)
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


class PhysicsLayer2(nn.Module):
    def __init__(self, device, num_filters):
        super(PhysicsLayer2, self).__init__()
        self.physics_filter = torch.FloatTensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]]).to(device).unsqueeze(0).unsqueeze(0).repeat(num_filters, num_filters, 1, 1)
        self.p2d = (1, 1, 1, 1)

    def forward(self, x):
        out = F.pad(x, self.p2d, 'constant', 0)        
        out = F.conv2d(out, self.physics_filter, stride=1, padding=0, bias=None)
        return out


class Resblock2(nn.Module):
    def __init__(self, num_filters, dropout_rate, device):
        super(Resblock2, self).__init__()
        self.layer1 = nn.Sequential(
            nn.BatchNorm2d(num_filters),
            nn.ReLU(inplace=True),
            PhysicsLayer(device),
            nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Dropout(dropout_rate)
        ) 
        self.layer2 = nn.Sequential(
            nn.BatchNorm2d(num_filters),
            nn.ReLU(inplace=True),
            PhysicsLayer(device),
            nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Dropout(dropout_rate)
        )
        
    def forward(self, x):
        residual = x
        out = self.layer2(self.layer1(x))
        return out + residual


class PhyResNet2(nn.Module):
    def __init__(self, args):
        super(PhyResNet2, self).__init__()
        self.physics_layer = PhysicsLayer(args.device)
        self.input_layer = nn.Conv2d(args.input_channels, args.num_filters, kernel_size=3, stride=1, padding=1, bias=False)
        layers = [Resblock2(args.num_filters, args.dropout_rate, args.device) for _ in range(args.num_layers)]
        self.middle_layer = nn.Sequential(*layers)
        self.output_layer = LastLayer(args.num_filters, args.output_channels)
        self._count_params()

    def forward(self, x):
        out = self.physics_layer(x)
        out = self.input_layer(out)
        out = self.middle_layer(out)
        out = self.output_layer(out)
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


import argparse
def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch framework for NNs')
    parser.add_argument('--input-channels', type=int, default=3)
    parser.add_argument('--output-channels', type=int, default=1)
    parser.add_argument('--num-filters', type=int, default=64)
    parser.add_argument('--num-layers', type=int, default=3)
    parser.add_argument('--dropout-rate', type=float, default=0.)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_parser()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = PhyResNet(args)
    summary(model, (args.input_channels, 64, 64), device='cpu')


