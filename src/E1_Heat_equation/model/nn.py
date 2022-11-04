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

from model.FNO import FNO2d
from model.DenseNet import DenseNet
from model.ResNet import ResNet, PhyResNet, PhyResNet2
from torch.optim.lr_scheduler import StepLR, ExponentialLR, ReduceLROnPlateau


def getModel(args):
    if args.model == 'FNO':
        model = FNO2d(args)
        model = model.to(args.device)
    # elif args.model == 'PhyFNO':
    #     model = PhyFNO2d(args)
    #     model = model.to(args.device)
    elif args.model == 'DenseNet':
        model = DenseNet(args.nic)
        model = model.to(args.device)
    elif args.model == 'ResNet':
        model = ResNet(args)
        model = model.to(args.device)
    elif args.model == 'PhyResNet':
        model = PhyResNet(args)
        model = model.to(args.device)
    elif args.model == 'PhyResNet2':
        model = PhyResNet2(args)
        model = model.to(args.device)
    else:
        raise NotImplementedError
    return model


def getOpt(args, model):
    criterion = torch.nn.MSELoss(reduction='sum')
    
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)

    if args.lrs == 'StepLR':
        scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    elif args.lrs == 'ExponentialLR':
        scheduler = ExponentialLR(optimizer, gamma=0.995)
    elif args.lrs == 'ReduceLROnPlateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-8)
    else:
        raise NotImplementedError
        
    logger = {}
    logger['mse_train'] = []
    logger['mse_in'] = []
    logger['mse_ex'] = []
    logger['mse_test'] = []
    return criterion, optimizer, scheduler, logger


