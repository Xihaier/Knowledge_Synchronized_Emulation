import torch
import numpy as np


def toNumpy(tensor):
    '''
    Converts Pytorch tensor to numpy array
    '''
    return tensor.detach().cpu().numpy()


def toTuple(a):
    '''
    Converts array to tuple
    '''
    try:
        return tuple(toTuple(i) for i in a)
    except TypeError:
        return a

    
def assemblex(args, inputs):
    '''
    Expand input to match model in channels
    input [b, 2, x, y]
    '''
    inputs = torch.unsqueeze(inputs, 1)
    dims = torch.ones(len(inputs.shape))
    dims[1] = args.nic 
    inputs = inputs.repeat(toTuple(toNumpy(dims).astype(int))).to(args.device)
    return inputs


def medianMSE(u_out, u_target):
    '''
    Compute median MSE for test samples
    '''
    mse = []
    for idx in range(u_out.shape[0]):        
        mse_all = torch.sum((u_out[idx,:] - u_target[idx,:]) ** 2)
        mse.append(toNumpy(mse_all))
    return np.median(mse)
    
    
    