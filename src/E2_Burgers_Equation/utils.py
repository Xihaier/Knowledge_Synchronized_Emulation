# utils.py
# Utility functions.

import os
import random
import logging
import datetime

import torch
import numpy as np


def set_seed(args):
    """Set seed for reproducibility.
    Args:
        seed (int, optional): number to use as the seed. Defaults to 1225.
    """
    seed = args.randomseed
    # Set seeds
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)  # multi-GPU
    if args.log: 
        logging.info(f"[Seeds] random seeds: {args.randomseed}")


def set_device(args):
    """Set the device for computation.
    Args:
        cuda (bool): Determine whether to use GPU or not (if available).
    Returns:
        Device that will be use for compute.
    """
    args.num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.device == 'cpu': args.num_gpus = 0
    args.distributed = args.num_gpus > 1
    args.local_rank = 0
    if args.distributed:
        args.local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(args.local_rank)
        args.device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
    args.log = not args.distributed or args.local_rank == 0
    if args.log: 
        logging.info(f"[Device] device: {args.device}, num_gpus: {args.num_gpus}, distributed: {args.distributed}")
    return args


def set_logger(args):
    """Set the logging for computation.
    Args:
        An object with programs arguements.
    Returns:
        A file that logs will be written on.
    """
    log_name = args.model + '_' + 'n' + str(args.ntrain)
    log_para = args.optim_alg
    
    date = str(datetime.datetime.now())
    log_base = 'time_' + date[date.find("-"):date.rfind(".")].replace("-", "").replace(":", "").replace(" ", "_")

    log_root = args.logroot
    args.logdir = os.path.join(log_root, log_name)
    os.makedirs(args.logdir, exist_ok=True)
    
    filename = log_para + '_' + log_base + '.log'
    logging.basicConfig(level=logging.__dict__['INFO'], format="%(asctime)s [%(levelname)s] %(message)s", handlers=[logging.FileHandler(args.logdir+'/'+filename), logging.StreamHandler()])

    args.logger = {}
    args.logger['mse_train'] = []
    args.logger['mse_in'] = []
    args.logger['mse_ex'] = []
    args.logger['mse_test'] = []
    
    logging.info('-'*37)
    logging.info('Arguments summary')
    tmp = vars(args)
    for item in tmp:
        logging.info(f'[{item}] is {tmp[item]}')


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
    dims = torch.ones(len(inputs.shape))
    dims[1] = int(args.input_channels/2)
    inputs = inputs.repeat(toTuple(toNumpy(dims).astype(int))).to(args.device)
    return inputs


def medianMSE(u_out, u_target):
    '''
    Compute median MSE for test samples
    '''
    mse_in, mse_ex, mse = [], [], []
    for idx in range(u_out.shape[0]):        
        mse_interpolation = torch.sum((u_out[idx,:51] - u_target[idx,:51]) ** 2)
        mse_extrapolation = torch.sum((u_out[idx,51:] - u_target[idx,51:]) ** 2)
        mse_all = torch.sum((u_out[idx,:] - u_target[idx,:]) ** 2)
        mse_in.append(toNumpy(mse_interpolation))
        mse_ex.append(toNumpy(mse_extrapolation))
        mse.append(toNumpy(mse_all))
    return [np.median(mse_in), np.median(mse_ex), np.median(mse)]