# data.py
# Data processing operations.

import os
import logging

import torch
import numpy as np

from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler


class BurgerDataLoader():
    '''
    Class used for creating data loaders for the 2d burgers equation
    Args:
        dt (float): time-step size of the model
        shuffle (boolean): shuffle the training data or not
    '''
    def __init__(self, dt=0.01, shuffle=True):

        self.dt = dt
        self.shuffle = shuffle

    def createTrainingLoader(self, ncases, nel, distributed, batch_size=64):
        '''
        Loads in training data from Fenics simulator
        Args:
            data_dir (string): directory of data
            ncases (int): number of training cases to use
            n_init (int): number of intial conditions to use from each case
            batch_size (int): mini-batch size
        '''
        dataSet = InitPeriodicCond2d(nel, ncases)
        if distributed:
            train_sampler = train_sampler = DistributedSampler(dataset=dataSet)
            training_loader = DataLoader(dataSet, batch_size=batch_size, sampler=train_sampler, shuffle=self.shuffle, num_workers=4, pin_memory=True, drop_last=True)
        else:
            training_loader = DataLoader(dataSet, batch_size=batch_size, shuffle=self.shuffle, num_workers=4, pin_memory=True, drop_last=True)
        self.training_loader0 = training_loader
        return training_loader

    def createTestingLoader(self, data_dir, cases, tMax=1.0, simdt=0.001, save_every=2, batch_size=1):
        '''
        Loads in training data from Fenics simulator, assumes simulator has saved each time-step at specified delta t

        Args:
            data_dir (string): directory of data
            cases (np.array): array of training cases, must be integers
            tMax (float): maximum time value to load simulator data up to
            simdt (float): time-step size used in the simulation
            save_every (int): Interval to load the training data at (default is 2 to match FEM simulator)
            batch_size (int): mini-batch size
        Returns:
            test_loader (Pytorch DataLoader): Returns testing loader
        '''
        testing_data = []
        target_data = []

        logging.info('-'*37)
        logging.info("Test data summary")
        for i, val in enumerate(cases):
            case_dir = os.path.join(data_dir, "run{:d}".format(val))
            logging.info("Reading test case: {}".format(case_dir))
            seq = []
            for j in range(0, int(tMax/simdt)+1, save_every):
                file_dir = os.path.join(case_dir, "u{:d}.npy".format(j))
                u0 = np.load(file_dir)
                seq.append(u0[:,:,:])

            file_dir = os.path.join(case_dir, "u0.npy")
            uInit = np.load(file_dir)
            uTarget = np.stack(seq, axis=0)

            testing_data.append(torch.Tensor(uInit[:,:,:]).unsqueeze(0))
            target_data.append(torch.Tensor(uTarget))

        data_tuple = (torch.cat(testing_data, dim=0), torch.stack(target_data, dim=0))
        testing_loader = DataLoader(TensorDataset(*data_tuple), batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)
        logging.info('-'*37+'\n')
        return testing_loader


class InitPeriodicCond2d(torch.utils.data.Dataset):
    """
    Generate periodic initial condition on the fly.

    Args:
        order (int): order of Fourier series expansion
        ncells (int): spatial discretization over [0, 1]
        nsamples (int): total # samples
    """
    def __init__(self, ncells, nsamples, order=4):
        super().__init__()
        self.order = order
        self.nsamples = nsamples
        self.ncells = ncells
        x = np.linspace(0, 1, ncells+1)[:-1]
        xx, yy = np.meshgrid(x, x)
        aa, bb = np.meshgrid(np.arange(-order, order+1), np.arange(-order, order+1))
        k = np.stack((aa.flatten(), bb.flatten()), 1)
        self.kx_plus_ly = (np.outer(k[:, 0], xx.flatten()) + np.outer(k[:, 1], yy.flatten()))*2*np.pi
 
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            init_condition
        """
        np.random.seed(index+100000)
        lam = np.random.randn(2, 2, (2*self.order+1)**2)
        c = -1 + np.random.rand(2) * 2

        f = np.dot(lam[0], np.cos(self.kx_plus_ly)) + np.dot(lam[1], np.sin(self.kx_plus_ly))
        f = 2 * f / np.amax(np.abs(f), axis=1, keepdims=True) + c[:, None]
        return torch.FloatTensor(f).reshape(-1, self.ncells, self.ncells)

    def __len__(self):
        return self.nsamples


def getData(args):
    # Domain settings, matches solver settings
    x0 = 0
    x1 = 1.0
    args.dx = (x1 - x0)/args.nel

    # Create training loader
    burgerLoader = BurgerDataLoader(dt=args.dt)
    training_loader = burgerLoader.createTrainingLoader(args.ntrain, args.nel, args.distributed, batch_size=args.btrain)

    # Create test loader
    test_cases = np.array([idx for idx in range(args.ntest)]).astype(int)
    testing_loader = burgerLoader.createTestingLoader(args.test_data_dir, test_cases, simdt=0.005, batch_size=args.ntest)
    return training_loader, testing_loader
