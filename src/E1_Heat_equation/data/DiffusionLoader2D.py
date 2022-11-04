'''
=====
Distributed by: Computational Science Initiative, Brookhaven National Laboratory (MIT Liscense)
- Associated publication:
url: 
doi: 
github: 
=====
'''
import os
import torch
import numpy as np

from torch.utils.data import DataLoader, TensorDataset


class DiffusionLoader():
    '''
    Class used for creating data loaders for the 2d diffusion system
    Args:
        dt (float): time-step size of the model
        shuffle (boolean): shuffle the training data or not
    '''
    def __init__(self, dt=0.01, shuffle=True):

        self.dt = dt
        self.shuffle = shuffle

    def createTrainingLoader(self, ncases, nel, batch_size, prob_dis):
        # Create on the fly Dataset
        if prob_dis == 'uniform':
            dataSet = InitUniform(nel, ncases)
        elif prob_dis == 'gaussian1':
            dataSet = InitGaussian1(nel, ncases)
        elif prob_dis == 'gaussian2':
            dataSet = InitGaussian2(nel, ncases)
        elif prob_dis == 'point':
            dataSet = InitPoint(nel, ncases)
        else:
            TypeError('Initial condition is not defined')   
        
        training_loader = DataLoader(dataSet, batch_size=batch_size, shuffle=self.shuffle, num_workers=4, pin_memory=True, drop_last=True)
        
        # Save original training loader
        self.training_loader0 = training_loader
        return training_loader

    def createTestingLoader(self, data_dir, cases, batch_size=1):
        '''
        Loads in validation data from simulation, assumes simulator has saved each time-step at specified delta t

        Args:
            data_dir (string): directory of data
            cases (np.array): array of training cases, must be integers
            batch_size (int): mini-batch size
        Returns:
            test_loader (Pytorch DataLoader): Returns testing loader
        '''
        testing_data = []
        target_data = []

        # Loop through test cases
        print('-'*37)
        print('Test data summary')
        dt = 0.01
        T = 1.0

        for i, val in enumerate(cases):
            case_dir = os.path.join(data_dir, 'run{:d}'.format(val))
            print('Reading test case: {}'.format(case_dir))
            seq = []
            for j in range(0, int(T/dt)+1):
                file_dir = os.path.join(case_dir, 'u{:d}.npy'.format(j))
                u0 = np.load(file_dir)
                # Remove the periodic nodes
                seq.append(u0[:,:])

            file_dir = os.path.join(case_dir, 'u0.npy')
            uInit = np.load(file_dir)
            uTarget = np.stack(seq, axis=0)

            # Remove the periodic nodes and unsqueeze first dim
            testing_data.append(torch.Tensor(uInit[:,:]).unsqueeze(0))
            target_data.append(torch.Tensor(uTarget))        
        
        # Create data loader
        data_tuple = (torch.cat(testing_data, dim=0), torch.stack(target_data, dim=0))
        testing_loader = DataLoader(TensorDataset(*data_tuple), batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)
        
        print('-'*37+'\n')
        return testing_loader


# one Gaussian heat source
class InitGaussian1(torch.utils.data.Dataset):
    def __init__(self, ncells, nsamples):
        super().__init__()
        self.nsamples = nsamples
        self.ncells = ncells
        x = np.linspace(0, 1, ncells+2)
        self.xx, self.yy = np.meshgrid(x, x)
 
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            init_condition
        """
        np.random.seed(index+100000) # Make sure this is different than the seeds set in finite element solver!
        
        b = np.random.normal(0.5, 0.1, 2)
        c = 0.1
        u0 = np.exp(-((self.xx-b[0])**2 + (self.yy-b[1])**2)/(2*c**2))

        u0[0, :] = 0
        u0[1, :] = 0
        u0[:, 0] = 0
        u0[:, 1] = 0
        return torch.FloatTensor(u0[1:-1, 1:-1])
   
    def __len__(self):
        # generate on-the-fly
        return self.nsamples


# two Gaussian heat sources
class InitGaussian2(torch.utils.data.Dataset):
    def __init__(self, ncells, nsamples):
        super().__init__()
        self.nsamples = nsamples
        self.ncells = ncells
        x = np.linspace(0, 1, ncells+2)
        self.xx, self.yy = np.meshgrid(x, x)
 
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            init_condition
        """
        np.random.seed(index+100000) # Make sure this is different than the seeds set in finite element solver!
        
        b1 = np.random.normal(0.5, 0.1, 2)
        c1 = 0.1
        u00 = np.exp(-((self.xx-b1[0])**2 + (self.yy-b1[1])**2)/(2*c1**2))
        b2 = np.random.normal(0.5, 0.1, 2)
        c2 = 0.1
        u01 = np.exp(-((self.xx-b2[0])**2 + (self.yy-b2[1])**2)/(2*c2**2))
        u0 = u00 + u01

        u0[0, :] = 0
        u0[1, :] = 0
        u0[:, 0] = 0
        u0[:, 1] = 0
        return torch.FloatTensor(u0[1:-1, 1:-1])
   
    def __len__(self):
        # generate on-the-fly
        return self.nsamples

    
# uniform heat source
class InitUniform(torch.utils.data.Dataset):
    def __init__(self, ncells, nsamples):
        super().__init__()
        self.nsamples = nsamples
        self.ncells = ncells
 
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            init_condition
        """
        np.random.seed(index+100000) # Make sure this is different than the seeds set in finite element solver!
        
        u0 = np.random.uniform(low=0.0, high=1.0, size=(self.ncells+2, self.ncells+2))

        u0[0, :] = 0
        u0[1, :] = 0
        u0[:, 0] = 0
        u0[:, 1] = 0
        return torch.FloatTensor(u0[1:-1,1:-1])
   
    def __len__(self):
        # generate on-the-fly
        return self.nsamples
    
    
# point heat source
class InitPoint(torch.utils.data.Dataset):
    def __init__(self, ncells, nsamples):
        super().__init__()
        self.nsamples = nsamples
        self.ncells = ncells
 
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            init_condition
        """
        np.random.seed(index+100000) # Make sure this is different than the seeds set in finite element solver!
        
        u0 = np.zeros((self.ncells+2+2, self.ncells+2+2))
        
        b = np.random.randint(0, 64, (2, 1))
        u0[b[0], b[1]] = 1
        
        u0[0, :] = 0
        u0[1, :] = 0
        u0[:, 0] = 0
        u0[:, 1] = 0
        return torch.FloatTensor(u0[1:-1,1:-1])
   
    def __len__(self):
        # generate on-the-fly
        return self.nsamples

    
    