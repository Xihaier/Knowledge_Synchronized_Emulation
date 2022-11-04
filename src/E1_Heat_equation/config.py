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
import errno
import torch
import random
import argparse

from pprint import pprint


class Parser(argparse.ArgumentParser):
    def __init__(self):
        '''
        Program arguments.
        '''
        super(Parser, self).__init__(description='2D Diffusion System')
        self.add_argument('--exp-dir', type=str, default="./results", help='directory to save experiments')
        self.add_argument('--seed', type=int, default=3407, help='manual seed used in PyTorch and Numpy')
        self.add_argument('--grad-filter', type=str, default='f3', choices=['sf3', 'f3', 'f5'], help='types of gradient filter')
        self.add_argument('--simulator', type=bool, default=False, help='simulate validation data')
        self.add_argument('--train-dis', type=str, default='uniform', choices=['uniform', 'gaussian1', 'gaussian2'], help='probability distribution of the training data')
        self.add_argument('--test-dis', type=str, default='uniform', choices=['uniform', 'gaussian1', 'gaussian2'], help='probability distribution of the test data')

        # data
        self.add_argument('--alpha', type=float, default=0.0003, help='diffusion constant, at most 0.005 to ensure dt is stable, 0.0003 for uniform')
        self.add_argument('--theta', type=float, default=0.5, help='theta rule for time integration')
        self.add_argument('--ncases', type=int, default=128, help="number of validation data")
        self.add_argument('--ntrain', type=int, default=4096, choices=[256, 512, 1024, 2048, 4096, 8192, 16384], help="number of training data")
        self.add_argument('--nic', type=int, default=3, help="number of input channels")
        self.add_argument('--noc', type=int, default=1, help="number of output channels")
        self.add_argument('--nel', type=int, default=64, help="number of elements/ collocation points")

        # model
        self.add_argument('--model', type=str, default='FNO', choices=['FNO', 'DenseNet', 'ResNet', 'PhyResNet', 'PhyResNet2'], help='choose the model')
        self.add_argument('--modes', type=int, default=12, help='number of Fourier modes to multiply')
        self.add_argument('--width', type=int, default=20, help='width of the Fourier layer')
        self.add_argument('--dt', type=float, default=0.01, help='discrete time step')
        self.add_argument('--test-every', type=int, default=1, help='time-step interval to test (must match simulator')
        self.add_argument('--test-step', type=int, default=100, help='number of timesteps to predict for')

        self.add_argument('--input-channels', type=int, default=3, help='number of input channels')
        self.add_argument('--output-channels', type=int, default=1, help='number of output channels')
        self.add_argument('--num-filters', type=int, default=64, help='number of channels in the middle of the network')
        self.add_argument('--num-layers', type=int, default=3, help='number of layers')
        self.add_argument('--dropout-rate', type=float, default=0., help='dropout rate')

        # optimization
        self.add_argument('--epochs', type=int, default=120, help='number of epochs to train')
        self.add_argument('--btrain', type=int, default=128, help='training batch size')
        self.add_argument('--lr', type=float, default=0.001, help='ADAM learning rate')
        self.add_argument('--weight-decay', type=float, default=1e-4, help="weight decay")
        self.add_argument('--gamma', type=float, default=0.99, help='ADAM learning rate')
        self.add_argument('--step-size', type=int, default=3, help='step size')
        self.add_argument('--lrs', type=str, default='ExponentialLR', choices=['StepLR', 'ReduceLROnPlateau', 'ExponentialLR'], help="learning rate scheduler")

        # logging
        self.add_argument('--save-epoch', type=int, default=10, help='how many epochs to wait before save')
        self.add_argument('--test-freq', type=int, default=1, help='how many epochs to wait before test')
        self.add_argument('--plot-freq', type=int, default=5, help='how many epochs to wait before plotting test output')

    def parse(self):
        '''
        Parse program arguments
        '''
        # Load basic configuration 
        args = self.parse_args()

        # Experiment save directory
        args.data_dir = './data/pdf_test_{}'.format(args.test_dis)
        # args.save_dir = args.exp_dir + '/' + args.model + '/in_{}_theta_{}'.format(args.nic, args.theta)
        args.save_dir = args.exp_dir + '/' + args.model + '/in_{}_theta_{}_{}'.format(args.nic, args.theta, args.grad_filter)
        self.mkdirs(args.save_dir)

        # Set random seed
        if args.seed is None:
            args.seed = random.randint(1, 10000)
        random.seed(args.seed)
        torch.manual_seed(args.seed)

        # Set device
        args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # Print arguments
        print('-'*37)
        print('Arguments summary')
        pprint(vars(args))
        print('-'*37+'\n')

        args.sepLineS = '-'*37
        args.sepLineE = '-'*37+'\n'

        return args

    def mkdirs(self, *directories):
        '''
        Makes a directory if it does not exist
        '''
        for directory in list(directories):
            try:
                os.makedirs(directory)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise


if __name__ == '__main__':
    args = Parser().parse()