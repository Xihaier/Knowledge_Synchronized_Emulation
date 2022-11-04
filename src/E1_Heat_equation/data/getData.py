'''
=====
Distributed by: Computational Science Initiative, Brookhaven National Laboratory (MIT Liscense)
- Associated publication:
url: 
doi: 
github: 
=====
'''
import numpy as np

from data.simulator import diffusionSim
from data.DiffusionLoader2D import DiffusionLoader


def getDataloaders(args):
    # Domain settings, matches solver settings
    x0 = 0.0
    x1 = 1.0
    args.dx = (x1 - x0)/(args.nel + 1)

    # Create training loader
    diffuLoader = DiffusionLoader(dt=args.dt)
    training_loader = diffuLoader.createTrainingLoader(args.ntrain, args.nel, args.btrain, args.train_dis)

    # Create test loader
    if args.simulator:
        diffusionSim(args.alpha, args.ncases, args.test_dis, args.data_dir)
        
    test_cases = np.array([idx for idx in range(args.ncases)]).astype(int)
    testing_loader = diffuLoader.createTestingLoader(args.data_dir, test_cases, batch_size=args.ncases)
    return training_loader, testing_loader


