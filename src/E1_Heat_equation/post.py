import torch
import numpy as np

from time import time
from config import Parser
from model.nn import getModel
from utils.viz import vizPred, vizErr
from data.DiffusionLoader2D import DiffusionLoader

import warnings
warnings.filterwarnings("ignore")


# set base configuration
args = Parser().parse()
x0 = 0
x1 = 1.0
args.dx = (x1 - x0)/(args.nel + 1)
args.data_dir = 'data/pdf_test_uniform/'
args.save_dir = 'results/FNO/in_3_theta_0.5/model_118.pth'

# create data loader
numTest = 128
DiffusionLoader = DiffusionLoader(dt=args.dt)
test_cases = np.array([idx for idx in range(numTest)]).astype(int)
testing_loader = DiffusionLoader.createTestingLoader('{}'.format(args.data_dir), test_cases, batch_size=numTest)

# load trained model
model = getModel(args)
if torch.cuda.is_available():
    model.load_state_dict(torch.load('{}'.format(args.save_dir)))
else:
    model.load_state_dict(torch.load('{}'.format(args.save_dir), map_location=torch.device('cpu')))
print('-'*37)
print('Model {} is loaded'.format(args.model))

# validation
n_test = 100
with torch.no_grad():
    start = time()
    uPred, uTarget = vizPred(args, model, testing_loader, tstep=n_test)        
    stop = time()
    print('Inference time: {}'.format(stop-start))        
    vizErr(uPred, uTarget)
    del model