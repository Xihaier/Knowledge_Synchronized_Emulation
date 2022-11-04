import torch
import warnings
import numpy as np

from time import time
from config import Parser
from utils.utils import assemblex, medianMSE
from model.nn import getModel, getOpt
from data.getData import getDataloaders
from model.DiffusionFD import Diffusion2DIntegrate

warnings.filterwarnings("ignore")


#####################################
# base configuration
#####################################
args = Parser().parse()

#####################################
# load data
#####################################
train_loader, test_loader = getDataloaders(args)

#####################################
# define model
#####################################
model = getModel(args)
criterion, optimizer, scheduler, logger = getOpt(args, model)
DiffusionInt = Diffusion2DIntegrate(args.dx, grad_filter=args.grad_filter, alpha=args.alpha, device=args.device)

################################################################
# training and test
################################################################
dtStep = 25
dtArr = np.linspace(np.log10(args.dt)-1, np.log10(args.dt), dtStep)
dtArr = 10**(dtArr)

for epoch in range(1, args.epochs + 1):
    # training phase
    print(args.sepLineS)
    print('Training epoch {} summary'.format(epoch))
    dt =  dtArr[min(epoch, dtArr.shape[0]-1)]
    tsteps = np.zeros(len(train_loader)).astype(int) + int(90*min(epoch/75, 1) + 10)
    tback = np.zeros((len(train_loader))) + np.random.randint(2,5,tsteps.shape[0])
    tstart  = np.zeros(tsteps.shape[0])
    
    model.train()
    start = time()
    mse_train = 0.
    
    for batch_idx, inputs in enumerate(train_loader):        
        inputs = assemblex(args, inputs)        
        loss, mse_step = 0., 0.
        optimizer.zero_grad()
        
        for t in range(tsteps[batch_idx]):
            uPred = model(inputs[:,-args.nic:,:]) 
            ustar = DiffusionInt.theta_rule(uPred, inputs[:,-1:,:], dt, args.theta)
            loss += criterion(uPred, ustar)
            mse_step += criterion(uPred, ustar).item()

            if((t+1)%tback[batch_idx] == 0):
                loss.backward()
                loss = 0
                optimizer.step()
                optimizer.zero_grad()
                inputs = inputs[:,-1*int(args.nic-1):,:].detach()
                input0 = uPred.detach()
                inputs = torch.cat([inputs, input0], dim=1)
            else:
                input0 = uPred
                inputs = torch.cat([inputs, input0], dim=1)
        mse_train += (mse_step/tsteps[batch_idx])
        
        if((batch_idx+1) % 5 == 0):
            print('Batch {}, Averaged MSE: {:.6f}'.format(batch_idx, mse_step/tsteps[batch_idx]))
            
    logger['mse_train'].append(mse_train/len(train_loader))
    
    stop = time()
    
    print('Training epoch {} Loss: {:.6f}  Time: {:.3f}'.format(epoch, mse_train/len(train_loader), stop-start))
    print(args.sepLineE)

    # test phase
    print(args.sepLineS)
    print('Test epoch {} summary'.format(epoch))
    
    if(epoch % args.test_freq == 0):
        with torch.no_grad():
            model.eval()
            start = time()

            mb_size = int(len(test_loader.dataset)/len(test_loader))
            u_out = torch.zeros(len(test_loader.dataset), args.test_step//args.test_every+1, 1, args.nel, args.nel)
            u_target = torch.zeros(len(test_loader.dataset), args.test_step//args.test_every+1, 1, args.nel, args.nel)
                
            for batch_idx, (input0, uTarget0) in enumerate(test_loader):
                inputs = assemblex(args, input0)
                input0 = torch.unsqueeze(input0, 1)
                uTarget0 = torch.unsqueeze(uTarget0, 2)
                                
                u_out[batch_idx*mb_size:(batch_idx+1)*mb_size, 0] = input0
                u_target[batch_idx*mb_size:(batch_idx+1)*mb_size] = uTarget0[:,:(args.test_step//args.test_every+1)].cpu()

                for t in range(args.test_step):
                    uPred = model(inputs[:,-args.nic:,:])
                                        
                    if((t+1)%args.test_every == 0):
                        u_out[batch_idx*mb_size:(batch_idx+1)*mb_size, (t+1)//args.test_every, :, :] = uPred
                    inputs = inputs[:,-1*int(args.nic-1):,:].detach()                    
                    input0 = uPred.detach()
                    inputs = torch.cat([inputs,  input0], dim=1)
                                        
            mseTest = medianMSE(u_out, u_target)
            logger['mse_test'].append(mseTest)

            if (epoch >= args.save_epoch) and (mseTest<=min(logger['mse_test'])):
                torch.save(model.state_dict(), args.save_dir +'/model_{}.pth'.format(epoch))
        
            stop = time()
        print('Test epoch {} --- Time: {:.3f}'.format(epoch, stop-start))   
        print('Test epoch {} --- MSE: {:.6f} '.format(epoch, mseTest))
    print(args.sepLineE)
    
    if args.lrs == 'ReduceLROnPlateau':
        scheduler.step(mse_train/len(train_loader))
    else:
        scheduler.step()