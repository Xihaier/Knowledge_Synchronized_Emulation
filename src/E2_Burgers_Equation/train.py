# train.py
# Training operations.

import math
import logging
import torch
import numpy as np

from time import time
from tqdm import tqdm
from models import Burger2DIntegrate
from utils import assemblex, medianMSE
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import StepLR, ExponentialLR, ReduceLROnPlateau


class LogCoshLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_t, y_prime_t):
        ey_t = y_t - y_prime_t
        return torch.mean(torch.log(torch.cosh(ey_t + 1e-12)))


class XTanhLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_t, y_prime_t):
        ey_t = y_t - y_prime_t
        return torch.mean(ey_t * torch.tanh(ey_t))


class XSigmoidLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_t, y_prime_t):
        ey_t = y_t - y_prime_t
        return torch.mean(2 * ey_t / (1 + torch.exp(-ey_t)) - ey_t)


def get_loss(args):
    """Set loss function.
    Args:
        optim_loss: Determine the loss function.
    Returns:
        Loss function will be use for optimization.
    """
    if args.optim_loss == "LogCoshLoss":
        criterion = LogCoshLoss().to(args.device)
    elif args.optim_loss == "XTanhLoss":
        criterion = XTanhLoss().to(args.device)
    elif args.optim_loss == "XSigmoidLoss":
        criterion = XTanhLoss().to(args.device)
    elif args.optim_loss == "MSELoss":
        criterion =torch.nn.MSELoss(reduction='sum').to(args.device)
    if args.log: 
        logging.info(f"[Loss] {args.optim_loss}")
    return criterion


def get_optimizer(model, args):
    """Set optimizer.
    Args:
        optim_alg: Determine the optimization algorithm.
    Returns:
        Algorithm will be use for optimization.
    """
    if args.optim_alg == "Adam":
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.optim_lr)
        if args.log: 
            logging.info(f"[Optimizer] {args.optim_alg}, lr: {args.optim_lr}")
    elif args.optim_alg == "AdamL2":
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.optim_lr, weight_decay=args.optim_wd)
        if args.log: 
            logging.info(f"[Optimizer] {args.optim_alg}, lr: {args.optim_lr}, wd: {args.optim_wd}")
    elif args.optim_alg == "AdamW":
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.optim_lr, weight_decay=args.optim_wd)
        if args.log: 
            logging.info(f"[Optimizer] {args.optim_alg}, lr: {args.optim_lr}, wd: {args.optim_wd}")
    return optimizer


class SquareRootScheduler:
    def __init__(self, base_lr):
        self.lr = base_lr

    def __call__(self, num_update):
        return self.lr * pow(num_update + 1.0, -0.5)


class CosineScheduler:
    def __init__(self, max_update, base_lr, final_lr, warmup_steps, warmup_begin_lr):
        self.base_lr_orig = base_lr
        self.max_update = max_update
        self.final_lr = final_lr
        self.warmup_steps = warmup_steps
        self.warmup_begin_lr = warmup_begin_lr
        self.max_steps = self.max_update - self.warmup_steps

    def get_warmup_lr(self, epoch):
        increase = (self.base_lr_orig - self.warmup_begin_lr) * float(epoch) / float(self.warmup_steps)
        return self.warmup_begin_lr + increase

    def __call__(self, epoch):
        if epoch < self.warmup_steps:
            return self.get_warmup_lr(epoch)
        if epoch <= self.max_update:
            self.base_lr = self.final_lr + (self.base_lr_orig - self.final_lr) * (1 + math.cos(math.pi * (epoch - self.warmup_steps) / self.max_steps)) / 2
        return self.base_lr


def get_scheduler(args, optimizer):
    """get learning scheduler.
    Args:
        optim_sche: get the learning scheduler.
    Returns:
        Learning scheduler will be use for adjusting the learning rate.
    """
    s = args.optim_scheduler
    if s == "Cosine":
        scheduler = CosineScheduler(max_update=args.epochs, base_lr=args.optim_lr, final_lr=args.optim_lr_final, warmup_steps=args.optim_warmup, warmup_begin_lr=args.optim_lr_init)
    elif s == "SquareRoot":
        scheduler = SquareRootScheduler(base_lr=args.optim_lr)
    elif s == "StepLR":
        scheduler = StepLR(optimizer, step_size=3, gamma=0.99)
    elif s == "ExponentialLR":
        scheduler = ExponentialLR(optimizer, gamma=0.995)
    elif s == "ReduceLROnPlateau":
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-8)
    if args.log:
        logging.info(f"[Scheduler] {args.optim_scheduler}")
        logging.info(f"[Warm up] {args.optim_warmup} steps")
        logging.info(f"[Epochs] {args.epochs} epochs")
    return scheduler


class Trainer(object):
    def __init__(self, model, criterion, optimizer, scheduler, args):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        
        self.args = args
        self.device = args.device
        self.logger = args.logger
        self.log_freq = self.args.log_freq

        self.burgerInt = Burger2DIntegrate(args.dx, nu=args.nu, grad_kernels=[3, 3], device=args.device)
        self.dtStep = 25
        self.dtArr = np.linspace(np.log10(self.args.dt)-1, np.log10(self.args.dt), self.dtStep)
        self.dtArr = 10**(self.dtArr)
        
        self._checkpoint()

    def train_step(self, epoch, dataloader):
        logging.info("-"*37)
        logging.info(f"Training epoch {epoch} summary")

        dt =  self.dtArr[min(epoch, self.dtArr.shape[0]-1)]
        tsteps = np.zeros(len(dataloader)).astype(int) + int(90*min(epoch/75, 1) + 10)
        tback = np.zeros((len(dataloader))) + np.random.randint(2,5,tsteps.shape[0])

        self.model.train()
        start = time()
        mse_train = 0.0

        for batch_idx, inputs in enumerate(tqdm(dataloader)):
            inputs = assemblex(self.args, inputs)
            loss, mse_step = 0., 0.
            self.optimizer.zero_grad() 

            for t in range(tsteps[batch_idx]):
                uPred = self.model(inputs[:,-self.args.input_channels:,:])                
                ustar = self.burgerInt.crankNicolson(uPred, inputs[:,-2:,:], dt)
                loss += self.criterion(uPred, ustar)
                mse_step += self.criterion(uPred, ustar).item()

                if((t+1)%tback[batch_idx] == 0):
                    loss.backward()
                    loss = 0
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    inputs = inputs[:,-int(self.args.input_channels-2):,:].detach()
                    input0 = uPred.detach()
                    inputs = torch.cat([inputs, input0], dim=1)
                else:
                    input0 = uPred
                    inputs = torch.cat([inputs, input0], dim=1)
            mse_train += (mse_step/tsteps[batch_idx])

            if((batch_idx+1) % self.log_freq == 0):
                logging.info("Batch {}, Averaged MSE: {:.6f}".format(batch_idx, mse_step/tsteps[batch_idx]))
        
        self.logger['mse_train'].append(mse_train/len(dataloader))
        stop = time()            
        logging.info("Training epoch {} Loss: {:.6f} Time: {:.3f}".format(epoch, mse_train/len(dataloader), stop-start))
        logging.info("-"*37+"\n")

    def eval_step(self, epoch, dataloader):
        logging.info("-"*37)
        logging.info(f"Test epoch {epoch} summary")
                
        self.model.eval()
        start = time()

        with torch.no_grad():
            mb_size = int(len(dataloader.dataset)/len(dataloader))
            u_out = torch.zeros(len(dataloader.dataset), self.args.test_step//self.args.test_every+1, 2, self.args.nel, self.args.nel)
            u_target = torch.zeros(len(dataloader.dataset), self.args.test_step//self.args.test_every+1, 2, self.args.nel, self.args.nel)
    
            for batch_idx, (input0, uTarget0) in enumerate(tqdm(dataloader)):
                inputs = assemblex(self.args, input0)
                u_out[batch_idx*mb_size:(batch_idx+1)*mb_size,0] = input0
                u_target[batch_idx*mb_size:(batch_idx+1)*mb_size] = uTarget0[:,:(self.args.test_step//self.args.test_every+1)].cpu()

                for t in range(self.args.test_step):
                    uPred = self.model(inputs[:,-self.args.input_channels:,:,:])
                    if((t+1)%self.args.test_every == 0):
                        u_out[batch_idx*mb_size:(batch_idx+1)*mb_size, (t+1)//self.args.test_every,:,:,:] = uPred
                    inputs = inputs[:,-int(self.args.input_channels-2):,:].detach()
                    input0 = uPred.detach()
                    inputs = torch.cat([inputs,  input0], dim=1)
            
        mseTest = medianMSE(u_out, u_target)
        self.logger['mse_in'].append(mseTest[0])
        self.logger['mse_ex'].append(mseTest[1])
        self.logger['mse_test'].append(mseTest[2])
        if (epoch > 1) and (mseTest[2]<=min(self.logger['mse_test'])):
            if self.scheduler.__module__ == lr_scheduler.__name__:
                torch.save({'model_state_dict': self.model.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict(), 'scheduler_state_dict': self.scheduler.state_dict(), 'epoch': epoch, 'logger': self.logger}, self.args.logdir +'/model_{}.pth'.format(epoch))
            else:
                torch.save({'model_state_dict': self.model.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict(), 'epoch': epoch, 'logger': self.logger}, self.args.logdir +'/model_{}.pth'.format(epoch))
        
        stop = time()
        logging.info("Test epoch {} --- Time: {:.3f}".format(epoch, stop-start))   
        logging.info("Interpolation: {:.6f} --- Extrapolation: {:.6f}".format(mseTest[0], mseTest[1]))
        logging.info("-"*37+"\n")

    def update_lr(self, epoch):
        if self.scheduler.__module__ == lr_scheduler.__name__:
            self.scheduler.step()
        else:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.scheduler(epoch)

    def _checkpoint(self):
        if self.args.ckpt:
            checkpoint = torch.load(self.args.logdir + "/model_{}.pth".format(self.args.ckpt_epoch), map_location=self.args.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
