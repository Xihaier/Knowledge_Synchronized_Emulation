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
import torch.nn.functional as F


class LapLaceFilter2d(object):
    """
    Laplacian 2D Filter
    Args:
        grad_filter (str): gradient filter
        device (PyTorch device): active device
    """
    def __init__(self, grad_filter, device='cpu'):
        super().__init__()
        self.grad_filter = grad_filter
        
        # smooth filter 3*3
        if self.grad_filter == 'sf3':
            WEIGHT_3x3 = torch.FloatTensor([[[[1, 2, 1],
                                            [-2, -4, -2],
                                            [1, 2, 1]]]]).to(device) / 4.

            WEIGHT_3x3 = WEIGHT_3x3 + torch.transpose(WEIGHT_3x3, -2, -1)
        # filter 3*3
        elif self.grad_filter == 'f3':
            WEIGHT_3x3 = torch.FloatTensor([[[[0, 1, 0],
                                            [1, -4, 1],
                                            [0, 1, 0]]]]).to(device)
        # filter 5*5
        elif self.grad_filter == 'f5':
            WEIGHT_5x5 = torch.FloatTensor([[[[0, 0, -1, 0, 0],
                                              [0, 0, 16, -0, 0],
                                              [-1, 16, -60, 16, -1],
                                              [0, 0, 16, 0, 0],
                                              [0, 0, -1, 0, 0]]]]).to(device) / 12.
        else:
            TypeError('Gradient filter is not defined')   
        
        if '3' in self.grad_filter:
            self.weight = WEIGHT_3x3
        elif '5' in self.grad_filter:
            self.weight = WEIGHT_5x5
        else:
            TypeError('Gradient filter is not defined')  
            
    def __call__(self, u):
        """
        Args:
            u (torch.Tensor): [B, C, H, W]
        Returns:
            div_u(torch.Tensor): [B, C, H, W]
        """
        p2d = (1, 1, 1, 1)
        inputs = F.pad(u, p2d, 'constant', 0)
        filters = self.weight
        outputs = F.conv2d(inputs, filters, stride=1, padding=0, bias=None)
        return outputs


class Diffusion2DIntegrate(object):
    '''
    Performs time-integration of the 2D diffusion equation
    Args:
        dx (float): spatial discretization
        grad_filter (str): gradient filter
        alpha (float): diffusion constant
        device (PyTorch device): active device
    '''
    def __init__(self, dx, grad_filter, alpha, device='cpu'):
        self.dx = dx
        self.alpha = alpha
        self.grad = LapLaceFilter2d(grad_filter, device=device)

    def theta_rule(self, uPred1, uPred0, dt, theta):
        """
        The theta rule: (1) theta = 0    gives the Forward Euler scheme in time
                        (2) theta = 1    gives the Backward Euler scheme in time
                        (3) theta = 1/2  gives the Crank-Nicolson scheme in time
        Args:
            uPred1 (torch.Tensor): predicted quantity at t+dt
            uPred0 (torch.Tensor): prior timestep/input at t
            dt (float): delta t
        Returns:
            ustar (torch.Tensor): u integrated to time-step t+dt
        """   
        # Fourier numbers Fx = Fy = F
        F = self.alpha * dt / (self.dx * self.dx)
        
        # time t
        div_u0 = self.grad(uPred0[:,:,:,:])
        
        # time t+dt
        div_u1 = self.grad(uPred1[:,:,:,:])
        
        # update
        ustar = uPred0[:,:,:] + div_u0 * (1-theta) * F + div_u1 * theta * F
        return ustar


