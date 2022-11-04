# models.py
# Model architectures.

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.modules.utils import _pair, _quadruple

import logging

from collections import OrderedDict


class Resblock(nn.Module):
    def __init__(self, num_filters, dropout_rate):
        super(Resblock, self).__init__()
        self.layer1 = nn.Sequential(
            nn.BatchNorm2d(num_filters),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Dropout(dropout_rate)
        )

        self.layer2 = nn.Sequential(
            nn.BatchNorm2d(num_filters),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Dropout(dropout_rate)
        )
        
    def forward(self, x):
        residual = x
        out = self.layer2(self.layer1(x))
        return out + residual


class LastLayer(nn.Module):
    def __init__(self, num_filters, output_channels):
        super(LastLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(num_filters, output_channels, kernel_size=7, stride=1, padding=3, bias=False)

    def forward(self, x):
        return self.conv1(self.relu(self.bn1(x)))


class ResNet(nn.Module):
    def __init__(self, args):
        super(ResNet, self).__init__()
        self.input_layer = nn.Conv2d(args.input_channels, args.num_filters, kernel_size=7, stride=1, padding=3, bias=False)
        layers = [Resblock(args.num_filters, args.dropout_rate) for _ in range(args.num_layers)]
        self.middle_layer = nn.Sequential(*layers)
        self.output_layer = LastLayer(args.num_filters, args.output_channels)

    def forward(self, x):
        out = self.input_layer(x)
        out = self.middle_layer(out)
        out = self.output_layer(out)
        return out


class SpectralConv2d_fast(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d_fast, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class FNO2d(nn.Module):
    def __init__(self, args):
        super(FNO2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        input shape: (batchsize, x=64, y=64, c=12)
        output: the solution of the next timestep
        output shape: (batchsize, x=64, y=64, c=1)
        """

        self.modes1 = args.modes1
        self.modes2 = args.modes2
        self.width = args.width
        self.input_channels = args.input_channels
        self.output_channels = args.output_channels
        # self.padding = 2 # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(8, self.width)
        # input channel is 8: the solution of the previous 3 timesteps + 2 locations

        self.conv0 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)
        self.bn0 = torch.nn.BatchNorm2d(self.width)
        self.bn1 = torch.nn.BatchNorm2d(self.width)
        self.bn2 = torch.nn.BatchNorm2d(self.width)
        self.bn3 = torch.nn.BatchNorm2d(self.width)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 2)
        # output channel is 2: the solution of the next timestep

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)

        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        # x = F.pad(x, [0,self.padding, 0,self.padding]) # pad the domain if input is non-periodic

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        # x = x[..., :-self.padding, :-self.padding] # pad the domain if input is non-periodic
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)

        x = x.permute(0, 3, 1, 2)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)


def summary(model, input_size, batch_size=-1, device=torch.device('cuda:0'), dtypes=None):
    result, params_info = summary_string(model, input_size, batch_size, device, dtypes)
    logging.info( f"[Model Summary]  \n{result}")

    return params_info


def summary_string(model, input_size, batch_size=-1, device=torch.device('cuda:0'), dtypes=None):
    if dtypes == None:
        dtypes = [torch.FloatTensor]*len(input_size)

    summary_str = ''

    def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = "%s-%i" % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = batch_size
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = batch_size

            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params

        if (
            not isinstance(module, nn.Sequential)
            and not isinstance(module, nn.ModuleList)
        ):
            hooks.append(module.register_forward_hook(hook))

    # multiple inputs to the network
    if isinstance(input_size, tuple):
        input_size = [input_size]

    # batch_size of 2 for batchnorm
    x = [torch.rand(2, *in_size).type(dtype).to(device=device)
         for in_size, dtype in zip(input_size, dtypes)]

    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    # print(x.shape)
    model(*x)

    # remove these hooks
    for h in hooks:
        h.remove()

    summary_str += "----------------------------------------------------------------" + "\n"
    line_new = "{:>20}  {:>25} {:>15}".format(
        "Layer (type)", "Output Shape", "Param #")
    summary_str += line_new + "\n"
    summary_str += "================================================================" + "\n"
    total_params = 0
    total_output = 0
    trainable_params = 0
    for layer in summary:
        # input_shape, output_shape, trainable, nb_params
        line_new = "{:>20}  {:>25} {:>15}".format(
            layer,
            str(summary[layer]["output_shape"]),
            "{0:,}".format(summary[layer]["nb_params"]),
        )
        total_params += summary[layer]["nb_params"]

        total_output += np.prod(summary[layer]["output_shape"])
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"] == True:
                trainable_params += summary[layer]["nb_params"]
        summary_str += line_new + "\n"

    # assume 4 bytes/number (float on cuda).
    total_input_size = abs(np.prod(sum(input_size, ()))
                           * batch_size * 4. / (1024 ** 2.))
    total_output_size = abs(2. * total_output * 4. /
                            (1024 ** 2.))  # x2 for gradients
    total_params_size = abs(total_params * 4. / (1024 ** 2.))
    total_size = total_params_size + total_output_size + total_input_size

    summary_str += "================================================================" + "\n"
    summary_str += "Total params: {0:,}".format(total_params) + "\n"
    summary_str += "Trainable params: {0:,}".format(trainable_params) + "\n"
    summary_str += "Non-trainable params: {0:,}".format(total_params -
                                                        trainable_params) + "\n"
    summary_str += "----------------------------------------------------------------" + "\n"
    summary_str += "Input size (MB): %0.2f" % total_input_size + "\n"
    summary_str += "Forward/backward pass size (MB): %0.2f" % total_output_size + "\n"
    summary_str += "Params size (MB): %0.2f" % total_params_size + "\n"
    summary_str += "Estimated Total Size (MB): %0.2f" % total_size + "\n"
    summary_str += "----------------------------------------------------------------" + "\n"
    # return summary
    return summary_str, (total_params, trainable_params)


def get_model(args):
    """Set model.
    Args:
        model: Determine the deep learning model.
    Returns:
        Model will be use for modeling.
    """
    if args.model == "FNO":
        model = FNO2d(args)
        if args.log:
            logging.info(f"[Model] name: {args.model}, modes1: {args.modes1}, modes2: {args.modes2}, width: {args.width}")
            summary(model, (6, 64, 64), device='cpu')
        
    elif args.model == "ResNet":
         model = ResNet(args)
         if args.log: 
             logging.info(f"[Model] name: {args.model}, input-channels: {args.input_channels}, output-channels: {args.output_channels}, num-filters: {args.num_filters}, dropout-rate: {args.dropout_rate}")
             summary(model, (6, 64, 64), device='cpu')

    if args.distributed: 
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)

    return model


class LapLaceFilter2d(object):
    """
    Smoothed Laplacian 2D, assumes periodic boundary condition.
    https://en.wikipedia.org/wiki/Finite_difference_coefficient
    Args:
        dx (float): spatial discretization, assumes dx = dy
        kernel_size (int): choices=[3, 5]
        device (PyTorch device): active device
    """
    def __init__(self, dx, kernel_size=3, device='cpu'):
        super().__init__()
        self.dx = dx
#         # no smoothing
#         WEIGHT_3x3 = torch.FloatTensor([[[[0, 1, 0],
#                                           [1, -4, 1],
#                                           [0, 1, 0]]]]).to(device)
        # smoothed
        WEIGHT_3x3 = torch.FloatTensor([[[[1, 2, 1],
                                          [-2, -4, -2],
                                          [1, 2, 1]]]]).to(device) / 4.

        WEIGHT_3x3 = WEIGHT_3x3 + torch.transpose(WEIGHT_3x3, -2, -1)

        # print(WEIGHT_3x3)

        WEIGHT_5x5 = torch.FloatTensor([[[[0, 0, -1, 0, 0],
                                          [0, 0, 16, -0, 0],
                                          [-1, 16, -60, 16, -1],
                                          [0, 0, 16, 0, 0],
                                          [0, 0, -1, 0, 0]]]]).to(device) / 12.
        if kernel_size == 3:
            self.padding = _quadruple(1)
            self.weight = WEIGHT_3x3
        elif kernel_size == 5:
            self.padding = _quadruple(2)
            self.weight = WEIGHT_5x5

    def __call__(self, u):
        """
        Args:
            u (torch.Tensor): [B, C, H, W]
        Returns:
            div_u(torch.Tensor): [B, C, H, W]
        """
        u_shape = u.shape
        u = u.view(-1, 1, *u_shape[-2:])
        u = F.conv2d(F.pad(u, self.padding, mode='circular'), 
            self.weight, stride=1, padding=0, bias=None) / (self.dx**2)
        return u.view(u_shape)


class SobelFilter2d(object):
    """
    Sobel filter to estimate 1st-order gradient in horizontal & vertical 
    directions. Assumes periodic boundary condition.
    Args:
        dx (float): spatial discretization, assumes dx = dy
        kernel_size (int): choices=[3, 5]
        device (PyTorch device): active device
    """
    def __init__(self, dx, kernel_size=3, device='cpu'):
        self.dx = dx
        # smoothed central finite diff
        WEIGHT_H_3x3 = torch.FloatTensor([[[[-1, 0, 1],
                                            [-2, 0, 2],
                                            [-1, 0, 1]]]]).to(device) / 8.

        # larger kernel size tends to smooth things out
        WEIGHT_H_5x5 = torch.FloatTensor([[[[1, -8, 0, 8, -1],
                                            [2, -16, 0, 16, -2],
                                            [3, -24, 0, 24, -3],
                                            [2, -16, 0, 16, -2],
                                            [1, -8, 0, 8, -1]]]]).to(device) / (9*12.)
        if kernel_size == 3:
            self.weight_h = WEIGHT_H_3x3
            self.weight_v = WEIGHT_H_3x3.transpose(-1, -2)
            self.weight = torch.cat((self.weight_h, self.weight_v), 0)
            self.padding = _quadruple(1)
        elif kernel_size == 5:
            self.weight_h = WEIGHT_H_5x5
            self.weight_v = WEIGHT_H_5x5.transpose(-1, -2)
            self.padding = _quadruple(2)        

    def __call__(self, u):
        """
        Compute both hor and ver grads
        Args:
            u (torch.Tensor): (B, C, H, W)
        Returns:
            grad_u: (B, C, 2, H, W), grad_u[:, :, 0] --> grad_h
                                     grad_u[:, :, 1] --> grad_v
        """
        # (B*C, 1, H, W)
        u_shape = u.shape
        u = u.view(-1, 1, *u_shape[-2:])
        u = F.conv2d(F.pad(u, self.padding, mode='circular'), self.weight, 
                        stride=1, padding=0, bias=None) / (self.dx)
        return u.view(*u_shape[:2], *u.shape[-3:])

    def grad_h(self, u):
        """
        Get image gradient along horizontal direction, or x axis.
        Perioid padding before conv.
        Args:
            u (torch.Tensor): [B, C, H, W]
        Returns:
            ux (torch.Tensor): [B, C, H, W] calculated gradient
        """
        ux = F.conv2d(F.pad(u, self.padding, mode='circular'), self.weight_h, 
                        stride=1, padding=0, bias=None) / (self.dx)
        return ux
    
    def grad_v(self, u):
        """
        Get image gradient along vertical direction, or y axis.
        Perioid padding before conv.
        Args:
            u (torch.Tensor): [B, C, H, W]
        Returns:
            uy (torch.Tensor): [B, C, H, W] calculated gradient
        """
        uy = F.conv2d(F.pad(u, self.padding, mode='circular'), self.weight_v, 
                        stride=1, padding=0, bias=None) / (self.dx)
        return uy

class Burger2DIntegrate(object):
    '''
    Performs time-integration of the 2D Burger equation
    Args:
        dx (float): spatial discretization
        nu (float): hyper-viscosity
        grad_kernels (list): list of kernel sizes for first, second and forth order gradients
        device (PyTorch device): active device
    '''
    def __init__(self, dx, nu=1.0, grad_kernels=[3, 3], device='cpu'):
        
        self.nu = nu

        # Create gradients
        self.grad1 = SobelFilter2d(dx, kernel_size=grad_kernels[0], device=device)
        self.grad2 = LapLaceFilter2d(dx, kernel_size=grad_kernels[1], device=device)

    def backwardEuler(self, uPred, uPred0, dt):
        """
        Time integration of the 2D Burger system using implicit euler method
        Args:
            uPred (torch.Tensor): predicted quantity at t+dt
            uPred0 (torch.Tensor): prior timestep/input at t
            dt (float): delta t
        Returns:
            ustar (torch.Tensor): u integrated to time-step t+dt
        """
        grad_ux = self.grad1.grad_h(0.5*uPred[:,:1,:,:]**2)
        grad_uy = self.grad1.grad_v(uPred[:,:1,:,:])

        grad_vx = self.grad1.grad_h(uPred[:,1:,:,:])
        grad_vy = self.grad1.grad_v(0.5*uPred[:,1:,:,:]**2)

        div_u = self.nu * self.grad2(uPred[:,:1,:,:])
        div_v = self.nu * self.grad2(uPred[:,1:,:,:])

        burger_rhs_u = -grad_ux - uPred[:,1:,:,:]*grad_uy + div_u
        burger_rhs_v = -uPred[:,:1,:,:]*grad_vx - grad_vy + div_v

        ustar_u = uPred0[:,:1,:,:] + dt * burger_rhs_u
        ustar_v = uPred0[:,1:,:,:] + dt * burger_rhs_v

        return torch.cat([ustar_u, ustar_v], dim=1)

    def crankNicolson(self, uPred, uPred0, dt):
        """
        Time integration of the 2D Burger system using crank-nicolson
        Args:
            uPred (torch.Tensor): predicted quantity at t+dt
            uPred0 (torch.Tensor): prior timestep/input at t
            dt (float): delta t
        Returns:
            ustar (torch.Tensor): u integrated to time-step t+dt
        """
        grad_ux = self.grad1.grad_h(0.5*uPred[:,:1,:,:]**2)
        grad_uy = self.grad1.grad_v(uPred[:,:1,:,:])

        grad_vx = self.grad1.grad_h(uPred[:,1:,:,:])
        grad_vy = self.grad1.grad_v(0.5*uPred[:,1:,:,:]**2)

        div_u = self.nu * self.grad2(uPred[:,:1,:,:])
        div_v = self.nu * self.grad2(uPred[:,1:,:,:])

        grad_ux0 = self.grad1.grad_h(0.5*uPred0[:,:1,:,:]**2)
        grad_uy0 = self.grad1.grad_v(uPred0[:,:1,:,:])

        grad_vx0 = self.grad1.grad_h(uPred0[:,1:,:,:])
        grad_vy0 = self.grad1.grad_v(0.5*uPred0[:,1:,:,:]**2)

        div_u0 = self.nu * self.grad2(uPred0[:,:1,:,:])
        div_v0 = self.nu * self.grad2(uPred0[:,1:,:,:])
        
        burger_rhs_u = -grad_ux - uPred[:,1:,:,:]*grad_uy + div_u
        burger_rhs_v = -uPred[:,:1,:,:]*grad_vx - grad_vy + div_v
        burger_rhs_u0 = -grad_ux0 - uPred0[:,1:,:,:]*grad_uy0 + div_u0
        burger_rhs_v0 = -uPred0[:,:1,:,:]*grad_vx0 - grad_vy0 + div_v0

        ustar_u = uPred0[:,:1,:,:] + 0.5 * dt * (burger_rhs_u + burger_rhs_u0)
        ustar_v = uPred0[:,1:,:,:] + 0.5 * dt * (burger_rhs_v + burger_rhs_v0)

        return torch.cat([ustar_u, ustar_v], dim=1)

# import argparse
# def get_parser():
#     parser = argparse.ArgumentParser(description='PyTorch framework for NNs')
#     parser.add_argument('--input-channels', type=int, default=6)
#     parser.add_argument('--output-channels', type=int, default=2)
#     parser.add_argument('--modes1', type=int, default=12)
#     parser.add_argument('--modes2', type=int, default=12)
#     parser.add_argument('--width', type=int, default=20)
#     parser.add_argument('--num-filters', type=int, default=96)
#     parser.add_argument('--num-layers', type=int, default=3)
#     parser.add_argument('--dropout-rate', type=float, default=0.)
#     args = parser.parse_args()
#     return args


# if __name__ == '__main__':
#     args = get_parser()
#     args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

#     model = FNO2d(args)
#     # model = ResNet(args)

#     logging.basicConfig(filename = 'test.log', level=logging.DEBUG)
#     summary(model, (6, 64, 64), device='cpu')
    

