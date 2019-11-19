import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from stheno.torch import EQ

from src.utils import kernel_interpolate, EQKernel


class SimpleCNN(nn.Module):
    """ Implements the smaller CNN used for 1D experiments
    in the ConvCNP paper.

    Parameters
    ----------
    in_channels : int
        number of channels coming in as input. usually 1 + dim_y + dim_x

    dimension : int
        dimensionality of the x's (i.e. the number of extra dimensions to expect)
    """

    def __init__(self, in_channels, dimension):
        super(SimpleCNN, self).__init__()

        kernel_size = 5
        padding = math.floor(kernel_size/2)
        stride = 1
        num_channels = [in_channels] + [16, 32, 16, 2]

        layers = []

        # Set the correct conv type for the x dimension
        if dimension == 1:
            conv = nn.Conv1d
        elif dimension == 2:
            conv = nn.Conv2d

        for i in range(len(num_channels) - 1):
            layers.append(conv(num_channels[i], num_channels[i+1], kernel_size, stride, padding))
            layers.append(nn.ReLU())

        layers = layers[:-1]

        self.model = nn.Sequential(*layers)

    def forward(self, t_h):
        """ Computes the forward pass for concatedated input locations and
        evaluated representation function.

        Parameters
        ----------
        t_h : torch.Tensor
            The concatenated input location + representation function evaluated at the point
            Shape (batch_size, in_channel, (grid_shape))

        return : torch.Tensor, torch.Tensor
            y_mu and y_sigma for the gridded points
            Shape (batch_size, 1, (grid_shape)), Shape (batch_size, 1, (grid_shape))
        """

        y = self.model(t_h)
        y_mu = y.select(1, 0).unsqueeze(1)
        y_sigma = y.select(1, 1).unsqueeze(1)
        y_sigma = F.softplus(y_sigma)

        return y_mu, y_sigma


class XLCNN(nn.Module):

    def __init__(self, in_channels, dimension):
        super(XLCNN, self).__init__()

        kernel_size = 5
        padding = math.floor(kernel_size/2)
        stride = 1
        num_channels = [in_channels, 
                        2*in_channels,
                        4*in_channels,
                        8*in_channels,
                        16*in_channels,
                        32*in_channels,
                        64*in_channels,
                        32*in_channels,
                        16*in_channels,
                        8*in_channels,
                        4*in_channels,
                        2*in_channels,
                        2]

        # Set the correct conv type for the x dimension
        if dimension == 1:
            conv = nn.Conv1d
        elif dimension == 2:
            conv = nn.Conv2d

        self.L1 = conv(num_channels[0], num_channels[1], kernel_size, stride, padding)
        self.L2 = conv(num_channels[1], num_channels[2], kernel_size, stride, padding)
        self.L3 = conv(num_channels[2], num_channels[3], kernel_size, stride, padding)
        self.L4 = conv(num_channels[3], num_channels[4], kernel_size, stride, padding)
        self.L5 = conv(num_channels[4], num_channels[5], kernel_size, stride, padding)
        self.L6 = conv(num_channels[5], num_channels[6], kernel_size, stride, padding)
        self.L7 = conv(num_channels[6], num_channels[7], kernel_size, stride, padding)
        self.L8 = conv(num_channels[7]*2, num_channels[8], kernel_size, stride, padding)
        self.L9 = conv(num_channels[8]*2, num_channels[9], kernel_size, stride, padding)
        self.L10 = conv(num_channels[9]*2, num_channels[10], kernel_size, stride, padding)
        self.L11 = conv(num_channels[10]*2, num_channels[11], kernel_size, stride, padding)
        self.L12 = conv(num_channels[11]*2, num_channels[12], kernel_size, stride, padding)

    def forward(self, input):
        L1_out = F.relu(self.L1(input))
        L2_out = F.relu(self.L2(L1_out))
        L3_out = F.relu(self.L3(L2_out))
        L4_out = F.relu(self.L4(L3_out))
        L5_out = F.relu(self.L5(L4_out))
        L6_out = F.relu(self.L6(L5_out))
        L7_out = F.relu(self.L7(L6_out))

        L8_out = F.relu(self.L8(
            torch.cat((
                L5_out,
                L7_out
            ), dim=1)
        ))

        L9_out = F.relu(self.L9(
            torch.cat((
                L4_out,
                L8_out
            ), dim=1)
        ))

        L10_out = F.relu(self.L10(
            torch.cat((
                L3_out,
                L9_out
            ), dim=1)
        ))

        L11_out = F.relu(self.L11(
            torch.cat((
                L2_out,
                L10_out
            ), dim=1)
        ))

        L12_out = self.L12(
            torch.cat((
                L1_out,
                L11_out
            ), dim=1)
        )

        y_mu = L12_out.select(1, 0).unsqueeze(1)
        y_sigma = L12_out.select(1, 1).unsqueeze(1)
        y_sigma = F.softplus(y_sigma)

        return y_mu, y_sigma



# class EQkernel(nn.Module):
#     """ Implementation of the EQ keranl with a optionally learnable length
#     scale. Didn't use Stheno as it doesn't handle multiple batches well.

#     Parameters 
#     ----------
#     length_scale : torch.Tensor
#         The intial length scales to use for the data. Sould either be a scalar or 
#     """

#     def __init__(self, length_scale, trainable=True):
#         super(EQkernel, self).__init__()

#         if trainable:
#             self.length_scale = nn.Parameter(torch.tensor(length_scale))
#         else:
#             self.length_scale = torch.tensor(length_scale)

class ConvCNP(nn.Module):


    def __init__(self, cnn='simple'):
        super(ConvCNP, self).__init__()

        self.kernel_x_lengh_scale = nn.Parameter(torch.tensor(.5))
        self.kernel_rho_lengh_scale = nn.Parameter(torch.tensor(.5))
        self.kernel_x = EQ() > self.kernel_x_lengh_scale
        self.kernel_rho = EQ() > self.kernel_rho_lengh_scale
        # self.kernel_x = EQKernel(length_scale=.15, trainable=True)
        # self.kernel_rho = EQKernel(length_scale=.15, trainable=True)
        if cnn == 'simple':
            self.rho_cnn = SimpleCNN(2, 1)
        elif cnn == 'xl':
            self.rho_cnn = XLCNN(2, 1)


    def forward(self, x_context, y_context, x_target, y_target=None):
        """
        
        """

        num_batches, y_dim, num_context = y_context.shape
        _, x_dim, num_targets = x_target.shape

        x_context = x_context.transpose(1,2)
        y_context = y_context.transpose(1,2)
        x_target = x_target.transpose(1,2)
        y_target = y_target.transpose(1,2)
        
        # append the channel of ones to the y vector to give it the density channel.
        # This is the reqired kernel when the multiplicity of x_context is 1
        phi_y_context = torch.cat(
            (
                torch.ones_like(y_context),
                y_context
            ),
            dim=1
        )

        # Produces a 1D grid of input points. 
        # TODO:  make this depend on the target set's required support
        # (maybe min - 10% of range, max + 10% of range), and a density of
        # something on 10x the order of context points?
        # TODO: make this work for 2D or more x dimensions

        # x_min = torch.min(torch.tensor([torch.min(x_context), torch.min(x_target)]))
        # x_max = torch.max(torch.tensor([torch.max(x_context), torch.max(x_target)]))
        # x_range = x_max - x_min
        t_grid = torch.linspace(-2.2, 2.2, 1000).unsqueeze(-1)
        # t_grid = torch.linspace(x_min - 0.05 * x_range, x_max + 0.05 * x_range, 100).unsqueeze(-1)
        
        # Expand the t_grid to match the number of batches
        t_grid = t_grid.T.unsqueeze(0).repeat_interleave(num_batches, dim=0)

        # Calculate the repersentation function at each location of the grid
        # Need the transpositions as conv ops take one order of dimensions
        # and Stheno kernels the opposite.
        h_grid = kernel_interpolate(
            phi_y_context.transpose(1,2), 
            x_context.transpose(1,2), 
            t_grid.transpose(1,2),
            self.kernel_x
        ).transpose(1,2)

        # divide h_1 by h_0 for stability
        h_density_channel = h_grid[:, 0, :].unsqueeze(1)
        h_rest = h_grid[:, 1:, :]
        h_rest = h_rest / (h_density_channel + 1e-8)
        h_grid = torch.cat((h_density_channel, h_rest), dim=1)

        # Concatenate the t_grid locations with the evaluated represnetation
        # functions
        # rep = torch.cat((t_grid, h_grid), dim=1)

        # Pass the representation through the decoder.
        y_mu_grid, y_sigma_grid = self.rho_cnn(h_grid)

        y_grid = torch.cat((y_mu_grid, y_sigma_grid), dim=1)

        y_pred_target = kernel_interpolate(
            y_grid.transpose(1,2), 
            t_grid.transpose(1,2),
            x_target.transpose(1,2), 
            self.kernel_rho
        ).transpose(1, 2)

        y_pred_target_mu, y_pred_target_sigma = torch.chunk(y_pred_target, 2, dim=1)

        # If we have a y_target, then return the loss. Else do not.
        if y_target is not None:
            loss = -Normal(y_pred_target_mu, y_pred_target_sigma).log_prob(y_target).mean(dim=-1).sum()
        else:
            loss = None

        return y_pred_target_mu.transpose(1,2), y_pred_target_sigma.transpose(1,2), loss