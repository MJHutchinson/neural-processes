import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from stheno.torch import EQ

from src.utils import kernel_interpolate


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


    def __init__(self, process_dimension, cnn='simple', unit_density=32):
        super(ConvCNP, self).__init__()

        # Initialise at double grid spacing
        self.kernel_x_lengh_scale = nn.Parameter(torch.tensor(2./unit_density))
        self.kernel_rho_lengh_scale = nn.Parameter(torch.tensor(2./unit_density))
        self.kernel_x = EQ() > self.kernel_x_lengh_scale
        self.kernel_rho = EQ() > self.kernel_rho_lengh_scale
        self.unit_density = unit_density
        # self.kernel_x = EQKernel(length_scale=.15, trainable=True)
        # self.kernel_rho = EQKernel(length_scale=.15, trainable=True)
        if cnn == 'simple':
            self.rho_cnn = SimpleCNN(process_dimension+1, process_dimension)
        elif cnn == 'xl':
            self.rho_cnn = XLCNN(process_dimension+1, process_dimension)


    def forward(self, x_context, y_context, x_target, y_target=None):
        """ Forward pass on the context and target set 
        
        Arguments:
            x_context {torch.Tensor} -- Shape (batch_size, num_context, x_dim)
            y_context {torch.Tensor} -- Shape (batch_size, num_context, y_dim)
            x_target {torch.Tensor} -- Shape (batch_size, num_target, x_dim). Assumes this is a superset of x_context.
        
        Keyword Arguments:
            y_target {torch.Tensor} -- Shape (batch_size, num_target, y_dim). Assumes this is a superset of y_context. (default: {None})
        
        Returns:
            [y_pred_mu, y_pred_sigma, loss] -- [Mean and variance of the predictions for the y_target. The loss if we have y_targets to test against]
        """
        num_batches, num_context, y_dim = y_context.shape
        _, num_targets, x_dim = x_target.shape
        
        # append the channel of ones to the y vector to give it the density channel.
        # This is the reqired kernel when the multiplicity of x_context is 1

        t_grid_i = []

        # Loop through the x_dimensions and create a grid uniformly spaced with a 
        # density specified via self.unit_density and a range that definity covers
        # the range of the targets
        for i in range(x_dim):
            # get the x's in the desired dimension
            x_i = torch.select(x_target, dim=-1, index=i)

            # find the integer that lower and upper bound the min and max of the 
            # target x's in this dimension. Multiplying by 1.1 to give a bit of extra 
            # room
            x_min_i = torch.floor(torch.min(x_i) * 1.1)
            x_max_i = torch.ceil(torch.max(x_i) * 1.1)

            # create a uniform linspace
            t_grid_i.append(torch.linspace(x_min_i, x_max_i, int(self.unit_density * (x_max_i - x_min_i))))

        t_grid = torch.meshgrid(*t_grid_i)
        t_grid = torch.stack(t_grid, dim=-1)

        t_grid_shape = t_grid.shape
        t_grid = t_grid.view(-1, x_dim)

        # Expand the t_grid to match the number of batches
        t_grid = t_grid.unsqueeze(0).repeat_interleave(num_batches, dim=0)

        # Calculate the repersentation function at each location of the grid
        # Need the transpositions as conv ops take one order of dimensions
        # and Stheno kernels the opposite.
        h_grid = kernel_interpolate(
            y_context,
            x_context,
            t_grid,
            self.kernel_x,
            keep_density_channel=True
        )

        # Concatenate the t_grid locations with the evaluated represnetation
        # functions
        # rep = torch.cat((t_grid, h_grid), dim=1)

        # Pass the representation through the decoder.
        y_mu_grid, y_sigma_grid = self.rho_cnn(h_grid.transpose(1,2).view(num_batches, y_dim + 1, *list(t_grid_shape)[:-1]))
        y_mu_grid = y_mu_grid.view(num_batches, 1, -1).transpose(1,2)
        y_sigma_grid = y_sigma_grid.view(num_batches, 1, -1).transpose(1,2)

        y_grid = torch.cat((y_mu_grid, y_sigma_grid), dim=-1)

        y_pred_target = kernel_interpolate(
            y_grid,
            t_grid,
            x_target,
            self.kernel_rho,
            keep_density_channel=False
        )

        y_pred_target_mu, y_pred_target_sigma = torch.chunk(y_pred_target, 2, dim=-1)

        # If we have a y_target, then return the loss. Else do not.
        if y_target is not None:
            loss = - Normal(y_pred_target_mu, y_pred_target_sigma).log_prob(y_target).mean()
        else:
            loss = None

        return y_pred_target_mu, y_pred_target_sigma, loss, None, None