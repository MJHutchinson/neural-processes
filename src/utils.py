import torch
from torch import nn
import collections

NPRegressionDescription = collections.namedtuple(
    "NPRegressionDescription",
    ("query", "target_y", "num_total_points", "num_context_points"),
)

class MLP(nn.Module):
    """ Defines a basic MLP network.

    Parameters
    ----------
    in_dim : int 
        Dimension of the input

    out_dim : int
        The dimension of the output

    hid_dim : int
        Dimension of the hidden layers

    num_hid : int
        The number of hidden layers to use

    activation : torch.nn.Module
        The activation function to use between layers
    """

    def __init__(self, in_dim, out_dim, hid_dim, num_hid, activation=nn.ReLU):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hid_dim = hid_dim
        self.num_hid = num_hid
        self.activation = activation

        layers = []
        sizes = [self.in_dim] + [self.hid_dim] * num_hid + [self.out_dim]

        # append layers with acitvation functions
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i+1]))

        # append final layer with no activations
        layers.append(nn.Linear(sizes[-2], sizes[-1]))

        self.model = nn.Sequential(layers)

    def forward(self, input):
        return self.model(input)