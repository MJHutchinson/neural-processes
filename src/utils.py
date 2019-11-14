import torch
from torch import nn
import collections

NPRegressionDescription = collections.namedtuple(
    "NPRegressionDescription",
    ("query", "target_y", "num_total_points", "num_context_points"),
)

class MLP(nn.Module):
    """ Defines a basic MLP network.

    Usage:

        test = torch.ones(2,2,8)
        mlp = MLP(8, 3, 3, 3)
        mlp(test)

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

    def __init__(self, in_dim, out_dim, hid_dim, num_hid, activation=nn.ReLU()):
        super(MLP, self).__init__()
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
<<<<<<< HEAD
            layers.append(self.activation)
=======
            layers.append(activation)
>>>>>>> cee11a25cc8c1678dae3725e625682587ee6bf3c

        # append final layer with no activations
        layers.append(nn.Linear(sizes[-2], sizes[-1]))

        self.model = nn.Sequential(*layers)

    def forward(self, input):
        return self.model(input)


class BatchMLP(nn.Module):
    """ Defines a basic MLP network, except this will 
    expect the inputs to have an extra dimension that
    will need to be flattened before going through the 
    MLP and then recovered after.

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

    def __init__(self, in_dim, out_dim, hid_dim, num_hid, activation=nn.ReLU()):
        super(BatchMLP, self).__init__()
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
            layers.append(self.activation)

        # append final layer with no activations
        layers.append(nn.Linear(sizes[-2], sizes[-1]))

        self.model = nn.Sequential(*layers)

    def forward(self, input):
        """ Passes the 3D tensor through the network

        input : torch.Tensor
            Shape (batch_size, num_points, input_dim)      

        return : torch.Tensor
            Shape (batch_size, num_points, output_dim)      
        """

        batch_size, num_points, _ = input.size()

        input = input.view((batch_size * num_points, self.in_dim))
        output = self.model(input)
        return output.view(batch_size, num_points, self.out_dim)