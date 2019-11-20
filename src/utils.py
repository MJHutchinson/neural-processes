import torch
from torch import nn
import collections
import matplotlib.pyplot as plt

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
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            layers.append(self.activation)

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
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
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


def plot_functions(
    target_x,
    target_y,
    context_x,
    context_y,
    pred_y,
    std_y,
    save=False,
    show=False,
    experiment_name=None,
    iter=None,
):
    """Plots the predicted mean and variance and the context points.

    DISCLAIMER: not my own code.
    Credits to: https://github.com/deepmind/neural-processes/blob/master/attentive_neural_process.ipynb

    Args: 
    target_x: An array of shape [B,num_targets,1] that contains the
        x values of the target points.
    target_y: An array of shape [B,num_targets,1] that contains the
        y values of the target points.
    context_x: An array of shape [B,num_contexts,1] that contains 
        the x values of the context points.
    context_y: An array of shape [B,num_contexts,1] that contains 
        the y values of the context points.
    pred_y: An array of shape [B,num_targets,1] that contains the
        predicted means of the y values at the target points in target_x.
    std: An array of shape [B,num_targets,1] that contains the
        predicted std dev of the y values at the target points in target_x.
    """
    # Plot everything
    plt.figure()
    plt.plot(target_x[0], pred_y[0].data, "b", linewidth=2)
    plt.plot(
        target_x[0], target_y[0], "k:", linewidth=2
    )  # the .data converts it back to tensor
    plt.plot(context_x[0], context_y[0], "ko", markersize=10)
    plt.fill_between(
        target_x[0, :, 0],
        pred_y.data[0, :, 0] - std_y.data[0, :, 0],
        pred_y.data[0, :, 0] + std_y.data[0, :, 0],
        alpha=0.2,
        facecolor="#65c9f7",
        interpolate=True,
    )

    # Make the plot pretty
    plt.yticks([-2, 0, 2], fontsize=16)
    plt.xticks([-2, 0, 2], fontsize=16)
    plt.ylim([-2, 2])
    plt.grid("off")
    plt.gca()

    if save:
        plt.savefig("results/{}/{}.png".format(experiment_name, iter))
    elif show:
        plt.show()


def kernal_interpolate(values, value_locations, target_locations, kernal):
        """ Takes in some values at some given location, and computes the 
        kernal interpolated values at the target locations.

        Parameters
        ----------
        values : torch.Tensor
            Shape (num_batches, num_values, values_dim)

        value_locations : torch.Tensor
            Shape (num_batches, num_values, location_dim)

        target_locations : torch.Tensor
            Shape (num_batches, num_targets, location_dim)

        kernel : Stheno.Kernal

        returns : torch.Tensor
            Shape (num_batches, num_targets, values_dim)
        """

        num_batches, num_values, values_dim = values.shape
        num_batches, num_targets, location_dim = target_locations.shape

        gramm_targets_values = [
            # kernal(target_loc, value_loc)
            kernal(target_loc , value_loc).mat
            for 
            target_loc, value_loc 
            in 
            zip(target_locations.unbind(dim=0), value_locations.unbind(dim=0))
        ]

        gramm_targets_values = torch.stack(gramm_targets_values, dim=0)

        targets = torch.einsum('bvd,btv->btd', values, gramm_targets_values)

        return targets


class EQKernel(nn.Module):

    def __init__(self, length_scale, trainable=True):
        super(EQKernel, self).__init__()
        
        self.length_scale = torch.tensor(length_scale)
        if trainable:
            self.length_scale = nn.Parameter(self.length_scale)

    def __call__(self, x, y):
        euclid_norms = (x-y).norm()

        return torch.exp(-0.5 * euclid_norms / (self.length_scale ** 2))

def kernal_interpolate_multidim(values, value_locations, target_locations, kernal):
        """ Takes in some values at some given location, and computes the 
        kernal interpolated values at the target locations.

        Parameters
        ----------
        values : torch.Tensor
            Shape (num_batches, num_values, values_dim)

        value_locations : torch.Tensor
            Shape (num_batches, num_values, location_dim)

        target_locations : torch.Tensor
            Shape (num_batches, num_targets, location_dim)

        kernel : Stheno.Kernal

        returns : torch.Tensor
            Shape (num_batches, num_targets, values_dim)
        """

        num_batches, num_values, values_dim = values.shape
        num_batches, num_targets, location_dim = target_locations.shape

        gramm_targets_values = torch.zeros()
        for t in range(target_locations.shape[1]):

        gramm_targets_values = [
            # kernal(target_loc, value_loc)
            kernal(target_loc , value_loc).mat
            for 
            target_loc, value_loc 
            in 
            zip(target_locations.unbind(dim=0), value_locations.unbind(dim=0))
        ]

        gramm_targets_values = torch.stack(gramm_targets_values, dim=0)

        targets = torch.einsum('bvd,btv->btd', values, gramm_targets_values)

        return targets
