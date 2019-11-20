import os
import collections

import torch
import torch.nn as nn
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


def plot_function(
    target_x,
    target_y,
    context_x,
    context_y,
    pred_y,
    std_y,
    save=False,
    show=False,
    dir=None,
    name=None,
):
    """Plots the predicted mean and variance and the context points.

    ONLY plots the first function given.

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

    file = os.path.join(dir, name)

    if save:
        plt.savefig(file + '.pdf')
        plt.savefig(file + '.png')
        plt.close()
    elif show:
        plt.show()

def plot_function(
    target_x,
    target_y,
    context_x,
    context_y,
    pred_y,
    std_y,
    save=False,
    show=False,
    dir=None,
    name=None,
):
    """Plots the predicted mean and variance and the context points.

    ONLY plots the first function given.

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

    file = os.path.join(dir, name)

    if save:
        plt.savefig(file + '.pdf')
        plt.savefig(file + '.png')
        plt.close()
    elif show:
        plt.show()


def plot_compare_processes_gp(
    target_x,
    target_y,
    context_x,
    context_y,
    mean_y,
    std_y,
    mean_gp,
    std_gp,
    save=False,
    show=False,
    dir=None,
    name=None,
):
    """Plots the predicted mean and variance and the context points.

    ONLY plots the first function given.

    DISCLAIMER: not my own code.
    Credits to: https://github.com/deepmind/neural-processes/blob/master/attentive_neural_process.ipynb

    Args: 
    target_x: An array of shape [num_targets,1] that contains the
        x values of the target points.
    target_y: An array of shape [num_targets,1] that contains the
        y values of the target points.
    context_x: An array of shape [num_contexts,1] that contains 
        the x values of the context points.
    context_y: An array of shape [num_contexts,1] that contains 
        the y values of the context points.
    mean_y: An array of shape [num_targets,1] that contains the
        predicted means of the y values at the target points in target_x.
    std_y: An array of shape [num_targets,1] that contains the
        predicted std dev of the y values at the target points in target_x.
    mean_gp: An array of shape [num_targets,1] that contains the
        GP means of the y values at the target points in target_x.
    std_gp: An array of shape [num_targets,1] that contains the
        GP std dev of the y values at the target points in target_x.
    """
    # Plot the target line
    plt.figure()
    plt.plot(target_x, target_y, "k:", linewidth=2)  # the .data converts it back to tensor

    # Plot the context set
    plt.plot(context_x, context_y, "ko", markersize=10)
    
    # Plot the process posterior function
    plt.plot(target_x, mean_y.data, "b", linewidth=2)
    plt.fill_between(
        target_x,
        mean_y - std_y,
        mean_y + std_y,
        alpha=0.2,
        facecolor="b",
        interpolate=True,
    )

    # Plot the GP posterior function on the context
    plt.plot(target_x, mean_gp, "g", linewidth=2)
    plt.fill_between(
        target_x,
        mean_gp - std_gp,
        mean_gp + std_gp,
        alpha=0.2,
        facecolor="g",
        interpolate=True,
    )

    # Make the plot pretty
    plt.yticks([-2, 0, 2], fontsize=16)
    plt.xticks([-2, 0, 2], fontsize=16)
    plt.ylim([-3, 3])
    plt.grid("off")
    plt.gca()

    file = os.path.join(dir, name)

    if save:
        plt.savefig(file + '.pdf')
        plt.savefig(file + '.png')
        plt.close()
    elif show:
        plt.show()

def plot_compare_processes_gp_latent(
    target_x,
    target_y,
    context_x,
    context_y,
    mean_y,
    std_y,
    mean_gp,
    std_gp,
    save=False,
    show=False,
    dir=None,
    name=None,
    ):
    """Plots the predicted mean and variance and the context points.

    ONLY plots the first function given.

    DISCLAIMER: not my own code.
    Credits to: https://github.com/deepmind/neural-processes/blob/master/attentive_neural_process.ipynb

    Args: 
    target_x: An array of shape [num_targets,1] that contains the
        x values of the target points.
    target_y: An array of shape [num_targets,1] that contains the
        y values of the target points.
    context_x: An array of shape [num_contexts,1] that contains 
        the x values of the context points.
    context_y: An array of shape [num_contexts,1] that contains 
        the y values of the context points.
    mean_y: An array of shape [samples, num_targets,1] that contains the
        predicted means of the y values at the target points in target_x.
    std_y: An array of shape [samples, num_targets,1] that contains the
        predicted std dev of the y values at the target points in target_x.
    mean_gp: An array of shape [num_targets,1] that contains the
        GP means of the y values at the target points in target_x.
    std_gp: An array of shape [num_targets,1] that contains the
        GP std dev of the y values at the target points in target_x.
    """
    
    # Plot the target line
    plt.figure()
    plt.plot(target_x, target_y, "k:", linewidth=2)  # the .data converts it back to tensor

    # Plot the context set
    plt.plot(context_x, context_y, "ko", markersize=10)
    
    # Plot the process posterior function

    means_y = torch.unbind(mean_y, dim=0)
    stds_y = torch.unbind(std_y, dim=0)
    for m_y, s_y in zip(means_y, stds_y):
        plt.plot(target_x, m_y.data, "b", linewidth=2, alpha=0.2)
        plt.fill_between(
            target_x,
            m_y - s_y,
            m_y + s_y,
            alpha=0.05,
            facecolor="b",
            interpolate=True,
        )

    # Plot the GP posterior function on the context
    plt.plot(target_x, mean_gp, "g", linewidth=2)
    plt.fill_between(
        target_x,
        mean_gp - std_gp,
        mean_gp + std_gp,
        alpha=0.2,
        facecolor="g",
        interpolate=True,
    )

    # Make the plot pretty
    plt.yticks([-2, 0, 2], fontsize=16)
    plt.xticks([-2, 0, 2], fontsize=16)
    plt.ylim([-3, 3])
    plt.grid("off")
    plt.gca()

    file = os.path.join(dir, name)

    if save:
        plt.savefig(file + '.pdf')
        plt.savefig(file + '.png')
        plt.close()
    elif show:
        plt.show()


def kernel_interpolate(values, value_locations, target_locations, kernel):
        """ Takes in some values at some given location, and computes the 
        kernel interpolated values at the target locations.

        Parameters
        ----------
        values : torch.Tensor
            Shape (num_batches, num_values, values_dim)

        value_locations : torch.Tensor
            Shape (num_batches, num_values, location_dim)

        target_locations : torch.Tensor
            Shape (num_batches, num_targets, location_dim)

        kernel : Stheno.kernel

        returns : torch.Tensor
            Shape (num_batches, num_targets, values_dim)
        """

        num_batches, num_values, values_dim = values.shape
        num_batches, num_targets, location_dim = target_locations.shape

        gramm_targets_values = [
            # kernel(target_loc, value_loc)
            kernel(target_loc, value_loc).mat
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

# def kernal_interpolate_multidim(values, value_locations, target_locations, kernal):
#         """ Takes in some values at some given location, and computes the 
#         kernal interpolated values at the target locations.

#         Parameters
#         ----------
#         values : torch.Tensor
#             Shape (num_batches, num_values, values_dim)

#         value_locations : torch.Tensor
#             Shape (num_batches, num_values, location_dim)

#         target_locations : torch.Tensor
#             Shape (num_batches, num_targets, location_dim)

#         kernel : Stheno.Kernal

#         returns : torch.Tensor
#             Shape (num_batches, num_targets, values_dim)
#         """

#         num_batches, num_values, values_dim = values.shape
#         num_batches, num_targets, location_dim = target_locations.shape

#         gramm_targets_values = torch.zeros()
#         for t in range(target_locations.shape[1]):

#         gramm_targets_values = [
#             # kernal(target_loc, value_loc)
#             kernal(target_loc , value_loc).mat
#             for 
#             target_loc, value_loc 
#             in 
#             zip(target_locations.unbind(dim=0), value_locations.unbind(dim=0))
#         ]

#         gramm_targets_values = torch.stack(gramm_targets_values, dim=0)

#         targets = torch.einsum('bvd,btv->btd', values, gramm_targets_values)

#         return targets
