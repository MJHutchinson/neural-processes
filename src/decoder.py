import torch
from torch import nn
from torch.nn import functional as F

from src.utils import MLP, BatchMLP


class HeteroskedasticDecoder(nn.Module):
    """ Maps target inputs x and samples of the latent representaion z
    (which encodes the context points) to target outputs y. Predicts both
    the mean and variance of the process.

    Parameters
    ----------
    x_dim : int
        Dimension of the x values.

    rep_dim : int
        Dimension of the representation variable (r+z).

    y_dim : int
        Dimension of the y values.

    hid_dim : int
        Dimension of the hidden layers.

    num_hid : int
        Number of hidden layers in the decoder.
    """

    def __init__(self, x_dim, rep_dim, y_dim, hid_dim, num_hid):
        super(HeteroskedasticDecoder, self).__init__()
        self.x_dim = x_dim
        self.rep_dim = rep_dim
        self.y_dim = y_dim 
        self.hid_dim = hid_dim
        self.num_hid = num_hid

        self.x_rep_to_hidden = MLP(self.x_dim + self.rep_dim, self.hid_dim, self.hid_dim, self.num_hid)
        self.hidden_to_mu  = nn.Linear(self.hid_dim, self.y_dim)
        self.hidden_to_pre_sigma = nn.Linear(self.hid_dim, self.y_dim)

    def forward(self, x, rep):
        """
        x : torch.Tensor
            Shape (batch_size, num_points, x_dim)

        rep : torch.Tensor
            Shape (batch_size, num_points, rep_dim)
        """

        batch_size, num_points, _ = x.size()

        # flatten x and z to be concatenated and passed through the decoder
        rep = rep.view(batch_size * num_points, self.rep_dim)
        x = x.view(batch_size * num_points, self.x_dim)
        input = torch.cat((x, rep), dim=1)

        # pass the state through the decoder to get output
        hidden = F.relu(self.x_rep_to_hidden(input))
        mu = self.hidden_to_mu(hidden)
        pre_sigma = self.hidden_to_pre_sigma(hidden)

        mu = mu.view(batch_size, num_points, self.y_dim)
        pre_sigma = pre_sigma.view(batch_size, num_points, self.y_dim)

        # Formulation from Attentive Neural Processes, Empirical Evaluation of Neural Process Objectives
        sigma = 0.1 + 0.9 * F.softplus(pre_sigma)

        return mu, sigma


class HomoskedasticDecoder(nn.Module):
    """ Maps target inputs x and samples of the latent representaion z
    (which encodes the context points) to target outputs y. Predicts only
    the mean of the process, with a fixed variance.

    Parameters
    ----------
    x_dim : int
        Dimension of the x values.

    z_dim : int
        Dimension of the latent variable z.

    y_dim : int
        Dimension of the y values.

    hid_dim : int
        Dimension of the hidden layers.

    num_hid : int
        Number of hidden layers in the decoder.

    sigma : float
        The fixed standard deviation of the process 
    """

    def __init__(self, x_dim, z_dim, y_dim, hid_dim, num_hid, sigma):
        super(HomoskedasticDecoder, self).__init__()
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.y_dim = y_dim 
        self.hid_dim = hid_dim
        self.num_hid = num_hid
        self.sigma = sigma

        # num_hid + 1 for consistency with the heteroskedastic decoder
        self.xz_to_mu = MLP(self.x_dim + self.z_dim, self.y_dim, self.hid_dim, self.num_hid + 1)

    def forward(self, x, z):
        """
        x : torch.Tensor
            Shape (batch_size, num_points, x_dim)

        z : torch.Tensor
            Shape (batch_size, num_points, z_dim)
        """

        batch_size, num_points, _ = x.size()

        # flatten x and z to be concatenated and passed through the decoder
        z = z.view(batch_size * num_points, self.z_dim)
        x = x.view(batch_size * num_points, self.x_dim)
        input = torch.cat((x, z), dim=1)

        # pass the state through the decoder to get output
        mu = self.xz_to_mu(input)

        mu = mu.view(batch_size, num_points, self.y_dim)

        return mu, self.sigma * torch.ones_like(mu)

