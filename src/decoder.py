import torch
from torch import nn
from torch.nn import functional as F

from src.utils import MLP


class MLPDecoder(nn.Module):
    """ Maps target inputs x and samples of the latent representaion z
    (which encodes the context points) to target outputs y. 

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
    """

    def __init__(self, x_dim, z_dim, y_dim, hid_dim, num_hid):
        super(MLPDecoder, self).__init__()
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.y_dim = y_dim 
        self.hid_dim = hid_dim
        self.num_hid = num_hid

        self.xz_to_hidden = MLP(self.x_dim + self.z_dim, self.hid_dim, self.hid_dim, self.num_hid)
        self.hidden_to_mu  = nn.Linear(self.hid_dim, self.y_dim)
        self.hidden_to_pre_sigma = nn.Linear(self.hid_dim, self.y_dim)

    def forward(self, x, z):
        """
        x : torch.Tensor
            Shape (batch_size, num_points, x_dim)

        z : torch.Tensor
            Shape (batch_size, z_dim)
        """

        batch_size, num_points, _ = x.size()

        # one z for all the target points of a given set, expand z to 
        # be concatenated with each x from the relevant process. In the
        # future we might want multiple samples from the latent variable,
        # so may need to add another dimension (samples) to the incoming
        # z vector

        z = z.unsqueeze(1).repeat(1, num_points, 1)

        # flatten x and z to be concatenated and passed through the decoder
        z = z.view(batch_size * num_points, self.x_dim)
        x = x.view(batch_size * num_points, self.z_dim)
        input = torch.cat((x, z), dim=1)

        # pass the state through the decoder to get output
        hidden = self.xz_to_hidden(input)
        mu = self.hidden_to_mu(hidden)
        pre_sigma = self.hidden_to_pre_sigma(hidden)

        mu = mu.view(batch_size, num_points, self.y_dim)
        pre_sigma = pre_sigma.view(batch_size, num_points, self.y_dim)

        # Formulation from Attentive Neural Processes, Empirical Evaluation of Neural Process Objectives
        sigma = 0.1 + 0.9 * F.softplus(pre_sigma)

        return mu, sigma

