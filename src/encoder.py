import torch
from torch import nn
from torch.nn import functional as F

from src.utils import MLP, BatchMLP


class DeterministicAttentionEncoder(nn.Module):
    """ Maps (x_i, y_i) pairs to a representation vector r_i using a
    MLP network.

    Parameters
    ----------
    x_dim : int 
        Dimension of the x values
    
    y_dim : int
        Dimension of the y values

    r_dim : int
        The dimension of the representation vector

    hid_dim : int
        Dimension of the hidden layers

    num_hid : int
        The number of hidden layers to use
    """

    def __init__(self, x_dim, y_dim, r_dim, attention, hid_dim=50, num_hid=3):
        super(DeterministicAttentionEncoder, self).__init__()

        self.x_dim = x_dim
        self.y_dim = y_dim
        self.r_dim = r_dim
        self.hid_dim = hid_dim
        self.num_hid = num_hid

        self.attention = attention
        self.encoder = BatchMLP(self.x_dim + self.y_dim, self.r_dim, self.hid_dim, self.num_hid, nn.ReLU())

    def forward(self, x, y):
        """ passes a batch of context x and y throught the encoder 
        to get a represnetation vector.

        x : torch.Tensor
            Shape (batch_size, num_points, x_dim)

        y : torch.Tensor
            Shape (batch_size, num_points, y_dim)

        return : torch.Tensor
            Shape (batch_size, num_points, r_dim) 

        """
        xy = torch.cat((x, y), dim=-1)
        r_i = self.model(xy)
        r_i = self.attention(r_i)

        return r_i


class LatentEncoder(nn.Module):
    """ Maps a represnetation vector r to a Gaussian 
    distributed latent variable.

    Parameters
    ----------
    r_dim : int
        The dimension of the representation vector

    z_dim : int
        The dimension of the latent variable z

    TODO: possibility of making this mapping more complex
    """

    def __init__(self, x_dim, y_dim, z_dim, hid_dim=50, num_hid=3):
        super(LatentEncoder, self).__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.z_dim = z_dim

        self.xy_to_hidden = BatchMLP(self.x_dim + self.y_dim, hid_dim, num_hid)
        self.hidden_to_mu = nn.Linear(hid_dim, z_dim)
        self.hidden_to_pre_sigma = nn.Linear(hid_dim, z_dim)

    def forward(self, r):
        """
        r : torch.Tensor
            Shape (batch_size, r_dim)

        return : torch.Tensor, torch.Tensor
            Shape (batch_size, z_dim), (batch_size, z_dim)
        """
        hidden = torch.relu(self.xy_to_hidden(r))
        mu = self.hidden_to_mu(hidden)
        pre_sigma = self.hidden_to_pre_sigma(hidden)

        # formulation from "Attentive Neral Processes" and "Empirical Evaluation of Neural Process Objectives"
        # Seems silly to call it log sigma in this case. Might think about switching to an actual log encoding.
        sigma = 0.1 + 0.9 * F.softplus(pre_sigma)

        return mu, sigma


