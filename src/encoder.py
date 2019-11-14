import torch
from torch import nn
from torch.nn import functional as F

from src.utils import MLP, BatchMLP


class DeterministicEncoder(nn.Module):
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

    def __init__(self, x_dim, y_dim, r_dim, hid_dim=50, num_hid=3):
        super(DeterministicEncoder, self).__init__()

        self.x_dim = x_dim
        self.y_dim = y_dim
        self.r_dim = r_dim
        self.hid_dim = hid_dim
        self.num_hid = num_hid

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
        xy_context = torch.cat((x, y), dim=-1)
        r_i = self.encoder(xy_context)

        return r_i


class LatentEncoder(nn.Module):
    """ Maps (x_i, y_i) pairs to a latent vector z using an
    MLP network which describes a Gaussian distribution.

    Parameters
    ----------
    x_dim : int 
        Dimension of the x values
    
    y_dim : int
        Dimension of the y values

    z_dim : int
        The dimension of the representation vector

    hid_dim : int
        Dimension of the hidden layers

    num_hid : int
        The number of hidden layers to use

    TODO: possibility of making this mapping more complex
    """

    def __init__(self, x_dim, y_dim, z_dim, hid_dim=50, num_hid=3):
        super(LatentEncoder, self).__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.z_dim = z_dim
        self.hid_dim = hid_dim
        self.num_hid = num_hid

        self.xy_to_r_i = BatchMLP(x_dim + y_dim, hid_dim, hid_dim, num_hid)
        self.r_to_hidden = nn.Linear(hid_dim, hid_dim)
        self.hidden_to_mu = nn.Linear(hid_dim, z_dim)
        self.hidden_to_pre_sigma = nn.Linear(hid_dim, z_dim)

    def forward(self, x, y):
        """
        x : torch.Tensor
            Shape (batch_size, num_points, x_dim)

        y : torch.Tensor
            Shape (batch_size, num_points, y_dim)

        return : torch.Tensor, torch.Tensor
            Shape (batch_size, z_dim), (batch_size, z_dim)
        """

        # concat the xs and ys to input to the encoder
        xy = torch.cat((x, y), dim=-1)

        # map the inputs to the encoding per point r_i
        r_i = self.xy_to_r_i(xy)
        # take the average of the points representations per context set
        r = torch.mean(r_i, dim=1)
        # map the represnetations to the mus and sigmas
        hidden = F.relu(self.r_to_hidden(r))
        mu = self.hidden_to_mu(hidden)
        pre_sigma = self.hidden_to_pre_sigma(hidden)

        # formulation from "Attentive Neral Processes" and "Empirical Evaluation of Neural Process Objectives"
        # Seems silly to call it log sigma in this case. Might think about switching to an actual log encoding.
        sigma = 0.1 + 0.9 * F.softplus(pre_sigma)

        return mu, sigma


