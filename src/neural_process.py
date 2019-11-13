import torch
from torch import nn
from torch.distributions import Normal

class NeuralProcess(nn.Module):
    """ Implements the Neural Process in a general form.
    
    """

    def __init__(self, x_dim, y_dim, r_dim, z_dim, encoder, aggregator, latent_encoder, decoder, use_deterministic_path):
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.r_dim = r_dim
        self.z_dim = z_dim

        self.encoder = encoder
        self.aggregator = aggregator
        self.latent_encoder = latent_encoder
        self.decoder = decoder

        self.use_deterministic_path = use_deterministic_path

    def xy_to_r(self, x, y):
        """
        Maps sets of (x,y) pairs to the repreentation vector r for each set of pairs.

        Parameters
        ----------
        x : torch.Tensor
            Shape (batch_size, num_points, x_dim)

        y : torch.Tensor
            Shape (batch_size, num_points, y_dim)

        return : torch.Tensor
            Shape (batch_size, r_dim)
        """
        batch_size, num_points, _ = x.size()

        # flatten x's and y's and aggregate to go through the decoder
        x = x.view(batch_size * num_points, self.x_dim)
        y = y.view(batch_size * num_points, self.y_dim)

        r_i = self.encoder(x, y)
        r_i = r_i.view(batch_size, num_points, self.r_dim)

        # aggregate the r_i for each batch into the overall 
        # representation r
        r = self.aggregator(r_i)
        return r

    def r_to_z(self, r):
        return self.latent_encoder(r)

    def forward(self, x_context, y_context, x_target, y_target=None):

        batch_size, num_context, y_dim = y_context.size()
        _, num_target, x_size = x_target.size()

        # If we are training and have y_targets, then we want 
        # to encode the training and context set, as we need 
        # the context distribution to compute the KL
        if self.training and y_target is not None:
            r_target = self.xy_to_r(x_target, y_target)
            z_mu_target , z_sigam_target = self.r_to_z(r_target)

            r_context = self.xy_to_r(x_context, y_context)
            z_mu_context , z_sigam_context = self.r_to_z(r_context)

            q_target = Normal(z_mu_target, z_sigam_target)
            q_context = Normal(z_mu_context, z_sigam_context)

            z_