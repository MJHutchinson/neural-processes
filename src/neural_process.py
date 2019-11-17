import torch
from torch import nn
from torch.distributions import Normal


class AttentiveNeuralProcess(nn.Module):
    """ Implements the Neural Process in a general form.
    
    """

    def __init__(
        self,
        deterministic_encoder,
        attention,
        latent_encoder,
        decoder,
        use_deterministic_path,
    ):
        super(AttentiveNeuralProcess, self).__init__()
        self.deterministic_encoder = deterministic_encoder
        self.attention = attention
        self.latent_encoder = latent_encoder
        self.decoder = decoder

        self.use_deterministic_path = use_deterministic_path

    def xyx_to_r(self, x_context, y_context, x_target):
        """
        Maps sets of (x_context ,y_context) sets and x_targets to the reprsentation vector r_i
        for each pair of contexts and target.

        Parameters
        ----------
        x_context : torch.Tensor
            Shape (batch_size, num_context_points, x_dim)

        y_context : torch.Tensor
            Shape (batch_size, num_context_points, y_dim)

        x_target : torch.Tensor
            Shape (batch_size, num_target_points, x_dim)

        return : torch.Tensor
            Shape (batch_size, num_target_points, r_dim)
        """
        batch_size, num_context_points, y_size = y_context.size()
        _, num_target_points, _ = x_target.size()

        # map the context points to their deterministic represnetations.
        # r_i : Shape (batch_size, num_context_points, r_dim)
        r_i = self.deterministic_encoder(x_context, y_context)

        # apply attention to the representations and x_targets to get a
        # representation per target. r_j : Shape (batch_size, num_target_points r_dim)
        r_j = self.attention(x_target, x_context, r_i)
        return r_j

    def xy_to_z(self, x, y):
        return self.latent_encoder(x, y)

    def forward(self, x_context, y_context, x_target, y_target=None):

        batch_size, num_context, _ = y_context.size()
        _, num_target, x_size = x_target.size()

        # If we are training and have y_targets, then we want
        # to encode the training and context set, as we need
        # the context distribution to compute the KL
        z_mu_context, z_sigam_context = self.xy_to_z(x_context, y_context)

        q_context = Normal(z_mu_context, z_sigam_context)

        # If we don't have a y_target, we are are in prediction mode.
        # If we do have a y_target, we are in training mode.

        training = y_target is not None
        if training:
            z_mu_target, z_sigma_target = self.xy_to_z(x_target, y_target)

            q_target = Normal(z_mu_target, z_sigma_target)

            latent_sample = q_target.sample()

        else:
            latent_sample = q_context.sample()

        latent_sample = latent_sample.unsqueeze(dim=1).expand(-1, num_target, -1)

        if self.use_deterministic_path:
            deterministic_representation = self.xyx_to_r(x_context, y_context, x_target)
            rep = torch.cat((deterministic_representation, latent_sample), dim=-1)
        else:
            rep = latent_sample

        y_target_mu, y_target_sigma = self.decoder(x_target, rep)

        # TODO: Make this loss function flexible to use the vairous proposals
        # in Empirical Evaluation of Neural Process Objectives. See there for
        # details of this loss too.
        if training:
            # Log predictive probability of the observations
            y_dist = Normal(y_target_mu, y_target_sigma)
            log_pred = y_dist.log_prob(y_target).sum(dim=-1)
            # KL divergence between the context latent and target latent
            kl_target_context = (
                torch.distributions.kl_divergence(q_target, q_context)
                .sum(dim=-1, keepdim=True)
                .expand(-1, num_target)
            )
            # prior = Normal(0, 1)
            # kl_target_prior = torch.distributions.kl_divergence(q_target, prior).sum(dim=-1, keepdim=True).expand(-1, num_target)
            # kl_context_prior = torch.distributions.kl_divergence(context, prior).sum(dim=-1, keepdim=True).expand(-1, num_target)
            loss = -torch.mean(log_pred - kl_target_context / num_target)
        else:
            log_pred = None
            kl_target_context = (None,)
            loss = None

        return y_target_mu, y_target_sigma, log_pred, kl_target_context, loss

