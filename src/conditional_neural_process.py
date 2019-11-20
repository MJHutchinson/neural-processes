import torch
from torch import nn
from torch.distributions import Normal


class ConditionalNeuralProcess(nn.Module):
    """ Implements the Conditional Neural Process in a general form.
    
    """

    def __init__(
        self,
        deterministic_encoder,
        attention,
        decoder,
    ):
        super(ConditionalNeuralProcess, self).__init__()
        self.deterministic_encoder = deterministic_encoder
        self.attention = attention
        self.decoder = decoder

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

        deterministic_representation = self.xyx_to_r(x_context, y_context, x_target)

        y_target_mu, y_target_sigma = self.decoder(x_target, deterministic_representation)

        if y_target is not None:
            y_dist = Normal(y_target_mu, y_target_sigma)
            log_pred = y_dist.log_prob(y_target).sum(dim=-1)
            loss = -torch.mean(log_pred)
        else:
            loss = None

        return y_target_mu, y_target_sigma, loss, _, _

    # def evaluate(self, x_context, y_context, x_target, y_target):

    #     batch_size, num_context, _ = y_context.size()
    #     _, num_target, x_size = x_target.size()

    #     # remove the context points from the targets for predictive LL

    #     x_target_sub_context = x_target[:, num_context:, :]
    #     y_target_sub_context = y_target[:, num_context:, :]

    #     # compute the latent representations of the context and targets
    #     z_mu_context, z_sigam_context = self.xy_to_z(x_context, y_context)
    #     q_context = Normal(z_mu_context, z_sigam_context)

    #     z_mu_target, z_sigma_target = self.xy_to_z(x_target, y_target)
    #     q_target = Normal(z_mu_target, z_sigma_target)

    #     # Take a number of samples from the latent reps to compute more accurate 
    #     # bounds on the objectives

    #     samples = 10

    #     for i in range(samples):
    #         target_latent_sample = q_target.sample().unsqueeze(dim=1).expand(-1, num_target, -1)
    #         context_latent_sample = q_context.sample().unsqueeze(dim=1).expand(-1, num_target, -1)

    #         if self.use_deterministic_path:
    #             deterministic_representation = self.xyx_to_r(x_context, y_context, x_target)
    #             target_rep = torch.cat((deterministic_representation, target_latent_sample), dim=-1)
    #             context_rep = torch.cat((deterministic_representation, context_latent_sample), dim=-1)
    #         else:
    #             target_rep = target_latent_sample
    #             context_rep = context_latent_sample

            

    #     return y_target_mu, y_target_sigma, log_pred, kl_target_context, loss

