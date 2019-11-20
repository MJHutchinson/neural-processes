import os
import pickle
import argparse

import torch

from stheno.torch import EQ, RQ, Matern32, Matern52, GP, Delta, NoisyKernel
from stheno.torch import B
B.epsilon = 1e-6 

from src.utils import plot_compare_processes_gp, plot_compare_processes_gp_latent

kernel_noise = 2e-2

torch.manual_seed(0)

parser = argparse.ArgumentParser()

# parser.add_argument('-m', '--model', type=str, required=True)

# parser.add_argument('-g', '--GP-type', type=str, default='RBF')
# parser.add_argument('-r', '--random-kernel', action='store_true')

# parser.add_argument('-e', '--epochs', type=int, default=50_000)
# parser.add_argument('-b', '--batch-size', type=int, default=16)
# parser.add_argument('-c', '--max-context', type=int, default=16)
# parser.add_argument('-lr', '--learning-rate', type=float, default=0.001)
# parser.add_argument('--test-epoch', type=int, default=500)

# parser.add_argument('--name-suffix', type=str, default=None)

parser.add_argument('-m', '--model', type=str, default='ConvCNP')

parser.add_argument('-g', '--GP-type', type=str, default='RBF')
parser.add_argument('-r', '--random-kernel', action='store_true')

parser.add_argument('-e', '--epochs', type=int, default=50000)
parser.add_argument('-b', '--batch-size', type=int, default=16)
parser.add_argument('--max-context', type=int, default=50)
parser.add_argument('--min-context', type=int, default=3)
parser.add_argument('--max-points', type=int, default=100)
parser.add_argument('-lr', '--learning-rate', type=float, default=0.001)
parser.add_argument('--test-epoch', type=int, default=500)

parser.add_argument('--name-suffix', type=str, default='test')

args = parser.parse_args()

print(args)

results_dir = os.path.join('results', 'GP_' + args.GP_type, args.model )
if args.name_suffix is not None:
    results_dir = results_dir + '_' + args.name_suffix

os.makedirs(results_dir, exist_ok=True)

if args.model == 'CNP':
    from src.decoder import HeteroskedasticDecoder
    from src.encoder import DeterministicEncoder
    from src.aggregator import Attention
    from src.conditional_neural_process import ConditionalNeuralProcess
    x_dim = 1
    y_dim = 1
    r_dim = 64
    z_dim = 64
    hid_dim = 128
    num_hid = 3
    detencoder = DeterministicEncoder(x_dim, y_dim, r_dim, hid_dim=hid_dim, num_hid=num_hid)
    decoder = HeteroskedasticDecoder(x_dim, r_dim, y_dim, hid_dim=hid_dim, num_hid=2)
    att = Attention('identity','uniform', x_dim, r_dim, hid_dim=hid_dim, num_hid=num_hid)
    model = ConditionalNeuralProcess(detencoder, att, decoder)
elif args.model == 'ANP_CNP':
    from src.decoder import HeteroskedasticDecoder
    from src.encoder import DeterministicEncoder
    from src.aggregator import Attention
    from src.conditional_neural_process import ConditionalNeuralProcess
    x_dim = 1
    y_dim = 1
    r_dim = 64
    z_dim = 64
    hid_dim = 128
    num_hid = 3
    detencoder = DeterministicEncoder(x_dim, y_dim, r_dim, hid_dim=hid_dim, num_hid=num_hid)
    decoder = HeteroskedasticDecoder(x_dim, r_dim, y_dim, hid_dim=hid_dim, num_hid=2)
    att = Attention('mlp','multihead', x_dim, r_dim, hid_dim=hid_dim, num_hid=num_hid)
    model = ConditionalNeuralProcess(detencoder, att, decoder)
elif args.model == 'ANP':
    from src.decoder import HeteroskedasticDecoder
    from src.encoder import LatentEncoder, DeterministicEncoder
    from src.aggregator import Attention
    from src.neural_process import AttentiveNeuralProcess
    x_dim = 1
    y_dim = 1
    r_dim = 64
    z_dim = 64
    hid_dim = 128
    num_hid = 3
    detencoder = DeterministicEncoder(x_dim, y_dim, r_dim, hid_dim=hid_dim, num_hid=num_hid)
    latencoder = LatentEncoder(x_dim, y_dim, z_dim, hid_dim=hid_dim, num_hid=num_hid)
    decoder = HeteroskedasticDecoder(x_dim, r_dim + z_dim, y_dim, hid_dim=hid_dim, num_hid=2)
    att = Attention('mlp','multihead', x_dim, r_dim, hid_dim=hid_dim, num_hid=num_hid)
    model = AttentiveNeuralProcess(detencoder, att, latencoder, decoder, True)
elif args.model == 'NP':
    from src.decoder import HeteroskedasticDecoder
    from src.encoder import LatentEncoder, DeterministicEncoder
    from src.aggregator import Attention
    from src.neural_process import AttentiveNeuralProcess
    x_dim = 1
    y_dim = 1
    r_dim = 64
    z_dim = 64
    hid_dim = 128
    num_hid = 3
    detencoder = DeterministicEncoder(x_dim, y_dim, r_dim, hid_dim=hid_dim, num_hid=num_hid)
    latencoder = LatentEncoder(x_dim, y_dim, z_dim, hid_dim=hid_dim, num_hid=num_hid)
    decoder = HeteroskedasticDecoder(x_dim, r_dim + z_dim, y_dim, hid_dim=hid_dim, num_hid=2)
    att = Attention('identity','uniform', x_dim, r_dim, hid_dim=hid_dim, num_hid=num_hid)
    model = AttentiveNeuralProcess(detencoder, att, latencoder, decoder, True)
elif args.model == 'ConvCNP':
    from src.conv_cnp import ConvCNP
    model = ConvCNP(cnn='simple')
elif args.model == 'ConvCNPXL':
    from src.conv_cnp import ConvCNP
    model = ConvCNP(cnn='xl')
else:
    raise ValueError('Unrecognised model type')


if args.GP_type == 'RBF':
    from src.datagen.gpcurve import RBFGPCurvesReader
    datagen_train = RBFGPCurvesReader(
        batch_size=args.batch_size, 
        max_num_context=args.max_context,
        min_num_context=args.min_context,
        max_points=args.max_points,
        random_kernel_parameters=args.random_kernel
    )
    datagen_test = RBFGPCurvesReader(
        batch_size=1, 
        max_num_context=args.max_context,
        min_num_context=args.min_context,
        max_points=args.max_points,
        random_kernel_parameters=args.random_kernel,
        testing=True
    ) 
    datagen_validate = RBFGPCurvesReader(
        batch_size=1000, 
        max_num_context=args.max_context,
        min_num_context=args.min_context,
        max_points=args.max_points, 
        random_kernel_parameters=args.random_kernel,
    ) 
elif args.GP_type == 'Matern':
    from src.datagen.gpcurve import MaternGPCurvesReader
    datagen_train = MaternGPCurvesReader(
        batch_size=args.batch_size, 
        max_num_context=args.max_context,
        min_num_context=args.min_context,
        max_points=args.max_points,
        random_kernel_parameters=args.random_kernel,
        nu=2.5
    )
    datagen_test = MaternGPCurvesReader(
        batch_size=1, 
        max_num_context=args.max_context,
        min_num_context=args.min_context,
        max_points=args.max_points,
        random_kernel_parameters=args.random_kernel,
        nu=2.5,
        testing=True
    ) 
    datagen_validate = MaternGPCurvesReader(
        batch_size=1000, 
        max_num_context=args.max_context,
        min_num_context=args.min_context,
        max_points=args.max_points,
        random_kernel_parameters=args.random_kernel,
        nu=2.5,
    )    
else:
    raise ValueError('Unrecognised GP type')


optimiser = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

validation_losses = []
validation_epochs = []

data_valid, params = datagen_validate.generate_curves()
y_context_valid = data_valid.query[0][1].contiguous()
x_context_valid = data_valid.query[0][0].contiguous()
x_target_valid = data_valid.query[1].contiguous()
y_target_valid = data_valid.target_y.contiguous()

for epoch in range(args.epochs+1):
    data_train, _ = datagen_train.generate_curves()
    x_context = data_train.query[0][0].contiguous()
    x_context, x_context_sorted_indices = x_context.sort(1)
    y_context = data_train.query[0][1].contiguous()
    y_context = torch.gather(y_context, 1, x_context_sorted_indices)
    x_target = data_train.query[1].contiguous()
    x_target, x_target_sorted_indices = x_target.sort(1)
    y_target = data_train.target_y.contiguous()
    y_target = torch.gather(y_target, 1, x_target_sorted_indices)

    optimiser.zero_grad()

    if args.model in ['ANP', 'NP']:
        y_target_mu, y_target_sigma, loss, log_pred, kl_target_context  = model.forward(
            x_context, y_context, x_target, y_target
        )
    else:
        y_target_mu, y_target_sigma, loss, _, _ = model.forward(x_context, y_context, x_target, y_target)
        loss = loss.sum()
    loss.backward()
    optimiser.step()

    if epoch % args.test_epoch == 0:
        # plot some samples
        # refix the seed
        random_state = torch.get_rng_state()
        for i in range(10):
            torch.manual_seed(i)
            # Sample a single GP from the distribution
            data_test, params = datagen_test.generate_curves()
            y_context = data_test.query[0][1].contiguous()[0].unsqueeze(0)
            x_context = data_test.query[0][0].contiguous()[0].unsqueeze(0)
            x_target = data_test.query[1].contiguous()[0].unsqueeze(0)
            y_target = data_test.target_y.contiguous()[0].unsqueeze(0)

            # If our model has a latent part, we want to plot many 
            # samples from the function
            if args.model in ['ANP', 'NP']:
                y_target_mu = []
                y_target_sigma = []
                for j in range(10):
                    y_target_mu_i, y_target_sigma_i, _, _, _ = model.forward(
                        x_context, y_context, x_target, y_target
                    )
                    y_target_mu.append(y_target_mu_i)
                    y_target_sigma.append(y_target_sigma_i)

                y_target_mu = torch.cat(tuple(y_target_mu), dim=0)
                y_target_sigma = torch.cat(tuple(y_target_sigma), dim=0)
            # If the function is deterministic, we only want to plot one.
            else:
                y_target_mu, y_target_sigma, _, _, _ = model.forward(x_context, y_context, x_target, y_target)
                y_target_mu = y_target_mu
                y_target_sigma = y_target_sigma

            # Select only the first element of 
            y_context = y_context[0].data.squeeze()
            x_context = x_context[0].data.squeeze()
            x_target = x_target[0].data.squeeze()
            y_target = y_target[0].data.squeeze() 

            name = f'epoch_{epoch}_sample_{i}'

            # Fit the relevent GP to the data
            if args.GP_type == 'RBF':
                kernel = EQ().stretch(params[0].squeeze()) * (params[1].squeeze() ** 2)
            elif args.GP_type == 'Matern':
                kernel = Matern52().stretch(params[0].squeeze()) * (params[1].squeeze() ** 2)

            f = GP(kernel)
            e = GP(Delta()) * kernel_noise
            gp = f + e | (x_context, y_context)
            preds = gp(x_target)
            gp_mean , gp_lower, gp_upper = preds.marginals()
            gp_std = (gp_upper - gp_mean) / 2
            
            if args.model in ['ANP', 'NP']:
                plot_compare_processes_gp_latent(
                    x_target,
                    y_target,
                    x_context,
                    y_context,
                    y_target_mu.data.squeeze(),
                    y_target_sigma.data.squeeze(),
                    gp_mean,
                    gp_std,
                    save=True,
                    dir=results_dir,
                    name=name
                )
            else:
                y_target_mu = y_target_mu[0].data.squeeze()
                y_target_sigma = y_target_sigma[0].data.squeeze()
                plot_compare_processes_gp(
                    x_target,
                    y_target,
                    x_context,
                    y_context,
                    y_target_mu,
                    y_target_sigma,
                    gp_mean,
                    gp_std,
                    save=True,
                    dir=results_dir,
                    name=name
                )

        # refix the seed for the validation set to keep it constant
        validation_loss = 0       

        # If we have a model with a latent variable, sample a bunch of times to compute 
        # an accurate validation loss. Esle we only need one pass
        if args.model in ['ANP', 'NP']:
            validation_samples = 10
            for i in range(validation_samples):
                _, _, loss, _, _ = model.forward(
                    x_context_valid, y_context_valid, x_target_valid, y_target_valid
                )
                validation_loss = validation_loss + loss/validation_samples
        else:
            _, _, loss, _, _ = model.forward(
                    x_context_valid, y_context_valid, x_target_valid, y_target_valid
            )
            validation_loss = loss
        
        validation_epochs.append(epoch)
        validation_losses.append(float(validation_loss))

        # print(
        #     f"Iter: {epoch}, loss: {loss}, x_kernel: {model.kernel_x.length_scale}, rho_kernel: {model.kernel_rho.length_scale}"
        # )

        print(
            f"Iter: {epoch}, loss: {validation_loss}"
        )

        # put back the random state
        torch.set_rng_state(random_state)

output = {
    'args' : args,
    'epochs' : validation_epochs,
    'validation_losses' : validation_losses
}

with open(os.path.join(results_dir, 'results.pkl'), 'wb') as h:
    pickle.dump(output, h)

torch.save(model.state_dict(), os.path.join(results_dir, 'model.pt'))