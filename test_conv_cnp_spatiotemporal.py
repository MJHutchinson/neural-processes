import sys

import torch
import torch.optim as optim
import matplotlib.pyplot as plt

from src.conv_cnp import ConvCNP_multidim
from src.datagen.gpcurve import *
from src.utils import plot_functions

torch.manual_seed(0)

MAX_CONTEXT_POINTS = 50 #@param {type:"number"}
random_kernel_parameters=False #@param {type:"boolean"}

# Train dataset
datagen = ProductRBFCurvesReader(16, MAX_CONTEXT_POINTS, random_kernel_parameters=random_kernel_parameters)
# Test dataset
datagen_test = ProductRBFCurvesReader(batch_size=1, max_num_context=MAX_CONTEXT_POINTS, testing=True, random_kernel_parameters=random_kernel_parameters)


conv_cnp = ConvCNP_multidim(x_dim=2, y_dim=1)

optimizer = optim.Adam(conv_cnp.parameters(), lr=0.001)

EPOCHS = 50000

for epoch in range(EPOCHS):
    epoch_loss = 0
    # Train dataset
    data_train, _ = datagen.generate_curves()
    x_context = data_train.query[0][0].contiguous().transpose(1,2)
    y_context = data_train.query[0][1].contiguous().transpose(1,2)
    x_target = data_train.query[1].contiguous().transpose(1,2)
    y_target = data_train.target_y.contiguous().transpose(1,2)
    optimizer.zero_grad()
    
    y_target_mu, y_target_sigma, loss = conv_cnp.forward(x_context, y_context, x_target, y_target)
    loss.sum().backward()
    optimizer.step()
    
    if (epoch % 500) == 0:
        data_test, target_sigma = datagen_test.generate_curves()
        x_context = data_test.query[0][0].contiguous().transpose(1,2)
        y_context = data_test.query[0][1].contiguous().transpose(1,2)
        x_target = data_test.query[1].contiguous().transpose(1,2)
        y_target = data_test.target_y.contiguous().transpose(1,2)
        y_target_mu, y_target_sigma, _ = conv_cnp.forward(x_context, y_context, x_target, y_target)

        plt.figure()
        plt.scatter(
            x_target[0, :, 0].squeeze(-1),
            x_target[0, :, 1].squeeze(-1),
            c=y_target[0].squeeze(-1).data,
            marker="s",
        )
        plt.colorbar()
        plt.savefig("results/spatial_convnp/context_{}.png".format(epoch))


        plt.figure()
        plt.scatter(
            x_target[0, :, 0].squeeze(-1),
            x_target[0, :, 1].squeeze(-1),
            c=target_sigma[0].squeeze(-1).data,
            marker="s",
        )
        plt.colorbar()
        plt.savefig("results/spatial_convnp/context_sigma_{}.png".format(epoch))


        plt.figure()
        plt.scatter(
            x_target[0, :, 0].squeeze(-1),
            x_target[0, :, 1].squeeze(-1),
            c=y_target_mu[0].squeeze(-1).data,
            marker="s",
        )
        plt.colorbar()
        plt.savefig("results/spatial_convnp/pred_{}.png".format(epoch))
        
        plt.figure()
        plt.scatter(
            x_target[0, :, 0].squeeze(-1),
            x_target[0, :, 1].squeeze(-1),
            c=y_target_sigma[0].squeeze(-1).data,
            marker="s",
        )
        plt.colorbar()
        plt.savefig("results/spatial_convnp/pred_sigma_{}.png".format(epoch))
        
        print(
            f"Iter: {epoch}, loss: {loss.sum()}"
        )
