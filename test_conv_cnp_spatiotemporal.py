import sys

import torch
import torch.optim as optim
import matplotlib.pyplot as plt

from src.conv_cnp import ConvCNP
from src.datagen.product_gpcurve import *
from src.utils import plot_function

torch.manual_seed(0)

MAX_CONTEXT_POINTS = 50 #@param {type:"number"}
random_kernel_parameters=False #@param {type:"boolean"}

# Train dataset
datagen = ProductRBFCurvesReader(16, MAX_CONTEXT_POINTS, random_kernel_parameters=random_kernel_parameters)
# Test dataset
datagen_test = ProductRBFCurvesReader(batch_size=1, max_num_context=MAX_CONTEXT_POINTS, testing=True, random_kernel_parameters=random_kernel_parameters)


conv_cnp = ConvCNP(2, 1).cuda()

optimizer = optim.Adam(conv_cnp.parameters(), lr=0.001)

EPOCHS = 50000

for epoch in range(EPOCHS):
    epoch_loss = 0
    # Train dataset
    data_train, _ = datagen.generate_curves()
    x_context = data_train.query[0][0].contiguous().cuda()
    y_context = data_train.query[0][1].contiguous().cuda()
    x_target = data_train.query[1].contiguous().cuda()
    y_target = data_train.target_y.contiguous().cuda()
    optimizer.zero_grad()
    
    y_pred_target_mu, y_pred_target_sigma, loss, _, _ = conv_cnp.forward(x_context, y_context, x_target, y_target)
    loss.sum().backward()
    optimizer.step()
    
    if (epoch % 500) == 0:
        print(f'epoch: {epoch}, loss: {loss.sum()}, x_kernel_length: {conv_cnp.kernel_x.length_scale:0.4f}, rho_kernel_length: {conv_cnp.kernel_rho.length_scale:0.4f}')
        data_test, target_sigma = datagen_test.generate_curves()
        x_context = data_test.query[0][0].contiguous().cuda()
        y_context = data_test.query[0][1].contiguous().cuda()
        x_target = data_test.query[1].contiguous().cuda()
        y_target = data_test.target_y.contiguous().cuda()
        y_pred_target_mu, y_pred_target_sigma, loss, _, _ = conv_cnp.forward(x_context, y_context, x_target, y_target)

        x_target =  x_target.cpu()
        target_sigma = target_sigma.cpu()
        y_pred_target_mu = y_pred_target_mu.cpu()
        y_pred_target_sigma = y_pred_target_sigma.cpu()

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
            c=y_pred_target_mu[0].squeeze(-1).data,
            marker="s",
        )
        plt.colorbar()
        plt.savefig("results/spatial_convnp/pred_{}.png".format(epoch))
        
        plt.figure()
        plt.scatter(
            x_target[0, :, 0].squeeze(-1),
            x_target[0, :, 1].squeeze(-1),
            c=y_pred_target_sigma[0].squeeze(-1).data,
            marker="s",
        )
        plt.colorbar()
        plt.savefig("results/spatial_convnp/pred_sigma_{}.png".format(epoch))
        
