import sys

import torch
import torch.optim as optim
import matplotlib.pyplot as plt

from src.conv_cnp import ConvCNP
from src.datagen.gpcurve import *
from src.utils import plot_functions

torch.manual_seed(0)

MAX_CONTEXT_POINTS = 50 #@param {type:"number"}
random_kernel_parameters=True #@param {type:"boolean"}

# Train dataset
dataset_train = RBFGPCurvesReader(
    batch_size=16, max_num_context=MAX_CONTEXT_POINTS, random_kernel_parameters=random_kernel_parameters)
data_train = dataset_train.generate_curves()

# Test dataset
dataset_test = RBFGPCurvesReader(
    batch_size=1, max_num_context=MAX_CONTEXT_POINTS, testing=True, random_kernel_parameters=random_kernel_parameters)
data_test = dataset_test.generate_curves()


conv_cnp = ConvCNP()

optimizer = optim.Adam(conv_cnp.parameters(), lr=0.001)

EPOCHS = 50000

for epoch in range(EPOCHS):
    epoch_loss = 0
    # Train dataset
    dataset_train = RBFGPCurvesReader(
    batch_size=16, max_num_context=MAX_CONTEXT_POINTS, random_kernel_parameters=random_kernel_parameters)
    data_train = dataset_train.generate_curves()
    x_context = data_train.query[0][0].contiguous().transpose(1,2)
    y_context = data_train.query[0][1].contiguous().transpose(1,2)
    x_target = data_train.query[1].contiguous().transpose(1,2)
    y_target = data_train.target_y.contiguous().transpose(1,2)
    optimizer.zero_grad()
    
    y_target_mu, y_target_sigma, loss = conv_cnp.forward(x_context, y_context, x_target, y_target)
    loss.sum().backward()
    optimizer.step()
    
    if (epoch % 500) == 0:
        data_test = dataset_test.generate_curves()
        x_context = data_test.query[0][0].contiguous().transpose(1,2)
        y_context = data_test.query[0][1].contiguous().transpose(1,2)
        x_target = data_test.query[1].contiguous().transpose(1,2)
        y_target = data_test.target_y.contiguous().transpose(1,2)
        y_target_mu, y_target_sigma, _ = conv_cnp.forward(x_context, y_context, x_target, y_target)

        plot_functions(
            x_target.transpose(1,2),
            y_target.transpose(1,2),
            x_context.transpose(1,2),
            y_context.transpose(1,2), 
            y_target_mu.transpose(1,2), 
            y_target_sigma.transpose(1,2),
            save=True,
            experiment_name='conv_cnp_1d_test',
            iter=epoch
        )
        print(f'epoch: {epoch}, loss: {loss.sum()}, x_kernal_length: {conv_cnp.kernal_x.length_scale:0.4f}, rho_kernal_length: {conv_cnp.kernal_rho.length_scale:0.4f}')

