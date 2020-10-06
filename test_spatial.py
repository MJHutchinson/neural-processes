from src.decoder import HeteroskedasticDecoder
from src.encoder import LatentEncoder, DeterministicEncoder
from src.aggregator import Attention
from src.neural_process import AttentiveNeuralProcess
from src.datagen.spatial import SpatialDataset

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# argparse this
MAX_NUM_CONTEXT = 100
MAX_NUM_EXTRA_TARGET = 20
BATCH_SIZE = 100
EPOCHS = 500000
PLOT_AFTER = 10000

# CSV names saved wrong way around, need fixing
xtest = torch.Tensor(pd.read_csv("data/Xtrain.csv").values)
ytest = torch.Tensor(pd.read_csv("data/Ytrain.csv").values)
xtrain = torch.Tensor(pd.read_csv("data/Xtest.csv").values)
ytrain = torch.Tensor(pd.read_csv("data/Ytest.csv").values)
trainDataset = SpatialDataset(
    xtrain, ytrain, BATCH_SIZE, MAX_NUM_CONTEXT, MAX_NUM_EXTRA_TARGET
)
testDataset = SpatialDataset(
    xtest, ytest, BATCH_SIZE, MAX_NUM_CONTEXT, MAX_NUM_EXTRA_TARGET
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# define model
# Sizes of the layers of the MLPs for the encoders and decoder
# The final output layer of the decoder outputs two values, one for the mean and
# one for the variance of the prediction at the target location
x_dim = 2
y_dim = 1
r_dim = 64
z_dim = 64
hid_dim = 128
num_hid = 3
detencoder = DeterministicEncoder(x_dim, y_dim, r_dim, hid_dim=hid_dim, num_hid=num_hid)
latencoder = LatentEncoder(x_dim, y_dim, z_dim, hid_dim=hid_dim, num_hid=num_hid)
decoder = HeteroskedasticDecoder(
    x_dim, r_dim + z_dim, y_dim, hid_dim=hid_dim, num_hid=2
)
att = Attention("identity", "uniform", x_dim, r_dim, hid_dim=hid_dim, num_hid=num_hid)
model = AttentiveNeuralProcess(detencoder, att, latencoder, decoder, True)
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    model = nn.DataParallel(model)
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in tqdm(range(EPOCHS)):
    epoch_loss = 0
    # Train dataset
    x_context, y_context, x_target, y_target = trainDataset.generate()
    x_context = x_context.to(device)
    y_context = y_context.to(device)
    x_target = x_target.to(device)
    y_target = y_target.to(device)

    # compute loss and backpropagate
    model.train()
    optimizer.zero_grad()
    y_pred_target_mu, y_pred_target_sigma, loss, _, _ = model.forward(
        x_context, y_context, x_target, y_target
    )
    loss = loss.sum()
    loss.backward()
    optimizer.step()

    if epoch % PLOT_AFTER == 0:
        torch.save(model.state_dict(), f"results/spatial_2dGP/spatial_{epoch}.pt")
        model.eval()
        with torch.no_grad():
            y_pred, y_target_sigma, _, _, _ = model.forward(
                xtest.unsqueeze(0).to(device),
                ytest.unsqueeze(0).to(device),
                xtest.unsqueeze(0).to(device),
            )
            os.system(
                f"echo Iter: {epoch}, loss: {loss.sum()} Test {epoch} MSE Loss: {(y_pred.cpu().view(-1) - ytest.view(-1)).pow(2).mean()} >> results/spatial_2dGP/spatial_loss"
            )
            # print(
            #     f"Test {epoch} MSE Loss: {(y_pred.cpu().view(-1) - ytest.view(-1)).pow(2).mean()}"
            # )
        plt.figure()
        plt.plot(ytest.view(-1).data, ytest.view(-1).data, linewidth=2)
        plt.scatter(y_pred.cpu().view(-1).data, ytest.view(-1).data)
        plt.xlabel("Prediction")
        plt.ylabel("Label")
        plt.grid("off")
        plt.gca()
        name = f"results/spatial_2dGP/spatial_{epoch}.png"
        plt.savefig(name)
        name = f"results/spatial_2dGP/spatial_{epoch}.pdf"
        plt.savefig(name)
        plt.close()
        plt.show()