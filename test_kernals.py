import torch

from src.datagen.kernels import rbf_kernel, matern_kernel

from stheno.torch import EQ, RQ, Matern32, Matern52, GP, Delta, NoisyKernel

import matplotlib.pyplot as plt

x = torch.linspace(-4,4,101).unsqueeze(-1)

l1 = torch.tensor(0.3).unsqueeze(0)
sigma = torch.tensor(2.).unsqueeze(0)
noise = 2e-2


y_rbf = matern_kernel(x.unsqueeze(0), 2.5, l1.unsqueeze(0).unsqueeze(0), sigma.unsqueeze(0), noise)
y_rbf = y_rbf.squeeze()[50]

y_stheno = (Matern52().stretch(l1) * sigma**2)(x, 0).mat.squeeze()

plt.plot(x, y_rbf, label='generator')
plt.plot(x, y_stheno, label='stheno')
plt.legend()
plt.show()