import torch
import matplotlib.pyplot as plt

from stheno.torch import EQ

from src.utils import kernal_interpolate

# x = torch.linspace(1, 10, 10).unsqueeze(0).unsqueeze(-1)
# y = torch.linspace(1, 10, 10).unsqueeze(0).unsqueeze(-1)

x = torch.tensor([1.,2.,3.,6.,10.]).unsqueeze(0).unsqueeze(-1)
y = torch.tensor([5.,6.,4.,3.,7.]).unsqueeze(0).unsqueeze(-1)

y = torch.cat(
            (
                y,
                torch.ones_like(y)
            ),
            dim=2
        )

kernal = EQ() > .5

x_grid = torch.linspace(0, 11, 1000).unsqueeze(0).unsqueeze(-1)

y_grid = kernal_interpolate(y, x, x_grid, kernal)

print(y.squeeze())
print(y.squeeze().shape)

plt.scatter(x.squeeze(), y.squeeze()[:, 0])
plt.plot(x_grid.squeeze(), y_grid.squeeze())
plt.plot(x_grid.squeeze(), y_grid.squeeze()[:, 0] / y_grid.squeeze()[:, 1])
plt.show()