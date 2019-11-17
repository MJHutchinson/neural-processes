from src.conv_cnp import *

from stheno.torch import EQ

batch_size = 10
x_dim = 1
y_dim = 1
r_dim = x_dim + 1
grid_size = 100

batches = 100
num_target = 10
num_context = 20

X_context = torch.Tensor(batches, x_dim, num_context).normal_()
Y_context = torch.Tensor(batches, y_dim, num_context).normal_()
X_target = torch.Tensor(batches, x_dim, num_target).normal_()
Y_target = torch.Tensor(batches, y_dim, num_target).normal_()


simple_cnn = SimpleCNN(x_dim + r_dim, 1)

t_h = torch.Tensor(batch_size, x_dim + r_dim, grid_size)
y = simple_cnn(t_h)

print(f'SimpleCNN tst: output shape {y[0].shape}, should be {(batch_size, y_dim, grid_size)}')

convcnp = ConvCNP()

rep = convcnp(X_context, Y_context, X_target, Y_target)