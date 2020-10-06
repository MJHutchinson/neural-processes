import torch
from torch.utils import data
from torch import eye, randint, int32, arange, Tensor, float32, sqrt
import numpy as np
from src.datagen.kernels import rbf_kernel, matern_kernel, generate_lengthscale_sigma_f


class SpatialDataset:
    """Dataset to load the spatial dataset


    References: https://github.com/EmilienDupont/neural-processes/blob/master/utils.py
    """

    def __init__(
        self,
        X,
        Y,
        batch_size,
        max_num_context,
        max_num_extra_target,
        x_size=2,
        y_size=1,
        l1_size=1,
        l1_scale=1,
        nu=1.5,
        sigma_scale=1.0,
        random_kernel_parameters=True,
    ):
        """
        Args:
        """
        self.X = X
        self.Y = Y
        self._batch_size = batch_size
        self._max_num_context = max_num_context
        self._max_num_extra_target = max_num_extra_target
        self._x_size = x_size
        self._y_size = y_size
        self._random_kernel_parameters = random_kernel_parameters
        self._l1_size = l1_size
        self._l1_scale = l1_scale
        self._sigma_scale = sigma_scale
        self.nu = nu

    def __len__(self):
        return self.X.shape[0]

    def _generate_kernel_params(self):
        # Set kernel parameters
        # Either choose a set of random parameters for the mini-batch
        l1, sigma_f = generate_lengthscale_sigma_f(
            self._random_kernel_parameters,
            self._batch_size,
            self._y_size,
            self._l1_size,
            self._l1_scale,
            self._sigma_scale,
        )
        return l1, sigma_f

    def _kernel(self, xdata, l1, sigma_f, sigma_noise=2e-2):
        """Applies the Matern kernel to generate curve data.

        Args:
            xdata: Tensor of shape [B, num_total_points, x_size] with
                the values of the x-axis data.
            l1: Tensor of shape [B, y_size, x_size], the scale
                parameter of the Gaussian kernel.
            sigma_f: Tensor of shape [B, y_size], the magnitude
                of the std.
            sigma_noise: Float, std of the noise that we add for stability.

        Returns:
            The kernel, a float tensor of shape
            [B, y_size, num_total_points, num_total_points].
        """
        kernel = matern_kernel(xdata, self.nu, l1, sigma_f, sigma_noise=2e-2)

        return kernel

    def generate(self):
        """"""

        num_context = np.random.randint(low=5, high=self._max_num_context)
        num_extra_target = np.random.randint(self._max_num_extra_target)
        num_target = num_context + num_extra_target

        # shuffle index
        locations = randint(len(self), size=(self._batch_size * num_target,))
        x_target = self.X[locations, :].reshape(self._batch_size, num_target, self._x_size)
        y_target = self.Y[locations, :].reshape(self._batch_size, num_target, self._y_size)
        
        # now generate the context set labels based on a 2D GP
        params = self._generate_kernel_params()
        x_context = x_target[:, :num_context, :]
        K = self._kernel(x_context, *params, sigma_noise=2e-2)
        cholesky = torch.cholesky(K.double()).float()
        # Sample a curve
        # [batch_size, y_size, num_total_points, 1]
        y_context = (
            cholesky
            @ Tensor(self._batch_size, self._y_size, num_context, 1).normal_()
        )

        # [batch_size, num_total_points, y_size]
        y_context = y_context.squeeze(3).transpose(1, 2)
        y_target[:, :num_context, :] = y_context

        return x_context, y_context, x_target, y_target
