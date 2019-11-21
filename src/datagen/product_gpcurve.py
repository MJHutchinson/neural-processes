import collections
from torch import eye, randint, int32, arange, Tensor, float32, sqrt
import torch
from abc import abstractmethod
from src.utils import NPRegressionDescription
from src.datagen.kernels import rbf_kernel, matern_kernel, generate_lengthscale_sigma_f
import numpy as np

class ProductRBFCurvesReader(object):
    """Generates curves using a Gaussian Process (GP).

    Supports vector inputs (x) and vector outputs (y). Kernel is
    Materns 1/2, 3/2 and 5/2, using the x-value l2 coordinate distance scaled by
    some factor chosen randomly in a range. Outputs are independent gaussian
    processes.
    """

    def __init__(
        self,
        batch_size,
        max_num_context,
        x_size=2,
        y_size=1,
        l1_scale_space=0.6,
        l1_scale_time=0.6,
        sigma_scale_space=1.0,
        sigma_scale_time=1.0,
        random_kernel_parameters=True,
        testing=False,
    ):
        """Creates a regression dataset of functions sampled from a GP.

        Args:
            batch_size: An integer.
            max_num_context: The max number of observations in the context.
            x_size: Integer >= 1 for length of "x values" vector.
            y_size: Integer >= 1 for length of "y values" vector.
            l1_scale: Float; typical scale for kernel distance function.
            sigma_scale: Float; typical scale for variance.
            random_kernel_parameters: If `True`, the kernel parameters (l1 and sigma) 
                will be sampled uniformly within [0.1, l1_scale] and [0.1, sigma_scale].
            testing: Boolean that indicates whether we are testing. If so there are
                more targets for visualization.
        """
        self._batch_size = batch_size
        self._max_num_context = max_num_context
        self._x_size=x_size
        self._y_size=y_size
        self._random_kernel_parameters=random_kernel_parameters
        self._testing=testing
        self._l1_scale_space = l1_scale_space
        self._l1_scale_time = l1_scale_time
        self._sigma_scale_space = sigma_scale_space
        self._sigma_scale_time = sigma_scale_time

    def _generate_kernel_params(self):
        # Set kernel parameters
        # Either choose a set of random parameters for the mini-batch
        l1_space, sigma_f_space = generate_lengthscale_sigma_f(
            self._random_kernel_parameters,
            self._batch_size,
            self._y_size,
            self._x_size,
            self._l1_scale_space,
            self._sigma_scale_space,
        )

        l1_time, sigma_f_time = generate_lengthscale_sigma_f(
            self._random_kernel_parameters,
            self._batch_size,
            self._y_size,
            self._x_size,
            self._l1_scale_time,
            self._sigma_scale_time,
        )
        return l1_space, sigma_f_space, l1_time, sigma_f_time

    def _kernel(
        self,
        space_values,
        time_values,
        l1_space,
        sigma_f_space,
        l1_time,
        sigma_f_time,
        sigma_noise=2e-2,
    ):
        """Applies the product RBF kernel to generate curve data.

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
        kernel_space = rbf_kernel(
            space_values, l1_space, sigma_f_space, sigma_noise=2e-2
        )
        kernel_time = rbf_kernel(time_values, l1_time, sigma_f_time, sigma_noise=2e-2)

        return kernel_space, kernel_time

    def generate_curves(self):
        """Builds the op delivering the data.

        Generated functions are `float32` with x values between -2 and 2.

        Returns:
            A `CNPRegressionDescription` namedtuple.
        """
        num_context = randint(low=3, high=self._max_num_context, size=(1,), dtype=int32)

        # If we are testing we want to have more targets and have them evenly
        # distributed in order to plot the function.
        if self._testing:
            num_target = 50
            num_total_points = num_target
            space_values = (
                torch.linspace(-2, 2, int(num_total_points))
                .unsqueeze(0)
                .repeat_interleave(repeats=self._batch_size, axis=0)
            ).unsqueeze(-1)
            time_values = (
                torch.linspace(-2, 2, int(num_total_points))
                .unsqueeze(0)
                .repeat_interleave(repeats=self._batch_size, axis=0)
            ).unsqueeze(-1)
        # During training the number of target points and their x-positions are
        # selected at random
        else:
            num_target = randint(
                low=2,
                high=int(self._max_num_context - num_context),
                size=(1,),
                dtype=int32
            )
            num_total_points = num_context + num_target
            space_values = (
                torch.linspace(-2, 2, int(num_total_points))
                .unsqueeze(0)
                .repeat_interleave(repeats=self._batch_size, axis=0)
            ).unsqueeze(-1)
            time_values = (
                torch.linspace(-2, 2, int(num_total_points))
                .unsqueeze(0)
                .repeat_interleave(repeats=self._batch_size, axis=0)
            ).unsqueeze(-1)
        # Set kernel parameters
        # Either choose a set of random parameters for the mini-batch
        params = self._generate_kernel_params()

        # Pass the x_values through the kernel
        # [batch_size, y_size, num_total_points, num_total_points]
        kernel_space, kernel_time = self._kernel(
            space_values, time_values, *params, sigma_noise=2e-2
        )

        # Calculate Cholesky, using double precision for better stability:
        cholesky_space = torch.cholesky(kernel_space.double()).float()
        cholesky_time = torch.cholesky(kernel_time.double()).float()
        kron_cholesky = torch.zeros(
            self._batch_size, self._y_size, num_total_points ** 2, num_total_points ** 2
        )

        for i in range(self._batch_size):
            kron_cholesky[i][0] = Tensor(
                np.kron(
                    cholesky_time.squeeze(1).numpy()[i, :, :],
                    cholesky_space.squeeze(1).numpy()[i, :, :],
                )
            )
        # Sample a curve
        # [batch_size, y_size, num_total_points, 1]
        y_values = (
            kron_cholesky
            @ Tensor(self._batch_size, self._y_size, num_total_points ** 2, 1).normal_()
        )

        # [batch_size, num_total_points, y_size]
        y_values = y_values.squeeze(1)

        x_values = torch.zeros(self._batch_size, num_total_points ** 2, self._x_size)
        index = 0
        for i in range(num_total_points):
            x_values[:, index : index + num_total_points, 1] = space_values[
                :, i, 0
            ].unsqueeze(1)
            x_values[:, index : index + num_total_points, 0] = time_values[:, :, 0]
            index += num_total_points

        if self._testing:
            # Select the targets
            target_x = x_values
            target_y = y_values
            target_sigma = kron_cholesky.squeeze(1).diagonal(dim1=1, dim2=2)

            # Select the observations
            idx = arange(num_target**2)[torch.randperm(num_target**2)]
            context_x = target_x[:, idx[: num_context ** 2], :]
            context_y = y_values[:, idx[: num_context ** 2], :]

        else:
            # Select the targets which will consist of the context points as well as
            # some new target points
            target_x = x_values[:, : (num_target + num_context) ** 2, :]
            target_y = y_values[:, : (num_target + num_context) ** 2, :]
            target_sigma = kron_cholesky.squeeze(1)[:,: (num_target + num_context) ** 2,: (num_target + num_context) ** 2].diagonal(dim1=1, dim2=2)

            # Select the observations
            context_x = x_values[:, : num_context ** 2, :]
            context_y = y_values[:, : num_context ** 2, :]

        query = ((context_x, context_y), target_x)

        return NPRegressionDescription(
            query=query,
            target_y=target_y,
            num_total_points=target_x.shape[1],
            num_context_points=num_context,
        ), target_sigma