import collections
from torch import eye, randint, int32, arange, Tensor, float32, sqrt
import torch
from abc import abstractmethod
from src.utils import NPRegressionDescription
from src.datagen.kernels import rbf_kernel, matern_kernel, generate_lengthscale_sigma_f
import numpy as np

# The (A)NP takes as input a `NPRegressionDescription` namedtuple with fields:
#   `query`: a tuple containing ((context_x, context_y), target_x)
#   `target_y`: a tensor containing the ground truth for the targets to be
#     predicted
#   `num_total_points`: A vector containing a scalar that describes the total
#     number of datapoints used (context + target)
#   `num_context_points`: A vector containing a scalar that describes the number
#     of datapoints used as context
# The GPCurvesReader returns the newly sampled data in this format at each
# iteration


class GPCurvesReader(object):
    """Generates curves using a Gaussian Process (GP).

    Supports vector inputs (x) and vector outputs (y). 
    TODO: different to what Hyunjik Kim had, we will need to repeat the generate_curve
    in our training procedure
    """

    def __init__(
        self,
        batch_size,
        max_num_context,
        x_size=1,
        y_size=1,
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
        self._x_size = x_size
        self._y_size = y_size
        self._random_kernel_parameters = random_kernel_parameters
        self._testing = testing

    @abstractmethod
    def _kernel(self, xdata, params, sigma_noise=2e-2):
        raise NotImplementedError()

    @abstractmethod
    def _generate_kernel_params(self):
        raise NotImplementedError()

    @abstractmethod
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
            num_target = 400
            num_total_points = num_target
            x_values = (
                torch.arange(-2, 2, 1 / 100)
                .unsqueeze(0)
                .repeat_interleave(repeats=16, axis=0)
            )
            x_values = x_values.unsqueeze(-1)
        # During training the number of target points and their x-positions are
        # selected at random
        else:
            num_target = randint(
                low=0,
                high=int(self._max_num_context - num_context),
                size=(1,),
                dtype=int32,
            )
            num_total_points = num_context + num_target
            x_values = Tensor(
                self._batch_size, num_total_points, self._x_size
            ).uniform_(-2, 2)
        # Set kernel parameters
        # Either choose a set of random parameters for the mini-batch
        params = self._generate_kernel_params()

        # Pass the x_values through the kernel
        # [batch_size, y_size, num_total_points, num_total_points]
        kernel = self._kernel(x_values, *params, sigma_noise=2e-2)

        # Calculate Cholesky, using double precision for better stability:
        cholesky = torch.cholesky(kernel.double()).float()

        # Sample a curve
        # [batch_size, y_size, num_total_points, 1]
        y_values = (
            cholesky
            @ Tensor(self._batch_size, self._y_size, num_total_points, 1).normal_()
        )

        # [batch_size, num_total_points, y_size]
        y_values = y_values.squeeze(3).transpose(1, 2)

        if self._testing:
            # Select the targets
            target_x = x_values
            target_y = y_values

            # Select the observations
            idx = arange(num_target)[torch.randperm(num_target)]
            context_x = x_values[:, idx[:num_context], :]
            context_y = y_values[:, idx[:num_context], :]

        else:
            # Select the targets which will consist of the context points as well as
            # some new target points
            target_x = x_values[:, : num_target + num_context, :]
            target_y = y_values[:, : num_target + num_context, :]

            # Select the observations
            context_x = x_values[:, :num_context, :]
            context_y = y_values[:, :num_context, :]

        query = ((context_x, context_y), target_x)

        return NPRegressionDescription(
            query=query,
            target_y=target_y,
            num_total_points=target_x.shape[1],
            num_context_points=num_context,
        )


class RBFGPCurvesReader(GPCurvesReader):
    """Generates curves using a Gaussian Process (GP).

    Supports vector inputs (x) and vector outputs (y). Kernel is
    mean-squared exponential, using the x-value l2 coordinate distance scaled by
    some factor chosen randomly in a range. Outputs are independent gaussian
    processes.
    """

    def __init__(
        self,
        batch_size,
        max_num_context,
        x_size=1,
        y_size=1,
        l1_scale=0.6,
        sigma_scale=1.0,
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
        super(RBFGPCurvesReader, self).__init__(
            batch_size,
            max_num_context,
            x_size=x_size,
            y_size=y_size,
            random_kernel_parameters=random_kernel_parameters,
            testing=testing,
        )
        self._l1_scale = l1_scale
        self._sigma_scale = sigma_scale

    def _generate_kernel_params(self):
        # Set kernel parameters
        # Either choose a set of random parameters for the mini-batch
        l1, sigma_f = generate_lengthscale_sigma_f(
            self._random_kernel_parameters,
            self._batch_size,
            self._y_size,
            self._x_size,
            self._l1_scale,
            self._sigma_scale,
        )
        return l1, sigma_f

    def _kernel(self, xdata, l1, sigma_f, sigma_noise=2e-2):
        """Applies the Gaussian kernel to generate curve data.

        Args:
            xdata: Tensor of shape [B, num_total_points, x_size] with
                the values of the x-axis data.
            l1: Tensor of shape [B, y_size, x_size], the scale
                parameter of the Gaussian kernel.
            sigma_f: Tensor of shape [B, y_size], the magnitude
                of the std.†
            sigma_noise: Float, std of the noise that we add for stability.

        Returns:
            The kernel, a float tensor of shape
            [B, y_size, num_total_points, num_total_points].
        """
        kernel = rbf_kernel(xdata, l1, sigma_f, sigma_noise=2e-2)

        return kernel


class MaternGPCurvesReader(GPCurvesReader):
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
        x_size=1,
        y_size=1,
        l1_scale=0.6,
        nu=1.5,
        sigma_scale=1.0,
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
        super(MaternGPCurvesReader, self).__init__(
            batch_size,
            max_num_context,
            x_size=x_size,
            y_size=y_size,
            random_kernel_parameters=random_kernel_parameters,
            testing=testing,
        )
        self._l1_scale = l1_scale
        self._sigma_scale = sigma_scale
        self.nu = nu

    def _generate_kernel_params(self):
        # Set kernel parameters
        # Either choose a set of random parameters for the mini-batch
        l1, sigma_f = generate_lengthscale_sigma_f(
            self._random_kernel_parameters,
            self._batch_size,
            self._y_size,
            self._x_size,
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


class RBFGPCurvesReader(GPCurvesReader):
    """Generates curves using a Gaussian Process (GP).

    Supports vector inputs (x) and vector outputs (y). Kernel is
    mean-squared exponential, using the x-value l2 coordinate distance scaled by
    some factor chosen randomly in a range. Outputs are independent gaussian
    processes.
    """

    def __init__(
        self,
        batch_size,
        max_num_context,
        x_size=1,
        y_size=1,
        l1_scale=0.6,
        sigma_scale=1.0,
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
        super(RBFGPCurvesReader, self).__init__(
            batch_size,
            max_num_context,
            x_size=x_size,
            y_size=y_size,
            random_kernel_parameters=random_kernel_parameters,
            testing=testing,
        )
        self._l1_scale = l1_scale
        self._sigma_scale = sigma_scale

    def _generate_kernel_params(self):
        # Set kernel parameters
        # Either choose a set of random parameters for the mini-batch
        l1, sigma_f = generate_lengthscale_sigma_f(
            self._random_kernel_parameters,
            self._batch_size,
            self._y_size,
            self._x_size,
            self._l1_scale,
            self._sigma_scale,
        )
        return l1, sigma_f

    def _kernel(self, xdata, l1, sigma_f, sigma_noise=2e-2):
        """Applies the Gaussian kernel to generate curve data.

        Args:
            xdata: Tensor of shape [B, num_total_points, x_size] with
                the values of the x-axis data.
            l1: Tensor of shape [B, y_size, x_size], the scale
                parameter of the Gaussian kernel.
            sigma_f: Tensor of shape [B, y_size], the magnitude
                of the std.†
            sigma_noise: Float, std of the noise that we add for stability.

        Returns:
            The kernel, a float tensor of shape
            [B, y_size, num_total_points, num_total_points].
        """
        kernel = rbf_kernel(xdata, l1, sigma_f, sigma_noise=2e-2)

        return kernel


class ProductRBFCurvesReader(GPCurvesReader):
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
        super(ProductRBFCurvesReader, self).__init__(
            batch_size,
            max_num_context,
            x_size=x_size,
            y_size=y_size,
            random_kernel_parameters=random_kernel_parameters,
            testing=testing,
        )
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
            num_target = 400
            num_total_points = num_target
            space_values = (
                torch.linspace(-2, 2, int(num_total_points))
                .unsqueeze(0)
                .repeat_interleave(repeats=16, axis=0)
            ).unsqueeze(-1)
            time_values = (
                torch.linspace(-2, 2, int(num_total_points))
                .unsqueeze(0)
                .repeat_interleave(repeats=16, axis=0)
            ).unsqueeze(-1)
        # During training the number of target points and their x-positions are
        # selected at random
        else:
            num_target = randint(
                low=0,
                high=int(self._max_num_context - num_context),
                size=(1,),
                dtype=int32,
            )
            num_total_points = num_context + num_target
            space_values = (
                torch.linspace(-2, 2, int(num_total_points))
                .unsqueeze(0)
                .repeat_interleave(repeats=16, axis=0)
            ).unsqueeze(-1)
            time_values = (
                torch.linspace(-2, 2, int(num_total_points))
                .unsqueeze(0)
                .repeat_interleave(repeats=16, axis=0)
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
        kron_kernel = torch.zeros(
            self._batch_size, self._y_size, num_total_points ** 2, num_total_points ** 2
        )

        for i in range(self._batch_size):
            kron_kernel[i][0] = Tensor(
                np.kron(
                    cholesky_space.squeeze(1).numpy()[i, :, :],
                    cholesky_time.squeeze(1).numpy()[i, :, :],
                )
            )
        # Sample a curve
        # [batch_size, y_size, num_total_points, 1]
        y_values = (
            kron_kernel
            @ Tensor(self._batch_size, self._y_size, num_total_points ** 2, 1).normal_()
        )

        # [batch_size, num_total_points, y_size]
        y_values = y_values.squeeze(1)

        x_values = torch.zeros(self._batch_size, num_total_points ** 2, self._x_size)
        x_values[:, :, 1] = (
            time_values.squeeze(1).repeat_interleave(
                repeats=int(num_total_points), axis=1
            )
        ).squeeze(2)

        index = num_total_points
        for i in range(num_total_points):
            x_values[:, index : index + num_total_points, 0] = space_values[
                :, i, 0
            ].unsqueeze(1)
            index += num_total_points

        if self._testing:
            # Select the targets
            target_x = x_values
            target_y = y_values

            # Select the observations
            idx = arange(num_target)[torch.randperm(num_target)]
            context_x = target_x[:, idx[: num_context ** 2], :]
            context_y = y_values[:, idx[: num_context ** 2], :]

        else:
            # Select the targets which will consist of the context points as well as
            # some new target points
            target_x = x_values[:, : (num_target + num_context) ** 2, :]
            target_y = y_values[:, : (num_target + num_context) ** 2, :]

            # Select the observations
            context_x = x_values[:, : num_context ** 2, :]
            context_y = y_values[:, : num_context ** 2, :]

        query = ((context_x, context_y), target_x)

        return NPRegressionDescription(
            query=query,
            target_y=target_y,
            num_total_points=target_x.shape[1],
            num_context_points=num_context,
        )

