import collections
from torch import eye, randint, int32, arange, Tensor, float32, sqrt
import torch
from abc import abstractmethod

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

NPRegressionDescription = collections.namedtuple(
    "NPRegressionDescription",
    ("query", "target_y", "num_total_points", "num_context_points"),
)


class GPCurvesReader(object):
    """Generates curves using a Gaussian Process (GP).

    Supports vector inputs (x) and vector outputs (y). Kernel is
    mean-squared exponential, using the x-value l2 coordinate distance scaled by
    some factor chosen randomly in a range. Outputs are independent gaussian
    processes.
    TODO: different to what Hyunjik Kim had, we will need to repeat the generate_curve
    in our training procedure
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
        self._batch_size = batch_size
        self._max_num_context = max_num_context
        self._x_size = x_size
        self._y_size = y_size
        self._l1_scale = l1_scale
        self._sigma_scale = sigma_scale
        self._random_kernel_parameters = random_kernel_parameters
        self._testing = testing

    @abstractmethod
    def _kernel(self, xdata, l1, sigma_f, sigma_noise=2e-2):
        raise NotImplementedError()

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
        if self._random_kernel_parameters:
            l1 = Tensor(self._batch_size, self._y_size, self._x_size).uniform_(
                0.1, self._l1_scale
            )
            sigma_f = Tensor(self._batch_size, self._y_size).uniform_(
                0.1, self._sigma_scale
            )
        # Or use the same fixed parameters for all mini-batches
        else:
            l1 = eye(self._batch_size, self._y_size, self._x_size) * self._l1_scale
            sigma_f = eye(self._batch_size, self._y_size) * self._sigma_scale

        # Pass the x_values through the kernel
        # [batch_size, y_size, num_total_points, num_total_points]
        kernel = self._kernel(x_values, l1, sigma_f)

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
    TODO: different to what Hyunjik Kim had, we will need to repeat the generate_curve
    in our training procedure
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
        super(RBFGPCurvesReader, self).__init__(
            batch_size,
            max_num_context,
            x_size=1,
            y_size=1,
            l1_scale=0.6,
            sigma_scale=1.0,
            random_kernel_parameters=True,
            testing=False,
        )
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
        self._l1_scale = l1_scale
        self._sigma_scale = sigma_scale
        self._random_kernel_parameters = random_kernel_parameters
        self._testing = testing

    def _kernel(self, xdata, l1, sigma_f, sigma_noise=2e-2):
        """Applies the Gaussian kernel to generate curve data.

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
        num_total_points = xdata.shape[1]

        # Expand and take the difference
        xdata1 = xdata.unsqueeze(1)  # [B, 1, num_total_points, x_size]
        xdata2 = xdata.unsqueeze(2)  # [B, num_total_points, 1, x_size]
        diff = xdata1 - xdata2  # [B, num_total_points, num_total_points, x_size]

        # [B, y_size, num_total_points, num_total_points, x_size]
        norm = (diff[:, None, :, :, :] / l1[:, :, None, None, :]).pow(2)

        norm = norm.sum(-1)  # [B, data_size, num_total_points, num_total_points]

        # [B, y_size, num_total_points, num_total_points]
        kernel = sigma_f.pow(2)[:, :, None, None] * (-0.5 * norm).exp()

        # Add some noise to the diagonal to make the cholesky work.
        kernel += (sigma_noise ** 2) * eye(num_total_points)

        return kernel


class MaternGPCurvesReader(GPCurvesReader):
    """Generates curves using a Gaussian Process (GP).

    Supports vector inputs (x) and vector outputs (y). Kernel is
    mean-squared exponential, using the x-value l2 coordinate distance scaled by
    some factor chosen randomly in a range. Outputs are independent gaussian
    processes.
    TODO: different to what Hyunjik Kim had, we will need to repeat the generate_curve
    in our training procedure
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
        super(MaternGPCurvesReader, self).__init__(
            batch_size,
            max_num_context,
            x_size=1,
            y_size=1,
            l1_scale=0.6,
            sigma_scale=1.0,
            random_kernel_parameters=True,
            testing=False,
        )
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
        self._l1_scale = l1_scale
        self._sigma_scale = sigma_scale
        self.nu = nu
        self._random_kernel_parameters = random_kernel_parameters
        self._testing = testing

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
        num_total_points = xdata.shape[1]

        # Expand and take the difference
        xdata1 = xdata.unsqueeze(1)  # [B, 1, num_total_points, x_size]
        xdata2 = xdata.unsqueeze(2)  # [B, num_total_points, 1, x_size]
        diff = xdata1 - xdata2  # [B, num_total_points, num_total_points, x_size]

        # [B, data_size, num_total_points, num_total_points]
        norm = diff[:, None, :, :, :].norm(dim=-1) / l1[:, :, None, :]

        if self.nu == 0.5:
            # [B, y_size, num_total_points, num_total_points]
            kernel = sigma_f.pow(2)[:, :, None, None] * (-norm).exp()

        elif self.nu == 1.5:
            kernel = (
                sigma_f.pow(2)[:, :, None, None]
                * (1 + sqrt(Tensor([3])) * norm)
                * (-sqrt(Tensor([3])) * norm).exp()
            )

        elif self.nu == 2.5:
            kernel = (
                sigma_f.pow(2)[:, :, None, None]
                * (1 + sqrt(Tensor([5])) * norm + 5 / 3 * norm.pow(2))
                * (-sqrt(Tensor([5])) * norm).exp()
            )
        else:
            raise NotImplementedError()

        # Add some noise to the diagonal to make the cholesky work.
        kernel += (sigma_noise ** 2) * eye(num_total_points)

        return kernel
