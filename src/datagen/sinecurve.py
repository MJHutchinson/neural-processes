import numpy as np
import collections
from torch import eye, randint, int32, arange, Tensor, float32, sqrt
import torch
from src.utils import NPRegressionDescription


class SineCurvesReader(object):
    def __init__(
        self,
        batch_size=100,
        max_num_context=50,
        x_size=1,
        y_size=1,
        amplitude_scale=5,
        phase_scale=0.2 * np.pi,
        random_parameters=True,
        testing=False,
        restrict_test_range=True,
    ):
        """Creates a regression dataset of functions sampled from a GP.

        Args:
            batch_size: An integer.
            max_num_context: The max number of observations in the context.
            x_size: Integer >= 1 for length of "x values" vector.
            y_size: Integer >= 1 for length of "y values" vector.
            amplitude_scale: Float; typical scale for sine amplitude.
            phase_scale: Float; typical scale for sine phase.
            random_parameters: If `True`, the parameters (amplitude and phase)
                will be sampled uniformly within [0.1, amplitude] and [0.1, phase_scale].
            testing: Boolean that indicates whether we are testing. If so there are
                more targets for visualization.
            restrict_test_range: target data will only generated in the range [0, pi]
        """

        self._batch_size = batch_size
        self._max_num_context = max_num_context
        self._x_size = x_size
        self._y_size = y_size
        self._amplitude_scale = amplitude_scale
        self._phase_scale = phase_scale
        self._random_parameters = random_parameters
        self._testing = testing
        self._restrict_test_range = restrict_test_range

    def generate_curves(self):
        """Builds the op delivering the data.

        Generated functions are `float32` with x values between 0 and 2 * pi.

        Example:
            ## checking SineCurvesReader works
            this = SineCurvesReader().generate_curves()
            context_data_x, context_data_y = this.query[0]
            context_data_x = context_data_x.squeeze()
            context_data_y = context_data_y.squeeze()

            context_data_x.shape
            context_data_y.shape

            for curve_id in range(context_data_x.shape[0]):
                plt.scatter(context_data_x[curve_id, :].tolist(), context_data_y[curve_id, :].tolist())

        Returns:
            A `NPRegressionDescription` namedtuple.
        """
        num_context = randint(low=3, high=self._max_num_context, size=(1,), dtype=int32)

        # If we are testing we want to have more targets and have them evenly
        # distributed in order to plot the function.
        if self._testing:
            num_target = 400
            num_total_points = num_target
            x_values = (
                torch.arange(0, 2 * np.pi, 1 / 100)
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
            ).uniform_(0, 2 * np.pi)
        # Either choose a set of random parameters for the mini-batch
        if self._random_parameters:
            amplitude = Tensor(self._batch_size, self._y_size, self._x_size).uniform_(
                0.1, self._amplitude_scale
            )
            phase = Tensor(self._batch_size, self._y_size).uniform_(
                0, self._phase_scale
            )
        # Or use the same fixed parameters for all mini-batches
        else:
            # this is'nt working
            amplitude = (
                eye(self._batch_size, self._y_size, self._x_size)
                * self._amplitude_scale
            )
            phase = eye(self._batch_size, self._y_size) * self._phase_scale

        y_values = amplitude.reshape(100, 1, 1) * np.sin(
            x_values - phase.reshape(100, 1, 1)
        )

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
