import collections
from torch import eye, randint, int32, arange, Tensor, float32, sqrt, ones
import torch


def rbf_kernel(xdata, l1, sigma_f, sigma_noise=2e-2):
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


def matern_kernel(xdata, nu, l1, sigma_f, sigma_noise=2e-2):
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
    norm = (diff[:, None, :, :, :] / l1[:, :, None, None, :]).norm(dim=-1)

    if nu == 0.5:
        # [B, y_size, num_total_points, num_total_points]
        kernel = sigma_f.pow(2)[:, :, None, None] * (-norm).exp()

    elif nu == 1.5:
        kernel = (
            sigma_f.pow(2)[:, :, None, None]
            * (1 + sqrt(Tensor([3])) * norm)
            * (-sqrt(Tensor([3])) * norm).exp()
        )

    elif nu == 2.5:
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


def generate_lengthscale_sigma_f(
    random_kernel_parameters, batch_size, y_size, x_size, l1_scale, sigma_scale
):

    # Set kernel parameters
    # Either choose a set of random parameters for the mini-batch
    if random_kernel_parameters:
        l1 = Tensor(batch_size, y_size, x_size).uniform_(0.1, l1_scale)
        sigma_f = Tensor(batch_size, y_size).uniform_(0.1, sigma_scale)
    # Or use the same fixed parameters for all mini-batches
    else:
        l1 = ones(batch_size, y_size, x_size) * l1_scale
        sigma_f = ones(batch_size, y_size) * sigma_scale

    return l1, sigma_f
