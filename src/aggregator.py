from torch import Tensor, abs, einsum, normal
from math import sqrt
from torch.nn import (
    Softmax,
    Tanh,
    Sigmoid,
    Conv1d,
    MultiheadAttention,
    ReLU,
    Linear,
    Sequential,
)

# utility methods
# def create_batch_mlp(input, output_sizes):
#     """Apply MLP to the final axis of a 3D tensor (reusing already defined MLPs).

#     Args:
#     input: input tensor of shape [B,n,d_in].
#     output_sizes: An iterable containing the output sizes of the MLP as defined 
#         in `basic.Linear`.

#     Returns:
#     tensor of shape [B,n,d_out] where d_out=output_sizes[-1]
#     """
#     # Get the shapes of the input and reshape to parallelise across observations
#     batch_size, _, _ = input.shape[0], input.shape[1], input.shape[2]

#     modules = []
#     in_size = input.shape[-1]
#     # Pass through MLP
#     for _, size in enumerate(output_sizes[:-1]):
#         modules.append(Linear(in_size, size))
#         modules.append(ReLU())
#         in_size = size.copy()

#     # Last layer without a ReLu
#     modules.append(Linear(in_size, output_sizes[-1]))
#     mlp = Sequential(*modules)
#     mlp.batch_size = batch_size
#     mlp.final_output_size = output_sizes[-1]

#     return mlp


def uniform_attention(q, v):
    """Uniform attention. Equivalent to np.
containing 
    Args:
    q: queries. tensor of shape [B,m,d_k].
    v: values. tensor of shape [B,n,d_v].

    Returns:
    tensor of shape [B,m,d_v].

    """
    total_points = q.shape[1]
    rep = v.mean(dim=1, keepdim=True)  # [B,1,d_v]
    rep = rep.repeat_interleave(repeats=total_points, axis=1)
    return rep


def laplace_attention(q, k, v, scale, normalise):
    """Computes laplace exponential attention.

    Args:
    q: queries. tensor of shape [B,m,d_k].
    k: keys. tensor of shape [B,n,d_k].
    v: values. tensor of shape [B,n,d_v].
    scale: float that scales the L1 distance.
    normalise: Boolean that determines whether weights sum to 1.

    Returns:
    tensor of shape [B,m,d_v].
    """
    k = k.unsqueeze(1)  # [B,1,n,d_k]
    q = q.unsqueeze(2)  # [B,m,1,d_k]
    unnorm_weights = -abs((k - q) / scale)  # [B,m,n,d_k]
    unnorm_weights = unnorm_weights.sum(-1)  # [B,m,n]
    if normalise:
        weight_fn = Softmax()
    else:
        weight_fn = lambda x: 1 + Tanh(x)
        weights = weight_fn(unnorm_weights)  # [B,m,n]
    rep = einsum("bik,bkj->bij", weights, v)  # [B,m,d_v]
    return rep


def dot_product_attention(q, k, v, normalise):
    """Computes dot product attention.

    Args:
    q: queries. tensor of  shape [B,m,d_k].
    k: keys. tensor of shape [B,n,d_k].
    v: values. tensor of shape [B,n,d_v].
    normalise: Boolean that determines whether weights sum to 1.containing 

    Returns:
    tensor of shape [B,m,d_v].
    """
    d_k = q.shape[-1]
    scale = sqrt(d_k)
    unnorm_weights = einsum("bjk,bik->bij", k, q) / scale  # [B,m,n]
    if normalise:
        weight_fn = Softmax()
    else:
        weight_fn = Sigmoid()
        weights = weight_fn(unnorm_weights)  # [B,m,n]
    rep = einsum("bik,bkj->bij", weights, v)  # [B,m,d_v]
    return rep


class Attention(object):
    """The Attention module.
	"""

    def __init__(
        self,
        rep,
        output_sizes,
        att_type,
        scale=1.0,
        normalise=True,
        num_heads=8,
    ):
        """Create attention module.

        Takes in context inputs, target inputs and
        representations of each context input/output pairnice
        to output an aggregated representation of the context data.
        Args:
            rep: transformation to apply to contexts before computing attention. 
                One of: ['identity','mlp'].
            output_sizes: list of number of hidden units per layer of mlp.
                Used only if rep == 'mlp'.
            att_type: type of attention. One of the following:
                ['uniform','laplace','dot_product','multihead']
            batch_mlp: network class of batch mlp network, constructed by create_batch_mlp
            scale: scale of attention.
            normalise: Boolean determining whether to:
                1. apply softmax to weights so that they sum to 1 across context pts or
                2. apply custom transformation to have weights in [0,1].
            num_heads: number of heads for multihead.
        """

        self._rep = rep

        self._output_sizes = output_sizes

        self._type = att_type

        self._scale = scale
        self._normalise = normalise
        if self._type == "multihead":
            self._num_heads = num_heads
        # self.batch_mlp = create_batch_mlp(test, output_sizes = [2])

    def __call__(self, x1, x2, r):
        """Apply attention to create aggregated representation of r.

		Args:
			x1: tensor of shape [B,n1,d_x].
			x2: tensor of shape [B,n2,d_x].
			r: tensor of shape [B,n1,d].
			
		Returns:
			tensor of shape [B,n2,d]

		Raises:
			NameError: The argument for rep/type was invalid.
		"""
        if self._rep == "identity":
            k, q = (x1, x2)
        elif self._rep == "mlp":
            # Pass through MLP
            # TODO: integrate with Michael's MLP
            k = self.batch_mlp(x1, self._output_sizes, "attention")
            q = self.batch_mlp(x2, self._output_sizes, "attention")
        else:
            raise NameError("'rep' not among ['identity','mlp']")

        if self._type == "uniform":
            rep = uniform_attention(q, r)
        elif self._type == "laplace":
            rep = laplace_attention(q, k, r, self._scale, self._normalise)
        elif self._type == "dot_product":
            rep = dot_product_attention(q, k, r, self._normalise)
        elif self._type == "multihead":
            multihead_attention = MultiheadAttention(
                embed_dim=r.shape[-1], num_heads=self._num_heads
            )
            rep, _ = multihead_attention.forward(q, k, r)
        else:
            raise NameError(
                (
                    "'att_type' not among ['uniform','laplace','dot_product'"
                    ",'multihead']"
                )
            )

        return rep
