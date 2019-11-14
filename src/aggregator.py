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

from src.utils import BatchMLP


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

        def weight_fn(x):
            return 1 + Tanh(x)

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
        self, rep, att_type, x_dim, r_dim, scale=1.0, normalise=True, num_heads=8, hid_dim=None, num_hid=3,
    ):
        """Create attention module.

        Takes in context inputs, target inputs and
        representations of each context input/output pairnice
        to output an aggregated representation of the context data.
        Args:
            rep: transformation to apply to contexts before computing attention. 
                One of: ['identity','mlp'].
            att_type: type of attention. One of the following:
                ['uniform','laplace','dot_product','multihead']
            x_dim : dimension of the x vectors
            r_dim : dimension of the incoming representation vectors
            batch_mlp: network class of batch mlp network, constructed by create_batch_mlp
            scale: scale of attention.
            normalise: Boolean determining whether to:
                1. apply softmax to weights so that they sum to 1 across context pts or
                2. apply custom transformation to have weights in [0,1].
            num_heads: number of heads for multihead.
            hid_dim : dimension of the hidden layers if using an "mlp" represantation
            num_hid : number of hidden layers to use if using an "mlp" represantation
        """

        self._rep = rep

        self._type = att_type

        self._scale = scale
        self._normalise = normalise

        if self._rep == "identity":
            self.embed_func = lambda x: x
        elif self._rep == "mlp":
            # Im assuming the input/output of the encoding should be the same size?
            # Also assuming if you dont specify a hidden dim just set it to the rep size
            if hid_dim == None:
                hid_dim = r_dim
            self.batch_mlp = BatchMLP(x_dim, hid_dim, hid_dim, num_hid)
            # self.batch_mlp = create_batch_mlp(test, output_sizes = [2])

            self.embed_func = lambda x: self.batch_mlp(x)
        else:
            raise NameError("'rep' not among ['identity','mlp']")

        if self._type == "uniform":
            self.attention_func = lambda q, k, r: uniform_attention(q, r)
        elif self._type == "laplace":
            self.attention_func = lambda q, k, r: laplace_attention(q, k, r, self._scale, self._normalise)
        elif self._type == "dot_product":
            self.attention_func = lambda q, k, r: dot_product_attention(q, k, r, self._normalise)
        elif self._type == "multihead":
            self._num_heads = num_heads   
            self.multihead_attention = MultiheadAttention(
                embed_dim=r_dim, num_heads=self._num_heads
            )
            self.attention_func = lambda q, k, r: self.multihead_attention.forward(q, k, r)[0]
        else:
            raise NameError(
                (
                    "'att_type' not among ['uniform','laplace','dot_product'"
                    ",'multihead']"
                )
            )   

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
        
        q = self.embed_func(x1)
        k = self.embed_func(x2)

        rep = self.attention_func(q, k, r)

        return rep
