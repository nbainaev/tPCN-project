import torch
import torch.nn as nn
import numpy as np
from typing import Union, Any
import numpy.typing as npt


SetSdr = set[int]
SparseSdr = Union[list[int], npt.NDArray[np.int32], SetSdr]
DenseSdr = npt.NDArray[Union[np.int32, np.float32]]

class Softmax(nn.Module):
    def forward(self, inp):
        return torch.softmax(inp, dim=-1)

    def deriv(self, inp):
        # Compute the softmax output
        soft = self.forward(inp)
        # Initialize a tensor for the derivative, with the same shape as the softmax output
        s = soft.unsqueeze(-2)  # Add a dimension for broadcasting
        identity = torch.eye(s.size(-1)).unsqueeze(0).to(inp.device)  
        # The diagonal contains s_i * (1 - s_i) and off-diagonal s_i * (-s_j)
        deriv = identity * s - s * s.transpose(-1, -2) # shape (batch_size, N, N)
        return deriv

class Tanh(nn.Module):
    def forward(self, inp):
        return torch.tanh(inp)

    def deriv(self, inp):
        return 1.0 - torch.tanh(inp) ** 2.0

class ReLU(nn.Module):
    def forward(self, inp):
        return torch.relu(inp)

    def deriv(self, inp):
        out = self(inp)
        out[out > 0] = 1.0
        return out

class Sigmoid(nn.Module):
    def forward(self, inp):
        return torch.sigmoid(inp)

    def deriv(self, inp):
        out = self(inp)
        return out * (1 - out)
    
def isnone(x, default):
    """Return x if it's not None, or default value instead."""
    return x if x is not None else default

def clip(x: Any, low=None, high=None) -> Any:
    """Clip the value with the provided thresholds. NB: doesn't support vectorization."""

    # both x < None and x > None are False, so consider them as safeguards
    if x < low:
        x = low
    elif x > high:
        x = high
    return x

def sparse_to_dense(
        sdr: SparseSdr,
        size: int | tuple | DenseSdr = None,
        shape: int | tuple | DenseSdr = None,
        dtype=float,
        like: DenseSdr = None
) -> DenseSdr:
    """
    Converts SDR from sparse representation to dense.

    Size, shape and dtype define resulting dense vector params.
    The size should be at least inducible (from shape or like).
    The shape default is 1-D, dtype: float.

    Like param is a shorthand, when you have an array with all three params set correctly.
    Like param overwrites all others!
    """

    if like is not None:
        shape, size, dtype = like.shape, like.size, like.dtype
    else:
        if isinstance(size, np.ndarray):
            size = size.size
        if isinstance(shape, np.ndarray):
            shape = shape.shape

        # -1 for reshape means flatten.
        # It is also invalid size, which we need here for the unset shape case.
        shape = isnone(shape, -1)
        size = isnone(size, np.prod(shape))

    dense_vector = np.zeros(size, dtype=dtype)
    dense_vector[sdr] = 1
    return dense_vector.reshape(shape)

def safe_divide(x, y: int | float):
    """
    Return x / y or just x itself if y == 0 preventing NaNs.
    Warning: it may not work as you might expect for floats, use it only when you need exact match!
    """
    return x / y if y != 0 else x

def softmax(
        x: npt.NDArray[np.float32], *, temp: float = None, beta: float = None, axis: int = -1
) -> npt.NDArray[np.float32]:
    """
    Compute softmax values for a vector `x` with a given temperature or inverse temperature.
    The softmax operation is applied over the last axis by default, or over the specified axis.
    """
    beta = isnone(beta, 1.0)
    temp = isnone(temp, 1 / beta)
    temp = clip(temp, 1e-5, 1e+4)

    e_x = np.exp((x - np.max(x, axis=axis, keepdims=True)) / temp)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)
