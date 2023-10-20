from typing import Optional
import math
from inspect import isfunction

import torch
import torch.nn.functional as F


def default(val, default_val):
    """
    Returns the default value if the given value is None.
    
    Args:
        val: Value to check for None.
        default_val: Default value to return if val is None. If a function, the function is called.
        
    Returns:
        The value `val` if it's not None, otherwise `default_val`.
    """
    default_val = default_val() if isfunction(default_val) else default_val
    return val if val is not None else default_val

def cast_tuple(el):
    """
    Casts the given element to a tuple.
    
    Args:
        el: Element to cast.
        
    Returns:
        Tuple containing the element.
    """
    return el if isinstance(el, tuple) else (el,)


def top1(t: torch.Tensor):
    """
    Returns the top 1 value and its index from a tensor along the last dimension.
    
    Args:
        t (torch.Tensor): Input tensor.
        
    Returns:
        Tuple of top value and its index.
    """
    values, index = t.topk(k=1, dim=-1)
    values, index = map(lambda x: x.squeeze(dim=-1), (values, index))
    return values, index

def cumsum_exclusive(t: torch.Tensor, dim: Optional[int]=-1) -> torch.Tensor:
    """
    Computes the exclusive cumulative sum of a tensor along a specified dimension.
    
    Args:
        t (torch.Tensor): Input tensor.
        dim (int, optional): Dimension along which to perform cumsum. Defaults to -1.
        
    Returns:
        Torch.Tensor: Tensor with exclusive cumsum applied.
    """

    num_dims = len(t.shape)
    num_pad_dims = - dim - 1
    pre_padding = (0, 0) * num_pad_dims
    pre_slice   = (slice(None),) * num_pad_dims
    padded_t = F.pad(t, (*pre_padding, 1, 0)).cumsum(dim=dim)
    return padded_t[(..., slice(None, -1), *pre_slice)]


def safe_one_hot(indexes: torch.Tensor, max_length: int):
    """
    Creates a one-hot encoded tensor safely by handling out-of-bound indices.
    
    Args:
        indexes (torch.Tensor): Tensor containing indices.
        max_length (int): Maximum length for one-hot encoding.
        
    Returns:
        Torch.Tensor: One-hot encoded tensor.
    """
    max_index = indexes.max() + 1
    return F.one_hot(indexes, max(max_index + 1, max_length))[..., :max_length]

def init_tensor(t: torch.Tensor):
    """
    Initializes a tensor with uniform values within a range defined by its dimensions.
    
    Args:
        t (torch.Tensor): Tensor to initialize.
        
    Returns:
        Torch.Tensor: Initialized tensor.
    """
    dim = t.shape[-1]
    std = 1 / math.sqrt(dim)
    return t.uniform_(-std, std)