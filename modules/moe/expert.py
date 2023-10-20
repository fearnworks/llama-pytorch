from typing import Optional

import torch
import torch.nn as nn
from moe.activations import GELU
from moe.moe_util import default, cast_tuple, init_tensor

MIN_EXPERT_CAPACITY = 4

class Experts(nn.Module):
    def __init__(self,
                 dim: int,
                 num_experts: int = 16,
                 hidden_dim: Optional[int] = None,
                 activation: nn.Module = GELU):
        """
        Initialize the Experts module.

        Args:
            dim (int): The dimensionality of the input.
            num_experts (int, optional): The number of experts to use. Defaults to 16.
            hidden_dim (Optional[int], optional): The dimensionality of the hidden layer. Defaults to 4 times the input dimension if not provided.
            activation (nn.Module, optional): The activation function to use. Defaults to GELU.

        Notes:
            The weights for each expert are initialized during the construction of the module.
        """
        super().__init__()

        hidden_dim = default(hidden_dim, dim * 4)
        num_experts = cast_tuple(num_experts)

        w1 = torch.zeros(*num_experts, dim, hidden_dim)
        w2 = torch.zeros(*num_experts, hidden_dim, dim)

        w1 = init_tensor(w1)
        w2 = init_tensor(w2)

        self.w1 = nn.Parameter(w1)
        self.w2 = nn.Parameter(w2)
        self.act = activation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Experts module.

        Args:
            x (torch.Tensor): Input tensor with shape [..., num_tokens, dim].

        Returns:
            torch.Tensor: Output tensor after passing through the experts, with shape [..., num_tokens, dim].
        """
        hidden = torch.einsum('...nd,...dh->...nh', x, self.w1)
        hidden = self.act(hidden)
        out    = torch.einsum('...nh,...hd->...nd', hidden, self.w2)
        return out

