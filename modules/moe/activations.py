import math
import torch
import torch.nn as nn


class GELU_(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Gaussian Error Linear Unit (GELU) activation function.
        
        Args:
            x (torch.Tensor): Input tensor.
        
        Returns:
            torch.Tensor: Output tensor after applying GELU.
        """
        
        # Compute GELU activation
        gelu_out = 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        
        return gelu_out

# Use native PyTorch implementation if available, otherwise use custom implementation
GELU = nn.GELU if hasattr(nn, 'GELU') else GELU_
