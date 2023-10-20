import torch
import torch.nn as nn
import torch.nn.functional as F

from model import ModelArgs

class RMSNorm(nn.Module):
    def __init__(self, dimension: int, epsilon: float = 1e-6) -> None:
        """
        Initialize the RMSNorm layer.
        
        Parameters:
        - dimension (int): The size of the dimension to normalize.
        - epsilon (float): A small constant to avoid division by zero. Default is 1e-6.
        """
        super().__init__()
        
        # Epsilon to avoid division by zero
        self.epsilon = epsilon
        
        # Learnable scaling factor (gamma parameter)
        self.weight = nn.Parameter(torch.ones(dimension))
        
    def _normalize(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Perform RMS normalization on the input tensor.
        
        Parameters:
        - input_tensor (torch.Tensor): Input tensor of shape (Batch_Size, Sequence_Length, Dimension).
        
        Returns:
        - torch.Tensor: RMS-normalized tensor.
        """
        # Compute root mean square (RMS) normalization
        return input_tensor * torch.rsqrt(input_tensor.pow(2).mean(-1, keepdim=True) + self.epsilon)
        
    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for RMS normalization.
        
        Parameters:
        - input_tensor (torch.Tensor): Input tensor of shape (Batch_Size, Sequence_Length, Dimension).
        
        Returns:
        - torch.Tensor: RMS-normalized and scaled tensor.
        """
        # Apply RMS normalization and scaling
        return self.weight * self._normalize(input_tensor.float()).type_as(input_tensor)

class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        """
        Initialize the FeedForward layer.
        
        Parameters:
        - args (ModelArgs): Configuration arguments for the model.
        """
        super().__init__()
        
        # Calculate hidden dimension
        hidden_dim = 4 * args.dim
        hidden_dim = int(2 * hidden_dim / 3)
        if args.feed_forward_dim_multiplier is not None:
            hidden_dim = int(args.feed_forward_dim_multiplier * hidden_dim)
        # Round the hidden_dim to the nearest multiple of the multiple_of parameter
        hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)
        
        # Initialize weight matrices
        self.w1 = nn.Linear(args.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the feed-forward block.
        
        Parameters:
        - x (torch.Tensor): Input tensor of shape (Batch, Sequence_Length, Dimension).
        
        Returns:
        - torch.Tensor: The output tensor of the same shape as input.
        """
        
        # First linear transformation followed by swish activation
        swish = F.silu(self.w1(x))
        
        # Second linear transformation for gating mechanism
        x_V = self.w3(x)
        
        # Element-wise multiplication (gating)
        x = swish * x_V
        
        # Final linear transformation to match output dimension
        x = self.w2(x)
        
        return x