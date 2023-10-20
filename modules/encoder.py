import torch
import torch.nn as nn

from attention import SelfAttention
from layer import FeedForward, RMSNorm

from model import ModelArgs

class EncoderBlock(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        """
        Initialize the EncoderBlock layer.
        
        Parameters:
        - args (ModelArgs): Configuration arguments for the model.
        """
        super().__init__()
        
        # Initialize layer configurations based on passed arguments
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        
        # Initialize self-attention and feed-forward layers
        self.attention = SelfAttention(args)
        self.feed_forward = FeedForward(args)
        
        # Initialize normalization layers
        self.attention_norm = RMSNorm(args.dim, epsilon=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, epsilon=args.norm_eps)
    
    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the encoder block.
        
        Parameters:
        - x (torch.Tensor): Input tensor of shape (Batch, Sequence_Length, Dimension).
        - start_pos (int): Starting position for positional encoding.
        - freqs_complex (torch.Tensor): Complex frequencies for positional encoding.
        
        Returns:
        - torch.Tensor: The output tensor of the same shape as input.
        """
        
        # Apply normalization and self-attention
        norm_attention_input = self.attention_norm(x)
        attention_output = self.attention.forward(norm_attention_input, start_pos, freqs_complex)
        
        # Residual connection after attention
        h = x + attention_output
        
        # Apply normalization and feed-forward layer
        norm_ffn_input = self.ffn_norm(h)
        ffn_output = self.feed_forward.forward(norm_ffn_input)
        
        # Residual connection after feed-forward
        out = h + ffn_output
        
        return out
