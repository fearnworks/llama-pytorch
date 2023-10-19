import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional

from rope import precompute_positional_frequencies, apply_rotary_embeddings

@dataclass
class ModelArgs:
    dim: int = 4096 # dimensionality of the model's hidden states.
    n_layers: int = 32 # The number of transformer layers in the model. Each layer consists of self-attention and feed-forward neural network components.
    n_heads: int = 8 # The number of attention heads in the multi-head attention mechanism. More heads allow the model to focus on different parts of the input sequence simultaneously.
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1 # Set when tokenizer is loaded 
    multiple_of: int = 256
    feed_forward_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5 # A small constant added during layer normalization to prevent division by zero.
    # KV Cache
    max_batch_size = 32
    max_seq_len: int = 2048
    device: str = None

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
        self.scaling_factor = nn.Parameter(torch.ones(dimension))
        
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
        return self.scaling_factor * self._normalize(input_tensor.float()).type_as(input_tensor)



def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Repeat the key and value heads.
    
    Parameters:
    - x (torch.Tensor): Input tensor of shape (Batch, Sequence_Length, KV_Heads, Head_Dimension).
    - n_rep (int): Number of repetitions for each head.
    
    Returns:
    - torch.Tensor: The output tensor with keys and values repeated.
    """
    
    batch_size, seq_len, n_kv_heads, head_dim = x.shape

    if n_rep == 1:
        return x
    
    # Expand and repeat the last dimension
    expanded_x = x[:, :, :, None, :].expand(batch_size, seq_len, n_kv_heads, n_rep, head_dim)
    
    # Reshape to combine the KV heads and repetitions
    reshaped_x = expanded_x.reshape(batch_size, seq_len, n_kv_heads * n_rep, head_dim)
    
    return reshaped_x

class SelfAttention(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        """
        Initialize the SelfAttention layer.
        
        Parameters:
        - args (ModelArgs): Configuration arguments for the model.
        """
        super().__init__()

        # Initialize configurations based on arguments
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_heads_q = args.n_heads
        self.n_rep = self.n_heads_q // self.n_kv_heads
        self.head_dim = args.dim // args.n_heads

        # Initialize weight matrices
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        # Initialize key and value caches
        self.cache_k = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))
        self.cache_v = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_complex: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through the self-attention block.
        
        Parameters:
        - x (torch.Tensor): Input tensor of shape (Batch, Sequence_Length, Dimension).
        - start_pos (int): Starting position for positional encoding.
        - freqs_complex (torch.Tensor): Complex frequencies for positional encoding.
        
        Returns:
        - torch.Tensor: The output tensor of the same shape as input.
        """
        batch_size, seq_len, _ = x.shape  # (B, 1, Dim)

        # (B, 1, Dim) -> (B, 1, H_Q * Head_Dim)
        xq = self.wq(x)
        # (B, 1, Dim) -> (B, 1, H_KV * Head_Dim)
        xk = self.wk(x)
        # (B, 1, Dim) -> (B, 1, H_KV * Head_Dim)
        xv = self.wv(x)

        # (B, 1, H_Q * Head_Dim) -> (B, 1, H_Q, Head_Dim)
        xq = xq.view(batch_size, seq_len, self.n_heads_q, self.head_dim)
        # (B, 1, H_KV * Head_Dim) -> (B, 1, H_KV, Head_Dim)
        xk = xk.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        # (B, 1, H_KV * Head_Dim) -> (B, 1, H_KV, Head_Dim)
        xv = xv.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        # (B, 1, H_Q, Head_Dim) --> (B, 1, H_Q, Head_Dim)
        xq = apply_rotary_embeddings(xq, freqs_complex, device=x.device)
        # (B, 1, H_KV, Head_Dim) --> (B, 1, H_KV, Head_Dim)
        xk = apply_rotary_embeddings(xk, freqs_complex, device=x.device)

        # Replace the entry in the cache
        self.cache_k[:batch_size, start_pos : start_pos + seq_len] = xk
        self.cache_v[:batch_size, start_pos : start_pos + seq_len] = xv

        # (B, Seq_Len_KV, H_KV, Head_Dim)
        keys = self.cache_k[:batch_size, : start_pos + seq_len]
        # (B, Seq_Len_KV, H_KV, Head_Dim)
        values = self.cache_v[:batch_size, : start_pos + seq_len]

        # Since every group of Q shares the same K and V heads, just repeat the K and V heads for every Q in the same group.

        # (B, Seq_Len_KV, H_KV, Head_Dim) --> (B, Seq_Len_KV, H_Q, Head_Dim)
        keys = repeat_kv(keys, self.n_rep)
        # (B, Seq_Len_KV, H_KV, Head_Dim) --> (B, Seq_Len_KV, H_Q, Head_Dim)
        values = repeat_kv(values, self.n_rep)

        # (B, 1, H_Q, Head_Dim) -> (B, H_Q, 1, Head_Dim)
        xq = xq.transpose(1, 2)
        # (B, Seq_Len_KV, H_Q, Head_Dim) -> (B, H_Q, Seq_Len_KV, Head_Dim)
        keys = keys.transpose(1, 2)
        # (B, Seq_Len_KV, H_Q, Head_Dim) -> (B, H_Q, Seq_Len_KV, Head_Dim)
        values = values.transpose(1, 2)

        # (B, H_Q, 1, Head_Dim) @ (B, H_Q, Head_Dim, Seq_Len_KV) -> (B, H_Q, 1, Seq_Len_KV)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        # (B, H_Q, 1, Seq_Len_KV) -> (B, H_Q, 1, Seq_Len_KV)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)

        # (B, H_Q, 1, Seq_Len) @ (B, H_Q, Seq_Len_KV, Head_Dim) -> (B, H_Q, 1, Head_Dim)
        output = torch.matmul(scores, values)
        # (B, H_Q, 1, Head_Dim) -> (B, 1, H_Q, Head_Dim) -> (B, 1, Dim)
        output = (output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1))
        return self.wo(output) # (B, 1, Dim) -> (B, 1, Dim)

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
        if args.ffn_dim_multiplier is not None:
            hidden_dim = int(args.ffn_dim_multiplier * hidden_dim)
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
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
    
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


class Transformer(nn.Module):
    def __init__(self, model_args: ModelArgs) -> None:
        """
        Initialize the Transformer model.
        
        Parameters:
        - model_args (ModelArgs): Configuration arguments for the model.
        """
        super().__init__()
        
        # Validate vocabulary size
        assert model_args.vocab_size != -1, "vocab_size must be set"
        
        # Model configuration
        self.model_args = model_args
        self.vocab_size = model_args.vocab_size
        self.num_layers = model_args.n_layers  # Maintaining compatibility with pretrained model
        
        # Token Embeddings
        self.token_embeddings = nn.Embedding(self.vocab_size, model_args.dim)
        
        # Encoder Layers
        self.encoder_layers = nn.ModuleList()  # Maintaining compatibility with pretrained model
        for _ in range(model_args.n_layers):
            self.encoder_layers.append(EncoderBlock(model_args))
        
        # Normalization Layer
        self.layer_norm = RMSNorm(model_args.dim, eps=model_args.norm_eps)  # Maintaining compatibility with pretrained model
        
        # Output Layer
        self.output_layer = nn.Linear(model_args.dim, self.vocab_size, bias=False)  # Maintaining compatibility with pretrained model
        
        # Pre-compute frequency components for positional encoding
        self.frequency_components = precompute_positional_frequencies(
            model_args.dim // model_args.n_heads, 
            model_args.max_seq_len * 2, 
            device=model_args.device
        )  # Maintaining compatibility with pretrained model

    def forward(self, 
                input_tokens: torch.Tensor, 
                start_position: int) -> torch.Tensor:
        """
        Forward pass for the model.
        
        Parameters:
        - input_tokens (torch.Tensor): Input tokens tensor of shape (Batch_Size, Sequence_Length).
        - start_position (int): Starting position for positional encoding.
        
        Returns:
        - torch.Tensor: Model output.
        """
        # Validate input shape
        batch_size, sequence_length = input_tokens.shape
        assert sequence_length == 1, "Only one token can be processed at a time"
        
        # Token Embedding
        hidden_states = self.token_embeddings(input_tokens)  # Shape: (Batch_Size, Sequence_Length, Dimension)
        
        # Get frequency components for positional encoding
        frequency_components = self.frequency_components[start_position:start_position + sequence_length]  # Shape: (Sequence_Length, Dimension)
        
        # Pass through all encoder layers
        for layer in self.encoder_layers:
            hidden_states = layer(hidden_states, start_position, frequency_components)
            
        # Apply normalization layer
        normalized_states = self.layer_norm(hidden_states)
        
        # Generate output
        output_tensor = self.output_layer(normalized_states).float()
        
        return output_tensor