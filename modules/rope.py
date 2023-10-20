
import torch

def precompute_positional_frequencies(head_dimension: int, sequence_length: int, device: str, theta: float = 10000.0) -> torch.Tensor:
    """
    Precompute the frequency components for positional encoding based on the given parameters.
    
    Parameters:
    - head_dimension (int): Dimensionality of each attention head. Must be an even number.
    - sequence_length (int): Length of the sequence for which positional encoding is computed.
    - device (str): Device to allocate tensors ('cpu', 'cuda', etc.)
    - theta (float): Base parameter for the theta calculation. Default is 10000.0.
    
    Returns:
    - torch.Tensor: Precomputed frequency components for positional encoding.
    """
    
    # Validate that the head dimension is even, as specified in the paper
    assert head_dimension % 2 == 0, "Dimension must be divisible by 2"
    
    # Compute the theta parameter according to the formula
    theta_indices = torch.arange(0, head_dimension, 2).float()
    theta_values = 1.0 / (theta ** (theta_indices / head_dimension)).to(device)
    
    # Generate the sequence positions (the 'm' parameter in the paper)
    sequence_positions = torch.arange(sequence_length, device=device)
    
    # Compute frequencies by taking the outer product of sequence positions and theta values
    frequencies = torch.outer(sequence_positions, theta_values).float()
    
    # Compute complex numbers in polar form, where the magnitude is 1 and the angle is the frequency
    complex_frequencies = torch.polar(torch.ones_like(frequencies), frequencies)
    
    return complex_frequencies

def apply_rotary_embeddings(x: torch.Tensor, freqs_complex: torch.Tensor, device: str) -> torch.Tensor:
    """
    Apply rotary embeddings to the input tensor.
    
    Parameters:
    - x (torch.Tensor): Input tensor of shape (Batch, Sequence_Length, Heads, Head_Dimension).
    - freqs_complex (torch.Tensor): Complex frequencies for positional encoding.
    - device (str): Device specification ('cpu' or 'cuda').
    
    Returns:
    - torch.Tensor: The output tensor with rotary embeddings applied.
    """
    
    # Convert the last dimension to complex numbers
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    
    # Add batch and head dimensions to the freqs_complex tensor
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)
    
    # Multiply to rotate the complex numbers
    x_rotated = x_complex * freqs_complex
    
    # Convert back to real numbers and reshape to original dimensions
    x_out = torch.view_as_real(x_rotated).reshape(*x.shape)
    
    return x_out.type_as(x).to(device)