import torch
import torch.nn as nn

from rope import precompute_positional_frequencies

from model import ModelArgs
from encoder import EncoderBlock
from layer import RMSNorm

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
        self.tok_embeddings = nn.Embedding(self.vocab_size, model_args.dim)
        
        # Encoder Layers
        self.layers = nn.ModuleList()  # Maintaining compatibility with pretrained model
        for _ in range(model_args.n_layers):
            self.layers.append(EncoderBlock(model_args))
        
        # Normalization Layer
        self.norm = RMSNorm(model_args.dim, epsilon=model_args.norm_eps)  # Maintaining compatibility with pretrained model
        
        # Output Layer
        self.output = nn.Linear(model_args.dim, self.vocab_size, bias=False)  # Maintaining compatibility with pretrained model
        
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
        hidden_states = self.tok_embeddings(input_tokens)  # Shape: (Batch_Size, Sequence_Length, Dimension)
        
        # Get frequency components for positional encoding
        frequency_components = self.frequency_components[start_position:start_position + sequence_length]  # Shape: (Sequence_Length, Dimension)
        
        # Pass through all encoder layers
        for layer in self.layers:
            hidden_states = layer(hidden_states, start_position, frequency_components)
            
        # Apply normalization layer
        normalized_states = self.norm(hidden_states)
        
        # Generate output
        output_tensor = self.output(normalized_states).float()
        
        return output_tensor