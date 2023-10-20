from dataclasses import dataclass
from typing import Optional

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
    max_batch_size: int  = 32
    max_seq_len: int = 2048
    device: str = None
