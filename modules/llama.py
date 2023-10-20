from typing import List, Optional, Tuple

import time
from pathlib import Path
import json

import torch
from sentencepiece import SentencePieceProcessor
from tqdm import tqdm

from model import ModelArgs
from transformer import Transformer

class LLaMA:

    def __init__(self, model: Transformer, tokenizer: SentencePieceProcessor, model_args: ModelArgs):
        self.model = model
        self.tokenizer = tokenizer
        self.args = model_args

    @staticmethod
    def build(checkpoints_dir: str, tokenizer_path: str, load_model: bool, 
              max_seq_len: int, max_batch_size: int, device: str):
        """
        Build the LLaMA model along with its tokenizer and configuration.

        Args:
            checkpoints_dir (str): Directory where the model checkpoints are stored.
            tokenizer_path (str): Path to the tokenizer model.
            load_model (bool): Whether to load the model from checkpoints or not.
            max_seq_len (int): Maximum sequence length.
            max_batch_size (int): Maximum batch size.
            device (str): Device to run the model on ('cpu', 'cuda', etc.).

        Returns:
            LLaMA: An instance of the LLaMA class.

        Notes:
            - This function first checks if model checkpoints are available.
            - It then loads the configuration params and tokenizer.
            - Finally, it initializes the Transformer model and returns an instance of LLaMA.
        """

        # Initialize timing
        prev_time = time.time()

        # Load model from checkpoints if required
        if load_model:
            checkpoints = sorted(Path(checkpoints_dir).glob("*.pth"))
            assert len(checkpoints) > 0, f"no checkpoint files found in {checkpoints_dir}"
            ckpt_path = checkpoints[0]
            print(f'Loading checkpoint "{ckpt_path}"')
            checkpoint = torch.load(ckpt_path, map_location="cpu")
            print(f"Loaded checkpoint in {time.time() - prev_time:.2f}s")
            prev_time = time.time()

        # Load configuration params
        with open(Path(checkpoints_dir) / "params.json", "r") as f:
            params = json.loads(f.read())

        # Initialize model arguments and tokenizer
        model_args = ModelArgs(
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            device=device,
            **params
        )
        tokenizer = SentencePieceProcessor()
        tokenizer.load(tokenizer_path)
        model_args.vocab_size = tokenizer.vocab_size()

        # Set default device and initialize Transformer model
        torch.set_default_device(device)
        model = Transformer(model_args).to(device)

        # Load state dict if required
        if load_model:
            del checkpoint['rope.freqs']
            model.load_state_dict(checkpoint, strict=True)
            print(f"Loaded state dict in {time.time() - prev_time:.2f}s")
        
        return LLaMA(model, tokenizer, model_args)


    def text_completion(self, prompts: List[str], temperature: float = 0.6, 
                        top_p: float = 0.9, max_gen_len: Optional[int] = None) -> Tuple[List[List[int]], List[str]]:
        """
        Complete text based on given prompts.

        Args:
            prompts (List[str]): A list of string prompts to generate text from.
            temperature (float, optional): Controls randomness in Boltzmann distribution. Default is 0.6.
            top_p (float, optional): Cumulative probability for Top-P sampling. Default is 0.9.
            max_gen_len (Optional[int], optional): Maximum length for generated text. If None, will be set to max_seq_len - 1.

        Returns:
            Tuple[List[List[int]], List[str]]: A tuple containing a list of lists with token IDs and a list with the generated texts.
            
        Notes:
            - The function tokenizes the prompts and initializes output tokens.
            - It then iteratively generates new tokens and appends them to the output tokens.
            - Finally, it decodes the output tokens to generate the output text.
        """

        device = self.args.device
        if max_gen_len is None:
            max_gen_len = self.args.max_seq_len - 1

        # Tokenize prompts and initialize output tokens
        prompt_tokens = [self.tokenizer.encode(prompt, out_type=int, add_bos=True, add_eos=False) for prompt in prompts]
        batch_size = len(prompt_tokens)
        assert batch_size <= self.args.max_batch_size, f"batch size must be less than or equal to {self.args.max_batch_size}"
        max_prompt_len = max(len(prompt) for prompt in prompt_tokens)
        assert max_prompt_len <= self.args.max_seq_len, f"prompt length must be less than or equal to {self.args.max_seq_len}"
        total_len = min(self.args.max_seq_len, max_gen_len + max_prompt_len)
        pad_id = self.tokenizer.pad_id()
        tokens = torch.full((batch_size, total_len), pad_id, dtype=torch.long, device=device)
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device=device)

        # Generate text based on the prompts
        eos_reached = torch.tensor([False] * batch_size, device=device)
        prompt_tokens_mask = tokens != pad_id
        cur_iterator = tqdm(range(1, total_len), desc="Generating tokens")
        for cur_pos in cur_iterator:
            with torch.no_grad():
                logits = self.model.forward(tokens[:, cur_pos-1:cur_pos], cur_pos)
            if temperature > 0:
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                next_token = self._sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits[:, -1], dim=-1)

            # Update output tokens and check for end-of-sentence
            next_token = next_token.reshape(-1)
            next_token = torch.where(prompt_tokens_mask[:, cur_pos], tokens[:, cur_pos], next_token)
            tokens[:, cur_pos] = next_token
            eos_reached |= (~prompt_tokens_mask[:, cur_pos]) & (next_token == self.tokenizer.eos_id)
            if all(eos_reached):
                break

        # Decode tokens to generate output text
        out_tokens = []
        out_text = []
        for prompt_index, current_prompt_tokens in enumerate(tokens.tolist()):
            if self.tokenizer.eos_id in current_prompt_tokens:
                eos_idx = current_prompt_tokens.index(self.tokenizer.eos_id)
                current_prompt_tokens = current_prompt_tokens[:eos_idx]
            out_tokens.append(current_prompt_tokens)
            out_text.append(self.tokenizer.decode(current_prompt_tokens))

        return (out_tokens, out_text)
        

    def _sample_top_p(self, probs: torch.Tensor, p: float) -> torch.Tensor:
        """
        Sample a token from a distribution using Top-P sampling.

        Args:
            probs (Tensor): A tensor of shape (Batch_Size, vocab_size) containing token probabilities.
            p (float): The cumulative probability threshold for Top-P sampling.

        Returns:
            Tensor: A tensor of shape (Batch_Size, 1) containing the index of the next token.
            
        Notes:
            - The method first sorts the probabilities in descending order and calculates their cumulative sum.
            - It then masks the probabilities that exceed the Top-P threshold and redistributes them.
            - Finally, a token is sampled from the redistributed probabilities.
        """
        
        # Sort probabilities and calculate their cumulative sum
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        
        # Create a mask based on Top-P threshold
        mask = probs_sum - probs_sort > p  # Shifts the cumulative sum by 1 position to the right before masking
        
        # Zero out and redistribute probabilities based on the mask
        probs_sort[mask] = 0.0
        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
        
        # Sample and retrieve the next token
        next_token = torch.multinomial(probs_sort, num_samples=1)
        next_token = torch.gather(probs_idx, -1, next_token)
        
        return next_token



