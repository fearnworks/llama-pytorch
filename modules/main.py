import torch
from llama import LLaMA
if __name__ == '__main__':
    torch.manual_seed(0)

    allow_cuda = True
    device = 'cuda' if torch.cuda.is_available() and allow_cuda else 'cpu'
    print(device)
    prompts = [
        "Simply put, a linear layer is ",
        "If Microsoft was a Dutch company, it would",
        # Few shot promt
        """Translate English to Dutch:

        sea otter => zeeotter
        peppermint => pepermunt
        plush giraffe => pluchen giraf
        cheese => """, # should be kaas
        # Zero shot prompt
        """Generate a python program that functions as a calculator:
        """
    ]

    model = LLaMA.build(
        checkpoints_dir='llama-2-7b/',
        tokenizer_path='tokenizer.model',
        load_model=True,
        max_seq_len=1024,
        max_batch_size=len(prompts),
        device=device
    )

    out_tokens, out_texts = (model.text_completion(prompts, max_gen_len=64))
    assert len(out_texts) == len(prompts)
    for i in range(len(out_texts)):
        print(f'{out_texts[i]}')
        print('-' * 50)