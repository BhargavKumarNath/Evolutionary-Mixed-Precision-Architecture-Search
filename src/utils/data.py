import torch
from datasets import load_dataset
from transformers import AutoTokenizer

def get_calibration_dataset(tokenizer_name: str, seq_len: int = 2048, nsamples: int = 128, seed: int = 42):
    """Loads Wikitext2 and prepares a list of input_idx for calibration.
    Returns: List [torch.Tensor]"""
    try:
        data = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    except Exception:
        print("Warning: Could not load Wikitext2. Using dummy data.")
        return [torch.randint(0, 1000, (1, seq_len)) for _ in range(nsamples)]
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.model_max_length = int(1e30)

    # tokenize the entire dataset into one long stream
    encodings = tokenizer("\n\n".join(data['text']), return_tensors='pt', add_special_tokens=False)

    # Chunk into nsamples of seq_len
    num_tokens = encodings.input_ids.size(1)
    dataset = []

    # Use a datetministic generator for reproducibility
    rng = torch.Generator().manual_seed(seed)

    for _ in range(nsamples):
        i = torch.randint(0, num_tokens - seq_len, (1,), generator=rng).item()
        chunk = encodings.input_ids[:, i : i + seq_len] 
        dataset.append(chunk)
    
    return dataset
