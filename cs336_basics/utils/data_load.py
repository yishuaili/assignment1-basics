import torch
import numpy as np

def get_batch(dataset: np.ndarray, batch_size: int, context_length: int, device: str):
    start_indices = np.random.randint(0, len(dataset) - context_length, size=batch_size)
    x = torch.stack([torch.from_numpy(dataset[s : s + context_length]) for s in start_indices])
    y = torch.stack([torch.from_numpy(dataset[s + 1 : s + context_length + 1]) for s in start_indices])
    return x.to(device), y.to(device)

    