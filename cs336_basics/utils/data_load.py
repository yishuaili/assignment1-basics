import torch
import numpy as np

def get_batch(dataset: np.ndarray, batch_size: int, context_length: int, device: str):
    start_indices = np.random.randint(0, len(dataset) - context_length, size=batch_size)
    x = torch.stack([torch.from_numpy(dataset[s : s + context_length]) for s in start_indices])
    y = torch.stack([torch.from_numpy(dataset[s + 1 : s + context_length + 1]) for s in start_indices])
    return x.to(device), y.to(device)

class Dataset:
    def __init__(self, dataset_name: str, context_length: int, batch_size: int, device: str, **kwargs):
        dataset_path = f"data/{dataset_name}"
        self.train_data = np.memmap(f"{dataset_path}/train.bin", dtype=np.uint16, mode="r").astype(np.int64)
        self.val_data = np.memmap(f"{dataset_path}/val.bin", dtype=np.uint16, mode="r").astype(np.int64)
        self.context_length = context_length
        self.batch_size = batch_size
        self.device = device

    def get_batch(self, split: str):
        data = self.train_data if split == "train" else self.val_data
        return get_batch(data, self.batch_size, self.context_length, self.device)
