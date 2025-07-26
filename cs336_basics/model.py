import torch
import torch.nn as nn
import numpy as np
import math
from typing import Optional
from einops import rearrange

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        self.reset_parameters()
        
    def reset_parameters(self):
        # --- Weight Initialization (as per image) ---
        d_in = self.in_features
        d_out = self.out_features

        # Calculate variance: sigma^2 = 2 / (d_in + d_out)
        variance = 2.0 / (d_in + d_out)
        # Calculate standard deviation: sigma = sqrt(variance)
        std = math.sqrt(variance)

        # Initialize weights using the custom truncated normal distribution
        # Mean is 0, standard deviation is 'std', and truncation is at 3 * std
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3.0 * std, b=3.0 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.T
    

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.weight = nn.Parameter(torch.empty((num_embeddings, embedding_dim), **factory_kwargs))
        self.reset_parameters()


    def reset_parameters(self):
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1, a=-3.0, b=3.0)


    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]
    

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, **factory_kwargs))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * x / (x.pow(2).mean(dim=-1, keepdim=True) + self.eps).sqrt()


class PWFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()

        # d_ff_raw = int(d_model * (8 / 3))
        # # Round up to the nearest multiple of 64
        # self.d_ff = (d_ff_raw + 63) // 64 * 64
        self.d_ff = d_ff

        self.w_1 = nn.Parameter(torch.ones((self.d_ff, d_model), **factory_kwargs))
        self.w_2 = nn.Parameter(torch.ones((d_model, self.d_ff), **factory_kwargs))
        self.w_3 = nn.Parameter(torch.ones((self.d_ff, d_model), **factory_kwargs))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_1_x = x @ self.w_1.T
        w_3_x = x @ self.w_3.T
        silu = w_1_x * torch.sigmoid(w_1_x)
        silu_w_3 = silu * w_3_x
        return silu_w_3 @ self.w_2.T
    

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        factory_kwargs = {'device': device}
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        inv_freq = 1.0 / (self.theta ** (torch.arange(0, self.d_k, 2) / self.d_k))

        # Precompute cos and sin values for all possible positions and frequencies
        # positions = [0, 1, 2, ..., max_seq_len-1]
        positions = torch.arange(max_seq_len)

        # angles = positions * inv_freq_k (broadcasting happens)
        # Shape: (max_seq_len, d_k/2)
        angles = rearrange(positions, 's -> s 1') * rearrange(inv_freq, 'k -> 1 k')

        # Store cos and sin values as non-persistent buffers
        # cos_cached[i, k] will be cos(theta_i,k)
        # sin_cached[i, k] will be sin(theta_i,k)
        self.register_buffer('cos_cached', torch.cos(angles), persistent=False)
        self.register_buffer('sin_cached', torch.sin(angles), persistent=False)


    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # Ensure token_positions are on the same device as cached buffers
        # token_positions = token_positions.to(self.device)


        seq_cos = self.cos_cached[token_positions]
        seq_sin = self.sin_cached[token_positions]

        x_re = rearrange(x, '... s (d c) -> ... s d c', c=2)
        
        x0 = x_re[..., 0]
        x1 = x_re[..., 1]

        new_x0 = x0 * seq_cos - x1 * seq_sin
        new_x1 = x0 * seq_sin + x1 * seq_cos

        # combined = rearrange([new_x0, new_x1], '... d -> ... d')
        combined = torch.stack((new_x0, new_x1), dim=-1)
        back_to_shape = rearrange(combined, '... d n -> ... (d n)')
        return back_to_shape