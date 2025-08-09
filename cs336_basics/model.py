import torch
import torch.nn as nn
import numpy as np
import math
from typing import Optional
from einops import rearrange, einsum
import torch.nn.functional as F

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
    

def softmax(x: torch.Tensor, i: int):
    max_v = x.max(dim=i, keepdim=True).values

    x_sub = x - max_v

    exp_x = torch.exp(x_sub)
    result = exp_x / exp_x.sum(dim=i, keepdim=True)
    return result


def scaled_dot_product_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask):
    d_k = q.shape[-1]
    q_k = einsum(q, k, "... q d, ... k d -> ... q k")
    scaled_q_k = q_k / np.sqrt(k.size(-1))
    scaled_q_k = scaled_q_k.masked_fill(~mask, float('-inf'))
    q_k_s = F.softmax(scaled_q_k, dim=-1)
    v_o = einsum(q_k_s, v, "... q k, ... k d -> ... q d")
    return v_o

    # # Compute the dot product of Q and K
    # dot_product = torch.matmul(q, k.transpose(-1, -2))
    
    # # Scale the dot product
    # scaled_dot_product = dot_product / np.sqrt(k.size(-1))
    
    # # Apply the mask
    # if mask is not None:
    #     scaled_dot_product = scaled_dot_product.masked_fill(mask, float('-inf'))

    # # Compute the attention weights
    # attention_weights = F.softmax(scaled_dot_product, dim=-1)

    
    # # Compute the weighted sum of the values
    # output = torch.matmul(attention_weights, v)
    
    # return output

def scaled_dot_product_attention_2(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask):
    # Compute the dot product of Q and K
    dot_product = torch.matmul(q, k.transpose(-1, -2))
    
    # Scale the dot product
    scaled_dot_product = dot_product / np.sqrt(k.size(-1))
    
    # Apply the mask
    if mask is not None:
        scaled_dot_product = scaled_dot_product.masked_fill(mask, float('-inf'))

    # Compute the attention weights
    attention_weights = F.softmax(scaled_dot_product, dim=-1)

    
    # Compute the weighted sum of the values
    output = torch.matmul(attention_weights, v)
    
    return output



class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.d_model = d_model

        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads
        self.num_heads = num_heads
        # self.rope = RotaryPositionalEmbedding(theta=10000, d_k=self.d_k, max_seq_len=1024, device=device)

        self.w_q = nn.Linear(self.d_model, self.d_model, bias=False, **factory_kwargs)
        self.w_k = nn.Linear(self.d_model, self.d_model, bias=False, **factory_kwargs)
        self.w_v = nn.Linear(self.d_model, self.d_model, bias=False, **factory_kwargs)
        self.w_o = nn.Linear(self.d_model, self.d_model, bias=False, **factory_kwargs)

        # causal_mask_base = torch.tril(torch.ones(self.d_model, self.d_model, dtype=torch.bool), diagonal=0)
        # self.register_buffer('causal_mask', causal_mask_base, persistent=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        print(f"w_q values: {self.w_q.weight}")

        print(f"Input tensor x shape: {x.shape}")

        Q = self.w_q(x)
        K = self.w_k(x)
        V = self.w_v(x)
        
        print(f"Q, K, V shapes after linear projection: {Q.shape}, {K.shape}, {V.shape}")


        Q = rearrange(Q, "... s (h d) -> ... h s d", h=self.num_heads)
        K = rearrange(K, "... s (h d) -> ... h s d", h=self.num_heads)
        V = rearrange(V, "... s (h d) -> ... h s d", h=self.num_heads)

        print(f"Q, K, V shapes after rearrange: {Q.shape}, {K.shape}, {V.shape}")

        # Add rope
        # token_positions = torch.arange(x.shape[-2])
        # q_p = self.rope.forward(Q, token_positions)
        # k_p = self.rope.forward(K, token_positions)

        # Scale dot product attention
        causal_mask = torch.tril(torch.ones(x.shape[-2], x.shape[-2], dtype=torch.bool), diagonal=0)
        print(f"Causal mask shape: {causal_mask.shape}")
        #out = scaled_dot_product_attention(Q, K, V, causal_mask)
        out = scaled_dot_product_attention(Q, K, V, causal_mask)
        print(f"Output of scaled_dot_product_attention shape: {out.shape}")

        if torch.any(torch.isnan(out)):
            print("Warning: The out tensor contains NaNs.")
        else:
            print("No NaNs found in out.")
        # Combine heads
        out = rearrange(out, "... h s d -> ... s (h d)")
        print(f"Output after combining heads shape: {out.shape}")
        final_output = self.w_o(out)
        print(f"Final output shape: {final_output.shape}")

        return final_output
    
# class TMultiHeadAttention(nn.Module):
#     def __init__(self, d_model: int, num_heads: int):
#         super(TMultiHeadAttention, self).__init__()
#         self.d_model = d_model
#         self.num_heads = num_heads
#         self.d_k = d_model // num_heads
#         self.q_proj = nn.Linear(d_model, d_model, bias=False)
#         self.k_proj = nn.Linear(d_model, d_model, bias=False)
#         self.v_proj = nn.Linear(d_model, d_model, bias=False)
#         self.output_proj = nn.Linear(d_model, d_model, bias=False)

#     def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
#         B, T, _ = x.size()
#         q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        
#         q = q.view(B, T, self.num_heads, self.d_k).transpose(1, 2)
#         k = k.view(B, T, self.num_heads, self.d_k).transpose(1, 2)
#         v = v.view(B, T, self.num_heads, self.d_k).transpose(1, 2)

#         mask = torch.triu(torch.ones(T, T), diagonal=1).bool().to(x.device)
#         x = scaled_dot_product_attention(q, k, v, mask=mask)
#         x = x.transpose(1, 2)
#         x = x.contiguous().view(B, T, self.d_model)
#         x = x.view(B, T, self.d_model)
#         x = self.output_proj(x)
#         return x



class MultiHeadAttentionRope(nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int, theta: float, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.d_model = d_model

        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads
        self.num_heads = num_heads
        self.rope = RotaryPositionalEmbedding(theta=theta, d_k=self.d_k, max_seq_len=max_seq_len, device=device)

        self.w_q = nn.Linear(self.d_model, self.d_model, bias=False, **factory_kwargs)
        self.w_k = nn.Linear(self.d_model, self.d_model, bias=False, **factory_kwargs)
        self.w_v = nn.Linear(self.d_model, self.d_model, bias=False, **factory_kwargs)
        self.w_o = nn.Linear(self.d_model, self.d_model, bias=False, **factory_kwargs)

        # causal_mask_base = torch.tril(torch.ones(self.d_model, self.d_model, dtype=torch.bool), diagonal=0)
        # self.register_buffer('causal_mask', causal_mask_base, persistent=False)
        
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        print(f"w_q values: {self.w_q.weight}")

        print(f"Input tensor x shape: {x.shape}")

        Q = self.w_q(x)
        K = self.w_k(x)
        V = self.w_v(x)
        
        print(f"Q, K, V shapes after linear projection: {Q.shape}, {K.shape}, {V.shape}")


        Q = rearrange(Q, "... s (h d) -> ... h s d", h=self.num_heads)
        K = rearrange(K, "... s (h d) -> ... h s d", h=self.num_heads)
        V = rearrange(V, "... s (h d) -> ... h s d", h=self.num_heads)

        print(f"Q, K, V shapes after rearrange: {Q.shape}, {K.shape}, {V.shape}")

        # Add rope
        # token_positions = torch.arange(x.shape[-2])
        q_p = self.rope.forward(Q, token_positions)
        k_p = self.rope.forward(K, token_positions)

        # Scale dot product attention
        causal_mask = torch.tril(torch.ones(x.shape[-2], x.shape[-2], dtype=torch.bool), diagonal=0)
        print(f"Causal mask shape: {causal_mask.shape}")
        #out = scaled_dot_product_attention(Q, K, V, causal_mask)
        out = scaled_dot_product_attention(q_p, k_p, V, causal_mask)
        print(f"Output of scaled_dot_product_attention shape: {out.shape}")

        if torch.any(torch.isnan(out)):
            print("Warning: The out tensor contains NaNs.")
        else:
            print("No NaNs found in out.")
        # Combine heads
        out = rearrange(out, "... h s d -> ... s (h d)")
        print(f"Output after combining heads shape: {out.shape}")
        final_output = self.w_o(out)
        print(f"Final output shape: {final_output.shape}")

        return final_output