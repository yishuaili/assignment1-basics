import torch
import numpy as np


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=None, weight_decay=0.001, betas=(0.9, 0.999), eps=1e-8, **kwargs):

        defaults = dict(lr=lr, weight_decay=weight_decay, betas=betas, eps=eps)
        super().__init__(params, defaults)

    def set_lr(self, lr):
        for group in self.param_groups:
            group['lr'] = lr
        
    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            weight_decay = group['weight_decay']
            beta1, beta2 = group['betas']
            eps = group['eps']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('AdamW does not support sparse gradients, please consider SparseAdam instead')
                
                state = self.state[p]
                if len(state) == 0:
                    state['t'] = 0
                    state['m'] = torch.zeros_like(p.data)
                    state['v'] = torch.zeros_like(p.data)

                t = state['t'] + 1
                m, v = state['m'], state['v']
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * grad * grad
                lr_t = lr * (1 - beta2 ** t) ** 0.5 / (1 - beta1 ** t)
                p.data -= (lr_t * m / (v ** 0.5 + eps))
                p.data -= lr * weight_decay * p.data

                state['m'] = m
                state['v'] = v
                state['t'] = t

                
            
def learning_rate_schedule(t, lr_max, lr_min, t_w, t_c):
    if t< t_w:
        return lr_max * (t/t_w)
    elif t_w <= t and t <= t_c:
        return lr_min + 0.5 * (1 + np.cos((t - t_w) / (t_c - t_w) * np.pi)) * (lr_max - lr_min)
    else:
        return lr_min
    
    