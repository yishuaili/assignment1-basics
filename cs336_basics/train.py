from dataclasses import dataclass, field, asdict
from transformers import HfArgumentParser
from typing import Optional
import torch
import logging
import time
from cs336_basics.model import TransformerLM, cross_entropy_loss
from cs336_basics.utils.data_load import Dataset
from cs336_basics.utils.io import load_checkpoint
from cs336_basics.optimizer import AdamW, learning_rate_schedule, gradient_clipping


@dataclass
class TrainingConfig:
    # Dataset 
    dataset_name: str
    batch_size: int
    device: Optional[str] = field(default="cuda" if torch.cuda.is_available() else "cpu")
    
    # model parameters
    vocab_size: Optional[int] = field(default=10000)
    context_length: Optional[int] = field(default=256)
    num_layers: Optional[int] = field(default=4)
    d_model: Optional[int] = field(default=512)
    num_heads: Optional[int] = field(default=16)
    d_ff: Optional[int] = field(default=1344)
    rope_theta: Optional[float] = field(default=10000)
    #attn_pdrop: Optional[float] = field(default=0.1)
    #resid_pdrop: Optional[float] = field(default=0.1)
    init_from: str = field(default='scratch')

    # training
    total_iters: Optional[int] = field(default=10000)
    warmup_iters: Optional[int] = field(default=None)
    cooldown_iters: Optional[int] = field(default=None)
    lr_max: Optional[float] = field(default=3e-4)
    lr_min: Optional[float] = field(default=0)
    weight_decay: Optional[float] = field(default=0.001)

    # logging
    wandb_logging: Optional[bool] = field(default=False)
    wandb_project: Optional[str] = field(default=None)
    wandb_run_name: Optional[str] = field(default=None)
    log_interval: Optional[int] = field(default=None)
    eval_interval: Optional[int] = field(default=None)
    eval_iters: Optional[int] = field(default=100)
    
    # ablation studies
    no_rmsnorm: Optional[bool] = field(default=False)
    parallel_layers: Optional[bool] = field(default=False)
    post_norm: Optional[bool] = field(default=False)

    def __post_init__(self):
        if self.warmup_iters is None:
            self.warmup_iters = max(1, int(self.total_iters * 0.01))
        if self.log_interval is None:
            self.log_interval = max(1, int(self.total_iters * 0.001))
        if self.eval_interval is None:
            self.eval_interval = max(1, int(self.total_iters * 0.01))
        if self.wandb_logging:
            assert self.wandb_project is not None, 'wandb_project must be provided if wandb_logging is True'
            assert self.wandb_run_name is not None, 'wandb_run_name must be provided if wandb_logging is True'
        self.ablation = self.no_rmsnorm or self.parallel_layers or self.post_norm

    
# parse config
parser = HfArgumentParser(TrainingConfig)
config = parser.parse_args_into_dataclasses()[0]
if config.wandb_logging:
    import wandb
    wandb.init(project=config.wandb_project, name=config.wandb_run_name)
logging.info(f'Training with config: {asdict(config)}')

# loading the dataset
dataset = Dataset(**asdict(config))

# loading the model
model_config = {
    "vocab_size": config.vocab_size,
    "context_length": config.context_length,
    "d_model": config.d_model,
    "num_layers": config.num_layers,
    "num_heads": config.num_heads,
    "d_ff": config.d_ff,
    "rope_theta": config.rope_theta,
}
model = TransformerLM(**model_config)
model.to(config.device)
if config.init_from != 'scratch':
    ckpt_dir = f'data/out/checkpoints/{config.init_from}'
    iter_num = load_checkpoint(model, optimizer, ckpt_dir)

optimizer = AdamW(model.parameters(), **asdict(config))


def eval():
    total_loss = 0.0
    for _ in range(config.eval_iters):
        x, y = dataset.get_batch('val')
        x, y = x.to(config.device), y.to(config.device)
        with torch.no_grad():
            logits = model(x)
            loss = cross_entropy_loss(logits, y)

iter_num = 0
curr_time = time.time()

# training loop
while iter_num < config.total_iters:
    optimizer.zero_grad()

    x, y = dataset.get_batch('train')
    logits = model(x)
    loss = cross_entropy_loss(logits, y)
    gradient_clipping(model.parameters(), 1.0)
    lr = learning_rate_schedule(iter_num, lr_max=config.lr_max, lr_min=config.lr_min, t_w=config.warmup_iters, t_c=config.cooldown_iters)
    optimizer.set_lr(lr)
    optimizer.step()
    finish_time = time.time()

    # logging
    if iter_num % config.log_interval == 0:
        logging.info(f'Iter: {iter_num}, Train loss: {loss.item():.4f}, LR: {lr:.6f}, Time: {1000*(finish_time - curr_time):.2f}ms')
    # evaluation
    if iter_num % config.eval_interval == 0:
        eval()

    curr_time = finish_time
    iter_num += 1
    
    