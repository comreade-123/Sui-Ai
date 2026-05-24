import torch

class ModelConfig:
    vocab_size = None
    n_embd = 512
    n_head = 8
    n_layer = 8
    block_size = 256
    dropout = 0.1

class TrainConfig:
    batch_size = 8
    learning_rate = 5e-5
    max_steps = 60000
    eval_interval = 500
    save_interval = 2000
    gradient_accumulation_steps = 2
    warmup_steps = 1000
    max_grad_norm = 1.0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'