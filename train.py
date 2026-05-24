import os
import math
import torch
import numpy as np
from tqdm import tqdm
from model.model import MiniGPT
from model.config import ModelConfig, TrainConfig
from utils import TokenizerWrapper, prepare_data, load_data, get_batch, split_data

# -------------------- 1. 加载分词器 --------------------
tokenizer = TokenizerWrapper('tokenizer/my_tokenizer.model')
ModelConfig.vocab_size = tokenizer.vocab_size
print(f"Vocab size: {ModelConfig.vocab_size}")

# -------------------- 2. 准备数据 --------------------
bin_file = 'data/processed/train.bin'
if not os.path.exists(bin_file):
    os.makedirs('data/processed', exist_ok=True)
    prepare_data('data/processed/mixed_skypile_lccc.txt', tokenizer, bin_file)

train_data = load_data(bin_file)
train_ids, val_ids = split_data(train_data, split=0.95)
print(f"训练集 tokens: {len(train_ids)}, 验证集 tokens: {len(val_ids)}")

# -------------------- 3. 初始化模型、优化器 --------------------
model = MiniGPT(ModelConfig).to(TrainConfig.device)
optimizer = torch.optim.AdamW(model.parameters(), lr=TrainConfig.learning_rate, weight_decay=0.1)

# -------------------- 从检查点恢复训练 --------------------
import glob
# 手动指定要恢复的检查点
latest_checkpoint = 'checkpoint_20000.pt'  # 改成你想恢复的那个
if os.path.exists(latest_checkpoint):
    print(f"发现检查点: {latest_checkpoint}")
    checkpoint = torch.load(latest_checkpoint, map_location=TrainConfig.device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_step = checkpoint['step']
    best_val_loss = checkpoint.get('val_loss', float('inf'))
    no_improve = 0   # ← 关键！重置早停计数器
    print(f"已从 step {start_step} 恢复训练，当前最佳验证损失: {best_val_loss:.4f}")
else:
    start_step = 0
    best_val_loss = float('inf')
    no_improve = 0   # ← 从头训练时也确保初始化为 0
    print("未发现检查点，从头开始训练。")

no_improve = 0   # 重置早停计数器

# -------------------- 4. 学习率调度（预热+余弦衰减） --------------------
def get_lr(step, warmup_steps, max_steps, max_lr):
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    if step > max_steps:
        return 0.0
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return max_lr * coeff

# -------------------- 5. 混合精度训练 --------------------
scaler = torch.amp.GradScaler('cuda', enabled=(TrainConfig.device == 'cuda'))

# -------------------- 6. 早停相关变量 --------------------
best_val_loss = float('inf')
patience = 25               # 连续 25 个验证周期不改善就停止
no_improve = 0
early_stop = False

# -------------------- 7. 训练循环 --------------------
model.train()
pbar = tqdm(range(start_step + 1, TrainConfig.max_steps + 1), desc="Training")
for step in pbar:
    if early_stop:
        break

    # 调整学习率
    lr = get_lr(step, TrainConfig.warmup_steps, TrainConfig.max_steps, TrainConfig.learning_rate)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # 梯度累积
    for micro_step in range(TrainConfig.gradient_accumulation_steps):
        xb, yb = get_batch(train_ids, ModelConfig.block_size, TrainConfig.batch_size, TrainConfig.device)
        with torch.amp.autocast('cuda', enabled=(TrainConfig.device == 'cuda')):
            logits, loss = model(xb, yb)
            loss = loss / TrainConfig.gradient_accumulation_steps
        scaler.scale(loss).backward()

    # 梯度裁剪与更新
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), TrainConfig.max_grad_norm)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    train_loss = loss.item() * TrainConfig.gradient_accumulation_steps

    # -------------------- 验证与早停判断 --------------------
    if step % TrainConfig.eval_interval == 0:
        model.eval()
        with torch.no_grad():
            xb_val, yb_val = get_batch(val_ids, ModelConfig.block_size, TrainConfig.batch_size, TrainConfig.device)
            with torch.amp.autocast('cuda', enabled=(TrainConfig.device == 'cuda')):
                _, val_loss = model(xb_val, yb_val)
        model.train()

        pbar.set_description(
            f"Step {step:05d} | Train Loss {train_loss:.4f} | Val Loss {val_loss.item():.4f} | LR {lr:.2e}"
        )

        # 判断验证损失是否改善
        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            no_improve = 0
            # 保存最佳模型（只存模型权重，避免自定义类问题）
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': best_val_loss,
                'config': {
                    'vocab_size': ModelConfig.vocab_size,
                    'n_embd': ModelConfig.n_embd,
                    'n_head': ModelConfig.n_head,
                    'n_layer': ModelConfig.n_layer,
                    'block_size': ModelConfig.block_size,
                    'dropout': ModelConfig.dropout,
                }
            }, 'best_model.pt')
            print(f"  --> 新最佳模型已保存 (Val Loss: {best_val_loss:.4f})")
        else:
            no_improve += 1
            print(f"  --> 验证损失未改善 ({no_improve}/{patience})")
            if no_improve >= patience:
                print(f"验证损失连续 {patience} 次未下降，在 step {step} 提前停止训练。")
                early_stop = True
                break

    # 定期保存检查点
    if step % TrainConfig.save_interval == 0:
        torch.save({
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss.item() if step % TrainConfig.eval_interval == 0 else None,
            'config': {
                'vocab_size': ModelConfig.vocab_size,
                'n_embd': ModelConfig.n_embd,
                'n_head': ModelConfig.n_head,
                'n_layer': ModelConfig.n_layer,
                'block_size': ModelConfig.block_size,
                'dropout': ModelConfig.dropout,
            }
        }, f'checkpoint_{step}.pt')
        print(f"  --> 检查点已保存至 checkpoint_{step}.pt")

print(f"训练完成！最佳验证损失: {best_val_loss:.4f}")