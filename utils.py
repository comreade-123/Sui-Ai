import os
import torch
import numpy as np
import sentencepiece as spm


class TokenizerWrapper:
    def __init__(self, model_path):
        self.sp = spm.SentencePieceProcessor(model_file=model_path)
        self.vocab_size = self.sp.get_piece_size()
        self.pad_id = self.sp.pad_id()
        self.bos_id = self.sp.bos_id()
        self.eos_id = self.sp.eos_id()

    def encode(self, text):
        return self.sp.encode(text, out_type=int)

    def decode(self, ids):
        return self.sp.decode(ids)


def prepare_data(text_file, tokenizer, out_bin_file):
    """逐行编码文本，并在每段对话前后自动添加 bos/eos token，直接写入二进制文件"""
    print(f"正在编码 {text_file} ...")
    with open(text_file, 'r', encoding='utf-8') as fin:
        lines = [line.strip() for line in fin if line.strip()]

    total_lines = len(lines)
    print(f"共 {total_lines} 行待编码")
    token_count = 0

    # 二进制写入，追加模式
    with open(out_bin_file, 'wb') as bf:
        for i, line in enumerate(lines):
            ids = tokenizer.encode(line)
            # 在序列开头加 bos_id，结尾加 eos_id
            ids = [tokenizer.bos_id] + ids + [tokenizer.eos_id]
            np.array(ids, dtype=np.uint16).tofile(bf)
            token_count += len(ids)

            if (i + 1) % 500000 == 0:
                print(f"  已处理 {i+1}/{total_lines} 行...")

    print(f"编码完成！总 token 数: {token_count}, 保存至 {out_bin_file}")
    return None


def load_data(bin_file):
    """从二进制文件加载 token 数组（numpy）"""
    return np.fromfile(bin_file, dtype=np.uint16)


def get_batch(data, block_size, batch_size, device):
    """随机采样输入 x 和目标 y"""
    data_len = len(data)
    ix = torch.randint(0, data_len - block_size, (batch_size,))
    x = torch.stack([torch.tensor(data[i:i+block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.tensor(data[i+1:i+block_size+1].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)


def split_data(data, split=0.95):
    """将 token 列表切分为训练集和验证集"""
    n = int(len(data) * split)
    return data[:n], data[n:]