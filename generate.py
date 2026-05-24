import torch
import glob
import os
import re
import warnings
from model.model import MiniGPT
from model.config import ModelConfig
from utils import TokenizerWrapper

# -------------------- 1. 加载分词器 --------------------
tokenizer = TokenizerWrapper('tokenizer/my_tokenizer.model')
ModelConfig.vocab_size = tokenizer.vocab_size

# -------------------- 2. 找到要加载的模型文件 --------------------
if os.path.exists('best_model.pt'):
    latest_file = 'best_model.pt'
    print("将加载最佳验证损失模型: best_model.pt")
else:
    list_of_files = glob.glob('checkpoint_*.pt')
    if not list_of_files:
        raise FileNotFoundError("未找到任何检查点文件")
    latest_file = max(list_of_files, key=os.path.getctime)
    print(f"将加载检查点: {latest_file}")

# -------------------- 3. 设备设置 --------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# -------------------- 4. 加载模型 --------------------
with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)
    checkpoint = torch.load(latest_file, map_location=device, weights_only=False)

model = MiniGPT(ModelConfig).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"已加载模型，训练步数: {checkpoint.get('step', '未知')}")
print("\n" + "=" * 50)
print("岁 (Sui) 已上线，开始对话吧！（规则记忆模式）")
print("输入 'quit' 或 'exit' 退出")
print("输入 '/clear' 清除记忆")
print("=" * 50 + "\n")

# -------------------- 5. 规则记忆系统 --------------------
memory_bank = {}  # 存储用户告诉岁的所有事实

def extract_fact_from_input(user_input):
    """从用户输入中提取可记忆的事实"""
    # 匹配姓名
    name_match = re.search(r'我叫(\S+)', user_input)
    if name_match:
        memory_bank["名字"] = name_match.group(1)
        return f"(记忆已更新：你叫{name_match.group(1)})"
    
    # 匹配爱好
    like_match = re.search(r'我喜欢(吃?)(\S+)', user_input)
    if like_match:
        thing = like_match.group(2).rstrip('。，！？')
        memory_bank["爱好"] = thing
        return f"(记忆已更新：你喜欢{thing})"
    
    # 匹配“我是你的创造者”
    if '创造者' in user_input:
        memory_bank["角色"] = "创造者"
        return "(记忆已更新：你是我的创造者)"
    
    return None

def answer_from_memory(user_input):
    """检查用户是否在问一个已知事实，如果是，直接返回答案"""
    if any(kw in user_input for kw in ['我叫什么', '我的名字', '我叫啥']):
        if "名字" in memory_bank:
            return f"你叫{memory_bank['名字']}呀！我记得呢。"
    
    if any(kw in user_input for kw in ['我喜欢吃什么', '我喜欢什么', '我的爱好']):
        if "爱好" in memory_bank:
            return f"你喜欢吃{memory_bank['爱好']}，对吧？"

    if any(kw in user_input for kw in ['我是谁', '我的角色', '我是你的什么']):
        if "角色" in memory_bank:
            return f"你是我的{memory_bank['角色']}，我一直记得。"
    
    return None

# -------------------- 6. 对话历史管理 --------------------
history = []
MAX_HISTORY = 3

def build_prompt(user_input):
    """构建包含背景信息的最终提示词"""
    # 如果有记忆的事实，注入上下文
    context = ""
    if memory_bank:
        facts = "，".join([f"{k}是{v}" for k, v in memory_bank.items()])
        context = f"[已知: {facts}]"
    
    # 拼接最近对话
    recent = history[-MAX_HISTORY * 2:] if history else []
    if recent:
        return '\t'.join(recent + [user_input]) if not context else context + '\t' + '\t'.join(recent + [user_input])
    return f"{context}\t{user_input}" if context else user_input

# -------------------- 7. 交互循环 --------------------
while True:
    try:
        user_input = input("你: ").strip()
        if user_input.lower() in ('quit', 'exit', 'q'):
            print("再见！")
            break
        if user_input == '/clear':
            history.clear()
            memory_bank.clear()
            print("[对话历史和记忆已清空]\n")
            continue
        if not user_input:
            continue

        # --- 步骤 A: 先检查是否有规则匹配的答案 ---
        memory_answer = answer_from_memory(user_input)
        if memory_answer:
            print(f"岁: {memory_answer}\n")
            history.append(user_input)
            history.append(memory_answer)
            continue

        # --- 步骤 B: 处理事实输入 ---
        fact_msg = extract_fact_from_input(user_input)
        if fact_msg:
            print(f"岁: {fact_msg}\n")
            history.append(user_input)
            history.append(fact_msg)
            continue

        # --- 步骤 C: 正常对话生成 ---
        prompt = build_prompt(user_input)
        prompt_ids = tokenizer.encode(prompt)
        context_ids = [tokenizer.bos_id] + prompt_ids
        context = torch.tensor([context_ids], dtype=torch.long).to(device)

        with torch.no_grad():
            output = model.generate(
                context, max_new_tokens=80, temperature=0.6, top_p=0.85,
                top_k=None, repetition_penalty=1.1
            )

        generated_ids = output[0].tolist()
        new_ids = generated_ids[len(context_ids):]
        eos_idx = next((i for i, tid in enumerate(new_ids) if tid == tokenizer.eos_id), None)
        if eos_idx is not None:
            new_ids = new_ids[:eos_idx]

        generated_text = tokenizer.decode(new_ids)
        generated_text = generated_text.replace(' ', '')
        generated_text = generated_text.replace('[BOS]', '').replace('[EOS]', '').replace('[PAD]', '').replace('[UNK]', '').replace('⁇', '')
        generated_text = re.sub(r'([！。，？；：、…～!?,.;:])\1{2,}', r'\1\1', generated_text)

        print(f"岁: {generated_text}\n")

        history.append(user_input)
        history.append(generated_text)

    except KeyboardInterrupt:
        print("\n再见！")
        break
    except Exception as e:
        print(f"出错了: {e}")