# Sui Ai
# Sui (岁) - A 47M Chinese Conversational AI Trained from Scratch

**Sui** is a 47-million-parameter Chinese conversational language model trained entirely from scratch. Built on a Decoder-only Transformer architecture, Sui was pretrained on a hybrid corpus of cleaned social-media dialogues (LCCC-base, 80%) and high-quality web text (SkyPile-150B subset, 20%), totaling 1.24 billion tokens. The entire training process was completed on a single NVIDIA GeForce RTX 4060 (8 GB VRAM) in under 24 hours.

Sui supports interactive multi-turn conversation, rule-based user memory (name, preferences, etc.), and controllable generation via temperature, top-k, top-p sampling, and repetition penalty.

---

## ✨ Features

- 🧠 **47M parameters**, Decoder-only Transformer (GPT-2 style), trained from random initialization
- 🗣️ **Fluent Chinese daily conversation** with basic knowledge recall
- 🧠 **Rule-based memory**: remembers user-provided facts (name, likes, etc.)
- ⚙️ **Controllable generation**: temperature, top-k, top-p, repetition penalty
- 🖥️ **Consumer-grade hardware**: single RTX 4060, ~3.5 hours training, <6.5 GB VRAM
- 📦 **Batteries included**: full training, inference, and data preprocessing scripts
- 📄 **Comprehensive technical report**: see [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md)

---

## 📸 Gallery

### Training Progress
![Training Loss Curve](images/training.png)
*Training and validation loss over 27,500 steps. Best validation loss: 4.14 at step ~21,000.*

### Example Conversations
![Example Conversations](images/chat_examples.png)
*Testing basic conversation ability. Prompt: "你好" (Hello) and "你喜欢我吗？" (Do you like me?).*

---

## 📂 Project Structure
```
mini-llm/
├── model/ # Model definition and configuration
│ ├── model.py # MiniGPT implementation
│ └── config.py # Hyperparameters
├── tokenizer/ # Tokenizer training script and model files
│ ├── train_tokenizer.py
│ ├── my_tokenizer.model
│ └── my_tokenizer.vocab
├── utils.py # Data loading, preprocessing, batching
├── train.py # Training script (resume, early stopping, mixed precision)
├── generate.py # Interactive chat (with rule-based memory)
├── requirements.txt # Python dependencies
├── TECHNICAL_REPORT.md # Full technical documentation (bilingual)
├── README.md # This file
├── LICENSE # MIT License (for code)
├── LICENSE-MODEL # CC BY-NC 4.0 (for model weights)
├── .gitignore
├── images/ # Images for documentation
└── scripts/ # (Optional) data preprocessing scripts
```
