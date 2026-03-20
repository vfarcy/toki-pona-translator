import sys
import torch

# --- 1. PATCH DE SURVIE ULTRA-PRÉCOCE ---
# On injecte les types manquants AVANT que Unsloth ne commence à scanner Torch
for i in range(1, 8):
    setattr(torch, f"int{i}", torch.int8)

# On s'assure que même les bibliothèques tierces (torchao) voient ce changement
sys.modules['torch'] = torch 

from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

# --- 2. CONFIGURATION ---
max_seq_length = 2048
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-3-8b-bnb-4bit",
    max_seq_length = max_seq_length,
    load_in_4bit = True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 16, 
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
)

# --- 3. FORMATAGE ET DONNÉES ---
def formatting_prompts_func(examples):
    texts = []
    for messages in examples["messages"]:
        parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                parts.append(f"### System:\n{content}")
            elif role == "user":
                parts.append(f"### User:\n{content}")
            elif role == "assistant":
                parts.append(f"### Assistant:\n{content}")
        texts.append("\n\n".join(parts) + tokenizer.eos_token)
    return { "text" : texts, }

dataset = load_dataset("json", data_files="train_michel.jsonl", split="train")
dataset = dataset.map(formatting_prompts_func, batched=True)

# Petit dataset tres uniforme: on garde une validation pour surveiller l'overfit.
split_dataset = dataset.train_test_split(test_size=0.1, seed=42, shuffle=True)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]

# --- 4. ENTRAÎNEMENT ---
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_dataset,
    eval_dataset = eval_dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        num_train_epochs = 3,
        warmup_ratio = 0.05,
        learning_rate = 5e-5,
        weight_decay = 0.01,
        lr_scheduler_type = "cosine",
        max_grad_norm = 0.3,
        fp16 = False,
        bf16 = True,
        logging_steps = 10,
        eval_strategy = "steps",
        eval_steps = 50,
        save_strategy = "steps",
        save_steps = 50,
        save_total_limit = 2,
        output_dir = "outputs",
        report_to = "none",
    ),
)

print("🚀 Lancement de l'entraînement final...")
trainer.train()

# --- 5. SAUVEGARDE ---
print("💾 Sauvegarde du modèle dans le dossier 'toki_lora'...")
model.save_pretrained("toki_lora")
tokenizer.save_pretrained("toki_lora")

print("🎉 C'est fini ! Ton IA est prête et rangée sur ton disque.")

