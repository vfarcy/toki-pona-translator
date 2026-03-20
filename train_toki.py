import sys
import torch

# --- 1. PATCH DE SURVIE ULTRA-PRÉCOCE ---
# On injecte les types manquants AVANT que Unsloth ne commence à scanner Torch
for i in range(1, 8):
    if not hasattr(torch, f"int{i}"):
        setattr(torch, f"int{i}", torch.int8)

# On s'assure que même les bibliothèques tierces (torchao) voient ce changement
sys.modules['torch'] = torch 

from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments


def resolve_precision_flags():
    if not torch.cuda.is_available():
        return False, False
    if torch.cuda.is_bf16_supported():
        return True, False
    return False, True

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
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        text = f"### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{output}"
        texts.append(text)
    return { "text" : texts, }

dataset = load_dataset("json", data_files="dataset.jsonl", split="train")
dataset = dataset.map(formatting_prompts_func, batched = True)

use_bf16, use_fp16 = resolve_precision_flags()
print(f"Precision activee: bf16={use_bf16}, fp16={use_fp16}")

# --- 4. ENTRAÎNEMENT ---
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 300,
        learning_rate = 2e-4,
        fp16 = use_fp16,
        bf16 = use_bf16,
        logging_steps = 1,
        output_dir = "outputs",
        save_strategy = "steps",
        save_steps = 50,
        save_total_limit = 2,
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

