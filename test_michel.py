import torch
import sys
import warnings

warnings.filterwarnings("ignore")

# --- 1. PATCH TORCH ---
for i in range(1, 8):
    if not hasattr(torch, f"int{i}"):
        setattr(torch, f"int{i}", torch.int8)
sys.modules['torch'] = torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# --- 2. CHARGEMENT DU MODÈLE ---
model_id = "unsloth/llama-3-8b-bnb-4bit"
lora_path = "toki_lora"

print("📦 Chargement du modèle fine-tuné (méthode Michel Thomas)...")

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model = PeftModel.from_pretrained(model, lora_path)
model.eval()
print("✅ Modèle prêt.\n")

# --- 3. FONCTION DE GÉNÉRATION ---
def repondre(system: str, user: str) -> str:
    prompt = (
        f"### System:\n{system}\n\n"
        f"### User:\n{user}\n\n"
        f"### Assistant:\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False,
            temperature=None,
            top_p=None,
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "### Assistant:" in decoded:
        return decoded.split("### Assistant:")[-1].strip()
    return decoded.strip()

# --- 4. TESTS ---
SYSTEM = (
    "Tu enseignes le Toki Pona selon la méthode Michel Thomas : "
    "progression orale, sans stress, construction active, guidance douce."
)

questions = [
    "Exercice de la leçon Introduction. Peux-tu m'aider à construire une phrase en toki pona ?",
    "Comment dit-on 'je mange' en toki pona ?",
    "Quelle est la structure de base d'une phrase en toki pona ?",
    "Exercice de la leçon Basic verbs. Peux-tu m'aider à construire une phrase en toki pona ?",
    "Comment utilise-t-on le marqueur 'e' ?",
]

for i, question in enumerate(questions, 1):
    print(f"[{i}] USER : {question}")
    reponse = repondre(SYSTEM, question)
    print(f"    ASSISTANT : {reponse}")
    print()
