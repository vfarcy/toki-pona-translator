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

print("📦 Chargement du professeur Michel Thomas virtuel...")

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model = PeftModel.from_pretrained(model, lora_path)
model.eval()

SYSTEM = (
    "Tu enseignes le Toki Pona selon la méthode Michel Thomas : "
    "progression orale, sans stress, construction active, guidance douce. "
    "Tu guides l'élève étape par étape, tu corriges avec bienveillance, "
    "tu encourages la construction de phrases par l'élève lui-même."
)

# --- 3. FONCTION DE GÉNÉRATION AVEC HISTORIQUE ---
def construire_prompt(historique: list[dict]) -> str:
    """Construit le prompt complet à partir de l'historique de la conversation."""
    parties = [f"### System:\n{SYSTEM}"]
    for msg in historique:
        if msg["role"] == "user":
            parties.append(f"### User:\n{msg['content']}")
        elif msg["role"] == "assistant":
            parties.append(f"### Assistant:\n{msg['content']}")
    parties.append("### Assistant:\n")
    return "\n\n".join(parties)

def repondre(historique: list[dict]) -> str:
    prompt = construire_prompt(historique)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2,
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extraire uniquement la dernière réponse de l'assistant
    if "### Assistant:" in decoded:
        reponse = decoded.split("### Assistant:")[-1].strip()
        # Couper si le modèle commence à halluciner un nouveau tour
        for stop in ["### User:", "### System:"]:
            if stop in reponse:
                reponse = reponse.split(stop)[0].strip()
        return reponse
    return decoded.strip()

# --- 4. BOUCLE DE CONVERSATION ---
print("\n" + "="*60)
print("  Bienvenue dans votre cours de Toki Pona — méthode Michel Thomas")
print("  Tapez 'quitter' ou 'exit' pour terminer la session.")
print("="*60 + "\n")

historique = []

# Message d'ouverture du professeur
ouverture = {"role": "user", "content": "Bonjour, je voudrais apprendre le Toki Pona."}
historique.append(ouverture)
print("Vous : Bonjour, je voudrais apprendre le Toki Pona.")

reponse_initiale = repondre(historique)
historique.append({"role": "assistant", "content": reponse_initiale})
print(f"\nMichel Thomas : {reponse_initiale}\n")

while True:
    try:
        user_input = input("Vous : ").strip()
    except (KeyboardInterrupt, EOFError):
        print("\n\nAu revoir !")
        break

    if not user_input:
        continue

    if user_input.lower() in ("quitter", "exit", "quit"):
        print("\nMerci pour cette leçon. À bientôt !")
        break

    historique.append({"role": "user", "content": user_input})
    reponse = repondre(historique)
    historique.append({"role": "assistant", "content": reponse})

    print(f"\nMichel Thomas : {reponse}\n")
