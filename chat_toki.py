import torch
import sys
import warnings

warnings.filterwarnings("ignore")

# --- 1. PATCH DE SECOURS (A EXÉCUTER AVANT TOUT IMPORT) ---
# On simule la présence de int1 à int7 pour satisfaire torchao
for i in range(1, 8):
    if not hasattr(torch, f"int{i}"):
        setattr(torch, f"int{i}", torch.int8)
# On force la mise à jour globale du module pour tout l'interpréteur
sys.modules['torch'] = torch

# Maintenant que le terrain est prêt, on importe les bibliothèques lourdes
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
except (AttributeError, ModuleNotFoundError) as e:
    print(f"Erreur d'importation : {e}")
    print("Tentative de contournement en cours...")

# --- 2. CONFIGURATION ---
model_id = "unsloth/llama-3-8b-bnb-4bit"
lora_path = "toki_lora" 

print("📦 Chargement de l'IA (Patientez environ 15s)...")

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Appliquer ton entraînement
model = PeftModel.from_pretrained(model, lora_path)
model.eval()

# --- 3. FONCTION DE TRADUCTION ---
def traduire(text):
    prompt = f"### Instruction:\nTraduire en Toki Pona\n\n### Input:\n{text}\n\n### Response:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=32,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False,
            # On enlève les valeurs par défaut qui causent les warnings :
            temperature=None,
            top_p=None
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "### Response:" in decoded:
        return decoded.split("### Response:")[1].strip()
    return decoded

# --- 4. BOUCLE ---
print("\n--- TRADUCTEUR TOKI PONA ACTIF ---")
while True:
    phrase = input("Français > ")
    if phrase.lower() in ["exit", "quit"]: break
    print(f"Toki Pona > {traduire(phrase)}\n")

