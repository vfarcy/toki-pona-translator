import torch
import sys
import warnings
from threading import Thread

warnings.filterwarnings("ignore")

# --- 1. PATCH TORCH ---
for i in range(1, 8):
    if not hasattr(torch, f"int{i}"):
        setattr(torch, f"int{i}", torch.int8)
sys.modules['torch'] = torch

from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, logging as hf_logging
from peft import PeftModel

hf_logging.set_verbosity_error()

# --- 2. LEÇONS DISPONIBLES (ordre pédagogique) ---
LECONS = [
    "Introduction",
    "Basic nouns and pronouns",
    "Basic verbs",
    "li structure",
    "e object marker",
    "Adjectives and order",
    "Prepositions",
    "Emotions",
    "Questions",
    "Conversation",
    "Advanced composition",
]

# --- 3. CHARGEMENT DU MODÈLE ---
print("📦 Chargement du professeur Michel Thomas virtuel...")

model_id = "unsloth/llama-3-8b-bnb-4bit"
lora_path = "toki_lora"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model = PeftModel.from_pretrained(model, lora_path)
model.eval()
# Avoid the repetitive warning about max_length + max_new_tokens.
model.generation_config.max_length = None

# --- 4. GÉNÉRATION ---
def construire_prompt(lecon_actuelle: str, historique: list[dict]) -> str:
    system = (
        f"Tu enseignes le Toki Pona selon la méthode Michel Thomas : "
        f"progression orale, sans stress, construction active, guidance douce. "
        f"Nous travaillons actuellement sur la leçon : '{lecon_actuelle}'. "
        f"Tu guides l'élève étape par étape dans cette leçon, tu poses des exercices simples, "
        f"tu corriges avec bienveillance, tu félicites les progrès. "
        f"Tu t'exprimes en français. "
        f"Important: ne mélange pas les noms des leçons entre eux. "
        f"Ne répète pas mot pour mot la même phrase d'accueil à chaque tour. "
        f"Après une réponse d'élève, donne une correction courte puis un mini exercice suivant."
    )
    parties = [f"### System:\n{system}"]
    for msg in historique:
        if msg["role"] == "user":
            parties.append(f"### User:\n{msg['content']}")
        else:
            parties.append(f"### Assistant:\n{msg['content']}")
    parties.append("### Assistant:\n")
    return "\n\n".join(parties)

def repondre(lecon_actuelle: str, historique: list[dict]) -> str:
    prompt = construire_prompt(lecon_actuelle, historique)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )

    generation_kwargs = dict(
        **inputs,
        max_new_tokens=220,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.2,
        streamer=streamer,
    )

    with torch.no_grad():
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()

        print("\nProfesseur : ", end="", flush=True)
        morceaux: list[str] = []
        deja_affiche = 0
        marqueurs_stop = ["### User:", "### System:", "### Assistant:"]

        for morceau in streamer:
            morceaux.append(morceau)
            texte_courant = "".join(morceaux)

            stop_index = len(texte_courant)
            for marqueur in marqueurs_stop:
                idx = texte_courant.find(marqueur)
                if idx != -1 and idx < stop_index:
                    stop_index = idx

            texte_affichable = texte_courant[:stop_index]
            if len(texte_affichable) > deja_affiche:
                print(texte_affichable[deja_affiche:], end="", flush=True)
                deja_affiche = len(texte_affichable)

            if stop_index != len(texte_courant):
                break

        thread.join()
        print("\n")

    reponse = "".join(morceaux)
    for stop in ["### User:", "### System:", "### Assistant:"]:
        if stop in reponse:
            reponse = reponse.split(stop)[0]
    reponse = reponse.strip()
    if reponse == "1":
        reponse = "Très bien. Continuons: propose une phrase simple en toki pona avec 'mi'."
    return reponse

# --- 5. INTERFACE ---
def afficher_lecons():
    print("\n📚 Leçons disponibles :")
    for i, lecon in enumerate(LECONS, 1):
        print(f"  {i:2d}. {lecon}")
    print()

def afficher_aide():
    print("""
Commandes disponibles :
  /lecons       — afficher la liste des leçons
  /lecon N      — passer à la leçon numéro N
  /suivante     — passer à la leçon suivante
  /reset        — recommencer la leçon actuelle depuis le début
  /aide         — afficher cette aide
  /quitter      — terminer la session
""")

# --- 6. BOUCLE PRINCIPALE ---
print("\n" + "="*65)
print("   Cours de Toki Pona — méthode Michel Thomas")
print("="*65)
print("Bienvenue ! Tapez /aide pour voir les commandes disponibles.")

afficher_lecons()

lecon_index = 0
historique: list[dict] = []

# Message d'ouverture automatique
print(f"📖 Leçon en cours : {LECONS[lecon_index]}\n")
msg_ouverture = {
    "role": "user",
    "content": f"Commençons la leçon '{LECONS[lecon_index]}'. Je suis prêt."
}
historique.append(msg_ouverture)
print(f"Vous       : {msg_ouverture['content']}")
rep = repondre(LECONS[lecon_index], historique)
historique.append({"role": "assistant", "content": rep})

while True:
    try:
        user_input = input("Vous       : ").strip()
    except (KeyboardInterrupt, EOFError):
        print("\n\nAu revoir !")
        break

    if not user_input:
        continue

    # --- Commandes ---
    if user_input.startswith("/"):
        cmd = user_input.lower()

        if cmd in ("/quitter", "/exit", "/quit"):
            print("\nAu revoir ! Bonne continuation dans votre apprentissage.")
            break

        elif cmd == "/aide":
            afficher_aide()
            continue

        elif cmd == "/lecons":
            afficher_lecons()
            continue

        elif cmd == "/suivante":
            if lecon_index < len(LECONS) - 1:
                lecon_index += 1
                historique = []
                print(f"\n📖 Passage à la leçon : {LECONS[lecon_index]}\n")
                msg = {"role": "user", "content": f"Commençons la leçon '{LECONS[lecon_index]}'. Je suis prêt."}
                historique.append(msg)
                print(f"Vous       : {msg['content']}")
                rep = repondre(LECONS[lecon_index], historique)
                historique.append({"role": "assistant", "content": rep})
            else:
                print("🎉 Vous avez terminé toutes les leçons ! Félicitations.\n")
            continue

        elif cmd.startswith("/lecon "):
            try:
                n = int(cmd.split()[1]) - 1
                if 0 <= n < len(LECONS):
                    lecon_index = n
                    historique = []
                    print(f"\n📖 Passage à la leçon : {LECONS[lecon_index]}\n")
                    msg = {"role": "user", "content": f"Commençons la leçon '{LECONS[lecon_index]}'. Je suis prêt."}
                    historique.append(msg)
                    print(f"Vous       : {msg['content']}")
                    rep = repondre(LECONS[lecon_index], historique)
                    historique.append({"role": "assistant", "content": rep})
                else:
                    print(f"Numéro invalide. Choisissez entre 1 et {len(LECONS)}.\n")
            except ValueError:
                print("Usage : /lecon N  (ex: /lecon 3)\n")
            continue

        elif cmd == "/reset":
            historique = []
            print(f"\n🔄 Reprise de la leçon : {LECONS[lecon_index]}\n")
            msg = {"role": "user", "content": f"Commençons la leçon '{LECONS[lecon_index]}'. Je suis prêt."}
            historique.append(msg)
            print(f"Vous       : {msg['content']}")
            rep = repondre(LECONS[lecon_index], historique)
            historique.append({"role": "assistant", "content": rep})
            continue

        else:
            print(f"Commande inconnue : {user_input}. Tapez /aide.\n")
            continue

    # --- Échange normal ---
    historique.append({"role": "user", "content": user_input})
    rep = repondre(LECONS[lecon_index], historique)
    historique.append({"role": "assistant", "content": rep})
