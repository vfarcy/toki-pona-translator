import argparse
import json
import random
import re
from pathlib import Path

LEVELS = ["debutant", "intermediaire", "avance"]
CORRECTION_STYLES = ["courte", "longue"]

LESSON_HINTS = {
    "Introduction": "Reste sur une phrase tres simple avec mi ou sina.",
    "Basic nouns and pronouns": "Utilise les pronoms mi et sina avec un nom simple.",
    "Basic verbs": "Concentre-toi sur un verbe frequent comme moku, lukin ou tawa.",
    "li structure": "Pense a la place de li apres le sujet (sauf mi/sina seuls).",
    "e object marker": "Ajoute e devant l'objet direct.",
    "Adjectives and order": "Rappelle l'ordre nom puis adjectif.",
    "Prepositions": "Ajoute une preposition simple comme lon ou tawa.",
    "Emotions": "Exprime un etat avec pilin ou pona/ike.",
    "Questions": "Transforme une phrase en question simple.",
    "Conversation": "Garde un ton naturel de dialogue court.",
    "Advanced composition": "Combine deux idees en restant clair et progressif.",
}

SYSTEM_TEMPLATES = [
    "Tu enseignes le Toki Pona selon la methode Michel Thomas : progression orale, sans stress, construction active, guidance douce.",
    "Tu es un professeur Michel Thomas de Toki Pona : progression pas a pas, rassurante et pratique.",
    "Tu aides l'eleve en Toki Pona avec la methode Michel Thomas : clarte, repetition utile, confiance et progression.",
]

USER_TEMPLATES = [
    "Exercice de la lecon {lesson}. Niveau {level}. Je veux progresser, peux-tu me guider ?",
    "Lecon {lesson}, niveau {level}. Donne-moi un exercice progressif et corrige-moi ensuite.",
    "On travaille {lesson}. Je suis niveau {level}. Aide-moi avec une phrase a construire.",
    "Dans la lecon {lesson}, propose un entrainement adapte a un eleve {level}.",
]

REFORMULATION_LINES = [
    "Ensuite, reformule la meme idee avec d'autres mots pour ancrer la structure.",
    "Puis propose une reformulation equivalente pour renforcer la memoire.",
    "Ajoute une reformulation simple afin de comparer deux formulations correctes.",
]

SHORT_CORRECTION_LINES = [
    "Correction courte : valide la phrase en une ligne puis donne un mini-defi.",
    "Correction courte : corrige vite, explique en une phrase et relance l'eleve.",
    "Correction courte : feedback direct, bienveillant et actionnable.",
]

LONG_CORRECTION_LINES = [
    "Correction longue : explique la regle, montre l'erreur frequente et donne un exemple correct.",
    "Correction longue : decompose la phrase et justifie chaque morceau calmement.",
    "Correction longue : donne une explication pedagogique en plusieurs phrases courtes.",
]

PROGRESSIVE_LINES = [
    "Exercice progressif : etape 1 phrase minimale, etape 2 ajout d'un element, etape 3 variante.",
    "Exercice progressif : commence tres simple puis augmente legerement la difficulte.",
    "Exercice progressif : propose une version de base puis une extension guidee.",
]

ASSISTANT_OPENERS = [
    "Tres bien, on avance sans stress.",
    "Excellent, on construit cela ensemble, pas a pas.",
    "Parfait, je te guide calmement et on progresse tout de suite.",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Augmente un dataset Michel Thomas Toki Pona avec des variations pedagogiques."
    )
    parser.add_argument("--input", default="train_michel.jsonl", help="Fichier source JSONL")
    parser.add_argument(
        "--output", default="train_michel_augmented.jsonl", help="Fichier de sortie JSONL"
    )
    parser.add_argument(
        "--variants-per-example",
        type=int,
        default=4,
        help="Nombre de variantes generees par exemple source",
    )
    parser.add_argument(
        "--keep-original",
        action="store_true",
        help="Conserver aussi les exemples originaux dans la sortie",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed aleatoire")
    return parser.parse_args()


def extract_lesson(user_content: str) -> str:
    # Couvre: "lecon X." et "leçon X."
    match = re.search(r"le[cc]on\s+(.+?)\.", user_content, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return "Introduction"


def build_system(rng: random.Random, level: str, correction_style: str, progressive: bool, reformulation: bool) -> str:
    parts = [rng.choice(SYSTEM_TEMPLATES)]
    parts.append(f"Niveau cible: {level}.")
    parts.append(f"Style de correction attendu: {correction_style}.")
    if progressive:
        parts.append("L'exercice doit etre progressif.")
    if reformulation:
        parts.append("Inclure une reformulation pedagogique en fin de reponse.")
    return " ".join(parts)


def build_user(rng: random.Random, lesson: str, level: str) -> str:
    return rng.choice(USER_TEMPLATES).format(lesson=lesson, level=level)


def build_assistant(
    rng: random.Random,
    lesson: str,
    level: str,
    correction_style: str,
    progressive: bool,
    reformulation: bool,
) -> str:
    parts = [rng.choice(ASSISTANT_OPENERS)]
    parts.append(f"Dans la lecon {lesson}, on travaille un objectif adapte au niveau {level}.")
    parts.append(LESSON_HINTS.get(lesson, "Reste clair et simple, puis augmente legerement la difficulte."))

    if correction_style == "courte":
        parts.append(rng.choice(SHORT_CORRECTION_LINES))
    else:
        parts.append(rng.choice(LONG_CORRECTION_LINES))

    if progressive:
        parts.append(rng.choice(PROGRESSIVE_LINES))

    parts.append("Propose maintenant une phrase en toki pona, et je te corrige juste apres.")

    if reformulation:
        parts.append(rng.choice(REFORMULATION_LINES))

    return " ".join(parts)


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def save_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def augment(records: list[dict], variants_per_example: int, keep_original: bool, seed: int) -> list[dict]:
    rng = random.Random(seed)
    output = []

    if keep_original:
        output.extend(records)

    for record in records:
        msgs = record.get("messages", [])
        user_msg = next((m for m in msgs if m.get("role") == "user"), {"content": ""})
        lesson = extract_lesson(user_msg.get("content", ""))

        # Evite les doublons stricts sur un meme exemple source.
        seen = set()

        for _ in range(variants_per_example):
            level = rng.choice(LEVELS)
            correction_style = rng.choice(CORRECTION_STYLES)
            progressive = rng.choice([True, False])
            reformulation = rng.choice([True, False])

            system = build_system(rng, level, correction_style, progressive, reformulation)
            user = build_user(rng, lesson, level)
            assistant = build_assistant(
                rng,
                lesson,
                level,
                correction_style,
                progressive,
                reformulation,
            )

            signature = (system, user, assistant)
            if signature in seen:
                continue
            seen.add(signature)

            output.append(
                {
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                        {"role": "assistant", "content": assistant},
                    ]
                }
            )

    return output


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    records = load_jsonl(input_path)
    augmented = augment(
        records=records,
        variants_per_example=args.variants_per_example,
        keep_original=args.keep_original,
        seed=args.seed,
    )
    save_jsonl(output_path, augmented)

    print(f"Source examples      : {len(records)}")
    print(f"Variants per example : {args.variants_per_example}")
    print(f"Keep original        : {args.keep_original}")
    print(f"Output examples      : {len(augmented)}")
    print(f"Saved to             : {output_path}")


if __name__ == "__main__":
    main()
