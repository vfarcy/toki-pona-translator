# Toki Pona IA - Guide Complet

Ce depot contient un pipeline complet pour:
1. Augmenter un dataset pedagogique style Michel Thomas
2. Fine-tuner un modele LoRA avec Unsloth
3. Tester la qualite du modele
4. Echanger en mode cours interactif

Le guide ci-dessous est aligne avec les scripts actuellement presents dans le projet.

## 1) Prerequis

Configuration testee:
- Ubuntu 24
- NVIDIA GeForce RTX 4070 Ti 12 Go
- RAM 32 Go
- Python 3.11

Verifier le GPU:

sudo apt update
sudo apt install -y nvtop
nvidia-smi
nvtop

## 2) Environnement Python

Installation Miniforge/Mamba (si besoin):

wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh
source ~/.bashrc

Creation de l'environnement du projet:

mamba create -n toki_pona_ia python=3.11 pip -y
conda activate toki_pona_ia

Installation des dependances de base:

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install datasets trl transformers peft accelerate sentencepiece

Verification rapide:

python -c "import torch; print(torch.__version__); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'GPU ABSENT')"

## 3) Fichiers importants

- [train_michel.jsonl](train_michel.jsonl): dataset source au format messages
- [augment_michel_dataset.py](augment_michel_dataset.py): script d'augmentation pedagogique
- [train_toki.py](train_toki.py): fine-tuning LoRA avec split train/eval
- [test_michel.py](test_michel.py): test batch rapide du modele fine-tune
- [cours_michel.py](cours_michel.py): interface de cours interactif en terminal
- [toki_lora](toki_lora): dossier de sortie final du LoRA
- [outputs](outputs): checkpoints d'entrainement

## 4) Format de dataset attendu

Chaque ligne JSONL doit contenir:

{
	"messages": [
		{"role": "system", "content": "..."},
		{"role": "user", "content": "..."},
		{"role": "assistant", "content": "..."}
	]
}

Le script [train_toki.py](train_toki.py) est actuellement configure pour lire par defaut [train_michel.jsonl](train_michel.jsonl).

## 5) Augmenter le dataset

Generation standard (fichier augmente):

python augment_michel_dataset.py --variants-per-example 4 --keep-original --output train_michel_augmented.jsonl

Options utiles:
- --variants-per-example: nombre de variantes par exemple source
- --keep-original: conserve les exemples initiaux
- --seed: reproductibilite

Ce script cree des variations sur:
- niveau: debutant/intermediaire/avance
- style de correction: courte/longue
- exercice progressif
- reformulation pedagogique

## 6) Fine-tuning

Lancer l'entrainement:

python train_toki.py

Comportement actuel de [train_toki.py](train_toki.py):
- mode LoRA sur unsloth/llama-3-8b-bnb-4bit
- split 90/10 train/eval
- 3 epochs
- learning rate 5e-5
- evaluation tous les 50 steps
- checkpoints tous les 50 steps
- sauvegarde finale dans [toki_lora](toki_lora)

Important:
- Pour entrainer sur un dataset augmente, modifier la variable data_files dans [train_toki.py](train_toki.py) (ligne de load_dataset) vers train_michel_augmented.jsonl.

## 7) Tester le modele

Test batch rapide:

python test_michel.py

Ce script charge la base + LoRA [toki_lora](toki_lora) puis execute plusieurs prompts de verification.

## 8) Cours interactif

Lancer le cours terminal:

python cours_michel.py

Commandes disponibles dans [cours_michel.py](cours_michel.py):
- /aide
- /lecons
- /lecon N
- /suivante
- /reset
- /test
- /quitter

Notes:
- Le mode reponse est en streaming token par token (generation en direct).
- La commande /test lance une batterie de tests et affiche un score global.

## 9) Depannage rapide

1. Erreur CUDA ou GPU absent:
- verifier nvidia-smi
- verifier l'environnement actif
- verifier la version torch CUDA

2. Reponses trop repetitives:
- augmenter la diversite du dataset via [augment_michel_dataset.py](augment_michel_dataset.py)
- relancer un fine-tuning

3. Modele non trouve:
- verifier la presence de [toki_lora](toki_lora)
- verifier que l'entrainement est termine sans erreur

4. Verification syntaxe scripts:

python -m py_compile train_toki.py
python -m py_compile augment_michel_dataset.py
python -m py_compile test_michel.py
python -m py_compile cours_michel.py


