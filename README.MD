# Mode d'emploi

Teste sur la configuration suivante : Ubuntu 24, RTX 4070 Ti 12 Go, RAM 32 Go, SSD 750 Go.

## Compatibilite

- Systeme d'exploitation teste : Ubuntu 24
- GPU teste : NVIDIA GeForce RTX 4070 Ti 12 Go
- Memoire vive testee : 32 Go
- Stockage teste : SSD 750 Go
- Version Python cible : 3.11
- Cible PyTorch : CUDA 12.1

Verification CUDA / driver NVIDIA :

```bash
nvidia-smi
```

Sortie indicative :

```text
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 570.211.01             Driver Version: 570.211.01     CUDA Version: 12.8     |
|-----------------------------------------+------------------------+----------------------|
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 4070        Off |   00000000:01:00.0  On |                  N/A |
|  0%   40C    P8             10W /  200W |       8MiB /  12282MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+
```

Suivi GPU en temps reel avec nvtop :

```bash
# Installation (Ubuntu)
sudo apt update
sudo apt install -y nvtop

# Lancement
nvtop
```

Utilite :
- Visualiser l'occupation GPU pendant l'entrainement ou l'inference.
- Identifier rapidement un processus qui monopolise la VRAM.
- Controler l'evolution de la charge et de la temperature sans relancer `nvidia-smi`.

## 1. Installer Mamba via Miniforge

Cette méthode installe `mamba` par défaut et configure le dépôt Conda-Forge.

1. Télécharger le script d'installation :

```bash
wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
```

2. Lancer l'installation :

```bash
bash Miniforge3-$(uname)-$(uname -m).sh
```

Pendant l'installation :
- Appuyer sur `Entree` pour parcourir la licence.
- Saisir `yes` pour accepter les termes.
- Conserver l'emplacement par defaut.
- Repondre `yes` a la mise a jour du profil shell.

3. Recharger la session shell :

```bash
source ~/.bashrc
```

4. Verifier l'installation :

```bash
mamba --version
```

## 2. Utiliser Mamba

Remplacer `conda` par `mamba` pour les operations de resolution de paquets.

- Creer un environnement : `mamba create -n mon_env python=3.10`
- Installer des paquets : `mamba install numpy pandas`
- Rechercher un paquet : `mamba search scipy`

Activation et desactivation d'environnement : utiliser `conda activate` et `conda deactivate`.

## 3. Creer l'environnement du projet

```bash
# Creer l'environnement avec Python 3.11 et pip
mamba create -n toki_pona_ia python=3.11 pip -y

# Activer l'environnement
conda activate toki_pona_ia
```

## 4. Installer PyTorch et Unsloth

```bash
# PyTorch CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Unsloth
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```

## 5. Verifier l'installation GPU

```bash
python -c "import torch; print(f'Python: {torch.__version__}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"ABSENT\"}')"
```

Si le nom de la carte NVIDIA s'affiche, la configuration est operationnelle.

## 6. Rappel rapide des commandes

Action | Commande `conda` | Commande `mamba`
--- | --- | ---
Creer un environnement | `conda create -n test` | `mamba create -n test`
Installer un paquet | `conda install pytorch` | `mamba install pytorch`
Mettre a jour | `conda update --all` | `mamba update --all`
Activer un environnement | `conda activate ...` | `conda activate ...`


