# Détection de contenus synthétiques par IA

Projet ENSMR – Mines Rabat. Application mobile d'analyse d'images, de vidéos et de messages vocaux pour estimer la probabilité qu'un contenu soit généré par IA, avec explication de la décision.

Cahier des charges : [`docs/cahier_des_charges.pdf`](docs/cahier_des_charges.pdf)

## Équipe

BENABOU Aya · BELMABKHOUT Chaimae · CARIOU Baptiste · HAJJI Amine

## Structure du dépôt

| Dossier | Contenu |
|---|---|
| `image/` | Modèle image (EfficientNet-B2 / ConvNeXt-Tiny, Grad-CAM) |
| `audio/` | Modèle audio (LFCC / log-Mel, ResNet-18 / RawNet2) |
| `video/` | Modèle vidéo (EfficientNet + Bi-LSTM / GRU, timeline des segments) |
| `fusion/` | Fusion des scores en indice global (F6) |
| `frontend/` | Interface web HTML/CSS/JavaScript servie par Django |
| `backend/` | API Django REST et inférence vidéo EfficientNet-B0 + ResNet50 |
| `mobile/` | Application React Native |
| `docs/` | Cahier des charges, rapport, figures |
| `data/` | Données locales : **non versionnées** |
| `models/` | Poids des modèles : **non versionnés** |

## Contrat de sortie commun (à respecter par image, audio et vidéo)

```json
{
  "score": 0.87,
  "label": "probablement généré",
  "segments_suspects": [{"t_debut": 3.2, "t_fin": 5.0, "score": 0.93}],
  "explication": "texte ou chemin vers une carte de chaleur"
}
```

`score` ∈ [0, 1] (1 = généré par IA). Les niveaux : *probablement réel*, *incertain*, *probablement généré*.

## Installation

```bash
git clone <url-du-depot>
cd deepfake-detector
python -m venv .venv
source .venv/bin/activate      # Windows : .venv\Scripts\activate
pip install -r requirements.txt
```

## Travailler en équipe

Voir [CONTRIBUTING.md](CONTRIBUTING.md).

## Lancer l’interface web

Depuis la racine du dépôt, activez l’environnement Python, installez les dépendances et démarrez Django :

```powershell
python -m venv backend\venv
.\backend\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python backend\manage.py runserver
```

Ouvrez ensuite <http://127.0.0.1:8000/>. L’API reçoit les vidéos sur `/api/video/analyze/`. Les checkpoints attendus sont `backend/models/efficientnet_b0_best.pt` et `backend/models/resnet50_best.pt` (le nom avec tiret `resnet50-best.pt` est aussi accepté). Voir [frontend/README.md](frontend/README.md) pour le détail du prétraitement et de la réponse API.
