# Interface web DeepFake Detector

L’interface HTML/CSS/JavaScript responsive est servie par Django à la racine du site afin que les fichiers statiques, l’API et le cookie CSRF partagent la même origine.

## Démarrer en local

Depuis `backend/`, activez l’environnement virtuel puis démarrez Django :

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r ..\requirements.txt
python manage.py runserver
```

Ouvrez ensuite `http://127.0.0.1:8000/`. L’API reçoit les vidéos sur `POST /api/video/analyze/` avec le champ multipart `video`. `GET /api/video/status/` indique si les deux fichiers de poids sont présents. La réponse d’analyse contient le score moyen `score` (0–1), le verdict `label`, la durée et le nombre d’images traitées, ainsi qu’un score séparé pour chaque modèle dans `models`.

## Modèles et prétraitement

Les checkpoints se trouvent dans `backend/models/` : `efficientnet_b0_best.pt` et `resnet50_best.pt` (le serveur accepte aussi `resnet50-best.pt`). Le backend reconstruit les architectures binaires torchvision EfficientNet-B0 et ResNet50, charge les poids strictement, extrait jusqu’à 16 images réparties dans la vidéo, puis moyenne les probabilités sigmoid des deux modèles. Un score proche de `1` indique la classe fake.

Les checkpoints du dépôt sont des `state_dict` sans métadonnées de prétraitement. L’implémentation utilise donc une entrée RGB 224×224, un redimensionnement à 256, un recadrage central et la normalisation ImageNet. Si l’entraînement a employé d’autres transformations ou si la sortie positive n’est pas la classe fake, adaptez `backend/config/inference.py` pour correspondre à cet entraînement.
