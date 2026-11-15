# Règles de travail en équipe

## Branches

- `main` : toujours fonctionnel. **On ne pousse jamais directement dessus.**
- Une branche par tâche : `video/preprocessing`, `audio/baseline`, `backend/api-video`, `docs/rapport`...
- Depuis `main` à jour :

```bash
git checkout main
git pull
git checkout -b video/preprocessing
```

## Cycle de travail

1. `git pull` avant de commencer.
2. Petits commits fréquents, messages clairs (`video: extraction des frames à 3 FPS`).
3. `git push -u origin <branche>`.
4. Ouvrir une **Pull Request** sur GitHub, demander la relecture d'un autre membre.
5. Après validation : *Squash and merge*, puis supprimer la branche.

## À ne jamais commiter

- Datasets (FaceForensics++, DFDC, Celeb-DF...) : dossier `data/`
- Poids de modèles (`.pt`, `.pth`, `.h5`, `.tflite`, `.onnx`) : dossier `models/`
- Clés, mots de passe, fichiers `.env`

Partagez données et poids via Google Drive, Kaggle ou Hugging Face, et notez le lien dans le README du dossier concerné.

## Notebooks

Vider les sorties avant de commiter pour éviter les conflits :

```bash
pip install nbstripout
nbstripout --install
```

## Résultats reproductibles

- Fixer les seeds dans les scripts d'entraînement.
- Paramètres dans `configs/` (fichiers YAML), pas en dur dans le code.
- Découpage train/val/test **par vidéo / par identité**, jamais par frame.
