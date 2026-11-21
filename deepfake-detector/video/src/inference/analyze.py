"""Analyse d'une vidéo, de bout en bout (branche visages).

    extraction des visages -> score de chaque visage -> verdict (max des pistes)
    -> moments suspects en secondes -> résultat au format de l'API.

Le détecteur de visages et le modèle sont passés en paramètres : on peut ainsi tester
la logique sans GPU, avec de faux objets.
"""
import os
import shutil
import tempfile

import cv2
import numpy as np

from video.src.inference.aggregate import SEUIL, segments_suspects, verdict_video
from video.src.preprocessing.extract_faces import extraire_pistes


def charger_scoreur(chemin_poids, dispositif=None):
    """Charge EfficientNet-B0 (fichier .pt) et retourne une fonction
    scorer(liste d'images BGR 224x224) -> liste de scores entre 0 (réel) et 1 (généré)."""
    import torch
    import torch.nn as nn
    from torchvision.models import efficientnet_b0

    dispositif = dispositif or ("cuda" if torch.cuda.is_available() else "cpu")
    modele = efficientnet_b0(weights=None)
    modele.classifier[1] = nn.Linear(modele.classifier[1].in_features, 1)
    modele.load_state_dict(torch.load(chemin_poids, map_location=dispositif))
    modele.to(dispositif).eval()
    moyenne = torch.tensor([0.485, 0.456, 0.406], device=dispositif).view(1, 3, 1, 1)
    ecart = torch.tensor([0.229, 0.224, 0.225], device=dispositif).view(1, 3, 1, 1)

    @torch.no_grad()
    def scorer(images_bgr, taille_lot=32):
        scores = []
        for i in range(0, len(images_bgr), taille_lot):
            lot = np.stack([im[:, :, ::-1] for im in images_bgr[i:i + taille_lot]])   # BGR -> RGB
            x = torch.from_numpy(np.ascontiguousarray(lot)).permute(0, 3, 1, 2).float() / 255
            x = ((x.to(dispositif) - moyenne) / ecart)
            scores += torch.sigmoid(modele(x).squeeze(1)).cpu().tolist()
        return scores

    return scorer


def _explication(verdict, segments):
    if verdict["score"] is None:
        return "Aucun visage détecté : la partie visages ne peut pas analyser cette vidéo."
    if verdict["piste_suspecte"] is None:
        return "Aucune personne ne paraît générée (score le plus élevé : %.2f)." % verdict["score"]
    texte = "Une personne paraît générée (score %.2f)." % verdict["score"]
    if segments:
        passages = "; ".join("de %.1f s à %.1f s (score moyen %.2f)"
                             % (s["t_debut"], s["t_fin"], s["score"]) for s in segments)
        return texte + " Passages suspects : " + passages + "."
    return texte + " Le score est élevé sur l'ensemble de la piste, sans passage isolé."


def analyser_video(chemin_video, detecteur, scorer, seuil=SEUIL):
    """Retourne {score, label, segments_suspects[{t_debut, t_fin, score}], explication}."""
    dossier = tempfile.mkdtemp()
    try:
        pistes = extraire_pistes(chemin_video, os.path.join(dossier, "faces"), detecteur)
        scores_pistes, temps_pistes = {}, {}
        for p in pistes:
            fichiers = sorted(f for f in os.listdir(p["chemin_visages"]) if f.endswith(".jpg"))
            images = [cv2.imread(os.path.join(p["chemin_visages"], f)) for f in fichiers]
            scores_pistes[p["piste_id"]] = [float(s) for s in scorer(images)]
            temps_pistes[p["piste_id"]] = p["temps_s"]
    finally:
        shutil.rmtree(dossier, ignore_errors=True)

    verdict = verdict_video(scores_pistes, seuil=seuil)
    segments = []
    for pid, moyenne in verdict["scores_pistes"].items():
        if moyenne >= seuil:                                  # seulement les personnes suspectes
            segments += segments_suspects(scores_pistes[pid], temps_pistes[pid], seuil)
    segments.sort(key=lambda s: s["t_debut"])
    return dict(score=verdict["score"], label=verdict["label"],
                segments_suspects=segments, explication=_explication(verdict, segments))
