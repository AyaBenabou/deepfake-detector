"""Analyse UNE vraie vidéo avec le vrai modèle, depuis le terminal.

Lancer depuis la racine du dépôt :
    python video/scripts/analyser_une_video.py chemin\\vers\\ma_video.mp4
Option : --poids chemin\\vers\\effb0_meilleur.pt   (par défaut video/models/effb0_meilleur.pt)
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from video.src.inference.analyze import analyser_video, charger_scoreur      # noqa: E402
from video.src.preprocessing.extract_faces import creer_detecteur            # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Analyse d'une vidéo (branche visages)")
    ap.add_argument("video")
    ap.add_argument("--poids", default=os.path.join("video", "models", "effb0_meilleur.pt"))
    args = ap.parse_args()
    if not os.path.isfile(args.video):
        sys.exit("Vidéo introuvable : " + args.video)
    if not os.path.isfile(args.poids):
        sys.exit("Fichier du modèle introuvable : " + args.poids)

    t0 = time.time()
    print("1/3 chargement du détecteur de visages...")
    detecteur = creer_detecteur()
    print("2/3 chargement du modèle...")
    scorer = charger_scoreur(args.poids)
    print("3/3 analyse de la vidéo...")
    resultat = analyser_video(args.video, detecteur, scorer)
    print(json.dumps(resultat, ensure_ascii=False, indent=2))
    print("durée totale : %.1f s" % (time.time() - t0))


if __name__ == "__main__":
    main()
