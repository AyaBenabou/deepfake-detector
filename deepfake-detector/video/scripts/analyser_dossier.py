"""Analyse TOUTES les vidéos d'un dossier et écrit un tableau resultats.csv.

Lancer depuis la racine du dépôt :
    python video/scripts/analyser_dossier.py C:\\chemin\\vers\\dossier_de_videos
Option : --poids chemin\\vers\\effb0_meilleur.pt   (par défaut video/models/effb0_meilleur.pt)
Le détecteur et le modèle ne sont chargés qu'une fois : c'est plus rapide que de lancer
analyser_une_video.py plusieurs fois.
"""
import argparse
import csv
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from video.src.inference.analyze import analyser_video, charger_scoreur      # noqa: E402
from video.src.preprocessing.extract_faces import creer_detecteur            # noqa: E402

EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def analyser_dossier(dossier, detecteur, scorer):
    lignes = []
    for nom in sorted(os.listdir(dossier)):
        if os.path.splitext(nom)[1].lower() not in EXTENSIONS:
            continue
        t0 = time.time()
        r = analyser_video(os.path.join(dossier, nom), detecteur, scorer, detail=True)
        lignes.append(dict(video=nom,
                           score="" if r["score"] is None else round(r["score"], 3),
                           label=r["label"],
                           scores_pistes=r["scores_pistes"],
                           n_segments=len(r["segments_suspects"]),
                           secondes=round(time.time() - t0, 1)))
        print(lignes[-1])
    return lignes


def main():
    ap = argparse.ArgumentParser(description="Analyse d'un dossier de vidéos")
    ap.add_argument("dossier")
    ap.add_argument("--poids", default=os.path.join("video", "models", "effb0_meilleur.pt"))
    args = ap.parse_args()
    if not os.path.isdir(args.dossier):
        sys.exit("Dossier introuvable : " + args.dossier)
    if not os.path.isfile(args.poids):
        sys.exit("Fichier du modèle introuvable : " + args.poids)

    lignes = analyser_dossier(args.dossier, creer_detecteur(), charger_scoreur(args.poids))
    sortie = os.path.join(args.dossier, "resultats.csv")
    with open(sortie, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["video", "score", "label", "scores_pistes", "n_segments", "secondes"])
        w.writeheader()
        w.writerows(lignes)
    print("\n%d vidéos analysées -> %s" % (len(lignes), sortie))


if __name__ == "__main__":
    main()
