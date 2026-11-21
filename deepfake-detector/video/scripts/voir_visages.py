"""Montre les visages que le modèle reçoit, pour une vidéo : une image PNG avec une ligne par personne.

Lancer depuis la racine du dépôt :
    python video/scripts/voir_visages.py C:\\chemin\\vers\\ma_video.mp4
Crée voir_visages_<nom_de_la_video>.png dans le dossier courant. Ouvrez-la et regardez : les
visages sont-ils bien cadrés, droits, nets, pas coupés, pas couchés sur le côté ?
Si le fichier du modèle existe (video/models/effb0_meilleur.pt), le score de chaque visage
(0 = réel, 1 = généré) est écrit en jaune sur chaque visage.
"""
import os
import shutil
import sys
import tempfile

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from video.src.preprocessing.extract_faces import extraire_pistes     # noqa: E402

PAR_LIGNE = 10


def mosaique(chemin_video, detecteur, scorer=None):
    dossier = tempfile.mkdtemp()
    try:
        pistes = extraire_pistes(chemin_video, os.path.join(dossier, "faces"), detecteur)
        lignes = []
        for p in pistes:
            fichiers = sorted(f for f in os.listdir(p["chemin_visages"]) if f.endswith(".jpg"))
            pas = max(1, len(fichiers) // PAR_LIGNE)
            images = [cv2.imread(os.path.join(p["chemin_visages"], f)) for f in fichiers[::pas][:PAR_LIGNE]]
            scores = scorer(images) if scorer is not None else None
            images = [cv2.resize(im, (112, 112)) for im in images]
            if scores is not None:
                for im, sc in zip(images, scores):
                    cv2.putText(im, "%.2f" % sc, (4, 106), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            while len(images) < PAR_LIGNE:
                images.append(np.zeros((112, 112, 3), np.uint8))
            ligne = np.hstack(images)
            cv2.putText(ligne, "piste %d (%d visages)" % (p["piste_id"], p["n_visages"]), (4, 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            lignes.append(ligne)
        return np.vstack(lignes) if lignes else None
    finally:
        shutil.rmtree(dossier, ignore_errors=True)


def main():
    if len(sys.argv) < 2 or not os.path.isfile(sys.argv[1]):
        sys.exit("Usage : python video/scripts/voir_visages.py chemin\\vers\\video.mp4")
    from video.src.preprocessing.extract_faces import creer_detecteur
    poids = os.path.join("video", "models", "effb0_meilleur.pt")
    scorer = None
    if os.path.isfile(poids):
        from video.src.inference.analyze import charger_scoreur
        scorer = charger_scoreur(poids)
    image = mosaique(sys.argv[1], creer_detecteur(), scorer)
    if image is None:
        sys.exit("Aucun visage trouvé dans cette vidéo.")
    sortie = "voir_visages_%s.png" % os.path.splitext(os.path.basename(sys.argv[1]))[0]
    cv2.imwrite(sortie, image)
    print("Image écrite :", os.path.abspath(sortie))


if __name__ == "__main__":
    main()
