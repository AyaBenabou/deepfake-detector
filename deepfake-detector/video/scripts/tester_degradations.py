"""Diagnostic : le score d'une vidéo change-t-il quand on abîme un peu les visages ?

Lancer depuis la racine du dépôt :
    python video/scripts/tester_degradations.py C:\\chemin\\vers\\ma_video.mp4
Pour chaque version abîmée des visages (réduits, compressés, floutés), le script affiche le
score moyen et la part de visages au-dessus de 0,5. Si le score baisse beaucoup quand on
réduit ou qu'on compresse, le modèle réagit à la netteté ou à la résolution, qui ne sont pas
les mêmes dans nos vidéos d'entraînement et dans une vidéo de téléphone.
"""
import os
import shutil
import sys
import tempfile

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from video.src.preprocessing.extract_faces import extraire_pistes     # noqa: E402


def reduire(taille):
    def f(im):
        petit = cv2.resize(im, (taille, taille), interpolation=cv2.INTER_AREA)
        return cv2.resize(petit, (im.shape[1], im.shape[0]), interpolation=cv2.INTER_LINEAR)
    return f


def jpeg(qualite):
    def f(im):
        _, code = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, qualite])
        return cv2.imdecode(code, cv2.IMREAD_COLOR)
    return f


VERSIONS = [("originaux", lambda im: im),
            ("réduits à 112 px", reduire(112)),
            ("réduits à 64 px", reduire(64)),
            ("JPEG qualité 40", jpeg(40)),
            ("flou léger", lambda im: cv2.GaussianBlur(im, (5, 5), 0))]


def visages(chemin_video, detecteur):
    dossier = tempfile.mkdtemp()
    try:
        images = []
        for p in extraire_pistes(chemin_video, os.path.join(dossier, "faces"), detecteur):
            for f in sorted(os.listdir(p["chemin_visages"])):
                if f.endswith(".jpg"):
                    images.append(cv2.imread(os.path.join(p["chemin_visages"], f)))
        return images
    finally:
        shutil.rmtree(dossier, ignore_errors=True)


def tester(images, scorer):
    lignes = []
    for nom, f in VERSIONS:
        scores = np.array(scorer([f(im) for im in images]))
        lignes.append((nom, float(scores.mean()), float((scores > 0.5).mean())))
    return lignes


def main():
    if len(sys.argv) < 2 or not os.path.isfile(sys.argv[1]):
        sys.exit("Usage : python video/scripts/tester_degradations.py chemin\\vers\\video.mp4")
    poids = os.path.join("video", "models", "effb0_meilleur.pt")
    if not os.path.isfile(poids):
        sys.exit("Fichier du modèle introuvable : " + poids)
    from video.src.inference.analyze import charger_scoreur
    from video.src.preprocessing.extract_faces import creer_detecteur
    images = visages(sys.argv[1], creer_detecteur())
    if not images:
        sys.exit("Aucun visage trouvé dans cette vidéo.")
    print("%d visages\n" % len(images))
    print("%-20s %-12s %s" % ("version", "score moyen", "visages > 0,5"))
    for nom, moyenne, part in tester(images, charger_scoreur(poids)):
        print("%-20s %-12.3f %.0f %%" % (nom, moyenne, 100 * part))


if __name__ == "__main__":
    main()
