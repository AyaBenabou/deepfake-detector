"""Test sans GPU ni vraies vidéos : scène synthétique Alice / Bob / Charlie.

Lancer depuis la racine du dépôt :   python video/tests/test_pistes.py
Une fausse vidéo de 2 secondes est créée ; un faux détecteur retourne 3 visages par
image. On vérifie que le suivi crée 3 pistes de 6 visages, que l'alignement redresse
la ligne des yeux, et que le verdict du maximum donne l'exemple Alice/Bob/Charlie.
"""
import os
import shutil
import sys
import tempfile

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from video.src.inference.aggregate import verdict_video            # noqa: E402
from video.src.preprocessing import extract_faces as ef            # noqa: E402

LARGEUR, HAUTEUR, FPS, DUREE = 1280, 720, 24, 2.0
F = ef.LARGEUR_DETECTION / LARGEUR                                  # facteur de réduction


def fabriquer_video(chemin):
    out = cv2.VideoWriter(chemin, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (LARGEUR, HAUTEUR))
    for _ in range(int(FPS * DUREE)):
        out.write(np.full((HAUTEUR, LARGEUR, 3), 90, np.uint8))
    out.release()


class FauxDetecteur:
    """Retourne 3 visages par image : Alice (gauche), Bob (centre), Charlie (droite).
    Les visages bougent un peu d'une image à l'autre. Les yeux d'Alice sont inclinés."""
    def __init__(self):
        self.appels = 0

    def detect(self, image, landmarks=False):
        dx = 6 * self.appels                      # petit déplacement à chaque image
        self.appels += 1
        centres = [(200 + dx, 300), (480 + dx, 300), (760 + dx, 300)]   # en pixels réduits
        boites, probas, points = [], [], []
        for k, (cx, cy) in enumerate(centres):
            boites.append([cx - 60, cy - 60, cx + 60, cy + 60])
            probas.append(0.99)
            incl = 20 if k == 0 else 0            # Alice : oeil droit plus bas de 20 px
            points.append([[cx - 25, cy - 15], [cx + 25, cy - 15 + incl],
                           [cx, cy], [cx - 20, cy + 30], [cx + 20, cy + 30]])
        return np.array(boites), np.array(probas), np.array(points)


def main():
    dossier = tempfile.mkdtemp()
    try:
        video = os.path.join(dossier, "scene.mp4")
        fabriquer_video(video)
        sortie = os.path.join(dossier, "faces", "scene")
        pistes = ef.extraire_pistes(video, sortie, FauxDetecteur())

        print("pistes trouvées :", [(p["piste_id"], p["n_visages"]) for p in pistes])
        assert len(pistes) == 3, "il devrait y avoir 3 personnes suivies"
        assert all(p["n_visages"] == 6 for p in pistes), "6 images x 1 visage par personne"
        for p in pistes:
            assert len(os.listdir(p["chemin_visages"])) == 6
        # heure de chaque visage : 24 images/s, une image prélevée toutes les 8 -> 3 par seconde
        attendu = [0.0, 0.333, 0.667, 1.0, 1.333, 1.667]
        for p in pistes:
            assert p["temps_s"] == attendu, p["temps_s"]
        img = cv2.imread(os.path.join(sortie, "piste_0", "000.jpg"))
        assert img.shape == (ef.TAILLE, ef.TAILLE, 3)

        # alignement : sur une image à 2 yeux inclinés, la ligne des yeux devient horizontale
        frame = np.zeros((HAUTEUR, LARGEUR, 3), np.uint8)
        frame[:] = 50
        yg, yd = (300 - 15) / F, (300 - 15 + 20) / F
        xg, xd = (200 - 25) / F, (200 + 25) / F
        cv2.circle(frame, (int(xg), int(yg)), 6, (255, 255, 255), -1)
        cv2.circle(frame, (int(xd), int(yd)), 6, (255, 255, 255), -1)
        boite = np.array([140, 240, 260, 360]) / F
        pts = np.array([[xg, yg], [xd, yd], [0, 0], [0, 0], [0, 0]])
        aligne = ef.recadrer_visage(frame, boite, pts, aligner=True)
        brut = ef.recadrer_visage(frame, boite, pts, aligner=False)

        def pente(im):
            gris = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            ys, xs = np.where(gris > 200)
            gauche, droite = xs < xs.mean(), xs >= xs.mean()
            return abs(ys[gauche].mean() - ys[droite].mean())
        print("écart vertical entre les yeux : brut %.1f px -> aligné %.1f px"
              % (pente(brut), pente(aligne)))
        assert pente(aligne) < pente(brut) / 3, "l'alignement doit redresser les yeux"

        # verdict : exemple Alice / Bob / Charlie
        scores = {
            "Alice":   [0.04, 0.02, 0.06, 0.03, 0.01, 0.05],
            "Bob":     [0.88, 0.92, 0.85, 0.94, 0.91, 0.90],
            "Charlie": [0.02, 0.05, 0.01, 0.03, 0.04, 0.02],
        }
        v = verdict_video(scores)
        print("verdict :", v["label"], "| score %.3f" % v["score"], "| piste :", v["piste_suspecte"])
        print({k: round(s, 3) for k, s in v["scores_pistes"].items()})
        assert v["piste_suspecte"] == "Bob" and abs(v["score"] - 0.9) < 1e-9
        print("\nTOUS LES TESTS PASSENT")
    finally:
        shutil.rmtree(dossier, ignore_errors=True)


if __name__ == "__main__":
    main()
