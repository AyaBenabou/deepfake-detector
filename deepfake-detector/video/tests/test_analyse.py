"""Test de analyser_video sans GPU : faux détecteur et faux modèle.

Lancer depuis la racine du dépôt :   python video/tests/test_analyse.py
La scène synthétique a 3 personnes (Alice, Bob, Charlie), 6 visages chacune, à
0 ; 0,333 ; 0,667 ; 1,0 ; 1,333 ; 1,667 s. Le faux modèle trouve Bob suspect du 2e au 5e visage.
"""
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from video.src.inference.aggregate import segments_suspects          # noqa: E402
from video.src.inference.analyze import analyser_video               # noqa: E402
from video.tests.test_pistes import FauxDetecteur, fabriquer_video   # noqa: E402


def main():
    # 1. moments suspects d'une piste
    t = [0.0, 0.333, 0.667, 1.0, 1.333, 1.667]
    seg = segments_suspects([0.1, 0.9, 0.95, 0.92, 0.9, 0.1], t)
    assert len(seg) == 1 and seg[0]["t_debut"] == 0.333 and seg[0]["t_fin"] == 1.333, seg
    assert segments_suspects([0.1, 0.9, 0.1, 0.1, 0.1, 0.1], t) == []      # un seul visage : ignoré
    assert segments_suspects([0.1] * 6, t) == []
    print("segments :", seg)

    # 2. vidéo complète
    ordre = {"appel": 0}
    sorties = [[0.04, 0.02, 0.06, 0.03, 0.01, 0.05],      # piste 0 : Alice
               [0.1, 0.9, 0.95, 0.92, 0.9, 0.1],          # piste 1 : Bob
               [0.02, 0.05, 0.01, 0.03, 0.04, 0.02]]      # piste 2 : Charlie

    def faux_modele(images):
        sortie = sorties[ordre["appel"]]
        ordre["appel"] += 1
        assert len(images) == 6
        return sortie

    dossier = tempfile.mkdtemp()
    try:
        video = os.path.join(dossier, "scene.mp4")
        fabriquer_video(video)
        res = analyser_video(video, FauxDetecteur(), faux_modele)
    finally:
        shutil.rmtree(dossier, ignore_errors=True)
    print(res)
    assert res["label"] == "probablement généré"
    assert set(res) == {"score", "label", "segments_suspects", "explication"}
    assert len(res["segments_suspects"]) == 1
    assert res["segments_suspects"][0]["t_debut"] == 0.333 and res["segments_suspects"][0]["t_fin"] == 1.333
    assert "0.3 s à 1.3 s" in res["explication"]

    # 3. vidéo sans visage
    class SansVisage:
        def detect(self, image, landmarks=False):
            return None, None, None
    dossier = tempfile.mkdtemp()
    try:
        video = os.path.join(dossier, "vide.mp4")
        fabriquer_video(video)
        res = analyser_video(video, SansVisage(), faux_modele)
    finally:
        shutil.rmtree(dossier, ignore_errors=True)
    print(res)
    assert res["score"] is None and res["segments_suspects"] == []
    print("\nTOUS LES TESTS PASSENT")


if __name__ == "__main__":
    main()
