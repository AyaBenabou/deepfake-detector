"""Test du guichet FastAPI sans GPU : faux détecteur et faux modèle.

Lancer depuis la racine du dépôt :   python video/tests/test_api.py
Il faut : pip install fastapi python-multipart httpx
"""
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from fastapi.testclient import TestClient                            # noqa: E402
from video.src.api.main import creer_app                             # noqa: E402
from video.tests.test_pistes import FauxDetecteur, fabriquer_video   # noqa: E402

SORTIES = [[0.04, 0.02, 0.06, 0.03, 0.01, 0.05],      # Alice
           [0.1, 0.9, 0.95, 0.92, 0.9, 0.1],          # Bob
           [0.02, 0.05, 0.01, 0.03, 0.04, 0.02]]      # Charlie


class FauxModele:
    def __init__(self):
        self.appel = 0

    def __call__(self, images):
        sortie = SORTIES[self.appel % 3]
        self.appel += 1
        return sortie


def main():
    dossier = tempfile.mkdtemp()
    try:
        video = os.path.join(dossier, "scene.mp4")
        fabriquer_video(video)
        with TestClient(creer_app(FauxDetecteur(), FauxModele())) as client:
            r = client.get("/health")
            assert r.status_code == 200 and r.json() == {"statut": "ok"}

            with open(video, "rb") as f:
                r = client.post("/analyze/video", files={"file": ("scene.mp4", f, "video/mp4")})
            print(r.status_code, r.json())
            assert r.status_code == 200
            res = r.json()
            assert set(res) == {"score", "label", "segments_suspects", "explication"}
            assert res["label"] == "probablement généré"
            assert res["segments_suspects"][0]["t_debut"] == 0.333

            r = client.post("/analyze/video", files={"file": ("notes.txt", b"bonjour", "text/plain")})
            print(r.status_code, r.json())
            assert r.status_code == 400                                   # mauvais format

            r = client.post("/analyze/video", files={"file": ("faux.mp4", b"ce n'est pas une video", "video/mp4")})
            print(r.status_code, r.json())
            assert r.status_code == 400                                   # vidéo illisible
    finally:
        shutil.rmtree(dossier, ignore_errors=True)
    print("\nTOUS LES TESTS PASSENT")


if __name__ == "__main__":
    main()
