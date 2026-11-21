"""Guichet de l'application : reçoit une vidéo, renvoie le résultat de l'analyse.

Lancer depuis la racine du dépôt :
    uvicorn video.src.api.main:app --host 0.0.0.0 --port 8000
Le modèle est lu dans video/models/effb0_meilleur.pt (ou dans la variable d'environnement
VIDEO_MODEL_PATH). Il est chargé une seule fois, au démarrage du serveur.

Routes :
    GET  /health          -> {"statut": "ok"}
    POST /analyze/video   -> champ "file" (la vidéo) ; renvoie
                             {score, label, segments_suspects[{t_debut, t_fin, score}], explication}
"""
import os
import shutil
import tempfile
import threading
from contextlib import asynccontextmanager

import cv2
from fastapi import FastAPI, File, HTTPException, UploadFile

from video.src.inference.analyze import analyser_video, charger_scoreur
from video.src.preprocessing.extract_faces import creer_detecteur

POIDS_PAR_DEFAUT = os.path.join("video", "models", "effb0_meilleur.pt")
EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
TAILLE_MAX = 200 * 1024 * 1024          # 200 Mo


def creer_app(detecteur=None, scorer=None, chemin_poids=None):
    """Crée l'application. Les paramètres servent aux tests (faux détecteur, faux modèle) ;
    en production on les laisse vides et le vrai détecteur et le vrai modèle sont chargés."""
    etat = {"detecteur": detecteur, "scorer": scorer}
    verrou = threading.Lock()           # une seule analyse à la fois (le GPU et MTCNN ne sont pas partagés)

    @asynccontextmanager
    async def cycle(app):
        if etat["detecteur"] is None:
            etat["detecteur"] = creer_detecteur()
        if etat["scorer"] is None:
            etat["scorer"] = charger_scoreur(chemin_poids or os.environ.get("VIDEO_MODEL_PATH", POIDS_PAR_DEFAUT))
        yield

    app = FastAPI(title="Détection de deepfakes - vidéo", lifespan=cycle)

    @app.get("/health")
    def sante():
        return {"statut": "ok"}

    @app.post("/analyze/video")
    def analyser(file: UploadFile = File(...)):
        extension = os.path.splitext(file.filename or "")[1].lower()
        if extension not in EXTENSIONS:
            raise HTTPException(400, "Format non accepté. Formats : " + ", ".join(sorted(EXTENSIONS)))
        dossier = tempfile.mkdtemp()
        try:
            chemin = os.path.join(dossier, "video" + extension)
            taille = 0
            with open(chemin, "wb") as sortie:
                while True:
                    morceau = file.file.read(1024 * 1024)
                    if not morceau:
                        break
                    taille += len(morceau)
                    if taille > TAILLE_MAX:
                        raise HTTPException(413, "Vidéo trop lourde (maximum 200 Mo).")
                    sortie.write(morceau)
            cap = cv2.VideoCapture(chemin)
            lisible = cap.isOpened() and cap.grab()
            cap.release()
            if not lisible:
                raise HTTPException(400, "Vidéo illisible.")
            with verrou:
                return analyser_video(chemin, etat["detecteur"], etat["scorer"])
        finally:
            shutil.rmtree(dossier, ignore_errors=True)

    return app


app = creer_app()
