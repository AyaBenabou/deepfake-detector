"""Extraction des visages (version 3 : éclaircit les images sombres) : suivi de TOUTES les personnes de la vidéo.

Pour chaque vidéo :
  1. on prélève FPS_CIBLE images par seconde (au plus MAX_FRAMES images) ;
  2. MTCNN détecte tous les visages de chaque image (+ 5 points : yeux, nez, bouche) ;
  3. chaque visage est aligné (yeux à l'horizontale), recadré et redimensionné ;
  4. un suivi simple regroupe les visages d'une même personne en une "piste".

Sortie : <sortie>/<nom_video>/piste_0/000.jpg, 001.jpg, ...
                                piste_1/000.jpg, ...
Le verdict d'une vidéo se calcule ensuite par piste (voir inference/aggregate.py).

Utilisation dans un notebook Kaggle :

    from video.src.preprocessing.extract_faces import creer_detecteur, traiter_table
    det = creer_detecteur()
    pistes = traiter_table("video/configs/all_videos.csv", "/kaggle/working/faces", det, limite=5)
"""
import argparse
import os
import sys
import shutil

import cv2
import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except ImportError:  # tqdm absent : on continue sans barre de progression
    def tqdm(x, **kwargs):
        return x

FPS_CIBLE = 3            # images prélevées par seconde (le cahier demande 2 à 4)
MAX_FRAMES = 48          # images prélevées au maximum par vidéo (environ 16 s)
TAILLE = 224             # côté du visage enregistré (pixels)
MARGE = 0.25             # marge autour du visage (25 %)
LARGEUR_DETECTION = 960  # les images sont réduites à cette largeur pour la détection
SEUIL_CONFIANCE = 0.9    # confiance minimale du détecteur
MAX_ABSENCE = 3          # une piste est fermée après autant d'images sans visage
DIST_MAX = 1.0           # distance max (en tailles de visage) pour relier deux visages
MIN_VISAGES = 8          # une piste plus courte est écartée (seuil réduit pour les vidéos courtes)
MAX_PISTES = 4           # nombre maximal de pistes gardées par vidéo (les plus grandes)
SEUIL_SOMBRE = 60        # luminosité moyenne (0-255) en dessous de laquelle l'image est éclaircie
GAMMA_SOMBRE = 0.4       # correction gamma appliquée aux images sombres (< 1 = éclaircit)


def creer_detecteur(dispositif=None):
    """Crée le détecteur de visages MTCNN (GPU si disponible)."""
    import torch
    from facenet_pytorch import MTCNN

    if dispositif is None:
        dispositif = "cuda" if torch.cuda.is_available() else "cpu"
    return MTCNN(keep_all=True, device=dispositif)


# --------------------------------------------------------------------------- #
# Recadrage et alignement
# --------------------------------------------------------------------------- #
def recadrer_visage(frame, boite, points=None, taille=TAILLE, marge=MARGE, aligner=True):
    """Recadre un visage (carré centré, avec marge). Si `points` est fourni et
    `aligner` est vrai, la ligne des yeux est remise à l'horizontale."""
    x1, y1, x2, y2 = [float(v) for v in boite]
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    c = max(x2 - x1, y2 - y1) * (1 + marge) / 2

    angle = 0.0
    if aligner and points is not None:
        (lx, ly), (rx, ry) = points[0], points[1]      # oeil gauche, oeil droit
        angle = float(np.degrees(np.arctan2(ry - ly, rx - lx)))

    # matrice : rotation autour du centre du visage, puis le centre arrive en (c, c)
    m = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    m[0, 2] += c - cx
    m[1, 2] += c - cy
    cote = max(int(round(2 * c)), 2)
    sortie = cv2.warpAffine(frame, m, (cote, cote), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REPLICATE)
    return cv2.resize(sortie, (taille, taille))


# --------------------------------------------------------------------------- #
# Suivi (association des visages d'une image à l'autre)
# --------------------------------------------------------------------------- #
class Piste:
    def __init__(self, identifiant, boite, t):
        self.id = identifiant
        self.boite = boite          # dernière boîte connue (x1, y1, x2, y2)
        self.dernier_t = t          # dernier indice d'image où le visage est vu
        self.premier_t = t
        self.temps = []             # indice d'image prélevée de chaque visage enregistré
        self.n = 0                  # nombre de visages enregistrés
        self.surface_totale = 0.0

    def surface_moyenne(self):
        return self.surface_totale / max(self.n, 1)


def _centre_et_taille(b):
    return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2), max(b[2] - b[0], b[3] - b[1], 1.0)


def associer(pistes_actives, detections):
    """Relie chaque détection à la piste la plus proche (association gloutonne).
    Retourne {indice_detection: piste}. Les autres détections sont de nouvelles personnes."""
    candidats = []
    for i, boite in enumerate(detections):
        (dx, dy), _ = _centre_et_taille(boite)
        for p in pistes_actives:
            (px, py), taille_p = _centre_et_taille(p.boite)
            d = np.hypot(dx - px, dy - py) / taille_p
            if d <= DIST_MAX:
                candidats.append((d, i, p))
    candidats.sort(key=lambda x: x[0])

    liens, detections_prises, pistes_prises = {}, set(), set()
    for d, i, p in candidats:
        if i in detections_prises or p.id in pistes_prises:
            continue
        liens[i] = p
        detections_prises.add(i)
        pistes_prises.add(p.id)
    return liens


# --------------------------------------------------------------------------- #
# Images sombres
# --------------------------------------------------------------------------- #
_TABLE_GAMMA = (255.0 * (np.arange(256) / 255.0) ** GAMMA_SOMBRE).astype(np.uint8)


def eclaircir_si_sombre(frame):
    """Éclaircit (gamma) une image dont la luminosité moyenne est sous SEUIL_SOMBRE.
    Les images normales sont renvoyées telles quelles. Le détecteur MTCNN ne voit
    pas les visages dans le noir (luminosité ~35 : aucun visage ; ~106 : confiance 0,99)."""
    gris = cv2.cvtColor(frame[::8, ::8], cv2.COLOR_BGR2GRAY)      # sous-échantillon : rapide
    if float(gris.mean()) >= SEUIL_SOMBRE:
        return frame
    return cv2.LUT(frame, _TABLE_GAMMA)


# --------------------------------------------------------------------------- #
# Extraction d'une vidéo
# --------------------------------------------------------------------------- #
def extraire_pistes(chemin_video, dossier_sortie, detecteur, fps_cible=FPS_CIBLE,
                    max_frames=MAX_FRAMES, aligner=True):
    """Extrait et regroupe les visages d'une vidéo.
    Retourne une liste de dictionnaires (une ligne par piste gardée)."""
    if os.path.isdir(dossier_sortie):
        shutil.rmtree(dossier_sortie)
    os.makedirs(dossier_sortie)

    cap = cv2.VideoCapture(chemin_video)
    if not cap.isOpened():
        return []
    fps = cap.get(cv2.CAP_PROP_FPS) or 24
    pas = max(1, int(round(fps / fps_cible)))

    pistes, actives = [], []
    t, i = 0, 0                       # t : images prélevées ; i : images lues
    while t < max_frames:
        if not cap.grab():            # fin de la vidéo
            break
        if i % pas == 0:
            ok, frame = cap.retrieve()
            if ok:
                _traiter_image(frame, t, detecteur, pistes, actives, dossier_sortie, aligner)
                t += 1
        i += 1
    cap.release()
    # l'image prélevée numéro t vient de l'image lue t*pas, donc de la seconde t*pas/fps
    return _filtrer_pistes(pistes, dossier_sortie, n_images=t, seconde_par_image=pas / fps)


def _traiter_image(frame, t, detecteur, pistes, actives, dossier_sortie, aligner):
    frame = eclaircir_si_sombre(frame)       # détection ET recadrage sur l'image éclaircie
    h, w = frame.shape[:2]
    f = LARGEUR_DETECTION / w
    petit = cv2.resize(frame, (LARGEUR_DETECTION, int(h * f)))
    try:
        boites, probas, points = detecteur.detect(cv2.cvtColor(petit, cv2.COLOR_BGR2RGB),
                                                  landmarks=True)
    except Exception:
        boites = None

    detections, pts = [], []
    if boites is not None and probas is not None:
        for k in range(len(boites)):
            if probas[k] is not None and probas[k] >= SEUIL_CONFIANCE:
                detections.append(np.asarray(boites[k], dtype=float) / f)   # taille d'origine
                pts.append(None if points is None else np.asarray(points[k], dtype=float) / f)

    # une piste absente depuis trop longtemps est fermée
    actives[:] = [p for p in actives if t - p.dernier_t <= MAX_ABSENCE]
    liens = associer(actives, detections)

    for i, boite in enumerate(detections):
        piste = liens.get(i)
        if piste is None:                                  # nouvelle personne
            piste = Piste(len(pistes), boite, t)
            pistes.append(piste)
            actives.append(piste)
        visage = recadrer_visage(frame, boite, pts[i], aligner=aligner)
        dossier = os.path.join(dossier_sortie, f"piste_{piste.id}")
        os.makedirs(dossier, exist_ok=True)
        cv2.imwrite(os.path.join(dossier, f"{piste.n:03d}.jpg"), visage,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        piste.n += 1
        piste.temps.append(t)
        piste.boite, piste.dernier_t = boite, t
        piste.surface_totale += (boite[2] - boite[0]) * (boite[3] - boite[1])


def _filtrer_pistes(pistes, dossier_sortie, n_images, seconde_par_image=None):
    """Écarte les pistes trop courtes ; garde au plus MAX_PISTES (les plus grandes).
    Le minimum de visages s'adapte à la durée : au plus MIN_VISAGES, et au plus la
    moitié des images prélevées (une vidéo de 2 s donne seulement 6 images)."""
    minimum = min(MIN_VISAGES, max(3, n_images // 2))
    gardees = [p for p in pistes if p.n >= minimum]
    if not gardees and pistes:                             # vidéo très courte
        meilleure = max(pistes, key=lambda p: p.n)
        gardees = [meilleure] if meilleure.n >= 3 else []
    gardees = sorted(gardees, key=lambda p: p.surface_moyenne(), reverse=True)[:MAX_PISTES]
    ids = {p.id for p in gardees}
    for p in pistes:
        if p.id not in ids:
            shutil.rmtree(os.path.join(dossier_sortie, f"piste_{p.id}"), ignore_errors=True)
    return [dict(piste_id=p.id, n_visages=p.n, surface_moyenne=round(p.surface_moyenne()),
                 premiere_image=p.premier_t, derniere_image=p.dernier_t,
                 chemin_visages=os.path.join(dossier_sortie, f"piste_{p.id}"),
                 # seconde de chaque visage (000.jpg, 001.jpg, ...) dans la vidéo
                 temps_s=[round(t * seconde_par_image, 3) for t in p.temps]
                 if seconde_par_image else None)
            for p in sorted(gardees, key=lambda p: p.id)]


# --------------------------------------------------------------------------- #
# Traitement de la table complète
# --------------------------------------------------------------------------- #
def traiter_table(csv_videos, dossier_sortie, detecteur, limite=None, splits=None):
    """Traite les vidéos d'une table (colonnes : video, chemin, label, split, ...).

    Écrit <sortie>/pistes.csv (une ligne par personne suivie) et retourne cette table.
    Les vidéos déjà traitées (fichier termine.txt) sont ignorées : on peut relancer.
    """
    table = pd.read_csv(csv_videos)
    if splits is not None:
        table = table[table["split"].isin(splits)]
    if limite is not None:
        table = table.head(limite)

    os.makedirs(dossier_sortie, exist_ok=True)
    lignes = []
    for _, v in tqdm(table.iterrows(), total=len(table)):
        dossier = os.path.join(dossier_sortie, os.path.splitext(v["video"])[0])
        marqueur = os.path.join(dossier, "termine.txt")
        if os.path.exists(marqueur):                       # déjà fait
            pistes = _relire_pistes(dossier)
        else:
            pistes = extraire_pistes(v["chemin"], dossier, detecteur)
            with open(marqueur, "w") as fichier:
                fichier.write("ok")
        meta = {c: v[c] for c in ("video", "source", "label", "groupe", "split") if c in v}
        if not pistes:
            lignes.append({**meta, "piste_id": -1, "n_visages": 0})
        for p in pistes:
            lignes.append({**meta, **{k: x for k, x in p.items() if k != "temps_s"}})

    resultat = pd.DataFrame(lignes)
    resultat.to_csv(os.path.join(dossier_sortie, "pistes.csv"), index=False)
    return resultat


def _relire_pistes(dossier):
    pistes = []
    for nom in sorted(os.listdir(dossier)):
        chemin = os.path.join(dossier, nom)
        if nom.startswith("piste_") and os.path.isdir(chemin):
            n = len([f for f in os.listdir(chemin) if f.endswith(".jpg")])
            pistes.append(dict(piste_id=int(nom.split("_")[1]), n_visages=n, chemin_visages=chemin))
    return pistes


if __name__ == "__main__" and "ipykernel" not in sys.modules:   # seulement en terminal
    ap = argparse.ArgumentParser(description="Extraction des visages et suivi par personne")
    ap.add_argument("--csv", default="video/configs/all_videos.csv")
    ap.add_argument("--sortie", default="/kaggle/working/faces")
    ap.add_argument("--limite", type=int, default=None)
    args, _ = ap.parse_known_args()   # ignore les options ajoutées par Jupyter
    res = traiter_table(args.csv, args.sortie, creer_detecteur(), limite=args.limite)
    print(res.groupby("video")["piste_id"].count().value_counts().sort_index())
