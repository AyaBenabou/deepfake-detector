"""Du score par visage au verdict de la vidéo (étapes 5 et 6 du pipeline).

Pour chaque personne suivie (piste) on a une suite de scores entre 0 (authentique)
et 1 (deepfake). On résume chaque piste, puis la vidéo prend le score MAXIMUM des
pistes : il suffit qu'une seule personne soit truquée pour que la vidéo soit signalée.
"""
import numpy as np

SEUIL = 0.5


def score_piste(scores, methode="moyenne", k=3):
    """Résume les scores d'une piste.
    - "moyenne" : moyenne de tous les visages ;
    - "topk"    : moyenne des k scores les plus élevés (utile si le trucage n'est
                  visible que sur une partie de la piste)."""
    scores = np.asarray(scores, dtype=float)
    if methode == "topk":
        return float(np.sort(scores)[-k:].mean())
    return float(scores.mean())


def verdict_video(scores_par_piste, seuil=SEUIL, methode="moyenne"):
    """scores_par_piste : {identifiant_piste: [scores des visages de la piste]}.

    Retourne le score global, le label, la piste la plus suspecte et le score de
    chaque piste (pour encadrer en rouge la personne concernée)."""
    scores = {p: score_piste(s, methode) for p, s in scores_par_piste.items() if len(s) > 0}
    if not scores:
        return dict(score=None, label="incertain (aucun visage détecté)",
                    piste_suspecte=None, scores_pistes={})
    piste_max = max(scores, key=scores.get)
    score = scores[piste_max]
    return dict(score=score,
                label="probablement généré" if score >= seuil else "probablement réel",
                piste_suspecte=piste_max if score >= seuil else None,
                scores_pistes=scores)


def segments_suspects(scores, temps_s, seuil=SEUIL, longueur_min=2, fenetre=3):
    """Moments suspects d'UNE piste.

    scores  : score de chaque visage, dans l'ordre du temps ;
    temps_s : seconde de chaque visage dans la vidéo (même longueur que scores).
    On lisse la courbe (moyenne de `fenetre` visages voisins), puis un segment est une
    suite d'au moins `longueur_min` visages dont le score lissé dépasse `seuil`.
    Retourne [{"t_debut", "t_fin", "score"}] avec des temps en secondes."""
    scores = np.asarray(scores, dtype=float)
    demi = fenetre // 2
    lisse = np.array([scores[max(0, i - demi): i + demi + 1].mean() for i in range(len(scores))])
    segments, debut = [], None
    for k in range(len(lisse) + 1):
        dedans = k < len(lisse) and lisse[k] > seuil        # le dernier tour ferme le segment ouvert
        if dedans and debut is None:
            debut = k
        if not dedans and debut is not None:
            if k - debut >= longueur_min:
                segments.append(dict(t_debut=float(temps_s[debut]), t_fin=float(temps_s[k - 1]),
                                     score=float(lisse[debut:k].mean())))
            debut = None
    return segments
