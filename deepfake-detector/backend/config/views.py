import logging
from pathlib import Path
import tempfile
import time

from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import ensure_csrf_cookie

from .inference import (
    CHECKPOINT_CANDIDATES,
    MAX_SAMPLED_FRAMES,
    ModelUnavailableError,
    VideoDecodeError,
    checkpoint_paths,
    predict_video,
)


logger = logging.getLogger(__name__)
MAX_UPLOAD_BYTES = 500 * 1024 * 1024
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi"}


@ensure_csrf_cookie
def home(request):
    return render(request, "index.html")


def model_status(request):
    if request.method != "GET":
        return JsonResponse({"error": "Méthode non autorisée."}, status=405, headers={"Allow": "GET"})

    paths = checkpoint_paths()
    models = [
        {
            "name": name,
            "available": paths[name] is not None,
            "file": paths[name].name if paths[name] else CHECKPOINT_CANDIDATES[name][0],
        }
        for name in CHECKPOINT_CANDIDATES
    ]
    return JsonResponse({"weights_available": all(item["available"] for item in models), "models": models})


def analyze_video(request):
    if request.method != "POST":
        return JsonResponse({"error": "Méthode non autorisée."}, status=405, headers={"Allow": "POST"})

    uploaded_video = request.FILES.get("video")
    if uploaded_video is None:
        return JsonResponse({"error": "Ajoutez un fichier vidéo avec le champ « video »."}, status=400)

    extension = Path(uploaded_video.name).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        return JsonResponse({"error": "Formats acceptés : MP4, MOV et AVI."}, status=415)
    if uploaded_video.size <= 0:
        return JsonResponse({"error": "Le fichier vidéo est vide."}, status=400)
    if uploaded_video.size > MAX_UPLOAD_BYTES:
        return JsonResponse({"error": "La vidéo dépasse la limite de 500 Mo."}, status=413)

    temporary_path = None
    started_at = time.perf_counter()
    try:
        with tempfile.NamedTemporaryFile(prefix="clair-video-", suffix=extension, delete=False) as temporary_file:
            temporary_path = Path(temporary_file.name)
            for chunk in uploaded_video.chunks():
                temporary_file.write(chunk)

        result = predict_video(temporary_path)
        result["elapsed_seconds"] = round(time.perf_counter() - started_at, 3)
        result["filename"] = uploaded_video.name
        result["max_sampled_frames"] = MAX_SAMPLED_FRAMES
        return JsonResponse(result)
    except VideoDecodeError as exc:
        return JsonResponse({"error": str(exc)}, status=400)
    except ModelUnavailableError as exc:
        logger.exception("Les modèles de détection vidéo ne sont pas disponibles.")
        return JsonResponse({"error": str(exc)}, status=503)
    except Exception:
        logger.exception("Erreur pendant l’analyse de la vidéo %s", uploaded_video.name)
        return JsonResponse({"error": "Erreur interne pendant l’analyse vidéo."}, status=500)
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                logger.warning("Le fichier temporaire %s n’a pas pu être supprimé.", temporary_path)
