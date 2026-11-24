"""Frame-based inference for the local EfficientNet-B0 and ResNet50 checkpoints."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import threading
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms


MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
CHECKPOINT_CANDIDATES = {
    "EfficientNet-B0": ("efficientnet_b0_best.pt",),
    # Accept both spellings; this checkout currently contains resnet50_best.pt.
    "ResNet50": ("resnet50_best.pt", "resnet50-best.pt"),
}
IMAGE_SIZE = 224
MAX_SAMPLED_FRAMES = 16
INFERENCE_LOCK = threading.Lock()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PREPROCESS = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)


class VideoDecodeError(Exception):
    """Raised when OpenCV cannot decode usable frames from an uploaded video."""


class ModelUnavailableError(Exception):
    """Raised when a checkpoint cannot be loaded with its expected architecture."""


def checkpoint_paths() -> dict[str, Path | None]:
    paths: dict[str, Path | None] = {}
    for name, candidates in CHECKPOINT_CANDIDATES.items():
        paths[name] = next((MODEL_DIR / candidate for candidate in candidates if (MODEL_DIR / candidate).is_file()), None)
    return paths


def _read_state_dict(path: Path) -> dict[str, Any]:
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ModelUnavailableError(f"Le checkpoint {path.name} ne peut pas être chargé.") from exc

    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    elif isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    if not isinstance(state, dict):
        raise ModelUnavailableError(f"Le checkpoint {path.name} ne contient pas de state_dict.")

    keys = list(state)
    if keys and all(key.startswith("module.") for key in keys):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    return state


@lru_cache(maxsize=1)
def load_models() -> dict[str, nn.Module]:
    paths = checkpoint_paths()
    missing = [name for name, path in paths.items() if path is None]
    if missing:
        raise ModelUnavailableError("Checkpoint(s) manquant(s) : " + ", ".join(missing))

    efficientnet = models.efficientnet_b0(weights=None)
    efficientnet.classifier = nn.Sequential(nn.Dropout(p=0.2, inplace=True), nn.Linear(1280, 1))

    resnet = models.resnet50(weights=None)
    resnet.fc = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(2048, 1))

    networks = {"EfficientNet-B0": efficientnet, "ResNet50": resnet}
    try:
        for name, model in networks.items():
            path = paths[name]
            assert path is not None
            model.load_state_dict(_read_state_dict(path), strict=True)
            model.to(DEVICE).eval()
    except (RuntimeError, ModelUnavailableError) as exc:
        raise ModelUnavailableError(
            "Un checkpoint ne correspond pas à l’architecture binaire attendue "
            "(torchvision EfficientNet-B0/ResNet50, une sortie logit)."
        ) from exc
    return networks


def extract_video_frames(video_path: str | Path) -> tuple[torch.Tensor, float]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise VideoDecodeError("Le fichier n’a pas pu être ouvert comme vidéo.")

    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        duration_seconds = frame_count / fps if frame_count > 0 and np.isfinite(fps) and fps > 0 else 0.0
        if frame_count <= 0:
            raise VideoDecodeError("La vidéo ne contient aucune image décodable.")

        positions = np.unique(
            np.linspace(0, frame_count - 1, num=min(frame_count, MAX_SAMPLED_FRAMES), dtype=np.int64)
        )
        tensors = []
        for position in positions:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(position))
            ok, frame = capture.read()
            if not ok or frame is None:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tensors.append(PREPROCESS(Image.fromarray(rgb)))

        if not tensors:
            raise VideoDecodeError("Aucune image n’a pu être extraite de cette vidéo.")
        return torch.stack(tensors), duration_seconds
    finally:
        capture.release()


def _positive_probabilities(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    probabilities = []
    batch_size = 8
    with torch.inference_mode():
        for start in range(0, len(images), batch_size):
            batch = images[start : start + batch_size].to(DEVICE)
            logits = model(batch)
            if logits.ndim == 1:
                logits = logits.unsqueeze(1)
            if logits.ndim != 2 or logits.shape[1] != 1:
                raise ModelUnavailableError("Les modèles doivent produire un seul logit par image.")
            probabilities.append(torch.sigmoid(logits[:, 0]).cpu())
    return torch.cat(probabilities)


def predict_video(video_path: str | Path) -> dict[str, Any]:
    with INFERENCE_LOCK:
        models_by_name = load_models()

    images, duration_seconds = extract_video_frames(video_path)
    with INFERENCE_LOCK:
        model_scores = {
            name: float(_positive_probabilities(model, images).mean().item())
            for name, model in models_by_name.items()
        }

    fake_probability = sum(model_scores.values()) / len(model_scores)
    return {
        "score": fake_probability,
        "label": "probablement généré" if fake_probability >= 0.5 else "probablement réel",
        "elapsed_seconds": None,
        "duration_seconds": duration_seconds,
        "sampled_frames": len(images),
        "models": model_scores,
        "device": str(DEVICE),
    }
