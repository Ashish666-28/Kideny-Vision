"""CPU inference for the user's custom KidneySwin checkpoint.

Training reference: KidneySwin_Kaggle_Colab_NumPy_FIXED(1).ipynb.
Preprocessing matches its validation/test dataset: grayscale, aspect-preserving
black padding to 224 x 224, bilinear resampling, float32 scaling to [0, 1].
Class scores are uncalibrated softmax values, not diagnostic certainty.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import BinaryIO

import numpy as np
import torch
from PIL import Image, ImageOps

from custom_swin import KidneySwin


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = PROJECT_ROOT / "models" / "weights" / "finetuned_v2_best.pth"
SUPPORTED_CLASSES = {"normal", "stone", "cyst", "tumor"}
IMAGE_SIZE = 224


def preprocess_image(image_source: str | Path | BinaryIO) -> np.ndarray:
    """Return the exact grayscale float32 array used by the test loader."""
    with Image.open(image_source) as source:
        image = source.convert("L")
    image = ImageOps.pad(
        image,
        (IMAGE_SIZE, IMAGE_SIZE),
        method=Image.Resampling.BILINEAR,
        color=0,
    )
    return np.asarray(image, dtype=np.float32).copy() / 255.0


class KidneyPredictor:
    def __init__(self, weights_path: str | Path = DEFAULT_WEIGHTS):
        self.weights_path = Path(weights_path).resolve()
        if not self.weights_path.is_file():
            raise FileNotFoundError(f"Model file not found: {self.weights_path}")

        checkpoint = torch.load(
            self.weights_path, map_location="cpu", weights_only=True
        )
        if not isinstance(checkpoint, Mapping):
            raise ValueError("Expected a checkpoint containing model, classes and config.")
        missing = {"model", "classes", "config"} - set(checkpoint)
        if missing:
            raise ValueError(f"Checkpoint is missing: {', '.join(sorted(missing))}")

        names = checkpoint["classes"]
        if not isinstance(names, (list, tuple)) or not all(
            isinstance(name, str) for name in names
        ):
            raise ValueError("Checkpoint classes must be a list of class names.")
        self.classes = tuple(name.strip().lower() for name in names)
        if len(self.classes) != 4 or set(self.classes) != SUPPORTED_CLASSES:
            raise ValueError("Expected four distinct classes: normal, stone, cyst, tumor.")

        config = checkpoint["config"]
        if not isinstance(config, Mapping):
            raise ValueError("Checkpoint config must be a mapping.")
        self.config = dict(config)
        if (
            self.config.get("image_size") != IMAGE_SIZE
            or self.config.get("in_chans") != 1
            or self.config.get("num_classes") != len(self.classes)
        ):
            raise ValueError("This loader requires the trained 224 x 224, grayscale, four-class model.")
        if not isinstance(checkpoint["model"], Mapping) or not checkpoint["model"]:
            raise ValueError("Checkpoint model must contain a nonempty state dictionary.")

        self.model = KidneySwin(**self.config)
        # Never allow partially loaded weights or replace an unmatched layer.
        self.model.load_state_dict(checkpoint["model"], strict=True)
        self.model.eval()
        self._inference_lock = Lock()

    def predict(self, image_source: str | Path | BinaryIO) -> dict:
        image = preprocess_image(image_source)
        tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)
        with self._inference_lock, torch.inference_mode():
            logits = self.model(tensor)
            if tuple(logits.shape) != (1, len(self.classes)):
                raise RuntimeError("Model returned an unexpected output shape.")
            if not bool(torch.isfinite(logits).all()):
                raise RuntimeError("Model returned invalid class scores.")
            values = logits.float().softmax(dim=1)[0].cpu().tolist()

        index = max(range(len(values)), key=values.__getitem__)
        return {
            "label": self.classes[index],
            "class_index": index,
            "confidence": float(values[index]),
            "probabilities": {
                name: float(value) for name, value in zip(self.classes, values)
            },
            "score_type": "uncalibrated_softmax",
        }


@lru_cache(maxsize=1)
def get_predictor() -> KidneyPredictor:
    """Reuse the loaded checkpoint across requests in one Flask process."""
    return KidneyPredictor()


def predict_image(image_source: str | Path | BinaryIO) -> dict:
    return get_predictor().predict(image_source)


def main() -> int:
    parser = argparse.ArgumentParser(description="Load or test your trained KidneySwin model.")
    parser.add_argument("image", nargs="?", type=Path, help="Path to a PNG/JPG CT test image")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    args = parser.parse_args()
    try:
        predictor = KidneyPredictor(args.weights)
        print("Checkpoint loaded successfully; all saved weights matched.")
        print("Class order:", list(predictor.classes))
        print("Preprocessing: grayscale, 224 x 224 black padding, pixels / 255.")
        if args.image is not None:
            print(json.dumps(predictor.predict(args.image), indent=2))
        else:
            print("To predict a CT image: python model_predictor.py test_ct.png")
    except Exception as exc:
        parser.exit(1, f"Model test failed: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
