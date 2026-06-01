from __future__ import annotations

import base64
from collections import Counter
from pathlib import Path
from threading import Lock
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO

from .knowledge import CLASS_NAMES, COIN_KNOWLEDGE_BASE


BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "best_models" / "coin_v8s_768_best.pt"
FRONTEND_DIST = BASE_DIR / "frontend" / "dist"
IMGSZ = 768
JPEG_QUALITY = 92


class ModelService:
    def __init__(self, model_path: Path):
        self.model_path = model_path
        self._model: YOLO | None = None
        self._lock = Lock()

    @property
    def is_available(self) -> bool:
        return self.model_path.exists()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def class_names(self) -> list[str]:
        if self._model is None:
            return CLASS_NAMES

        names = self._model.names
        if isinstance(names, dict):
            return [names[index] for index in sorted(names)]
        return list(names)

    def _load(self) -> YOLO:
        if not self.model_path.exists():
            raise FileNotFoundError(str(self.model_path))
        if self._model is None:
            self._model = YOLO(str(self.model_path))
        return self._model

    def predict(self, image: np.ndarray, confidence: float):
        with self._lock:
            model = self._load()
            return model.predict(source=image, conf=confidence, imgsz=IMGSZ, verbose=False)[0]


model_service = ModelService(MODEL_PATH)
app = FastAPI(title="Ancient Coin Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model": {
            "name": MODEL_PATH.name,
            "available": model_service.is_available,
            "loaded": model_service.is_loaded,
            "imgsz": IMGSZ,
        },
        "classes": model_service.class_names(),
    }


@app.post("/api/detect")
async def detect_coin(
    file: UploadFile = File(...),
    confidence: float = Form(0.5),
) -> dict[str, Any]:
    if not 0 <= confidence <= 1:
        raise HTTPException(status_code=400, detail="confidence must be between 0 and 1")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="image file is empty")

    image = decode_image(content)
    try:
        result = model_service.predict(image, confidence)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=f"model file not found: {MODEL_PATH.name}") from exc

    annotated_image = encode_jpeg_data_url(result.plot())
    detections = build_detections(result)
    counts = Counter(detection["className"] for detection in detections)

    return {
        "annotatedImage": annotated_image,
        "detections": detections,
        "summary": {
            "total": len(detections),
            "byClass": dict(counts),
        },
        "model": {
            "name": MODEL_PATH.name,
            "imgsz": IMGSZ,
        },
    }


def decode_image(content: bytes) -> np.ndarray:
    image_array = np.frombuffer(content, dtype=np.uint8)
    if image_array.size == 0:
        raise HTTPException(status_code=400, detail="image file is empty")
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="uploaded file is not a valid image")
    return image


def encode_jpeg_data_url(image: np.ndarray) -> str:
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    if not ok:
        raise HTTPException(status_code=500, detail="failed to encode annotated image")
    payload = base64.b64encode(encoded).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"


def build_detections(result) -> list[dict[str, Any]]:
    detections = []
    names = result.names
    for box in result.boxes:
        class_id = int(box.cls[0])
        class_name = names[class_id]
        x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]
        detections.append(
            {
                "classId": class_id,
                "className": class_name,
                "confidence": float(box.conf[0]),
                "box": {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "width": x2 - x1,
                    "height": y2 - y1,
                },
                "knowledge": COIN_KNOWLEDGE_BASE.get(class_name, "抱歉，暂无该类钱币的详细科普资料。"),
            }
        )
    return detections


if FRONTEND_DIST.exists():
    assets_dir = FRONTEND_DIST / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def serve_frontend(path: str):
        requested = FRONTEND_DIST / path
        if requested.is_file():
            return FileResponse(requested)
        return FileResponse(FRONTEND_DIST / "index.html")
