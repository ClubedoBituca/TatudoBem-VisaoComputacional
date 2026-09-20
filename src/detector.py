"""Wrapper do detector YOLO pre-treinado.

FRENTE 1 (Deteccao) e dona deste arquivo.

Estado: FUNCIONAL no minimo necessario (carregar modelo + inferir num frame).
O que falta e da frente 1: desenho das caixas sobre o frame, tracking entre
frames e ajuste fino de limiares.

Nao ha treinamento em nenhum momento do MVP: usamos pesos COCO pre-treinados.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import load_config, resolve_path

# YOLO_CONFIG_DIR ja foi apontado para dentro do projeto em src/__init__.py,
# que roda antes deste modulo. Aqui so resta desligar o envio de telemetria.


def disable_telemetry() -> None:
    """Grava sync=False nas settings locais do Ultralytics (idempotente)."""
    from ultralytics import settings

    if settings.get("sync", True):
        settings.update({"sync": False})


@dataclass(frozen=True)
class Detection:
    """Um objeto detectado num frame.

    Coordenadas em pixels do frame de trabalho (ja redimensionado pelo
    VideoSource), no formato canto superior esquerdo / inferior direito.
    """

    class_id: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def bottom_center(self) -> tuple[float, float]:
        """Ponto inferior central da caixa.

        E o ponto de referencia da regra espacial: aproxima onde o objeto
        toca o chao, que e o que importa para saber se ele ocupa a faixa de
        circulacao. Usar o centro da caixa faria um objeto alto parecer estar
        mais adiante do que esta.
        """
        return ((self.x1 + self.x2) / 2.0, self.y2)

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)


class ObstacleDetector:
    """Carrega o YOLO pre-treinado e devolve deteccoes filtradas."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        conf_threshold: float | None = None,
        iou_threshold: float | None = None,
        imgsz: int | None = None,
        device: str | None = None,
    ) -> None:
        cfg = load_config()["detection"]
        self.model_path = resolve_path(model_path or cfg["model_path"])
        self.conf_threshold = conf_threshold or cfg["conf_threshold"]
        self.iou_threshold = iou_threshold or cfg["iou_threshold"]
        self.imgsz = imgsz or cfg["imgsz"]
        self.target_classes = set(cfg["target_classes"])
        self.ignored_classes = set(cfg["ignored_classes"])

        from ultralytics import YOLO  # import tardio: carregar torch custa segundos

        disable_telemetry()
        self.model = YOLO(str(self.model_path))
        self.device = device or self._pick_device()
        self.class_names: dict[int, str] = dict(self.model.names)

    @staticmethod
    def _pick_device() -> str:
        import torch

        if not torch.cuda.is_available():
            return "cpu"
        # Blackwell (sm_120) so roda em wheel compilado para essa capability.
        # Se a wheel instalada nao a inclui, e melhor cair para CPU do que
        # estourar um erro de kernel no meio da demo.
        capability = torch.cuda.get_device_capability(0)
        arch_list = torch.cuda.get_arch_list()
        if f"sm_{capability[0]}{capability[1]}" not in arch_list:
            return "cpu"
        return "cuda:0"

    def predict(
        self,
        frame: np.ndarray,
        only_targets: bool = True,
    ) -> list[Detection]:
        """Roda o detector num frame BGR e devolve a lista de deteccoes.

        `only_targets=True` mantem apenas as classes de `target_classes`.
        As classes de `ignored_classes` sao sempre descartadas: 'person' esta
        la porque pedestre em transito nao e barreira temporaria, e porque o
        projeto nao identifica pessoas.
        """
        results: Any = self.model.predict(
            source=frame,
            imgsz=self.imgsz,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
        )

        detections: list[Detection] = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                class_id = int(box.cls[0])
                class_name = self.class_names.get(class_id, str(class_id))
                if class_name in self.ignored_classes:
                    continue
                if only_targets and class_name not in self.target_classes:
                    continue
                x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
                detections.append(
                    Detection(
                        class_id=class_id,
                        class_name=class_name,
                        confidence=float(box.conf[0]),
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                    )
                )
        return detections
