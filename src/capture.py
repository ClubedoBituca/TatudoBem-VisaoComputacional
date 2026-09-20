"""Captura de frames a partir de arquivo de video ou webcam.

FRENTE 1 (Deteccao) e dona deste arquivo.

Estado: FUNCIONAL. E a base que as demais frentes consomem; evoluir com
cuidado para nao quebrar quem ja depende da interface.

Contrato:
    with VideoSource("data/samples/clipe.mp4") as source:
        for frame_index, frame_bgr in source:
            ...

`frame_bgr` e um numpy.ndarray BGR (padrao do OpenCV) ja redimensionado para
`work_width`. Nenhum frame e gravado em disco: o MVP nao persiste imagem.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from .config import load_config, resolve_path


class CaptureError(RuntimeError):
    """A fonte de video nao pode ser aberta ou lida."""


@dataclass(frozen=True)
class SourceInfo:
    """Metadados da fonte, uteis para log e para a interface."""

    source: str
    is_webcam: bool
    native_width: int
    native_height: int
    work_width: int
    work_height: int
    fps: float
    frame_count: int  # 0 quando desconhecido (webcam ou container sem indice)

    @property
    def duration_seconds(self) -> float:
        if self.fps > 0 and self.frame_count > 0:
            return self.frame_count / self.fps
        return 0.0


class VideoSource:
    """Itera frames de um arquivo .mp4 ou de uma webcam.

    Tres cuidados vem de trabalhar com gravacao de celular:
      * orientacao: o celular grava metadado de rotacao mesmo filmando na
        horizontal, e sem `CAP_PROP_ORIENTATION_AUTO` o frame pode sair deitado;
      * resolucao: 1080p/4K desperdicam CPU, entao reduzimos para `work_width`;
      * taxa de quadros: `frame_stride` permite pular frames em video de 60 fps.
    """

    def __init__(
        self,
        source: str | int | Path,
        work_width: int | None = None,
        frame_stride: int | None = None,
    ) -> None:
        cfg = load_config()["capture"]
        self.work_width = work_width or cfg["work_width"]
        self.frame_stride = max(1, frame_stride or cfg["frame_stride"])

        self.is_webcam = isinstance(source, int)
        if self.is_webcam:
            self._source_repr = f"webcam:{source}"
            self._capture = cv2.VideoCapture(source)
        else:
            path = resolve_path(source)
            if not path.exists():
                raise CaptureError(f"Arquivo de video nao encontrado: {path}")
            self._source_repr = str(path)
            self._capture = cv2.VideoCapture(str(path))

        if not self._capture.isOpened():
            raise CaptureError(
                f"Nao foi possivel abrir a fonte de video: {self._source_repr}. "
                "Em video de celular, o codec HEVC/H.265 e a causa mais comum; "
                "converta com: ffmpeg -i entrada.mov -c:v libx264 -crf 23 saida.mp4"
            )

        # Aplica a rotacao gravada no metadado, quando o backend suportar.
        self._capture.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)

        native_w = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        native_h = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(self._capture.get(cv2.CAP_PROP_FPS) or 0.0)
        count = int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        if native_w > 0 and self.work_width < native_w:
            work_w = self.work_width
            work_h = int(round(native_h * work_w / native_w))
        else:
            work_w, work_h = native_w, native_h

        self.info = SourceInfo(
            source=self._source_repr,
            is_webcam=self.is_webcam,
            native_width=native_w,
            native_height=native_h,
            work_width=work_w,
            work_height=work_h,
            fps=fps,
            frame_count=max(0, count),
        )

    def read(self) -> np.ndarray | None:
        """Le o proximo frame ja redimensionado, ou None no fim do video."""
        for _ in range(self.frame_stride):
            ok, frame = self._capture.read()
            if not ok:
                return None
        return self._resize(frame)

    def _resize(self, frame: np.ndarray) -> np.ndarray:
        height, width = frame.shape[:2]
        if width == self.info.work_width:
            return frame
        target_h = int(round(height * self.info.work_width / width))
        return cv2.resize(
            frame, (self.info.work_width, target_h), interpolation=cv2.INTER_AREA
        )

    def __iter__(self) -> Iterator[tuple[int, np.ndarray]]:
        index = 0
        while True:
            frame = self.read()
            if frame is None:
                return
            yield index, frame
            index += 1

    def release(self) -> None:
        if self._capture is not None:
            self._capture.release()

    def __enter__(self) -> "VideoSource":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.release()


def webcam_available(index: int = 0) -> bool:
    """Diz se existe webcam utilizavel no indice dado.

    No WSL2 isto retorna False: o kernel nao expoe /dev/video*. O projeto e
    alimentado por arquivo de video; a webcam fica como caminho alternativo
    para quem rodar em Linux nativo ou Windows.
    """
    capture = cv2.VideoCapture(index)
    try:
        return capture.isOpened()
    finally:
        capture.release()
