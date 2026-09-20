"""Regra espacial: o objeto detectado invade a faixa de circulacao?

FRENTE 2 (Regra espacial) e dona deste arquivo.

Estado: FUNCIONAL. As assinaturas publicadas no stub foram mantidas - quem
programou contra elas nao precisa mudar nada.

Regra combinada do MVP:
  1. o ponto de referencia do objeto e o inferior central da bounding box
     (`Detection.bottom_center`) - aproxima o contato com o chao;
  2. o objeto esta bloqueando se esse ponto cai dentro do poligono da zona;
  3. o bloqueio so e CONFIRMADO apos `confirm_frames` frames consecutivos, e
     so e LIBERADO apos `release_frames` frames consecutivos sem invasao.
     A histerese evita que uma deteccao piscante gere alerta e evento.

Saida esperada na interface: "ROTA LIVRE" ou "BARREIRA TEMPORARIA".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import cv2
import numpy as np

from .config import load_config
from .detector import Detection

Point = tuple[float, float]
Polygon = list[Point]


class RouteStatus(str, Enum):
    """Os dois unicos estados que a interface exibe."""

    LIVRE = "ROTA LIVRE"
    BLOQUEADA = "BARREIRA TEMPORARIA"


@dataclass
class BlockageState:
    """Resultado da avaliacao de um frame.

    `consecutive_frames` conta ha quantos frames seguidos a condicao ATUAL se
    mantem: com invasao, se ha intrusos; sem invasao, caso contrario. Serve
    para a interface mostrar o progresso ate a confirmacao ("3/8 frames").
    """

    status: RouteStatus
    intruders: list[Detection] = field(default_factory=list)
    consecutive_frames: int = 0
    just_confirmed: bool = False  # virou bloqueio neste frame -> abrir evento
    just_released: bool = False   # saiu do bloqueio neste frame -> fechar evento


def denormalize_polygon(polygon_norm: Polygon, width: int, height: int) -> Polygon:
    """Converte o poligono normalizado (0-1) para pixels do frame de trabalho.

    O JSON guarda o poligono normalizado justamente para sobreviver a troca de
    resolucao: o mesmo trecho pode ser filmado em 1080p e processado em 960 de
    largura. Aqui ele vira pixels do frame que a interface exibe.
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"Dimensoes invalidas do frame: {width}x{height}")
    return [(x * width, y * height) for x, y in polygon_norm]


def _as_contour(polygon: Polygon) -> np.ndarray:
    """Formata o poligono como contorno do OpenCV, validando o minimo.

    Menos de 3 vertices nao delimita area nenhuma - e um erro de configuracao
    que vale estourar cedo, nao virar "nunca detecta nada" no meio da demo.
    """
    if len(polygon) < 3:
        raise ValueError(
            f"Poligono precisa de ao menos 3 vertices, recebeu {len(polygon)}. "
            "Confira a zona ativa em config/zones.json."
        )
    return np.array(polygon, dtype=np.float32).reshape(-1, 1, 2)


def point_in_polygon(point: Point, polygon: Polygon) -> bool:
    """Diz se um ponto esta dentro do poligono.

    Ponto exatamente sobre a aresta conta como DENTRO (o teste do OpenCV
    devolve 0 nesse caso). A escolha e deliberada: em acessibilidade, alertar
    a mais custa menos do que deixar passar uma rota bloqueada.
    """
    return cv2.pointPolygonTest(_as_contour(polygon), (float(point[0]), float(point[1])), False) >= 0


def detections_in_zone(detections: list[Detection], polygon: Polygon) -> list[Detection]:
    """Filtra as deteccoes cujo ponto inferior central cai dentro da zona."""
    contorno = _as_contour(polygon)  # montado uma vez, nao por deteccao
    dentro: list[Detection] = []
    for deteccao in detections:
        x, y = deteccao.bottom_center
        if cv2.pointPolygonTest(contorno, (float(x), float(y)), False) >= 0:
            dentro.append(deteccao)
    return dentro


class BlockageTracker:
    """Acumula frames consecutivos e decide quando o bloqueio esta confirmado.

    Uso pretendido pela interface:

        tracker = BlockageTracker(polygon_px)
        state = tracker.update(detections)
        if state.just_confirmed:
            ...  # abrir evento

    Alem do `BlockageState`, o tracker expoe `blocked_frames`: quantos frames
    se passaram desde a confirmacao do bloqueio atual. E o que a frente 3 usa
    para calcular `duration_s` do evento ao fechar.
    """

    def __init__(
        self,
        polygon: Polygon,
        confirm_frames: int | None = None,
        release_frames: int | None = None,
    ) -> None:
        cfg = load_config()["blockage"]
        self.polygon = polygon
        self.confirm_frames = max(1, confirm_frames or cfg["confirm_frames"])
        self.release_frames = max(1, release_frames or cfg["release_frames"])
        self._contour = _as_contour(polygon)  # valida o poligono ja na construcao
        self.reset()

    def update(self, detections: list[Detection]) -> BlockageState:
        """Avalia um frame e devolve o estado atual da rota."""
        intrusos = detections_in_zone(detections, self.polygon)

        if intrusos:
            self._blocked_streak += 1
            self._clear_streak = 0
        else:
            self._clear_streak += 1
            self._blocked_streak = 0

        if self.status is RouteStatus.BLOQUEADA:
            self.blocked_frames += 1

        just_confirmed = False
        just_released = False

        if self.status is RouteStatus.LIVRE:
            if self._blocked_streak >= self.confirm_frames:
                self.status = RouteStatus.BLOQUEADA
                just_confirmed = True
                # O bloqueio ja existia durante os frames que o confirmaram;
                # contar so daqui subestimaria a duracao do evento.
                self.blocked_frames = self._blocked_streak
        elif self._clear_streak >= self.release_frames:
            self.status = RouteStatus.LIVRE
            just_released = True

        return BlockageState(
            status=self.status,
            intruders=intrusos,
            consecutive_frames=self._blocked_streak if intrusos else self._clear_streak,
            just_confirmed=just_confirmed,
            just_released=just_released,
        )

    def reset(self) -> None:
        """Zera os contadores (troca de video ou de zona)."""
        self.status = RouteStatus.LIVRE
        self.blocked_frames = 0
        self._blocked_streak = 0
        self._clear_streak = 0
