"""Regra espacial: o objeto detectado invade a faixa de circulacao?

FRENTE 2 (Regra espacial) e dona deste arquivo.

Estado: STUB. As assinaturas e o contrato estao fechados para que as frentes
1, 3 e 4 possam programar contra eles desde ja. A implementacao e da frente 2.

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

from .detector import Detection

Point = tuple[float, float]
Polygon = list[Point]


class RouteStatus(str, Enum):
    """Os dois unicos estados que a interface exibe."""

    LIVRE = "ROTA LIVRE"
    BLOQUEADA = "BARREIRA TEMPORARIA"


@dataclass
class BlockageState:
    """Resultado da avaliacao de um frame."""

    status: RouteStatus
    intruders: list[Detection] = field(default_factory=list)
    consecutive_frames: int = 0
    just_confirmed: bool = False  # virou bloqueio neste frame -> abrir evento
    just_released: bool = False   # saiu do bloqueio neste frame -> fechar evento


def denormalize_polygon(polygon_norm: Polygon, width: int, height: int) -> Polygon:
    """Converte o poligono normalizado (0-1) para pixels do frame de trabalho.

    TODO(frente-2): implementar.
    """
    raise NotImplementedError("frente-2: converter coordenadas normalizadas em pixels")


def point_in_polygon(point: Point, polygon: Polygon) -> bool:
    """Diz se um ponto esta dentro do poligono.

    Sugestao: cv2.pointPolygonTest(np.array(polygon, dtype=np.int32), point, False) >= 0.

    TODO(frente-2): implementar.
    """
    raise NotImplementedError("frente-2: teste de ponto em poligono")


def detections_in_zone(detections: list[Detection], polygon: Polygon) -> list[Detection]:
    """Filtra as deteccoes cujo ponto inferior central cai dentro da zona.

    TODO(frente-2): implementar.
    """
    raise NotImplementedError("frente-2: filtrar deteccoes dentro da zona")


class BlockageTracker:
    """Acumula frames consecutivos e decide quando o bloqueio esta confirmado.

    Uso pretendido pela interface:

        tracker = BlockageTracker(polygon_px)
        state = tracker.update(detections)
        if state.just_confirmed:
            ...  # abrir evento
    """

    def __init__(
        self,
        polygon: Polygon,
        confirm_frames: int | None = None,
        release_frames: int | None = None,
    ) -> None:
        raise NotImplementedError("frente-2: inicializar contadores e histerese")

    def update(self, detections: list[Detection]) -> BlockageState:
        """Avalia um frame e devolve o estado atual da rota.

        TODO(frente-2): implementar.
        """
        raise NotImplementedError("frente-2: maquina de estados do bloqueio")

    def reset(self) -> None:
        """Zera os contadores (troca de video ou de zona)."""
        raise NotImplementedError("frente-2: reset dos contadores")
