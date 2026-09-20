"""Componentes de interface Streamlit.

FRENTE 3 (Interface) e dona deste arquivo.

Estado: STUB. Mantem app.py enxuto: o app orquestra, este modulo desenha.

Telas previstas no MVP:
  * selecao da fonte (upload de .mp4 de data/samples ou caminho local);
  * player com o frame anotado - caixas das deteccoes e poligono da zona;
  * banner de status grande: verde "ROTA LIVRE" / vermelho "BARREIRA TEMPORARIA";
  * tabela dos eventos de outputs/events.csv.
"""

from __future__ import annotations

import numpy as np

from .blockage import BlockageState
from .detector import Detection

Polygon = list[tuple[float, float]]


def draw_overlay(
    frame: np.ndarray,
    detections: list[Detection],
    polygon: Polygon,
    state: BlockageState | None = None,
) -> np.ndarray:
    """Desenha poligono da zona e caixas das deteccoes sobre uma copia do frame.

    Nao altera o frame original. Marcar tambem o ponto inferior central de cada
    caixa ajuda muito a depurar a regra espacial.

    TODO(frente-3): implementar.
    """
    raise NotImplementedError("frente-3: overlay de zona e deteccoes")


def status_banner(state: BlockageState) -> None:
    """Renderiza o banner de status no Streamlit.

    TODO(frente-3): implementar.
    """
    raise NotImplementedError("frente-3: banner de status")


def events_table() -> None:
    """Renderiza a tabela de eventos a partir de outputs/events.csv.

    TODO(frente-3): implementar.
    """
    raise NotImplementedError("frente-3: tabela de eventos")


def sidebar_controls() -> dict[str, object]:
    """Controles de ajuste (confianca, frames de confirmacao, zona ativa).

    TODO(frente-3): implementar.
    """
    raise NotImplementedError("frente-3: controles da barra lateral")
