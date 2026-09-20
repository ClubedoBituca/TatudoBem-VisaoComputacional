"""Componentes de interface Streamlit.

FRENTE 3 (Interface) e dona deste arquivo.

Estado: `draw_overlay` FUNCIONAL (usado tanto pelo Streamlit quanto por
tools/render_video.py). O restante segue stub da frente 3.

Mantem app.py enxuto: o app orquestra, este modulo desenha.

Telas previstas no MVP:
  * selecao da fonte (upload de .mp4 de data/samples ou caminho local);
  * player com o frame anotado - caixas das deteccoes e poligono da zona;
  * banner de status grande: verde "ROTA LIVRE" / vermelho "BARREIRA TEMPORARIA";
  * tabela dos eventos de outputs/events.csv.
"""

from __future__ import annotations

import cv2
import numpy as np

from .blockage import BlockageState, RouteStatus
from .detector import Detection

Polygon = list[tuple[float, float]]

# Cores em BGR (padrao OpenCV).
VERDE = (60, 180, 75)
VERMELHO = (40, 40, 220)
LARANJA = (0, 150, 255)
BRANCO = (255, 255, 255)
PRETO = (0, 0, 0)


def _texto(frame: np.ndarray, txt: str, org: tuple[int, int], escala: float, cor, grossura: int = 1) -> None:
    """Escreve com contorno preto, para o texto sobreviver a fundo claro ou escuro.

    O corredor da UNIFEI e muito claro e estourado ao fundo: texto sem contorno
    some. Nao usar acento - as fontes Hershey do OpenCV sao ASCII.
    """
    cv2.putText(frame, txt, org, cv2.FONT_HERSHEY_SIMPLEX, escala, PRETO, grossura + 3, cv2.LINE_AA)
    cv2.putText(frame, txt, org, cv2.FONT_HERSHEY_SIMPLEX, escala, cor, grossura, cv2.LINE_AA)


def draw_overlay(
    frame: np.ndarray,
    detections: list[Detection],
    polygon: Polygon,
    state: BlockageState | None = None,
) -> np.ndarray:
    """Desenha poligono da zona e caixas das deteccoes sobre uma copia do frame.

    Nao altera o frame original. O ponto inferior central de cada caixa e
    marcado com um circulo: e o ponto que a regra espacial testa, e ve-lo na
    tela e o que permite entender por que um objeto contou ou nao como bloqueio.

    Objeto DENTRO da zona sai em vermelho; fora, em laranja. Assim da para
    conferir a decisao do sistema olhando um frame, sem ler log.
    """
    tela = frame.copy()
    bloqueada = state is not None and state.status is RouteStatus.BLOQUEADA
    cor_zona = VERMELHO if bloqueada else VERDE

    contorno = np.array(polygon, dtype=np.int32).reshape(-1, 1, 2)
    sombra = tela.copy()
    cv2.fillPoly(sombra, [contorno], cor_zona)
    tela = cv2.addWeighted(sombra, 0.25, tela, 0.75, 0)
    cv2.polylines(tela, [contorno], isClosed=True, color=cor_zona, thickness=2, lineType=cv2.LINE_AA)

    invasores = {id(d) for d in (state.intruders if state else [])}
    for deteccao in detections:
        invadindo = id(deteccao) in invasores
        cor = VERMELHO if invadindo else LARANJA
        p1 = (int(deteccao.x1), int(deteccao.y1))
        p2 = (int(deteccao.x2), int(deteccao.y2))
        cv2.rectangle(tela, p1, p2, cor, 2 if invadindo else 1, cv2.LINE_AA)

        bx, by = deteccao.bottom_center
        cv2.circle(tela, (int(bx), int(by)), 6, cor, -1, cv2.LINE_AA)
        cv2.circle(tela, (int(bx), int(by)), 6, PRETO, 1, cv2.LINE_AA)

        rotulo = f"{deteccao.class_name} {deteccao.confidence:.2f}"
        _texto(tela, rotulo, (p1[0], max(14, p1[1] - 6)), 0.5, cor)

    if state is not None:
        tela = _faixa_status(tela, state)
    return tela


def _faixa_status(frame: np.ndarray, state: BlockageState) -> np.ndarray:
    """Barra superior com o veredito, legivel de longe numa projecao."""
    altura_barra = 46
    bloqueada = state.status is RouteStatus.BLOQUEADA
    cor = VERMELHO if bloqueada else VERDE

    barra = frame.copy()
    cv2.rectangle(barra, (0, 0), (frame.shape[1], altura_barra), cor, -1)
    frame = cv2.addWeighted(barra, 0.85, frame, 0.15, 0)

    _texto(frame, state.status.value, (16, 33), 0.95, BRANCO, 2)

    if bloqueada:
        detalhe = ", ".join(sorted({d.class_name for d in state.intruders})) or "objeto"
    else:
        detalhe = f"{state.consecutive_frames} frame(s) sem invasao"
    # Medir o texto em vez de estimar pela contagem de caracteres: a fonte e
    # proporcional, e o chute cortava a frase na borda direita.
    (largura_txt, _), _ = cv2.getTextSize(detalhe, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    _texto(frame, detalhe, (max(8, frame.shape[1] - 14 - largura_txt), 31), 0.55, BRANCO)
    return frame


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
