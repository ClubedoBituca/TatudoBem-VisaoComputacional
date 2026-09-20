"""EXPERIMENTAL - deteccao de obstrucao por aparencia, sem depender de classe.

PROBLEMA QUE RESOLVE: o YOLO so conhece as 80 classes do COCO. Caixa de papelao,
cone de obra, tapume e entulho nao estao la, mas bloqueiam a passagem do mesmo
jeito. Este modulo pergunta outra coisa: "o que esta aqui ainda parece piso?".

ESTADO: PARCIAL. Nao esta ligado ao pipeline e nao deve entrar na demo sem mais
trabalho. Ver "limitacoes conhecidas" no fim desta docstring.

COMO FUNCIONA:
  1. Modelo de piso - dentro da faixa livre e fora do piso tatil, o percentil 70
     da luminancia de cada linha da o nivel do piso ali. Objeto e mais escuro que
     o piso, entao o percentil alto o ignora. Depois suaviza ao longo de y com um
     polinomio de grau 3, porque o piso varia devagar com a profundidade.
     Medir DENTRO da faixa e essencial: fora dela ha mobiliario e, nas linhas mais
     altas, nem sequer e piso - e o fundo do corredor.
  2. Modelo do piso tatil - um unico deslocamento de luminancia em relacao ao
     piso, mediana ao longo de toda a faixa. Localizado nao corrompe a mediana.
  3. Desvio - o que foge dos dois modelos e candidato a obstrucao. As bordas do
     piso tatil sao zona de transicao e ficam de fora, senao viram falso positivo
     continuo ao longo de todo o corredor.
  4. Componentes conexos acima de uma area minima viram obstrucoes, com o ponto
     inferior central como contato com o piso - mesma convencao do detector YOLO.

LIMITACOES CONHECIDAS - medidas, nao estimadas:
  * FALSO POSITIVO EM CAMINHO LIVRE. Em livre.mp4, que nao tem obstaculo nenhum,
    acusou 3 regioes. Esse e o bloqueio para entrar no pipeline: um detector que
    alarma com o caminho limpo e pior que nao ter detector.
  * Falso positivo em faixa estreita ao longo das bordas do piso tatil, mesmo com
    a margem de transicao.
  * Em caixas.mp4 pegou corretamente a caixa que invade o caminho pela esquerda -
    objeto que o YOLO nao ve de forma nenhuma. Mas nao pegou a caixa apoiada sobre
    o piso tatil no meio do corredor, onde a faixa e estreita e o objeto fica na
    fronteira de y_top.
  * Exige que a deteccao da faixa (src/lane.py) tenha acertado. Em clipe onde ela
    erra, isto erra junto.

O QUE PROVAVELMENTE FALTA: o modelo de piso por linha e simples demais para lidar
com reflexo no piso polido, sombra de luminaria e as faixas decorativas. Um
caminho mais promissor e comparar cada frame com a mediana temporal do proprio
clipe em vez de com um modelo parametrico - o que muda em relacao ao proprio
corredor e mais facil de medir do que o que "parece piso".
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .detector import Detection
from .lane import GuideLane

CLASSE_DESCONHECIDA = "obstrucao"
ID_DESCONHECIDO = -1


@dataclass(frozen=True)
class SurfaceModel:
    """Como o piso e o piso tatil deveriam parecer, por linha da imagem."""

    floor_level: np.ndarray  # luminancia esperada do piso, indexada por y em pixels
    strip_delta: float       # quanto o piso tatil escurece em relacao ao piso

    def expected_at(self, y: int, sobre_piso_tatil: bool) -> float:
        base = float(self.floor_level[y])
        return base - self.strip_delta if sobre_piso_tatil else base


def _luminancia(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(cv2.GaussianBlur(frame, (5, 5), 0), cv2.COLOR_BGR2LAB)[:, :, 0].astype(int)


def _limites(lane: GuideLane, y: int, largura: int, ratio: float) -> tuple[int, int, int, int]:
    yn = y / lane.frame_height
    centro = lane.center_at(yn)
    meia_livre = ratio * lane.half_width_at(yn)
    meia_tatil = lane.half_width_at(yn)
    return (
        max(0, int((centro - meia_livre) * largura)),
        min(largura, int((centro + meia_livre) * largura)),
        max(0, int((centro - meia_tatil) * largura)),
        min(largura, int((centro + meia_tatil) * largura)),
    )


def build_surface_model(
    fundo: np.ndarray,
    lane: GuideLane,
    ratio: float = 4.0,
    margem: int = 6,
) -> SurfaceModel:
    """Aprende o piso e o piso tatil a partir de uma imagem de fundo."""
    altura, largura = fundo.shape[:2]
    lum = _luminancia(fundo)

    nivel = np.full(altura, np.nan)
    nivel_tatil = np.full(altura, np.nan)
    for y in range(int(lane.y_top * altura), altura):
        e, d, se, sd = _limites(lane, y, largura, ratio)
        if d - e < 20:
            continue
        lados = np.concatenate([lum[y, e:max(e, se - margem)], lum[y, min(d, sd + margem):d]])
        if len(lados) >= 15:
            nivel[y] = np.percentile(lados, 70)
        if sd - se >= 5:
            nivel_tatil[y] = np.median(lum[y, se + 2:sd - 2])

    validos = ~np.isnan(nivel)
    if validos.sum() < 30:
        raise ValueError("Linhas insuficientes para modelar o piso.")
    ys = np.arange(altura)
    piso = np.polyval(np.polyfit(ys[validos], nivel[validos], 3), ys)
    delta = float(np.nanmedian(piso - nivel_tatil))
    return SurfaceModel(floor_level=piso, strip_delta=delta)


def find_obstructions(
    frame: np.ndarray,
    lane: GuideLane,
    modelo: SurfaceModel,
    ratio: float = 4.0,
    limiar: float = 38.0,
    area_minima: float = 0.0012,
    margem: int = 6,
    ignorar: list[Detection] | None = None,
) -> list[Detection]:
    """Regioes da faixa livre que nao parecem piso.

    `ignorar` recebe as deteccoes nao obstrutivas (pessoas) - a area delas e
    apagada da mascara, senao todo pedestre viraria obstrucao desconhecida.

    Devolve `Detection` com classe "obstrucao" e confianca proporcional a area,
    para fluir pelo mesmo `BlockageTracker` das deteccoes do YOLO.
    """
    altura, largura = frame.shape[:2]
    lum = _luminancia(frame)
    mascara = np.zeros((altura, largura), np.uint8)

    for y in range(int(lane.y_top * altura), altura):
        e, d, se, sd = _limites(lane, y, largura, ratio)
        if d - e < 8:
            continue
        # Tres trechos: piso a esquerda, piso tatil, piso a direita. As bordas do
        # piso tatil sao transicao e ficam fora - pixel intermediario nao casa com
        # nenhum dos dois modelos e viraria falso positivo ao longo do corredor.
        trechos = (
            (e, max(e, se - margem), modelo.expected_at(y, False)),
            (min(d, se + margem), max(0, sd - margem), modelo.expected_at(y, True)),
            (min(d, sd + margem), d, modelo.expected_at(y, False)),
        )
        for inicio, fim, esperado in trechos:
            if fim - inicio < 2:
                continue
            mascara[y, inicio:fim] = (np.abs(lum[y, inicio:fim] - esperado) > limiar) * 255

    for pessoa in ignorar or []:
        cv2.rectangle(
            mascara,
            (int(pessoa.x1), int(pessoa.y1)),
            (int(pessoa.x2), int(pessoa.y2)),
            0, -1,
        )

    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, np.ones((13, 13), np.uint8))

    total, _, stats, _ = cv2.connectedComponentsWithStats(mascara)
    minimo = area_minima * altura * largura
    achados: list[Detection] = []
    for i in range(1, total):
        x, y, largura_c, altura_c, area = stats[i]
        if area < minimo:
            continue
        achados.append(
            Detection(
                class_id=ID_DESCONHECIDO,
                class_name=CLASSE_DESCONHECIDA,
                confidence=min(0.99, area / (6 * minimo)),
                x1=float(x), y1=float(y),
                x2=float(x + largura_c), y2=float(y + altura_c),
            )
        )
    return achados
