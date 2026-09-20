"""Deteccao automatica da faixa guia (piso tatil) e da faixa livre de circulacao.

Modulo novo, compartilhado - avise no grupo ao alterar.

Por que existe: um poligono desenhado a mao serve para um enquadramento so.
Trocou o corredor, a camera ou o trecho, tem que desenhar de novo. Aqui o
sistema acha o piso tatil sozinho e deriva dele a faixa trafegavel.

Como funciona, em quatro passos:

  1. FUNDO. Camera estatica + mediana temporal de varios frames = corredor sem
     quem estava passando. Objeto parado permanece, e tudo bem: ele e obstaculo,
     nao ruido.

  2. MASCARA. Por linha da imagem, o piso e a referencia clara (percentil 85 da
     propria linha) e marcamos o que escurece em relacao a ela. Isso pega o piso
     tatil E os objetos - separa-los e o passo seguinte.

  3. EIXO. O piso tatil e a unica estrutura que atravessa TODA a profundidade do
     corredor; um objeto ocupa um trecho. Entao procuramos a reta (da base ate o
     ponto de fuga) com maior cobertura na mascara. Um feixe inteiro de retas
     dentro da faixa empata com cobertura 1,0 - o centro da faixa e a MEDIANA
     desse feixe, nao o primeiro empate.

  4. LARGURA. A partir do eixo, cresce para os lados enquanto a mascara continua,
     linha a linha, e ajusta largura = a*y + b. A rejeicao de outliers e
     ASSIMETRICA: fusao com um objeto encostado so infla a largura, nunca
     encolhe, entao o lado alto e cortado com mais rigor.

Limites conhecidos: exige camera estatica, piso tatil visivelmente mais escuro
que o piso ao redor, e um trecho reto. Corredor curvo, piso tatil da mesma cor
do piso ou camera em movimento quebram a premissa - nesses casos use o poligono
manual de config/zones.json.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .config import load_config

Point = tuple[float, float]
Polygon = list[Point]


class LaneNotFound(RuntimeError):
    """A faixa guia nao foi localizada com confianca suficiente."""


@dataclass(frozen=True)
class GuideLane:
    """Faixa guia localizada, em coordenadas normalizadas (0-1).

    O eixo e uma reta: `axis_top_x` na altura `y_top`, `axis_bottom_x` na base.
    A largura em pixels segue `width_coef = (a, b)` com `largura = a*y_px + b`.
    """

    axis_top_x: float
    axis_bottom_x: float
    y_top: float
    width_coef: tuple[float, float]
    coverage: float
    rows_used: int
    frame_width: int
    frame_height: int

    def center_at(self, y_norm: float) -> float:
        t = (y_norm - self.y_top) / (1.0 - self.y_top)
        return self.axis_top_x + (self.axis_bottom_x - self.axis_top_x) * t

    def half_width_at(self, y_norm: float) -> float:
        a, b = self.width_coef
        largura_px = max(0.0, a * y_norm * self.frame_height + b)
        return largura_px / self.frame_width / 2.0

    def _trapezio(self, fator: float) -> Polygon:
        topo, base = self.y_top, 1.0
        return [
            (self.center_at(base) - fator * self.half_width_at(base), base),
            (self.center_at(base) + fator * self.half_width_at(base), base),
            (self.center_at(topo) + fator * self.half_width_at(topo), topo),
            (self.center_at(topo) - fator * self.half_width_at(topo), topo),
        ]

    def strip_polygon(self) -> Polygon:
        """O piso tatil em si - util para desenhar na tela."""
        return self._trapezio(1.0)

    def free_path_polygon(self, ratio: float | None = None) -> Polygon:
        """A faixa livre de circulacao: o piso tatil alargado simetricamente.

        `ratio` e quantas vezes a largura do piso tatil corresponde a faixa que
        precisa ficar desobstruida. Quem se guia pelo piso tatil ocupa bem mais
        que a largura dele: precisa de folga para o corpo e para a bengala.
        O valor exato da norma (NBR 9050) deve ser conferido por quem a tenha -
        o padrao aqui e uma escolha de engenharia, nao citacao de norma.
        """
        if ratio is None:
            ratio = load_config().get("lane", {}).get("free_width_ratio", 4.0)
        return self._trapezio(float(ratio))


def background_median(video_path: str | Path, samples: int = 25) -> np.ndarray:
    """Mediana temporal de N frames: remove quem passa, mantem o cenario."""
    from .capture import VideoSource

    frames: list[np.ndarray] = []
    with VideoSource(video_path) as fonte:
        total = max(1, fonte.info.frame_count // max(1, fonte.frame_stride))
        passo = max(1, total // max(1, samples))
        for indice, frame in fonte:
            if indice % passo == 0:
                frames.append(frame)
            if len(frames) >= samples:
                break
    if not frames:
        raise LaneNotFound(f"Nenhum frame lido de {video_path}")
    return np.median(np.stack(frames), axis=0).astype(np.uint8)


def _mascara_nao_piso(fundo: np.ndarray, y_top: float, limiar: float) -> np.ndarray:
    altura, largura = fundo.shape[:2]
    luminancia = cv2.cvtColor(cv2.GaussianBlur(fundo, (5, 5), 0), cv2.COLOR_BGR2LAB)[:, :, 0].astype(int)
    mascara = np.zeros((altura, largura), dtype=bool)
    janela = slice(int(0.25 * largura), int(0.75 * largura))
    for y in range(int(y_top * altura), altura):
        linha = luminancia[y]
        referencia = np.percentile(linha[janela], 85)
        mascara[y] = (referencia - linha) > limiar * referencia
    return mascara


def _localizar_eixo(mascara: np.ndarray, y_top: float) -> tuple[float, float, float, int]:
    altura, largura = mascara.shape
    ys = np.arange(int(y_top * altura), altura)
    t = (ys - ys[0]) / max(1, ys[-1] - ys[0])
    base_cand = np.arange(0.30, 0.71, 0.003)
    fuga_cand = np.arange(0.40, 0.61, 0.003)

    grade = np.zeros((len(base_cand), len(fuga_cand)))
    for i, xb in enumerate(base_cand):
        for j, xv in enumerate(fuga_cand):
            xs = ((xv + (xb - xv) * t) * largura).astype(int)
            grade[i, j] = mascara[ys, xs].mean()

    otimos = np.argwhere(grade >= grade.max() - 1e-9)
    xb = float(np.median(base_cand[otimos[:, 0]]))
    xv = float(np.median(fuga_cand[otimos[:, 1]]))
    return xb, xv, float(grade.max()), len(otimos)


def _perfil_largura(mascara: np.ndarray, xb: float, xv: float, y_top: float) -> np.ndarray:
    altura, largura = mascara.shape
    ys = np.arange(int(y_top * altura), altura)
    t = (ys - ys[0]) / max(1, ys[-1] - ys[0])
    centros = ((xv + (xb - xv) * t) * largura).astype(int)

    obs: list[tuple[float, float]] = []
    for y, c in zip(ys, centros):
        if not mascara[y, c]:
            continue
        esquerda = c
        while esquerda > 0 and mascara[y, esquerda - 1]:
            esquerda -= 1
        direita = c
        while direita < largura - 1 and mascara[y, direita + 1]:
            direita += 1
        obs.append((float(y), float(direita - esquerda + 1)))
    return np.array(obs, dtype=float)


def _ajustar_largura(obs: np.ndarray) -> tuple[tuple[float, float], int]:
    y, larg = obs[:, 0], obs[:, 1]
    for _ in range(8):
        coef = np.polyfit(y, larg, 1)
        residuo = larg - np.polyval(coef, y)
        escala = max(1.5, 1.4826 * np.median(np.abs(residuo)))
        # Assimetrico: so a fusao com objeto infla a largura.
        manter = (residuo < 1.2 * escala) & (residuo > -3.0 * escala)
        if manter.sum() < 15:
            break
        y, larg = y[manter], larg[manter]
    a, b = np.polyfit(y, larg, 1)
    return (float(a), float(b)), len(y)


def detect_guide_lane(
    fundo: np.ndarray,
    y_top: float | None = None,
    limiar: float | None = None,
    min_coverage: float | None = None,
) -> GuideLane:
    """Localiza a faixa guia numa imagem de fundo ja limpa."""
    cfg = load_config().get("lane", {})
    y_top = y_top if y_top is not None else cfg.get("y_top", 0.58)
    limiar = limiar if limiar is not None else cfg.get("darkness_threshold", 0.13)
    min_coverage = min_coverage if min_coverage is not None else cfg.get("min_coverage", 0.9)

    altura, largura = fundo.shape[:2]
    mascara = _mascara_nao_piso(fundo, y_top, limiar)
    xb, xv, cobertura, _ = _localizar_eixo(mascara, y_top)

    if cobertura < min_coverage:
        raise LaneNotFound(
            f"Cobertura {cobertura:.2f} abaixo do minimo {min_coverage:.2f}. "
            "O piso tatil pode estar pouco contrastado, encoberto ou ausente."
        )

    obs = _perfil_largura(mascara, xb, xv, y_top)
    if len(obs) < 20:
        raise LaneNotFound(f"Apenas {len(obs)} linhas uteis; insuficiente para ajustar a largura.")

    coef, usadas = _ajustar_largura(obs)
    return GuideLane(
        axis_top_x=xv,
        axis_bottom_x=xb,
        y_top=float(y_top),
        width_coef=coef,
        coverage=cobertura,
        rows_used=usadas,
        frame_width=largura,
        frame_height=altura,
    )


def detect_from_video(video_path: str | Path, samples: int | None = None) -> GuideLane:
    """Atalho: mediana temporal do video e deteccao da faixa."""
    cfg = load_config().get("lane", {})
    samples = samples if samples is not None else cfg.get("background_samples", 25)
    return detect_guide_lane(background_median(video_path, samples))
