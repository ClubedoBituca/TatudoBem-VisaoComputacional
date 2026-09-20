"""Testes da regra espacial (src/blockage.py).

Nao precisa de video, modelo nem GPU: usa um poligono sintetico e deteccoes
falsas. Roda em milissegundos, entao da para rodar a cada alteracao.

Escrito com `assert` puro em vez de pytest porque o AGENTS.md limita as
dependencias do MVP - se a frente 4 decidir adotar pytest, estas funcoes
`test_*` ja sao compativeis e passam a rodar com `pytest tests/`.

Uso: .venv/bin/python tests/test_blockage.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.blockage import (  # noqa: E402
    BlockageTracker,
    RouteStatus,
    denormalize_polygon,
    detections_in_zone,
    point_in_polygon,
)
from src.detector import Detection  # noqa: E402

# Quadrado de 100x100 com canto em (100,100). Simples de conferir de cabeca.
QUADRADO = [(100.0, 100.0), (200.0, 100.0), (200.0, 200.0), (100.0, 200.0)]


def caixa(x_centro: float, y_base: float, classe: str = "chair") -> Detection:
    """Deteccao falsa cujo ponto inferior central e exatamente (x_centro, y_base)."""
    return Detection(
        class_id=56,
        class_name=classe,
        confidence=0.9,
        x1=x_centro - 20,
        y1=y_base - 50,
        x2=x_centro + 20,
        y2=y_base,
    )


def test_denormalize_polygon() -> None:
    triangulo = [(0.0, 0.0), (0.5, 0.0), (0.5, 1.0)]
    assert denormalize_polygon(triangulo, 960, 540) == [(0.0, 0.0), (480.0, 0.0), (480.0, 540.0)]

    # Frame invalido tem que estourar, nao devolver poligono degenerado.
    try:
        denormalize_polygon(triangulo, 0, 540)
    except ValueError:
        pass
    else:
        raise AssertionError("esperava ValueError para largura zero")


def test_point_in_polygon() -> None:
    assert point_in_polygon((150, 150), QUADRADO) is True, "centro esta dentro"
    assert point_in_polygon((50, 150), QUADRADO) is False, "a esquerda esta fora"
    assert point_in_polygon((150, 250), QUADRADO) is False, "abaixo esta fora"
    assert point_in_polygon((100, 150), QUADRADO) is True, "aresta conta como dentro"
    assert point_in_polygon((100, 100), QUADRADO) is True, "vertice conta como dentro"

    # Poligono degenerado e erro de configuracao: melhor estourar do que
    # silenciosamente nunca detectar nada.
    try:
        point_in_polygon((150, 150), [(0.0, 0.0), (1.0, 1.0)])
    except ValueError:
        pass
    else:
        raise AssertionError("esperava ValueError para poligono com 2 vertices")


def test_usa_ponto_inferior_central() -> None:
    """A caixa cruza a zona, mas a base esta fora: nao e bloqueio.

    Este e o teste que protege a invariante do projeto. Se alguem trocar a
    regra para centro da caixa ou IoU, ele quebra - e e para quebrar mesmo.
    """
    alta = Detection(
        class_id=56, class_name="chair", confidence=0.9,
        x1=140, y1=120,   # topo dentro do quadrado
        x2=160, y2=260,   # base abaixo dele, fora
    )
    assert alta.bottom_center == (150.0, 260.0)
    assert detections_in_zone([alta], QUADRADO) == []


def test_detections_in_zone() -> None:
    dentro = caixa(150, 150)
    fora = caixa(150, 400)
    resultado = detections_in_zone([dentro, fora], QUADRADO)
    assert resultado == [dentro]
    assert detections_in_zone([], QUADRADO) == []


def test_tracker_confirma_apos_n_frames() -> None:
    tracker = BlockageTracker(QUADRADO, confirm_frames=3, release_frames=3)
    intruso = [caixa(150, 150)]

    estado = tracker.update(intruso)
    assert estado.status is RouteStatus.LIVRE, "1 frame nao confirma"
    assert estado.consecutive_frames == 1
    assert estado.just_confirmed is False

    estado = tracker.update(intruso)
    assert estado.status is RouteStatus.LIVRE, "2 frames ainda nao confirmam"

    estado = tracker.update(intruso)
    assert estado.status is RouteStatus.BLOQUEADA, "o 3o frame confirma"
    assert estado.just_confirmed is True
    assert len(estado.intruders) == 1

    estado = tracker.update(intruso)
    assert estado.status is RouteStatus.BLOQUEADA
    assert estado.just_confirmed is False, "so dispara no frame da transicao"


def test_tracker_libera_apos_n_frames() -> None:
    tracker = BlockageTracker(QUADRADO, confirm_frames=2, release_frames=3)
    intruso = [caixa(150, 150)]
    for _ in range(2):
        tracker.update(intruso)
    assert tracker.status is RouteStatus.BLOQUEADA

    assert tracker.update([]).status is RouteStatus.BLOQUEADA, "1 frame limpo nao libera"
    assert tracker.update([]).status is RouteStatus.BLOQUEADA, "2 frames limpos nao liberam"

    estado = tracker.update([])
    assert estado.status is RouteStatus.LIVRE, "o 3o frame limpo libera"
    assert estado.just_released is True
    assert tracker.update([]).just_released is False, "so dispara na transicao"


def test_deteccao_piscante_nao_confirma() -> None:
    """Alterna presente/ausente. E o caso que justifica a persistencia temporal."""
    tracker = BlockageTracker(QUADRADO, confirm_frames=3, release_frames=3)
    intruso = [caixa(150, 150)]
    for i in range(10):
        estado = tracker.update(intruso if i % 2 == 0 else [])
        assert estado.status is RouteStatus.LIVRE, f"frame {i} nao devia confirmar"


def test_duracao_inclui_frames_de_confirmacao() -> None:
    """A barreira existia durante os frames que a confirmaram.

    Contar a duracao so a partir da confirmacao subestimaria todo evento em
    `confirm_frames` frames - e a metrica de duracao da frente 4 iria junto.
    """
    tracker = BlockageTracker(QUADRADO, confirm_frames=4, release_frames=2)
    intruso = [caixa(150, 150)]
    for _ in range(4):
        tracker.update(intruso)
    assert tracker.blocked_frames == 4

    tracker.update(intruso)
    assert tracker.blocked_frames == 5


def test_reset() -> None:
    tracker = BlockageTracker(QUADRADO, confirm_frames=2, release_frames=2)
    intruso = [caixa(150, 150)]
    tracker.update(intruso)
    tracker.update(intruso)
    assert tracker.status is RouteStatus.BLOQUEADA

    tracker.reset()
    assert tracker.status is RouteStatus.LIVRE
    assert tracker.blocked_frames == 0
    assert tracker.update(intruso).status is RouteStatus.LIVRE, "contadores zerados"


def main() -> int:
    testes = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    falhas = 0
    print("=" * 62)
    print("TESTES DA REGRA ESPACIAL")
    print("=" * 62)
    for teste in testes:
        try:
            teste()
        except AssertionError as exc:
            falhas += 1
            print(f"  FALHOU  {teste.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001
            falhas += 1
            print(f"  ERRO    {teste.__name__}: {type(exc).__name__}: {exc}")
        else:
            print(f"  ok      {teste.__name__}")
    print("-" * 62)
    print(f"{len(testes) - falhas}/{len(testes)} passaram.")
    return 1 if falhas else 0


if __name__ == "__main__":
    raise SystemExit(main())
