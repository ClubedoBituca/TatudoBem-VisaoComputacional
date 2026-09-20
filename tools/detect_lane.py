"""Localiza a faixa guia (piso tatil) num video e mostra o que foi detectado.

Uso:
    python tools/detect_lane.py data/samples/livre.mp4
    python tools/detect_lane.py data/samples/livre.mp4 --ratio 5
    python tools/detect_lane.py data/samples/livre.mp4 --salvar-zona corredor_b21

Sem --salvar-zona nada e gravado: so mede, desenha e imprime. Use o render em
outputs/demos/ para conferir o alinhamento antes de aceitar o resultado.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import src  # noqa: E402,F401
from src.config import CONFIG_PATH, PROJECT_ROOT, load_config  # noqa: E402
from src.lane import LaneNotFound, background_median, detect_guide_lane  # noqa: E402

CIANO = (255, 220, 0)
VERDE = (60, 200, 75)
MAGENTA = (255, 0, 255)


def desenhar(fundo, faixa, ratio: float):
    vis = fundo.copy()
    h, w = vis.shape[:2]
    for poly, cor in [(faixa.free_path_polygon(ratio), VERDE), (faixa.strip_polygon(), CIANO)]:
        pts = np.array([[int(x * w), int(y * h)] for x, y in poly], np.int32).reshape(-1, 1, 2)
        cv2.polylines(vis, [pts], True, cor, 2, cv2.LINE_AA)
    cv2.line(
        vis,
        (int(faixa.center_at(faixa.y_top) * w), int(faixa.y_top * h)),
        (int(faixa.center_at(1.0) * w), h),
        MAGENTA, 1, cv2.LINE_AA,
    )
    for txt, cor, y in [
        ("piso tatil detectado", CIANO, 22),
        (f"faixa livre ({ratio:g}x)", VERDE, 44),
        ("eixo", MAGENTA, 66),
    ]:
        cv2.putText(vis, txt, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(vis, txt, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor, 1, cv2.LINE_AA)
    return vis


def main() -> int:
    parser = argparse.ArgumentParser(description="Detecta a faixa guia num video.")
    parser.add_argument("video")
    parser.add_argument("--ratio", type=float, help="largura da faixa livre em multiplos do piso tatil")
    parser.add_argument("--salvar-zona", dest="zona", help="grava o poligono em config/zones.json com este nome")
    args = parser.parse_args()

    cfg = load_config()
    ratio = args.ratio if args.ratio is not None else cfg.get("lane", {}).get("free_width_ratio", 4.0)
    entrada = Path(args.video).resolve()

    try:
        fundo = background_median(entrada, cfg.get("lane", {}).get("background_samples", 25))
        faixa = detect_guide_lane(fundo)
    except LaneNotFound as exc:
        print(f"FALHOU: {exc}")
        print()
        print("Caminhos possiveis: baixar lane.min_coverage, ajustar lane.y_top, ou")
        print("desenhar a zona a mao com tools/draw_zone.py.")
        return 1

    print(f"Video       : {entrada.name}")
    print(f"Cobertura   : {faixa.coverage:.2f}   linhas usadas: {faixa.rows_used}")
    print(f"Eixo        : x={faixa.center_at(1.0):.4f} na base, x={faixa.center_at(faixa.y_top):.4f} no topo")
    print(f"Piso tatil  : {2 * faixa.half_width_at(1.0):.4f} de largura na base, "
          f"{2 * faixa.half_width_at(faixa.y_top):.4f} no topo")
    livre = faixa.free_path_polygon(ratio)
    print(f"Faixa livre : {livre[1][0] - livre[0][0]:.4f} de largura na base ({ratio:g}x o piso tatil)")
    print()
    print("Poligono da faixa livre (normalizado):")
    for x, y in livre:
        print(f"  [{x:.4f}, {y:.4f}]   folga ao eixo: {abs(x - faixa.center_at(y)):.4f}")

    destino = PROJECT_ROOT / "outputs" / "demos"
    destino.mkdir(parents=True, exist_ok=True)
    saida = destino / f"lane_{entrada.stem}.png"
    cv2.imwrite(str(saida), desenhar(fundo, faixa, ratio))
    print()
    print(f"Conferencia visual: {saida}")

    if args.zona:
        dados = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        dados.setdefault("zones", {})[args.zona] = {
            "description": (
                f"Faixa livre derivada automaticamente do piso tatil detectado em "
                f"{entrada.name} (tools/detect_lane.py), com largura {ratio:g}x a do piso tatil. "
                f"Cobertura {faixa.coverage:.2f}, {faixa.rows_used} linhas."
            ),
            "source_hint": f"data/samples/{entrada.name}",
            "polygon_norm": [[round(x, 4), round(y, 4)] for x, y in livre],
        }
        dados["active_zone"] = args.zona
        CONFIG_PATH.write_text(json.dumps(dados, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Zona '{args.zona}' gravada e ativada. config/zones.json e compartilhado - avise o time.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
