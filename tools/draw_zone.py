"""Desenha o poligono da zona acessivel sobre o primeiro frame de um video.

Abre uma janela do OpenCV com o primeiro frame, coleta os vertices por clique
e grava o poligono NORMALIZADO (0-1) em config/zones.json. Normalizado porque
o mesmo trecho pode ser reprocessado em outra resolucao.

Uso:
    python tools/draw_zone.py data/samples/obstaculo.mp4
    python tools/draw_zone.py data/samples/obstaculo.mp4 --zona corredor_ic
    python tools/draw_zone.py data/samples/obstaculo.mp4 --frame 90   # outro frame

Controles:
    clique esquerdo  adiciona vertice
    clique direito   desfaz o ultimo
    u                desfaz o ultimo
    r                recomeca do zero
    ENTER            salva e sai
    ESC / q          cancela sem salvar

Dica: clique seguindo a borda da faixa de passagem, no sentido horario,
comecando por um canto proximo a camera. Minimo de 3 pontos; 4 a 6 costumam
bastar para uma calcada em perspectiva.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402

import src  # noqa: E402,F401
from src.capture import CaptureError, VideoSource  # noqa: E402
from src.config import CONFIG_PATH  # noqa: E402

JANELA = "Caminho Livre - desenhe a zona (ENTER salva, ESC cancela)"

VERDE = (0, 200, 0)
AMARELO = (0, 220, 220)
PRETO = (0, 0, 0)
BRANCO = (255, 255, 255)


def desenhar(frame, pontos: list[tuple[int, int]]):
    """Redesenha o frame com os vertices e o poligono em construcao."""
    tela = frame.copy()

    if len(pontos) >= 2:
        cv2.polylines(tela, [_np_pontos(pontos)], isClosed=len(pontos) >= 3, color=VERDE, thickness=2)
    if len(pontos) >= 3:
        # Preenchimento translucido ajuda a enxergar a area coberta.
        sombra = tela.copy()
        cv2.fillPoly(sombra, [_np_pontos(pontos)], VERDE)
        tela = cv2.addWeighted(sombra, 0.25, tela, 0.75, 0)

    for i, (x, y) in enumerate(pontos):
        cv2.circle(tela, (x, y), 6, AMARELO, -1)
        cv2.circle(tela, (x, y), 6, PRETO, 1)
        cv2.putText(tela, str(i + 1), (x + 9, y - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.5, PRETO, 3)
        cv2.putText(tela, str(i + 1), (x + 9, y - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.5, BRANCO, 1)

    estado = f"{len(pontos)} ponto(s)  |  ENTER salva  u desfaz  r recomeca  ESC cancela"
    if len(pontos) < 3:
        estado = f"{len(pontos)}/3 pontos minimos  |  clique para adicionar"
    cv2.putText(tela, estado, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, PRETO, 4)
    cv2.putText(tela, estado, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, BRANCO, 1)
    return tela


def _np_pontos(pontos: list[tuple[int, int]]):
    import numpy as np

    return np.array(pontos, dtype=np.int32).reshape(-1, 1, 2)


def salvar(pontos: list[tuple[int, int]], largura: int, altura: int, zona: str, origem: Path) -> None:
    """Grava a zona em config/zones.json, preservando as demais zonas."""
    with CONFIG_PATH.open(encoding="utf-8") as f:
        cfg = json.load(f)

    cfg.setdefault("zones", {})[zona] = {
        "description": f"Zona desenhada sobre o primeiro frame de {origem.name}.",
        "source_hint": str(origem.relative_to(Path.cwd())) if origem.is_relative_to(Path.cwd()) else str(origem),
        "polygon_norm": [[round(x / largura, 4), round(y / altura, 4)] for x, y in pontos],
    }
    cfg["active_zone"] = zona

    with CONFIG_PATH.open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
        f.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Desenha a zona acessivel sobre um video.")
    parser.add_argument("video", help="caminho do .mp4")
    parser.add_argument("--zona", help="nome da zona em config/zones.json (padrao: nome do arquivo)")
    parser.add_argument("--frame", type=int, default=0, help="indice do frame a usar como fundo")
    args = parser.parse_args()

    origem = Path(args.video).resolve()
    zona = args.zona or origem.stem

    try:
        with VideoSource(origem) as fonte:
            frame = None
            for indice, atual in fonte:
                frame = atual
                if indice >= args.frame:
                    break
    except CaptureError as exc:
        print(f"ERRO: {exc}")
        return 1

    if frame is None:
        print("ERRO: nenhum frame lido do video.")
        return 1

    altura, largura = frame.shape[:2]
    pontos: list[tuple[int, int]] = []

    def ao_clicar(evento, x, y, _flags, _param):
        if evento == cv2.EVENT_LBUTTONDOWN:
            pontos.append((x, y))
        elif evento == cv2.EVENT_RBUTTONDOWN and pontos:
            pontos.pop()

    try:
        cv2.namedWindow(JANELA, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(JANELA, ao_clicar)
    except cv2.error as exc:
        # Sem servidor grafico nao da para desenhar. Melhor dizer isso com
        # todas as letras do que estourar um erro obscuro do Qt.
        print("ERRO: nao foi possivel abrir uma janela grafica.")
        print(f"       {exc}")
        print()
        print("No WSL, o WSLg costuma resolver (DISPLAY=:0). Sem interface grafica,")
        print("edite o polygon_norm da zona diretamente em config/zones.json:")
        print("  [[x1, y1], [x2, y2], ...] com valores entre 0 e 1,")
        print("  onde (0,0) e o canto superior esquerdo e (1,1) o inferior direito.")
        return 1

    print(f"Video : {origem.name}  ({largura}x{altura}, frame {args.frame})")
    print(f"Zona  : {zona}")
    print("Clique os vertices seguindo a faixa de passagem. ENTER salva, ESC cancela.")

    while True:
        cv2.imshow(JANELA, desenhar(frame, pontos))
        tecla = cv2.waitKey(20) & 0xFF

        if tecla in (27, ord("q")):
            cv2.destroyAllWindows()
            print("Cancelado. Nada foi gravado.")
            return 1
        if tecla in (ord("u"),) and pontos:
            pontos.pop()
        elif tecla == ord("r"):
            pontos.clear()
        elif tecla in (13, 10):  # ENTER
            if len(pontos) < 3:
                print(f"Faltam vertices: {len(pontos)}/3. Clique mais pontos.")
                continue
            break

    cv2.destroyAllWindows()
    salvar(pontos, largura, altura, zona, origem)
    print(f"Zona '{zona}' gravada em {CONFIG_PATH} com {len(pontos)} vertices.")
    print("Ela ja esta como active_zone. Avise o time: config/zones.json e compartilhado.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
