"""Renderiza um video ANOTADO: zona, objetos e veredito queimados na imagem.

Para que serve: ver o sistema decidindo, frame a frame, e ter um arquivo para
mostrar na apresentacao sem depender da demo ao vivo.

ATENCAO - privacidade. O pipeline em operacao NAO grava video (ver AGENTS.md
e PROJECT_CONTEXT.md): so registra eventos em CSV. Esta ferramenta e uma acao
deliberada, rodada a mao, fora do fluxo normal. Ela gera um arquivo que contem
as pessoas que aparecem na gravacao original, ainda que o sistema nao as
identifique. Antes de por num slide ou mandar em grupo, olhe quem aparece no
clipe - `pessoas.mp4` em especial - e prefira um trecho sem rostos legiveis.
A saida vai para outputs/demos/, que nao e versionada.

Uso:
    python tools/render_video.py data/samples/obstruido.mp4
    python tools/render_video.py data/samples/obstruido.mp4 --inicio 8 --fim 25
    python tools/render_video.py data/samples/borda.mp4 --zona borda --saida borda_anotado.mp4
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402

import src  # noqa: E402,F401
from src.blockage import BlockageTracker, RouteStatus, denormalize_polygon  # noqa: E402
from src.capture import CaptureError, VideoSource  # noqa: E402
from src.config import PROJECT_ROOT, load_config  # noqa: E402
from src.detector import ObstacleDetector  # noqa: E402
from src.ui import draw_overlay  # noqa: E402

SAIDA_PADRAO = PROJECT_ROOT / "outputs" / "demos"


def main() -> int:
    parser = argparse.ArgumentParser(description="Renderiza video anotado com a decisao do sistema.")
    parser.add_argument("video", help="caminho do .mp4 de entrada")
    parser.add_argument("--zona", help="nome da zona em config/zones.json (padrao: a zona ativa)")
    parser.add_argument("--saida", help="nome do arquivo de saida (padrao: <entrada>_anotado.mp4)")
    parser.add_argument("--inicio", type=float, default=0.0, help="segundo inicial")
    parser.add_argument("--fim", type=float, default=0.0, help="segundo final (0 = ate o fim)")
    args = parser.parse_args()

    cfg = load_config()
    zona = args.zona or cfg["active_zone"]
    if zona not in cfg["zones"]:
        print(f"ERRO: zona '{zona}' nao existe em config/zones.json.")
        print(f"       Disponiveis: {', '.join(cfg['zones'])}")
        return 1

    entrada = Path(args.video).resolve()
    SAIDA_PADRAO.mkdir(parents=True, exist_ok=True)
    saida = SAIDA_PADRAO / (args.saida or f"{entrada.stem}_anotado.mp4")

    detector = ObstacleDetector()

    try:
        fonte = VideoSource(entrada)
    except CaptureError as exc:
        print(f"ERRO: {exc}")
        return 1

    with fonte:
        info = fonte.info
        poligono = denormalize_polygon(
            [tuple(p) for p in cfg["zones"][zona]["polygon_norm"]],
            info.work_width,
            info.work_height,
        )
        rastreador = BlockageTracker(poligono)

        fps_saida = (info.fps / fonte.frame_stride) if info.fps > 0 else 15.0
        escritor = cv2.VideoWriter(
            str(saida), cv2.VideoWriter_fourcc(*"mp4v"), fps_saida,
            (info.work_width, info.work_height),
        )
        if not escritor.isOpened():
            print(f"ERRO: nao foi possivel criar {saida}. Codec mp4v indisponivel?")
            return 1

        print(f"Entrada : {entrada.name}  ({info.native_width}x{info.native_height}, {info.fps:.1f} fps)")
        print(f"Zona    : {zona}  ({len(poligono)} vertices)")
        print(f"Device  : {detector.device}   stride: {fonte.frame_stride}")
        print(f"Saida   : {saida}  a {fps_saida:.1f} fps")
        print()

        # Cada frame lido avanca frame_stride frames do video original.
        seg_por_frame = fonte.frame_stride / info.fps if info.fps > 0 else 0.0
        eventos, escritos = 0, 0
        inicio_evento = 0.0
        t0 = time.perf_counter()

        for indice, frame in fonte:
            segundo = indice * seg_por_frame
            if segundo < args.inicio:
                continue
            if args.fim and segundo > args.fim:
                break

            # Uma unica inferencia por frame: alem do custo, o overlay marca os
            # invasores por identidade de objeto, e uma segunda chamada devolveria
            # instancias diferentes das que o rastreador avaliou.
            deteccoes, pessoas = detector.predict_split(frame)
            estado = rastreador.update(deteccoes)
            escritor.write(draw_overlay(frame, deteccoes, poligono, estado, passersby=pessoas))
            escritos += 1

            if estado.just_confirmed:
                eventos += 1
                inicio_evento = segundo
                classes = ", ".join(sorted({d.class_name for d in estado.intruders}))
                print(f"  [{segundo:6.2f}s] BARREIRA confirmada  ({classes})")
            elif estado.just_released:
                print(f"  [{segundo:6.2f}s] rota liberada        (durou {segundo - inicio_evento:.1f}s)")

        escritor.release()
        decorrido = time.perf_counter() - t0

    if escritos == 0:
        print("ERRO: nenhum frame no intervalo pedido.")
        return 1

    if rastreador.status is RouteStatus.BLOQUEADA:
        print(f"  [fim] video termina com a rota ainda bloqueada")

    print()
    print(f"{escritos} frames em {decorrido:.1f}s ({escritos / decorrido:.1f} fps de processamento)")
    print(f"{eventos} evento(s) de barreira.")
    print(f"Pronto: {saida}  ({saida.stat().st_size / 1e6:.1f} MB)")
    print()
    print("Confira quem aparece no clipe antes de compartilhar.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
