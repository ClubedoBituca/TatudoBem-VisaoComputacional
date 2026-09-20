"""Portao 3: o YOLO carrega e infere?

Carrega o modelo pre-treinado, roda numa imagem e num punhado de frames de
video, e mede a latencia. Nenhum treinamento acontece aqui.

Uso: .venv/bin/python tests/smoke_inference.py [imagem_ou_video]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402

from src.capture import VideoSource  # noqa: E402
from src.config import PROJECT_ROOT  # noqa: E402
from src.detector import ObstacleDetector  # noqa: E402

IMAGEM_PADRAO = PROJECT_ROOT / "data" / "test" / "bus.jpg"
VIDEO_PADRAO = PROJECT_ROOT / "data" / "test" / "pipeline_check.mp4"
FRAMES_MEDIDOS = 20


def imprimir_deteccoes(deteccoes, titulo: str) -> None:
    print(f"{titulo}: {len(deteccoes)} objeto(s) de interesse")
    for d in deteccoes:
        cx, cy = d.bottom_center
        print(
            f"  - {d.class_name:<14} conf={d.confidence:.2f}  "
            f"caixa=({d.x1:.0f},{d.y1:.0f})-({d.x2:.0f},{d.y2:.0f})  "
            f"ponto_base=({cx:.0f},{cy:.0f})"
        )


def main() -> int:
    print("=" * 62)
    print("SMOKE 3/3 — INFERENCIA YOLO")
    print("=" * 62)

    inicio = time.perf_counter()
    detector = ObstacleDetector()
    print(f"Modelo      : {detector.model_path.name}")
    print(f"Device      : {detector.device}")
    print(f"imgsz       : {detector.imgsz}   conf: {detector.conf_threshold}")
    print(f"Classes     : {len(detector.class_names)} do COCO")
    print(f"Carga       : {(time.perf_counter() - inicio) * 1000:.0f} ms")
    print("-" * 62)

    alvo = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else IMAGEM_PADRAO

    if alvo.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
        frame = cv2.imread(str(alvo))
        if frame is None:
            print(f"FALHOU: nao foi possivel ler a imagem {alvo}")
            return 1
        print(f"Imagem      : {alvo.name} {frame.shape[1]}x{frame.shape[0]}")

        detector.predict(frame)  # descarta o primeiro: inclui warm-up do modelo
        t0 = time.perf_counter()
        alvos = detector.predict(frame)
        latencia = (time.perf_counter() - t0) * 1000
        todas = detector.predict(frame, only_targets=False)

        print(f"Latencia    : {latencia:.1f} ms/frame")
        imprimir_deteccoes(alvos, "Classes-alvo")
        print(f"Todas as classes (menos ignoradas): {len(todas)} objeto(s)")
        for d in todas:
            print(f"  - {d.class_name:<14} conf={d.confidence:.2f}")
        if not todas:
            print("FALHOU: nenhuma deteccao — inferencia suspeita.")
            return 1
    else:
        video = alvo if alvo.exists() else VIDEO_PADRAO
        if not video.exists():
            print(f"FALHOU: video nao encontrado: {video}")
            return 1
        print(f"Video       : {video.name}")
        with VideoSource(video) as fonte:
            latencias: list[float] = []
            total = 0
            for indice, frame in fonte:
                t0 = time.perf_counter()
                deteccoes = detector.predict(frame, only_targets=False)
                latencias.append((time.perf_counter() - t0) * 1000)
                total += len(deteccoes)
                if indice == 0:
                    imprimir_deteccoes(deteccoes, "Primeiro frame")
                if indice + 1 >= FRAMES_MEDIDOS:
                    break
        if not latencias:
            print("FALHOU: nenhum frame processado.")
            return 1
        util = latencias[1:] or latencias  # descarta o warm-up
        media = sum(util) / len(util)
        print(f"Frames      : {len(latencias)}")
        print(f"Latencia    : {media:.1f} ms/frame (~{1000 / media:.1f} fps) | "
              f"1o frame com warm-up: {latencias[0]:.0f} ms")
        print(f"Deteccoes   : {total} no total")

    print("-" * 62)
    print("OK — inferencia funcionando.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
