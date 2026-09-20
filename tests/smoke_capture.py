"""Portao 2: a leitura de video funciona?

Le alguns frames de um .mp4 e reporta resolucao, fps e o frame de trabalho.
Depois tenta a webcam - e reporta a ausencia como limitacao conhecida, sem
falhar: no WSL2 nao existe /dev/video*, e a entrada do projeto e arquivo.

Uso: .venv/bin/python tests/smoke_capture.py [caminho/do/video.mp4]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.capture import CaptureError, VideoSource, webcam_available  # noqa: E402
from src.config import PROJECT_ROOT  # noqa: E402

VIDEO_PADRAO = PROJECT_ROOT / "data" / "test" / "pipeline_check.mp4"
MAX_FRAMES = 30


def testar_arquivo(caminho: Path) -> bool:
    print(f"Arquivo     : {caminho}")
    if not caminho.exists():
        print("FALHOU: arquivo nao encontrado.")
        print("Gere o clipe de verificacao ou passe um video da UNIFEI como argumento.")
        return False

    try:
        with VideoSource(caminho) as fonte:
            info = fonte.info
            print(f"Resolucao   : {info.native_width}x{info.native_height} nativa")
            print(f"Frame util  : {info.work_width}x{info.work_height} (apos resize)")
            print(f"FPS         : {info.fps:.2f}")
            print(f"Frames      : {info.frame_count} (~{info.duration_seconds:.1f}s)")

            inicio = time.perf_counter()
            lidos = 0
            for _, frame in fonte:
                lidos += 1
                if lidos == 1:
                    print(f"Shape frame : {frame.shape} (altura, largura, canais BGR)")
                if lidos >= MAX_FRAMES:
                    break
            decorrido = time.perf_counter() - inicio
    except CaptureError as exc:
        print(f"FALHOU: {exc}")
        return False

    if lidos == 0:
        print("FALHOU: nenhum frame lido.")
        return False

    print(f"Lidos       : {lidos} frames em {decorrido * 1000:.0f} ms "
          f"({lidos / max(decorrido, 1e-9):.0f} fps de leitura pura)")
    return True


def testar_webcam() -> None:
    print("-" * 62)
    print("Webcam (caminho alternativo, nao e o fluxo principal)")
    devices = sorted(Path("/dev").glob("video*"))
    print(f"/dev/video* : {[d.name for d in devices] or 'nenhum'}")
    if webcam_available(0):
        print("Webcam      : disponivel no indice 0")
    else:
        print("Webcam      : INDISPONIVEL — limitacao conhecida do WSL2.")
        print("              O WSL2 nao expoe camera USB/integrada ao Linux. Para usar")
        print("              camera ao vivo: rodar em Linux/Windows nativo, ou anexar o")
        print("              dispositivo com usbipd-win (requer admin no Windows).")


def main() -> int:
    print("=" * 62)
    print("SMOKE 2/3 — CAPTURA DE VIDEO")
    print("=" * 62)
    caminho = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else VIDEO_PADRAO
    ok = testar_arquivo(caminho)
    testar_webcam()
    print("-" * 62)
    print("OK — leitura de video funcionando." if ok else "FALHOU — ver acima.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
