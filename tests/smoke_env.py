"""Portao 1: o ambiente esta de pe?

Imprime versoes, disponibilidade de GPU e o device que o projeto vai usar.
Nao falha por ausencia de GPU - reporta e segue, porque CPU e um caminho valido.

Uso: .venv/bin/python tests/smoke_env.py
"""

from __future__ import annotations

import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Importar o pacote do projeto antes de qualquer `import ultralytics`: e ele
# que aponta YOLO_CONFIG_DIR para dentro do repositorio, evitando que a
# biblioteca crie configuracao global em ~/.config/Ultralytics.
import src  # noqa: E402,F401


def main() -> int:
    print("=" * 62)
    print("SMOKE 1/3 — AMBIENTE")
    print("=" * 62)
    print(f"Python      : {platform.python_version()} ({sys.executable})")
    print(f"Sistema     : {platform.system()} {platform.release()}")

    falhas: list[str] = []

    try:
        import numpy

        print(f"numpy       : {numpy.__version__}")
    except Exception as exc:  # noqa: BLE001
        falhas.append(f"numpy: {exc}")

    try:
        import cv2

        print(f"opencv      : {cv2.__version__}")
    except Exception as exc:  # noqa: BLE001
        falhas.append(f"opencv: {exc}")

    try:
        import pandas

        print(f"pandas      : {pandas.__version__}")
    except Exception as exc:  # noqa: BLE001
        falhas.append(f"pandas: {exc}")

    try:
        import streamlit

        print(f"streamlit   : {streamlit.__version__}")
    except Exception as exc:  # noqa: BLE001
        falhas.append(f"streamlit: {exc}")

    try:
        import PIL

        print(f"pillow      : {PIL.__version__}")
    except Exception as exc:  # noqa: BLE001
        falhas.append(f"pillow: {exc}")

    try:
        import ultralytics

        print(f"ultralytics : {ultralytics.__version__}")
    except Exception as exc:  # noqa: BLE001
        falhas.append(f"ultralytics: {exc}")

    try:
        import torch

        print(f"torch       : {torch.__version__}")
        print(f"CUDA build  : {torch.version.cuda}")
        disponivel = torch.cuda.is_available()
        print(f"CUDA ativa  : {disponivel}")
        if disponivel:
            major, minor = torch.cuda.get_device_capability(0)
            arquiteturas = torch.cuda.get_arch_list()
            print(f"GPU         : {torch.cuda.get_device_name(0)}")
            print(f"Capability  : sm_{major}{minor}")
            print(f"Wheel cobre : {', '.join(a for a in arquiteturas if a.startswith('sm_'))}")
            if f"sm_{major}{minor}" not in arquiteturas:
                print(
                    "AVISO: a wheel do PyTorch nao foi compilada para esta GPU. "
                    "O projeto vai cair para CPU."
                )
        else:
            print("AVISO: rodando em CPU. Funciona, mas a demo fica mais lenta.")
    except Exception as exc:  # noqa: BLE001
        falhas.append(f"torch: {exc}")

    try:
        from src.detector import ObstacleDetector

        print(f"Device alvo : {ObstacleDetector._pick_device()}")
    except Exception as exc:  # noqa: BLE001
        falhas.append(f"src.detector: {exc}")

    print("-" * 62)
    if falhas:
        print("FALHOU:")
        for f in falhas:
            print(f"  - {f}")
        return 1
    print("OK — ambiente pronto.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
