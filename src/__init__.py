"""Caminho Livre - UNIFEI: pacote principal do MVP.

Importar qualquer coisa de `src` ja isola a configuracao do Ultralytics dentro
do projeto. Isso precisa acontecer ANTES do primeiro `import ultralytics` de
qualquer modulo, por isso mora aqui e nao em detector.py.
"""

from __future__ import annotations

import os
from pathlib import Path

__version__ = "0.1.0"

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# O Ultralytics grava settings em ~/.config/Ultralytics e, por padrao, envia
# eventos anonimos de uso para servidor externo (`sync: true`). Apontamos o
# diretorio para dentro do projeto: nenhuma configuracao global do usuario e
# tocada, e o desligamento do envio (ver detector.disable_telemetry) fica
# restrito a este repositorio.
ULTRALYTICS_CONFIG_DIR = PROJECT_ROOT / ".ultralytics"
ULTRALYTICS_CONFIG_DIR.mkdir(exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(ULTRALYTICS_CONFIG_DIR))
os.environ.setdefault("YOLO_VERBOSE", "false")
