"""Leitura da configuracao do projeto (config/zones.json).

Utilitario compartilhado pelas quatro frentes. Nao contem regra de negocio:
apenas carrega o JSON e resolve caminhos relativos a raiz do projeto.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "zones.json"


@lru_cache(maxsize=1)
def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Devolve o conteudo de config/zones.json como dicionario."""
    cfg_path = Path(path) if path else CONFIG_PATH
    with cfg_path.open(encoding="utf-8") as handle:
        return json.load(handle)


def active_polygon(cfg: dict[str, Any] | None = None) -> list[tuple[float, float]]:
    """Poligono da zona ativa, em coordenadas normalizadas (0-1).

    Normalizado de proposito: o video do celular pode vir em 1080p ou 4K e o
    pipeline trabalha num frame redimensionado. Guardar pixels quebraria a zona
    a cada troca de resolucao.
    """
    cfg = cfg or load_config()
    zone_name = cfg["active_zone"]
    return [tuple(point) for point in cfg["zones"][zone_name]["polygon_norm"]]


def resolve_path(relative: str | Path) -> Path:
    """Converte um caminho do JSON em caminho absoluto a partir da raiz."""
    relative = Path(relative)
    return relative if relative.is_absolute() else PROJECT_ROOT / relative
