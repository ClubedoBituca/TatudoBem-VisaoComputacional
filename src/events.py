"""Registro de eventos de bloqueio em CSV.

FRENTE 3 (Interface) e dona deste arquivo, em acordo com a frente 2.

Estado: STUB, com o esquema do CSV ja fechado - a frente 4 depende dele para
montar o ground truth.

Esquema de outputs/events.csv (uma linha por bloqueio confirmado e encerrado):

    event_id       identificador sequencial dentro da execucao
    timestamp      inicio do bloqueio, ISO 8601 local
    source         nome do arquivo de video ou "webcam:N"
    class_name     classe COCO do objeto que bloqueou
    confidence     confianca media do objeto durante o bloqueio
    duration_s     duracao do bloqueio em segundos
    frames         numero de frames em que o bloqueio persistiu
    zone           nome da zona ativa em config/zones.json

O MVP nao grava video nem frames: apenas estas linhas de texto.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path

from .config import PROJECT_ROOT

EVENTS_PATH = PROJECT_ROOT / "outputs" / "events.csv"

CSV_FIELDS = [
    "event_id",
    "timestamp",
    "source",
    "class_name",
    "confidence",
    "duration_s",
    "frames",
    "zone",
]


@dataclass
class BlockageEvent:
    """Uma ocorrencia de barreira temporaria ja encerrada."""

    event_id: int
    timestamp: str
    source: str
    class_name: str
    confidence: float
    duration_s: float
    frames: int
    zone: str

    def as_row(self) -> dict[str, object]:
        return asdict(self)


def ensure_csv(path: Path | None = None) -> Path:
    """Cria o CSV com cabecalho se ele ainda nao existir.

    TODO(frente-3): implementar.
    """
    raise NotImplementedError("frente-3: criar CSV com cabecalho")


def append_event(event: BlockageEvent, path: Path | None = None) -> None:
    """Acrescenta um evento ao CSV.

    TODO(frente-3): implementar.
    """
    raise NotImplementedError("frente-3: append de evento no CSV")


def load_events(path: Path | None = None):
    """Le o CSV como DataFrame do pandas, para exibir na interface.

    TODO(frente-3): implementar.
    """
    raise NotImplementedError("frente-3: leitura do CSV com pandas")
