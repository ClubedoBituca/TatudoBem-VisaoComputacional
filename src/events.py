"""Registro de eventos de bloqueio em CSV.

FRENTE 3 (Interface) e dona deste arquivo, em acordo com a frente 2.

Estado: FUNCIONAL. O esquema do CSV nao mudou em relacao ao que foi publicado
no stub - a frente 4 pode contar com ele para o ground truth.

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
    """Cria o CSV com cabecalho se ele ainda nao existir."""
    destino = path or EVENTS_PATH
    destino.parent.mkdir(parents=True, exist_ok=True)
    if not destino.exists() or destino.stat().st_size == 0:
        with destino.open("w", newline="", encoding="utf-8") as arquivo:
            csv.DictWriter(arquivo, fieldnames=CSV_FIELDS).writeheader()
    return destino


def next_event_id(path: Path | None = None) -> int:
    """Proximo id sequencial, continuando o que ja existe no arquivo.

    Le do arquivo em vez de contar em memoria: assim reprocessar um video na
    interface nao reinicia a numeracao e sobrescreve o historico.
    """
    destino = ensure_csv(path)
    with destino.open(newline="", encoding="utf-8") as arquivo:
        ids = [int(linha["event_id"]) for linha in csv.DictReader(arquivo) if linha.get("event_id")]
    return max(ids, default=0) + 1


def append_event(event: BlockageEvent, path: Path | None = None) -> None:
    """Acrescenta um evento ao CSV."""
    destino = ensure_csv(path)
    with destino.open("a", newline="", encoding="utf-8") as arquivo:
        csv.DictWriter(arquivo, fieldnames=CSV_FIELDS).writerow(event.as_row())


def load_events(path: Path | None = None):
    """Le o CSV como DataFrame do pandas, para exibir na interface."""
    import pandas as pd

    destino = ensure_csv(path)
    dados = pd.read_csv(destino)
    if not dados.empty:
        dados = dados.sort_values("event_id", ascending=False)
    return dados


def clear_events(path: Path | None = None) -> None:
    """Apaga os eventos, mantendo o cabecalho. Util antes de uma demo limpa."""
    destino = path or EVENTS_PATH
    destino.parent.mkdir(parents=True, exist_ok=True)
    with destino.open("w", newline="", encoding="utf-8") as arquivo:
        csv.DictWriter(arquivo, fieldnames=CSV_FIELDS).writeheader()
