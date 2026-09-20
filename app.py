"""Caminho Livre - UNIFEI: aplicacao Streamlit.

Executar:  streamlit run app.py

O app so orquestra; quem desenha e src/ui.py, quem decide e src/blockage.py.
Nenhum frame ou video e gravado em disco - so as linhas de outputs/events.csv.
"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

import cv2
import streamlit as st

from src.blockage import BlockageTracker, RouteStatus, denormalize_polygon
from src.capture import CaptureError, VideoSource
from src.config import PROJECT_ROOT, active_polygon, load_config
from src.detector import ObstacleDetector
from src.events import BlockageEvent, append_event, clear_events, next_event_id
from src.lane import LaneNotFound, detect_from_video
from src.ui import draw_overlay, events_table, sidebar_controls, status_banner

AMOSTRAS = PROJECT_ROOT / "data" / "samples"

st.set_page_config(page_title="Caminho Livre - UNIFEI", page_icon="•", layout="wide")


@st.cache_resource(show_spinner="Carregando o modelo YOLO...")
def carregar_detector() -> ObstacleDetector:
    """Uma instancia para toda a sessao: carregar o modelo custa segundos."""
    return ObstacleDetector()


@st.cache_data(show_spinner="Procurando o piso tátil no vídeo...")
def localizar_faixa(caminho: str, _mtime: float):
    """Detecta a faixa guia. Em cache por arquivo - custa ~25 frames de leitura.

    `_mtime` entra na chave para o cache cair sozinho se o vídeo for trocado.
    """
    try:
        return detect_from_video(caminho), None
    except LaneNotFound as exc:
        return None, str(exc)


def main() -> None:
    st.title("Caminho Livre — UNIFEI")
    st.caption(
        "Detecção de obstáculos temporários sobre a faixa de circulação acessível"
    )

    videos = sorted(p.name for p in AMOSTRAS.glob("*.mp4"))
    if not videos:
        st.error(
            f"Nenhum vídeo em `{AMOSTRAS}`. Grave com o celular **parado** e na "
            "horizontal, e salve ali. Ver README."
        )
        return

    controles = sidebar_controls(videos)
    if controles["limpar"]:
        clear_events()
        st.toast("Histórico de eventos apagado.")

    caminho = AMOSTRAS / str(controles["video"])
    detector = carregar_detector()

    # --- faixa acessivel -------------------------------------------------
    faixa, erro = (None, None)
    if controles["auto_lane"]:
        faixa, erro = localizar_faixa(str(caminho), caminho.stat().st_mtime)

    if faixa is not None:
        poligono_norm = faixa.free_path_polygon(float(controles["ratio"]))
        origem_zona = f"piso tátil detectado (cobertura {faixa.coverage:.2f})"
    else:
        poligono_norm = active_polygon()
        origem_zona = f"polígono fixo `{load_config()['active_zone']}`"
        if controles["auto_lane"]:
            st.warning(
                f"Não localizei o piso tátil neste vídeo ({erro}). "
                "Usando o polígono fixo — confira se ele bate com o enquadramento.",
                icon="⚠️",
            )

    st.caption(f"Faixa acessível: {origem_zona} · device `{detector.device}`")

    coluna_video, coluna_lado = st.columns([3, 2])
    with coluna_lado:
        area_status = st.empty()
        st.subheader("Eventos registrados")
        area_tabela = st.empty()
    with coluna_video:
        area_frame = st.empty()
        area_progresso = st.empty()

    with area_tabela.container():
        events_table()

    processar = st.sidebar.button("▶ Processar vídeo", type="primary", width="stretch")

    if not processar:
        with area_status.container():
            st.info("Escolha o vídeo e clique em **Processar vídeo**.", icon="▶")
        try:
            with VideoSource(caminho, frame_stride=int(controles["stride"])) as fonte:
                primeiro = fonte.read()
            if primeiro is not None:
                poligono = denormalize_polygon(poligono_norm, primeiro.shape[1], primeiro.shape[0])
                piso = (
                    denormalize_polygon(faixa.strip_polygon(), primeiro.shape[1], primeiro.shape[0])
                    if faixa is not None else None
                )
                area_frame.image(
                    cv2.cvtColor(draw_overlay(primeiro, [], poligono, None, guide_strip=piso), cv2.COLOR_BGR2RGB),
                    caption="Primeiro frame com a faixa acessível marcada",
                    width="stretch",
                )
        except CaptureError as exc:
            st.error(str(exc))
        return

    # --- laco de processamento -------------------------------------------
    detector.conf_threshold = float(controles["conf"])
    try:
        fonte = VideoSource(caminho, frame_stride=int(controles["stride"]))
    except CaptureError as exc:
        st.error(str(exc))
        return

    with fonte:
        info = fonte.info
        poligono = denormalize_polygon(poligono_norm, info.work_width, info.work_height)
        piso_tatil = (
            denormalize_polygon(faixa.strip_polygon(), info.work_width, info.work_height)
            if faixa is not None else None
        )
        rastreador = BlockageTracker(poligono, confirm_frames=int(controles["confirm_frames"]))

        seg_por_frame = fonte.frame_stride / info.fps if info.fps > 0 else 0.0
        total = max(1, info.frame_count // fonte.frame_stride)
        barra = area_progresso.progress(0.0)

        evento_id = next_event_id()
        abertura: dict[str, object] | None = None
        confiancas: list[float] = []
        novos = 0
        inicio = time.perf_counter()

        for indice, frame in fonte:
            deteccoes, pessoas = detector.predict_split(frame)
            estado = rastreador.update(deteccoes)
            segundo = indice * seg_por_frame

            if estado.just_confirmed:
                abertura = {"segundo": segundo, "quando": datetime.now().isoformat(timespec="seconds")}
                confiancas = []
            if estado.status is RouteStatus.BLOQUEADA and estado.intruders:
                confiancas.append(max(d.confidence for d in estado.intruders))
            if estado.just_released and abertura is not None:
                append_event(BlockageEvent(
                    event_id=evento_id,
                    timestamp=str(abertura["quando"]),
                    source=caminho.name,
                    class_name=", ".join(sorted({d.class_name for d in estado.intruders})) or "objeto",
                    confidence=round(sum(confiancas) / len(confiancas), 3) if confiancas else 0.0,
                    duration_s=round(segundo - float(abertura["segundo"]), 1),
                    frames=rastreador.blocked_frames,
                    zone=origem_zona,
                ))
                evento_id += 1
                novos += 1
                abertura = None
                with area_tabela.container():
                    events_table()

            area_frame.image(
                cv2.cvtColor(
                    draw_overlay(frame, deteccoes, poligono, estado, passersby=pessoas, guide_strip=piso_tatil),
                    cv2.COLOR_BGR2RGB,
                ),
                width="stretch",
            )
            with area_status.container():
                status_banner(estado)
            barra.progress(min(1.0, (indice + 1) / total), text=f"{segundo:.1f}s de {info.duration_seconds:.0f}s")

        # Video pode acabar com a rota ainda bloqueada: o evento existiu, so nao
        # chegou a fechar sozinho. Registrar assim mesmo, com a duracao ate o fim.
        if abertura is not None:
            fim = (info.frame_count // fonte.frame_stride) * seg_por_frame
            append_event(BlockageEvent(
                event_id=evento_id,
                timestamp=str(abertura["quando"]),
                source=caminho.name,
                class_name="em curso no fim do vídeo",
                confidence=round(sum(confiancas) / len(confiancas), 3) if confiancas else 0.0,
                duration_s=round(fim - float(abertura["segundo"]), 1),
                frames=rastreador.blocked_frames,
                zone=origem_zona,
            ))
            novos += 1

    decorrido = time.perf_counter() - inicio
    barra.empty()
    st.sidebar.success(f"{novos} evento(s) · {total / max(decorrido, 1e-9):.0f} fps")
    with area_tabela.container():
        events_table()


if __name__ == "__main__":
    main()
