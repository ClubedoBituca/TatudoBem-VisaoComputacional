"""Caminho Livre - UNIFEI: aplicacao Streamlit.

FRENTE 3 (Interface) e dona deste arquivo.

Estado: ESQUELETO. Sobe, mostra o ambiente e confirma que a stack esta de pe.
O laco de processamento e o overlay sao a entrega da frente 3 - ver TASKS.md.

Executar:  streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

from src.config import active_polygon, load_config

st.set_page_config(page_title="Caminho Livre - UNIFEI", page_icon="•", layout="wide")

st.title("Caminho Livre — UNIFEI")
st.caption("Detecção de obstáculos temporários em rotas de circulação acessível")

cfg = load_config()

st.info(
    "Esqueleto do MVP. O pipeline de detecção está sendo implementado pelas quatro "
    "frentes descritas em TASKS.md.",
    icon="ℹ️",
)

col_status, col_config = st.columns([2, 1])

with col_status:
    st.subheader("Status da rota")
    st.success("ROTA LIVRE", icon="✅")
    st.caption(
        "Placeholder. O status real virá de BlockageTracker.update() — frente 2."
    )

    st.subheader("Fonte de vídeo")
    st.file_uploader(
        "Vídeo do trecho (.mp4, celular estático na horizontal)",
        type=["mp4", "mov", "avi"],
        disabled=True,
        help="Habilitado pela frente 3. O vídeo não é gravado em disco.",
    )

with col_config:
    st.subheader("Configuração ativa")
    detection = cfg["detection"]
    st.metric("Zona ativa", cfg["active_zone"])
    st.write(f"**Modelo:** `{detection['model_path']}`")
    st.write(f"**Confiança mínima:** {detection['conf_threshold']}")
    st.write(f"**Frames para confirmar:** {cfg['blockage']['confirm_frames']}")
    st.write(f"**Classes ignoradas:** {', '.join(detection['ignored_classes'])}")
    st.write(f"**Vértices do polígono:** {len(active_polygon(cfg))}")

st.divider()
st.subheader("Eventos registrados")
st.caption("Tabela de outputs/events.csv — frente 3.")
