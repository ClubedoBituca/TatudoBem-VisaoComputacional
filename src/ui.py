"""Componentes de interface Streamlit.

FRENTE 3 (Interface) e dona deste arquivo.

Estado: FUNCIONAL. `draw_overlay` e usado tanto pelo Streamlit quanto por
tools/render_video.py, entao nao existem duas versoes do desenho.

Mantem app.py enxuto: o app orquestra, este modulo desenha.

Telas previstas no MVP:
  * selecao da fonte (upload de .mp4 de data/samples ou caminho local);
  * player com o frame anotado - caixas das deteccoes e poligono da zona;
  * banner de status grande: verde "ROTA LIVRE" / vermelho "BARREIRA TEMPORARIA";
  * tabela dos eventos de outputs/events.csv.
"""

from __future__ import annotations

import cv2
import numpy as np

from .blockage import BlockageState, RouteStatus
from .detector import Detection

Polygon = list[tuple[float, float]]

# Cores em BGR (padrao OpenCV).
VERDE = (60, 180, 75)
VERMELHO = (40, 40, 220)
LARANJA = (0, 150, 255)
AZUL = (200, 130, 30)
CIANO = (255, 220, 0)
BRANCO = (255, 255, 255)
PRETO = (0, 0, 0)


# Rotulos em portugues para a tela. As classes do COCO vem em ingles, e rotulo
# em ingles numa apresentacao brasileira parece coisa nao terminada. Sem acento:
# as fontes Hershey do OpenCV sao ASCII. Classe fora do mapa cai no nome original.
ROTULOS_PT = {
    "person": "pessoa",
    "chair": "cadeira",
    "bench": "banco",
    "couch": "sofa",
    "dining table": "mesa",
    "potted plant": "vaso de planta",
    "backpack": "mochila",
    "handbag": "bolsa",
    "suitcase": "mala",
    "bicycle": "bicicleta",
    "motorcycle": "moto",
    "car": "carro",
    "truck": "caminhao",
    "bus": "onibus",
    "tv": "monitor",
}


def rotulo(class_name: str) -> str:
    """Nome de exibicao da classe, em portugues quando houver traducao."""
    return ROTULOS_PT.get(class_name, class_name)


def _texto(frame: np.ndarray, txt: str, org: tuple[int, int], escala: float, cor, grossura: int = 1) -> None:
    """Escreve com contorno preto, para o texto sobreviver a fundo claro ou escuro.

    O corredor da UNIFEI e muito claro e estourado ao fundo: texto sem contorno
    some. Nao usar acento - as fontes Hershey do OpenCV sao ASCII.
    """
    cv2.putText(frame, txt, org, cv2.FONT_HERSHEY_SIMPLEX, escala, PRETO, grossura + 3, cv2.LINE_AA)
    cv2.putText(frame, txt, org, cv2.FONT_HERSHEY_SIMPLEX, escala, cor, grossura, cv2.LINE_AA)


def draw_overlay(
    frame: np.ndarray,
    detections: list[Detection],
    polygon: Polygon,
    state: BlockageState | None = None,
    passersby: list[Detection] | None = None,
    guide_strip: Polygon | None = None,
) -> np.ndarray:
    """Desenha poligono da zona e caixas das deteccoes sobre uma copia do frame.

    Nao altera o frame original. O ponto inferior central de cada caixa e
    marcado com um circulo: e o ponto que a regra espacial testa, e ve-lo na
    tela e o que permite entender por que um objeto contou ou nao como bloqueio.

    Objeto DENTRO da zona sai em vermelho; fora, em laranja. Assim da para
    conferir a decisao do sistema olhando um frame, sem ler log.

    `passersby` sao as deteccoes nao obstrutivas (pessoas), vindas de
    `ObstacleDetector.predict_split`. Saem em azul e rotuladas "nao obstrui",
    mesmo quando pisam na faixa. Sem desenha-las, uma pessoa atravessando a
    zona simplesmente sumiria da tela, e o sistema pareceria cego a ela.
    """
    tela = frame.copy()
    bloqueada = state is not None and state.status is RouteStatus.BLOQUEADA
    cor_zona = VERMELHO if bloqueada else VERDE

    if guide_strip is not None:
        # O piso tatil em ciano, so contorno: mostra de onde a faixa livre foi
        # derivada. Sem ele, a zona parece um retangulo arbitrario na tela.
        faixa = np.array(guide_strip, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(tela, [faixa], isClosed=True, color=CIANO, thickness=1, lineType=cv2.LINE_AA)

    contorno = np.array(polygon, dtype=np.int32).reshape(-1, 1, 2)
    sombra = tela.copy()
    cv2.fillPoly(sombra, [contorno], cor_zona)
    tela = cv2.addWeighted(sombra, 0.25, tela, 0.75, 0)
    cv2.polylines(tela, [contorno], isClosed=True, color=cor_zona, thickness=2, lineType=cv2.LINE_AA)

    invasores = {id(d) for d in (state.intruders if state else [])}
    for deteccao in detections:
        invadindo = id(deteccao) in invasores
        cor = VERMELHO if invadindo else LARANJA
        p1 = (int(deteccao.x1), int(deteccao.y1))
        p2 = (int(deteccao.x2), int(deteccao.y2))
        cv2.rectangle(tela, p1, p2, cor, 2 if invadindo else 1, cv2.LINE_AA)

        bx, by = deteccao.bottom_center
        cv2.circle(tela, (int(bx), int(by)), 6, cor, -1, cv2.LINE_AA)
        cv2.circle(tela, (int(bx), int(by)), 6, PRETO, 1, cv2.LINE_AA)

        etiqueta = f"{rotulo(deteccao.class_name)} {deteccao.confidence:.2f}"
        _texto(tela, etiqueta, (p1[0], max(14, p1[1] - 6)), 0.5, cor)

    for pessoa in passersby or []:
        p1 = (int(pessoa.x1), int(pessoa.y1))
        p2 = (int(pessoa.x2), int(pessoa.y2))
        cv2.rectangle(tela, p1, p2, AZUL, 1, cv2.LINE_AA)
        bx, by = pessoa.bottom_center
        cv2.circle(tela, (int(bx), int(by)), 5, AZUL, -1, cv2.LINE_AA)
        _texto(tela, f"{rotulo(pessoa.class_name)} - nao obstrui", (p1[0], max(14, p1[1] - 6)), 0.45, AZUL)

    if state is not None:
        tela = _faixa_status(tela, state, len(passersby or []))
    return tela


def _faixa_status(frame: np.ndarray, state: BlockageState, pessoas: int = 0) -> np.ndarray:
    """Barra superior com o veredito, legivel de longe numa projecao."""
    altura_barra = 46
    bloqueada = state.status is RouteStatus.BLOQUEADA
    cor = VERMELHO if bloqueada else VERDE

    barra = frame.copy()
    cv2.rectangle(barra, (0, 0), (frame.shape[1], altura_barra), cor, -1)
    frame = cv2.addWeighted(barra, 0.85, frame, 0.15, 0)

    _texto(frame, state.status.value, (16, 33), 0.95, BRANCO, 2)

    if bloqueada:
        detalhe = ", ".join(sorted({rotulo(d.class_name) for d in state.intruders})) or "objeto"
    elif pessoas:
        # Dizer que ha gente na cena e que isso nao conta e mais informativo do
        # que o contador de frames - e o ponto que a demo precisa deixar claro.
        detalhe = f"{pessoas} pessoa(s) em transito - nao conta como barreira"
    else:
        detalhe = f"{state.consecutive_frames} frame(s) sem invasao"
    # Medir o texto em vez de estimar pela contagem de caracteres: a fonte e
    # proporcional, e o chute cortava a frase na borda direita.
    (largura_txt, _), _ = cv2.getTextSize(detalhe, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    _texto(frame, detalhe, (max(8, frame.shape[1] - 14 - largura_txt), 31), 0.55, BRANCO)
    return frame


def status_banner(state: BlockageState) -> None:
    """Renderiza o banner de status no Streamlit.

    Grande e colorido de proposito: numa projecao, quem esta no fundo da sala
    tem que ler o veredito sem enxergar o video.
    """
    import streamlit as st

    if state.status is RouteStatus.BLOQUEADA:
        classes = ", ".join(sorted({rotulo(d.class_name) for d in state.intruders})) or "objeto"
        st.error(f"### BARREIRA TEMPORÁRIA\n\nObstruindo a passagem: **{classes}**", icon="🚫")
    else:
        st.success("### ROTA LIVRE\n\nNenhuma obstrução confirmada na faixa.", icon="✅")


def events_table(path=None) -> None:
    """Renderiza a tabela de eventos a partir de outputs/events.csv."""
    import streamlit as st

    from .events import load_events

    dados = load_events(path)
    if dados.empty:
        st.caption("Nenhum evento registrado ainda.")
        return

    st.dataframe(
        dados,
        hide_index=True,
        width="stretch",
        column_config={
            "event_id": st.column_config.NumberColumn("#", width="small"),
            "timestamp": st.column_config.TextColumn("Início"),
            "source": st.column_config.TextColumn("Vídeo"),
            "class_name": st.column_config.TextColumn("Objeto"),
            "confidence": st.column_config.NumberColumn("Confiança", format="%.2f"),
            "duration_s": st.column_config.NumberColumn("Duração (s)", format="%.1f"),
            "frames": st.column_config.NumberColumn("Frames", width="small"),
            "zone": st.column_config.TextColumn("Zona"),
        },
    )


def sidebar_controls(videos: list[str]) -> dict[str, object]:
    """Controles de ajuste na barra lateral.

    Os limiares ficam ajustaveis ao vivo porque calibrar na frente do juri e
    melhor do que travar num valor que nao serve para a cena do dia.
    """
    import streamlit as st

    from .config import load_config

    cfg = load_config()

    with st.sidebar:
        st.subheader("Fonte")
        # Abre no clipe que demonstra o sistema alarmando. O padrao alfabetico
        # caia num video onde a deteccao da faixa falha - pessima primeira tela.
        preferidos = ["obstruido.mp4", "caixas.mp4", "livre.mp4"]
        inicial = next((videos.index(v) for v in preferidos if v in videos), 0)
        video = st.selectbox("Vídeo", videos, index=inicial, help="Arquivos em data/samples/")

        st.subheader("Faixa acessível")
        automatica = st.toggle(
            "Detectar o piso tátil", value=True,
            help="Desligado, usa o polígono fixo de config/zones.json.",
        )
        ratio = st.slider(
            "Largura da faixa livre", 2.0, 6.0,
            float(cfg.get("lane", {}).get("free_width_ratio", 4.0)), 0.5,
            help="Em múltiplos da largura do piso tátil.",
            disabled=not automatica,
        )

        st.subheader("Detecção")
        confianca = st.slider(
            "Confiança mínima", 0.10, 0.90,
            float(cfg["detection"]["conf_threshold"]), 0.05,
        )
        confirmar = st.slider(
            "Frames para confirmar", 1, 40, int(cfg["blockage"]["confirm_frames"]),
            help="A 30 fps com salto 2, 8 frames ≈ 0,5 s.",
        )
        salto = st.slider(
            "Processar 1 frame a cada", 1, 6, int(cfg["capture"]["frame_stride"]),
            help="Maior = mais rápido, menos preciso no tempo.",
        )

        st.subheader("Eventos")
        limpar = st.button("Limpar histórico", width="stretch")

    return {
        "video": video,
        "auto_lane": automatica,
        "ratio": ratio,
        "conf": confianca,
        "confirm_frames": confirmar,
        "stride": salto,
        "limpar": limpar,
    }
