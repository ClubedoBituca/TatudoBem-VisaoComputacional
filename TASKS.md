# TASKS — quatro frentes paralelas

Regra de convivência: cada frente é dona dos seus arquivos. Precisa de algo de outra
frente? Programe contra a assinatura já publicada no stub — ela não vai mudar sem aviso.

Status do passo 0 (setup): **concluído**. `capture.py`, `detector.py` e `config.py` estão
funcionais; o resto é stub.

---

## Frente 1 — Detecção

**Dono de:** `src/capture.py`, `src/detector.py`
**Base já pronta:** `VideoSource` (arquivo/webcam, rotação, resize, stride) e
`ObstacleDetector.predict()` devolvendo `list[Detection]`.

**A fazer**
- [ ] Ajustar `conf_threshold` e a lista `target_classes` com os vídeos reais da UNIFEI —
      os valores atuais são chute inicial, não medição.
- [ ] Avaliar `yolo11s.pt` contra `yolo11n.pt`: ganho de recall vale a perda de FPS?
- [ ] Medir latência por frame no hardware da demo (GPU e CPU) e registrar aqui.
- [ ] Tratar vídeo HEVC do iPhone: detectar falha de abertura e orientar a conversão
      com ffmpeg (a mensagem de erro já existe em `CaptureError`).
- [ ] Opcional, se sobrar tempo: tracking por ID entre frames, para o mesmo objeto não
      gerar dois eventos.

**Pronto quando:** um `.mp4` da UNIFEI roda de ponta a ponta, as caixas saem estáveis nos
objetos de interesse e a latência média por frame está anotada.

---

## Frente 2 — Regra espacial

**Dono de:** `src/blockage.py`
**Depende de:** `Detection.bottom_center` (já existe) e do polígono em `config/zones.json`.

**A fazer**
- [ ] `denormalize_polygon()` — normalizado (0–1) → pixels do frame de trabalho.
- [ ] `point_in_polygon()` — sugestão: `cv2.pointPolygonTest(..., False) >= 0`.
- [ ] `detections_in_zone()` — filtra pelo ponto inferior central.
- [ ] `BlockageTracker` — contadores consecutivos com histerese
      (`confirm_frames` para confirmar, `release_frames` para liberar) e as flags
      `just_confirmed` / `just_released` que disparam abertura e fechamento de evento.
- [ ] Ferramenta para desenhar o polígono: abrir o primeiro frame numa janela OpenCV
      (o WSLg dá `DISPLAY=:0`), coletar cliques com `cv2.setMouseCallback` e gravar os
      pontos normalizados em `config/zones.json`. Sem isso, o polígono é ajustado no olho.
- [ ] Calibrar `confirm_frames` com o fps real do vídeo — 8 frames a 30 fps é ~0,27 s.

**Pronto quando:** um objeto parado na faixa vira `BARREIRA TEMPORÁRIA` de forma estável e
alguém passando pela borda do polígono não dispara alerta.

---

## Frente 3 — Interface

**Dono de:** `app.py`, `src/ui.py`, `src/events.py`
**Depende de:** `BlockageState` e `RouteStatus` (assinaturas já publicadas).

**A fazer**
- [ ] `app.py` — seleção da fonte (upload de `.mp4` ou caminho em `data/samples/`),
      laço de processamento e atualização do frame na tela.
- [ ] `draw_overlay()` — polígono da zona, caixas das detecções e o ponto inferior central
      de cada caixa (esse ponto é o que mais ajuda a depurar a frente 2).
- [ ] `status_banner()` — verde `ROTA LIVRE`, vermelho `BARREIRA TEMPORÁRIA`, bem grande.
- [ ] `events.py` — `ensure_csv()`, `append_event()`, `load_events()` conforme o esquema
      já documentado no módulo.
- [ ] `events_table()` — tabela dos eventos com pandas.
- [ ] `sidebar_controls()` — confiança, `confirm_frames` e zona ativa ajustáveis ao vivo.
      Vale ouro na hora da apresentação.

**Cuidado:** o upload do Streamlit vai para a memória. Não gravar o vídeo em disco —
`PROJECT_CONTEXT.md` proíbe persistir vídeo.

**Pronto quando:** `streamlit run app.py` processa um vídeo, mostra o status correndo e
lista os eventos gerados.

---

## Frente 4 — Testes

**Dono de:** `tests/`
**Base já pronta:** `smoke_env.py`, `smoke_capture.py`, `smoke_inference.py`.

**A fazer**
- [ ] Gravar vídeos controlados em `data/test/`, câmera estática, com roteiro conhecido:
      (a) rota livre o tempo todo; (b) objeto entra, fica, sai; (c) objeto na borda do
      polígono; (d) pessoa atravessando sem obstáculo — não pode gerar evento.
- [ ] Ground truth por vídeo: um CSV ao lado com os intervalos reais de bloqueio.
- [ ] Métricas: precisão, recall e erro de duração contra o ground truth.
- [ ] Testes unitários da frente 2 com polígono sintético — ponto dentro, fora e exatamente
      na aresta. Não precisa de vídeo e roda em milissegundos.
- [ ] Decidir se vale adicionar `pytest` (seria a única dependência nova aceitável; combinar
      com o time antes de instalar).

**Pronto quando:** existe um número defensável de acerto para mostrar na apresentação, e não
só "funcionou no vídeo que a gente testou".
