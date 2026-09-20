# TASKS — sprint de 3 horas, 4 pessoas

**Meta:** ao final, `streamlit run app.py` abre um vídeo real da UNIFEI, desenha a zona e as
detecções, alterna entre `ROTA LIVRE` e `BARREIRA TEMPORÁRIA` e grava os eventos em
`outputs/events.csv`. Com pelo menos um número de acerto medido.

**Congelamento de features: 2h30.** Os últimos 30 minutos são só demo e apresentação.
Funcionalidade que não estiver integrada às 2h30 fica de fora — não vale arriscar a demo.

## Regras de convivência no git

Cada pessoa mexe **só nos arquivos da sua frente**. A tabela de donos está no `AGENTS.md`.

```bash
git pull --rebase origin main     # antes de cada push
git push origin main
```

Push pequeno e frequente (a cada função pronta) vale mais que um push grande no fim.
Precisa mexer em arquivo de outra frente? Fale no grupo antes — não edite por conta.

`config/zones.json` é compartilhado: quem alterar avisa, porque conflito ali quebra todo mundo.

## Linha do tempo

| Horário | Bloco |
|---|---|
| 0:00 – 0:15 | Setup de todos: clonar, instalar, rodar os 3 smoke tests |
| 0:15 – 1:15 | **Bloco 1** — trabalho paralelo |
| 1:15 – 1:30 | **Checkpoint 1** — primeira integração, todo mundo com o mesmo `main` |
| 1:30 – 2:15 | **Bloco 2** — trabalho paralelo |
| 2:15 – 2:30 | **Checkpoint 2** — integração final e congelamento |
| 2:30 – 3:00 | Demo, métricas e apresentação |

### Setup (todos, 15 min)

```bash
git clone https://github.com/ClubedoBituca/TatudoBem-VisaoComputacional.git
cd TatudoBem-VisaoComputacional
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-cpu.txt      # ~200 MB — use este, a menos que você rode a demo
# pip install -r requirements.txt        # ~3 GB, só para a máquina da demo (GPU NVIDIA)
# regenerar o material de verificação: ver seção no README
python tests/smoke_env.py && python tests/smoke_capture.py && python tests/smoke_inference.py
```

Quem não conseguir rodar os três smoke tests em 15 minutos avisa no grupo na hora — não
fica tentando sozinho. Uma pessoa travada no setup custa um quarto da equipe.

---

## Frente 1 — Vídeo e detecção

**Dono de:** `src/capture.py`, `src/detector.py`, `data/samples/`
**Já pronto:** `VideoSource` e `ObstacleDetector.predict()` funcionam.

Esta frente tem pouco código e uma tarefa física urgente: **sem vídeo real, ninguém
integra nada**. Gravar vem primeiro.

### Bloco 1 (0:15 – 1:15)
- [ ] **Gravar 4 clipes**, celular **parado** (apoiado em tripé, muro ou banco), **horizontal**,
      30–60 s cada. Sem caminhar — o polígono é fixo, câmera em movimento invalida a regra.
  - `livre.mp4` — passagem desimpedida o tempo todo
  - `obstaculo.mp4` — objeto entra na faixa, fica parado, depois sai
  - `borda.mp4` — objeto encostado na borda da faixa, sem bloquear
  - `pessoas.mp4` — pessoas atravessando, sem obstáculo (não pode gerar evento)
- [ ] Copiar para `data/samples/` e **avisar no grupo** assim que o primeiro estiver lá.
      As frentes 2 e 3 estão esperando por isso.
- [ ] Se o celular gravar HEVC e o OpenCV não abrir:
      `ffmpeg -i entrada.mov -c:v libx264 -crf 23 saida.mp4`

### Bloco 2 (1:30 – 2:15)
- [ ] Investigar a mochila do `obstruido.mp4`: detectada como `mala`/`mochila` em parte
      dos frames e ausente em outros. O evento disparou pela cadeira, não por ela. Baixar
      `conf_threshold` ou documentar como limitação.
- [ ] Rodar o detector nos clipes reais e ajustar `detection.conf_threshold` e
      `detection.target_classes` em `config/zones.json`. Os valores atuais são chute.
- [ ] Ver o que o COCO **não** reconhece na filmagem (cone, tapume, entulho) e anotar —
      é a principal limitação honesta para a apresentação.
- [ ] Medir a latência média por frame no hardware da demo e registrar aqui:
      `latência: ___ ms/frame (___ fps), device: ___`
- [ ] Se estiver lento: baixar `capture.work_width` ou `detection.imgsz`.

**Pronto quando:** os 4 clipes estão em `data/samples/`, abrem no `VideoSource` e o detector
acha os objetos de interesse de forma estável.

---

## Frente 2 — Regra espacial ✅ CONCLUÍDA

**Dono de:** `src/blockage.py`, `tools/draw_zone.py`

Implementada e validada nos 4 clipes reais. Quem ia pegar esta frente **realoca para a
frente 4** (ground truth e métricas) ou reforça a 3.

Entregue:
- `denormalize_polygon()`, `point_in_polygon()`, `detections_in_zone()`
- `BlockageTracker` com histerese e as flags `just_confirmed` / `just_released`
- `tracker.blocked_frames` — frames desde a confirmação, para `duration_s` do evento
- `tools/draw_zone.py` — desenha a zona clicando sobre o primeiro frame
- `tests/test_blockage.py` — 9 testes, todos passando

Zona ativa `corredor_b21`: trapézio **simétrico em torno do eixo do piso tátil**
(`x=0.498`, medido no frame), com meia-largura de `0.034` no topo e `0.1865` na base,
cortado em `y=0.583`. A abertura em perspectiva reproduz a marcação feita a mão pela
equipe sobre um frame do corredor.

Resultado nos clipes reais (`confirm_frames=8`, `frame_stride=2`, ≈0,53 s):

| vídeo | esperado | eventos | % bloqueado |
|---|---|---|---|
| `livre.mp4` | LIVRE | 0 | 0% |
| `obstruido.mp4` | BARREIRA | 1 | 99% |
| `borda.mp4` | LIVRE | 0 | 0% |
| `pessoas.mp4` | LIVRE | 0 | 0% |

**Ajuste fino que sobrou**, se alguém tiver tempo: desenhar zona própria para o `borda.mp4`
com `python tools/draw_zone.py data/samples/borda.mp4 --zona borda`, já que cada clipe pode
ter enquadramento levemente diferente.

Se for preciso reposicionar a faixa, o eixo do piso tátil está em `x=0.498` — mantenha as
meias-larguras iguais dos dois lados, senão a folga fica desigual para quem se guia por ele.

---

## Frente 3 — Interface

**Dono de:** `app.py`, `src/ui.py`, `src/events.py`
**Não espere a frente 2.** As assinaturas de `BlockageState` e `RouteStatus` já estão
fechadas em `blockage.py` — programe contra elas e use um estado falso até a frente 2 entregar.

### Bloco 1 (0:15 – 1:15)
- [ ] `events.py`: `ensure_csv()`, `append_event()`, `load_events()` — o esquema do CSV já
      está documentado no módulo, não invente colunas.
- [x] ~~`ui.draw_overlay()`~~ — **pronto e validado**. Desenha zona, caixas (laranja fora /
      vermelho invadindo), ponto inferior central e barra de status. Só chamar.
- [ ] `ui.status_banner()` — verde `ROTA LIVRE`, vermelho `BARREIRA TEMPORÁRIA`, grande.
- [ ] `app.py`: seletor de vídeo em `data/samples/` e laço de processamento atualizando um
      `st.empty()` a cada frame.

### Bloco 2 (1:30 – 2:15)
- [ ] Ligar no `BlockageTracker` de verdade; `just_confirmed` / `just_released` escrevem no CSV.
- [ ] `ui.events_table()` com pandas.
- [ ] `ui.sidebar_controls()` — confiança, `confirm_frames` e zona ativa ajustáveis ao vivo.
      Vale ouro se a demo começar a falhar na frente do júri.

**Cuidado:** o upload do Streamlit fica em memória. **Não gravar o vídeo em disco** — o
`PROJECT_CONTEXT.md` proíbe persistir vídeo ou frame.

**Pronto quando:** dá para escolher um vídeo, ver o status mudando e a tabela de eventos crescendo.

---

## Frente 4 — Testes, métricas e integração

**Dono de:** `tests/`
Esta frente é também a **integradora**: é quem roda o `main` de ponta a ponta e acha o que
quebrou. Nos checkpoints, quem conduz é você.

### Bloco 1 (0:15 – 1:15)
- [ ] `tests/test_blockage.py` — testes da frente 2 com polígono sintético, sem vídeo
      nenhum: ponto dentro, fora, exatamente na aresta, e a histerese
      (N-1 frames não confirma, N confirma). Roda em milissegundos e pega bug cedo.
- [ ] Combinar com o time se entra `pytest` (única dependência nova aceitável). Se não
      entrar, escrever no mesmo estilo dos smoke tests, com `assert` e `main()`.
- [ ] Preparar o formato do ground truth: para cada vídeo, um CSV ao lado com os intervalos
      reais de bloqueio (`inicio_s,fim_s,classe`).

### Bloco 2 (1:30 – 2:15)
- [ ] Preencher o ground truth assistindo aos clipes da frente 1.
- [ ] `tests/metrics.py` — comparar eventos detectados com o ground truth: precisão, recall
      e erro de duração.
- [ ] Rodar o pipeline completo nos 4 clipes e reportar falhas no grupo **na hora**.

### Checkpoints — você conduz
- **1:15** — todo mundo dá push, você roda o `main` inteiro e reporta o que quebrou.
- **2:15** — integração final. A partir daqui, só correção de bug que ameace a demo.

**Pronto quando:** existe um número defensável para mostrar, e não só "funcionou no vídeo
que a gente testou".

---

## Bloco final (2:30 – 3:00) — todos

- [ ] Rodar a demo do começo ao fim **duas vezes**. Se falhar uma, tem conserto; se falhar
      na frente do júri, não tem.
- [ ] Decidir qual clipe é o da demo e deixá-lo pré-selecionado.
- [ ] Gravar um vídeo curto da tela como plano B, em `outputs/demos/` (não versionar).
- [ ] Fechar o roteiro: problema → como funciona → demo → números → limitações → próximos passos.
- [ ] Ser honesto sobre a limitação: o COCO não conhece cone de obra, tapume nem entulho.
      Dito de frente, isso vira "sabemos o próximo passo"; escondido, vira furo na arguição.

## Se o tempo apertar — o que cortar, nesta ordem

1. `sidebar_controls()` — bonito, não essencial
2. Métricas nos 4 clipes → fazer em 1 só
3. `tools/draw_zone.py` → ajustar o polígono editando o JSON na mão
4. **Nunca cortar:** o laço de detecção, a regra do polígono e o banner de status. Isso é o MVP.
