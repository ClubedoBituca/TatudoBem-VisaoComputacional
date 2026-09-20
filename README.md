# Caminho Livre — UNIFEI

**Detecção automática de obstáculos sobre rotas acessíveis, a partir do piso tátil.**

Uma bicicleta atravessada na calçada, uma cadeira deixada no corredor, uma caixa no meio da
passagem. Para quem enxerga, é um desvio de dois passos. Para quem se guia pelo **piso tátil
direcional**, é a rota inteira perdida — a faixa guia deixa de guiar.

Este projeto olha um trecho de circulação e responde uma pergunta só, em tempo real:

> **A rota está livre, ou existe uma barreira temporária sobre ela?**

---

## Como funciona

```
 vídeo  ─────────────────────────────────────────────────────────────────┐
   │                                                                     │
   ├─► 1. ONDE É O CAMINHO                                               │
   │      mediana temporal → modelo de segmentação acha o piso tátil     │
   │      → eixo + largura em perspectiva → faixa livre (4× a guia)      │
   │                                                                     │
   ├─► 2. O QUE TEM NA CENA                                              │
   │      YOLO11n (COCO) → objetos, com pessoas separadas à parte        │
   │                                                                     │
   ├─► 3. ESTÁ NO CAMINHO?                                               │
   │      ponto inferior central da caixa cai dentro da faixa?           │
   │                                                                     │
   └─► 4. É MESMO UMA BARREIRA?                                          │
          N frames consecutivos, com histerese ────────────────────────► │
                                                                         ▼
                                            ROTA LIVRE ┃ BARREIRA TEMPORÁRIA
                                                       └─► evento em CSV
```

**O passo 3 é o coração do projeto.** O detector diz "existe uma cadeira em tal lugar"; ele
não faz ideia do que é uma calçada. Quem transforma isso em "esta rota não está acessível" é
a nossa geometria — e ela usa o **ponto inferior central** da caixa, não o centro:

```
        ┌─────────┐
        │    ×    │  ← centro da caixa: em perspectiva, "flutua" e engana
        │         │
        └────●────┘  ← ponto inferior central: onde o objeto toca o chão
```

Um armário encostado na parede tem o centro da caixa sobre a faixa, mas a base fora dela.
Com o ponto de contato, ele não conta. Sem, contaria.

**O passo 4 existe porque detector pisca.** Numa sombra, um vaso vira "mochila" por um frame e
some no seguinte. Um bloqueio só é confirmado após `confirm_frames` frames consecutivos, e só é
liberado após `release_frames` seguidos sem invasão — histerese, para o status não oscilar com
um objeto exatamente na borda.

---

## Resultados medidos

Sete clipes gravados no corredor B.2.1 do campus, câmera estática, 1280×720 a 30 fps.

### Decisão final do sistema

| clipe | cena | esperado | obtido |
|---|---|---|---|
| `livre.mp4` | passagem desimpedida | ROTA LIVRE | ✅ 0 eventos |
| `obstruido.mp4` | cadeira e mochila na faixa | BARREIRA | ✅ 1 evento, 99% do tempo |
| `borda.mp4` | objeto encostado, sem bloquear | ROTA LIVRE | ✅ 0 eventos |
| `pessoas.mp4` | pedestres atravessando a faixa | ROTA LIVRE | ✅ 0 eventos |

O `pessoas.mp4` é o mais revelador: **três pessoas pisando na faixa** e o sistema segue em
ROTA LIVRE. Elas são detectadas e desenhadas na tela, rotuladas `não obstrui` — pedestre em
trânsito não é barreira temporária.

### Localização do piso tátil

Contra referência medida à régua sobre o frame (eixo em `x = 0,498`):

| clipe | heurística de luminância | modelo de segmentação |
|---|---|---|
| `obstruido` | erro 0,0030 | **erro 0,0001** |
| `livre` | erro 0,0030 | **erro 0,0006** |
| `objetos_variaveis` | eixo 0,321 — **errado** | **0,509 — correto** |
| `bicicleta_planta` | errado | errado (vaso cobre a faixa) |

---

## Setup

```bash
git clone https://github.com/ClubedoBituca/TatudoBem-VisaoComputacional.git
cd TatudoBem-VisaoComputacional

python3 -m venv .venv && source .venv/bin/activate

pip install -r requirements.txt        # GPU NVIDIA — ~3 GB
pip install -r requirements-cpu.txt    # sem GPU, ou só para desenvolver — ~200 MB
```

As duas listas travam as mesmas versões em tudo que não é PyTorch. Só a máquina da demo
precisa de CUDA.

Baixe os dois modelos (não versionados, ~12 MB no total):

```bash
mkdir -p models
curl -sL -o models/yolo11n.pt \
  https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt
curl -sL -o models/yolo11n_tactile.pt \
  https://raw.githubusercontent.com/DARoSLab/GuideTWSI/master/model_weights/yolo11n_tactile.pt
```

Verifique a instalação, na ordem:

```bash
python tests/smoke_env.py         # versões, GPU, device escolhido
python tests/smoke_capture.py     # leitura de vídeo e webcam
python tests/smoke_inference.py   # carga do modelo e inferência
python tests/test_blockage.py     # 9 testes da regra espacial, sem vídeo nem GPU
```

---

## Uso

```bash
streamlit run app.py
```

Escolha o vídeo na barra lateral e clique em **Processar vídeo**. O primeiro frame já aparece
com a faixa acessível marcada **antes** de processar, para conferir o enquadramento sem
esperar o clipe inteiro.

Ajustáveis ao vivo, sem reiniciar: largura da faixa livre, confiança mínima, frames para
confirmar, salto de frames e a alternância entre piso tátil detectado e polígono fixo. Os
eventos vão para `outputs/events.csv` e aparecem na tabela à direita.

> Um clipe de 50 s leva cerca de 2 minutos no padrão. Para a demo, use **salto 5 ou 6** — a
> decisão não muda, só a granularidade temporal. Não há botão de parar no meio: para
> interromper, recarregue a página.

### Ferramentas de linha de comando

```bash
# Onde está a faixa neste vídeo?
python tools/detect_lane.py data/samples/livre.mp4
python tools/detect_lane.py data/samples/livre.mp4 --ratio 5 --salvar-zona corredor_b21

# Vídeo anotado, para a apresentação ou como plano B da demo ao vivo
python tools/render_video.py data/samples/obstruido.mp4
python tools/render_video.py data/samples/obstruido.mp4 --inicio 8 --fim 25

# O YOLO reconhece este objeto? (rode ANTES de gravar)
python tools/check_objeto.py foto_do_objeto.jpg

# Desenhar a zona à mão, quando a detecção automática não servir
python tools/draw_zone.py data/samples/obstruido.mp4 --zona corredor_b21
```

---

## Como a faixa acessível é encontrada

O sistema **não usa um polígono desenhado à mão**. Um polígono desenhado vale para um
enquadramento só — trocou o corredor ou a câmera, tem que desenhar de novo. Aqui ele localiza
o piso tátil no próprio vídeo e deriva dele a faixa trafegável.

### Método principal — modelo de segmentação

Usamos o **`yolo11n_tactile.pt`**, um YOLOv11n-seg treinado para segmentar piso tátil,
publicado pelo projeto [GuideTWSI](https://guidedogrobot-tactile.github.io/) (DARoS Lab, UMass)
sob **licença MIT**. Ele foi treinado em **19.925 imagens de barras direcionais anotadas pixel
a pixel** — não procura "o que é escuro", aprendeu o que é piso tátil.

1. **Fundo limpo** — mediana temporal de 25 frames. Câmera é estática, então quem passava
   desaparece e sobra o corredor. Objeto parado permanece, e tudo bem: ele é obstáculo.
2. **Segmentação** — a rede marca, pixel a pixel, onde está o piso tátil.
3. **Medição** — para cada linha da imagem, onde a faixa começa e termina.
4. **Geometria** — ajusta eixo e largura, impondo que a faixa seja reta e convirja no ponto de
   fuga. A largura é constante no mundo real, logo tem de convergir a zero na imagem; sem essa
   restrição o ajuste sai quase horizontal em cena ruidosa, e o polígono vira um retângulo.
5. **Faixa livre** — o piso tátil alargado `lane.free_width_ratio` vezes (padrão 4), simétrico
   em torno do eixo.

**Sobre a largura.** Quem se guia pelo piso tátil ocupa bem mais que a largura dele: precisa de
folga para o corpo e para a bengala. O fator 4 é escolha de engenharia — o valor da **NBR 9050**
deve ser conferido por quem tenha a norma. A simetria não é opcional: folga desigual não
representa o caminho de quem usa a faixa.

**Por que o modelo é melhor.** Além da precisão, o ponto de corte superior (`y_top`) passou a
sair da própria máscara em vez de um valor fixo. É o que faz o `objetos_variaveis.mp4`
funcionar: gravado de outra posição, com o piso aparecendo só no terço inferior, o valor fixo
apontava para o teto.

**Só o passo 2 mudou** quando trocamos a heurística pelo modelo. Os passos 4 e 5 — onde está a
decisão de produto — continuam os mesmos, e por isso `BlockageTracker`, a regra do ponto
inferior central, o overlay e a interface não mudaram uma linha. O modelo é uma peça
substituível.

### Método reserva — heurística de luminância

Entra automaticamente quando o modelo não acha a faixa. Só OpenCV e NumPy: marca o que escurece
em relação ao piso da própria linha, e acha o eixo procurando a reta de maior cobertura — o piso
tátil atravessa toda a profundidade, um objeto ocupa só um trecho. Como um feixe inteiro de
retas dentro da faixa empata em cobertura, o eixo é a **mediana** do feixe, não o primeiro
empate.

Desligue o modelo com `lane.use_model = false` em `config/zones.json`.

---

## Saída visual

A interface mostra o frame anotado ao vivo, e `tools/render_video.py` grava um `.mp4` com tudo
queimado na imagem:

| elemento | significado |
|---|---|
| contorno **ciano** | piso tátil detectado pelo modelo |
| zona **verde** | faixa livre, rota desimpedida |
| zona **vermelha** | faixa livre, barreira confirmada |
| caixa **laranja** | objeto detectado, fora da faixa |
| caixa **vermelha** | objeto invadindo a faixa |
| caixa **azul** | pessoa — rotulada `não obstrui`, mesmo pisando na faixa |
| **círculo** na base | o ponto inferior central, que é o que a regra testa |

O círculo na base merece atenção: com ele na tela, dá para conferir **por que** o sistema
decidiu, olhando um frame. Sem ele, "por que essa cadeira não contou?" vira discussão.

> **Privacidade.** O pipeline em operação **não grava vídeo** — só linhas de CSV. O
> `render_video.py` é ação deliberada, fora do fluxo normal, e o arquivo gerado contém as
> pessoas que aparecem na gravação original, ainda que o sistema não as identifique. Confira
> quem aparece antes de pôr num slide ou mandar em grupo.

---

## Como gravar os vídeos

- **Celular parado.** Apoiado em tripé, muro ou banco. **Não caminhar** durante a gravação: a
  faixa é fixa em coordenadas de imagem, e câmera em movimento invalida a regra.
- **Orientação horizontal** (paisagem).
- Enquadrar o trecho de passagem de ponta a ponta, com o piso tátil visível.
- Salvar em `data/samples/`. Vídeos não são versionados nem persistidos pelo sistema.

**Teste o objeto antes de filmar.** O detector só conhece 80 classes do COCO, e **caixa de
papelão não é uma delas** — medido: aparece como `tv` a 0,34, ou não é detectada. Cone de obra,
tapume e entulho também estão fora.

```bash
python tools/check_objeto.py foto_do_objeto.jpg
```

Tire a foto do mesmo ângulo e iluminação da gravação — a resposta muda com a perspectiva.
Objetos que funcionam bem e são fáceis de achar num campus: **cadeira, mochila, bicicleta, vaso
de planta, mala, banco**.

Se o vídeo for HEVC/H.265 (padrão do iPhone) e o OpenCV não abrir:

```bash
ffmpeg -i entrada.mov -c:v libx264 -crf 23 saida.mp4
```

### Material de verificação

Mídia não é versionada. Para recriar o clipe usado pelos smoke tests:

```bash
curl -sL -o data/test/bus.jpg \
  https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/assets/bus.jpg
ffmpeg -y -loop 1 -i data/test/bus.jpg -t 5 -r 30 \
  -vf "scale=-2:720,pad=1280:720:(ow-iw)/2:0:color=gray" \
  -c:v libx264 -pix_fmt yuv420p -crf 23 data/test/pipeline_check.mp4
```

É uma imagem em loop: exercita o pipeline com detecções reais em todo frame, não serve para
avaliar qualidade.

---

## Configuração

Tudo que se ajusta está em [`config/zones.json`](config/zones.json).

| Chave | Padrão | O que faz |
|---|---|---|
| `capture.work_width` | 960 | largura do frame de trabalho; menor = mais rápido |
| `capture.frame_stride` | 2 | processa 1 frame a cada N |
| `lane.use_model` | `true` | usa o modelo de segmentação; `false` volta à heurística |
| `lane.free_width_ratio` | 4.0 | largura da faixa livre, em múltiplos do piso tátil |
| `lane.model_conf` | 0.10 | confiança mínima da segmentação do piso tátil |
| `lane.background_samples` | 25 | frames amostrados para a mediana temporal |
| `lane.min_coverage` | 0.80 | fração mínima de linhas cobertas para aceitar a faixa |
| `lane.y_top` | 0.58 | corte superior — só para a heurística; o modelo deriva o seu |
| `detection.conf_threshold` | 0.35 | confiança mínima para aceitar um objeto |
| `detection.target_classes` | 14 classes | classes COCO tratadas como possível obstáculo |
| `detection.ignored_classes` | `["person"]` | nunca contam como barreira |
| `blockage.confirm_frames` | 8 | frames consecutivos para confirmar um bloqueio |
| `blockage.release_frames` | 8 | frames consecutivos sem invasão para liberar |
| `zones.<nome>.polygon_norm` | — | polígono reserva, em coordenadas normalizadas (0–1) |

A 30 fps com salto 2, 8 frames ≈ **0,53 s** de persistência.

---

## Estrutura

```
app.py                      interface Streamlit — orquestra o pipeline
config/zones.json           faixa, classes, limiares
models/                     pesos (não versionados, ver Setup)

src/
  config.py                 leitura de config/zones.json
  capture.py                vídeo/webcam: rotação, resize, salto de frames
  detector.py               wrapper do YOLO COCO; separa obstáculos de pessoas
  lane.py                   localiza o piso tátil e deriva a faixa livre
  blockage.py               regra espacial e persistência temporal
  events.py                 escrita e leitura de outputs/events.csv
  ui.py                     overlay e componentes Streamlit
  surface.py                obstrução sem classe conhecida — EXPERIMENTAL, desligado

tools/
  detect_lane.py            localiza a faixa e grava a zona
  render_video.py           exporta vídeo anotado
  check_objeto.py           o YOLO reconhece este objeto?
  draw_zone.py              desenha a zona à mão, por cliques

tests/
  smoke_env.py              ambiente, GPU, device
  smoke_capture.py          leitura de vídeo e webcam
  smoke_inference.py        carga do modelo e latência
  test_blockage.py          9 testes da regra espacial, sem vídeo nem GPU

data/samples/               vídeos da UNIFEI (não versionados)
outputs/events.csv          um registro por bloqueio
outputs/demos/              vídeos anotados (não versionados)
```

---

## Privacidade e ética

- **Nenhum vídeo ou frame é gravado** pelo pipeline. Só as linhas de `outputs/events.csv`.
- **Não há reconhecimento facial nem identificação de pessoas**, em nenhuma forma. Pessoas são
  detectadas como objeto genérico e marcadas `não obstrui` — nada as identifica, rastreia entre
  frames ou extrai característica pessoal.
- **Telemetria desligada** nas duas bibliotecas que a trazem ativa: Ultralytics (configuração
  isolada em `.ultralytics/`, sem tocar em `~/.config`) e Streamlit (`gatherUsageStats = false`).
- O Streamlit escuta apenas em `localhost`, não em todas as interfaces de rede.

---

## Limitações conhecidas

**O vocabulário do detector.** O COCO tem cadeira, banco, vaso, bicicleta e carro. Não tem cone
de obra, tapume, placa de sinalização, entulho nem caixa — que são obstáculos comuns num campus.
O próximo passo real do projeto é anotar um dataset do próprio ambiente.

**Obstrução sem classe conhecida.** `src/surface.py` ataca isso por aparência, sem depender de
classe: em `caixas.mp4` acerta a caixa de papelão que o YOLO não vê. Mas acusa 3 falsos
positivos em caminho livre, então está **desligado do pipeline**. O módulo documenta o que foi
medido e o caminho sugerido.

**Um clipe com a faixa errada.** Em `bicicleta_planta.mp4` um vaso grande cobre o piso tátil por
um trecho longo e os dois métodos erram. O sistema cai no polígono fixo, avisando.

**Câmera precisa estar parada**, e o trecho, reto. Corredor curvo ou câmera em movimento quebram
a premissa.

**Sem webcam no WSL2.** O WSL2 não expõe `/dev/video*`. A entrada é arquivo; o caminho de webcam
existe em `capture.py` mas não foi exercitado. Para câmera ao vivo: Linux/Windows nativo, ou
`usbipd-win`.

**Sem GPS, mapa do campus, banco de dados ou autenticação.** Por decisão de escopo.

---

## Créditos e licenças

| Componente | Origem | Licença |
|---|---|---|
| `yolo11n_tactile.pt` | [GuideTWSI](https://guidedogrobot-tactile.github.io/) — DARoS Lab, UMass Amherst ([código](https://github.com/DARoSLab/GuideTWSI), [arXiv](https://arxiv.org/abs/2603.07060)) | MIT |
| `yolo11n.pt` e a biblioteca Ultralytics | [Ultralytics](https://github.com/ultralytics/ultralytics) | AGPL-3.0 |
| OpenCV, NumPy, Pandas, Streamlit, Pillow, PyTorch | — | BSD / Apache 2.0 |

> A Ultralytics é **AGPL-3.0**: quem distribui o software ou o oferece como serviço em rede
> precisa abrir o código. Este repositório é público, então está coberto. Se o projeto virar
> produto ou serviço institucional, alguém precisa revisar o licenciamento — a Ultralytics
> vende licença comercial para esse caso.

---

## Documentação do projeto

- [`PROJECT_CONTEXT.md`](PROJECT_CONTEXT.md) — o que o sistema é e as decisões de produto.
- [`AGENTS.md`](AGENTS.md) — invariantes e regras de código, para pessoas e IAs de IDE.
- [`TASKS.md`](TASKS.md) — frentes de trabalho e o que ficou em aberto.

Projeto desenvolvido em hackathon de acessibilidade na **UNIFEI**.
