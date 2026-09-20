# Caminho Livre — UNIFEI

MVP de visão computacional que identifica **obstáculos temporários** bloqueando áreas de
circulação acessível no campus da UNIFEI.

Uma câmera estática observa um trecho de passagem delimitado por um polígono. Um detector
YOLO pré-treinado encontra objetos no frame; se o **ponto inferior central** da caixa de um
objeto cai dentro do polígono por alguns frames seguidos, o sistema registra um evento e
mostra **BARREIRA TEMPORÁRIA**. Caso contrário, **ROTA LIVRE**.

> **Estado atual:** ambiente e esqueleto prontos, pipeline `captura → YOLO → inferência`
> verificado. A regra espacial, a interface e os testes estão distribuídos em quatro frentes
> paralelas — ver [`TASKS.md`](TASKS.md).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt        # com GPU NVIDIA — ~3 GB
pip install -r requirements-cpu.txt    # sem GPU, ou só para desenvolver — ~200 MB
```

As duas variantes travam as mesmas versões de tudo, mudando só o PyTorch. Só quem vai
rodar a demo precisa da versão com CUDA.

## Uso

```bash
source .venv/bin/activate     # ativar o ambiente
streamlit run app.py          # abrir a interface
```

Verificação rápida da instalação, na ordem:

```bash
python tests/smoke_env.py                      # versões, GPU, device escolhido
python tests/smoke_capture.py                  # leitura de vídeo e webcam
python tests/smoke_inference.py                # carga do modelo e inferência
python tests/smoke_inference.py data/test/pipeline_check.mp4   # inferência em vídeo
```

### Antes de gravar: teste o objeto

O detector só conhece 80 classes do COCO. **Caixa de papelão não é uma delas** — foi
testado: aparece como `tv` a 0.34, ou simplesmente não é detectada. Cone de obra, tapume
e entulho também estão fora.

Teste qualquer objeto com uma foto antes de gastar tempo filmando:

```bash
python tools/check_objeto.py foto_do_objeto.jpg
python tools/check_objeto.py clipe.mp4        # amostra 12 frames do vídeo
```

Tire a foto do mesmo ângulo, distância e iluminação da gravação — a resposta muda com a
perspectiva. Objetos que funcionam bem e são fáceis de conseguir num campus: **cadeira,
mochila, bicicleta, vaso de planta, mala, banco**.

### Material de verificação

Vídeos e imagens não são versionados. Para recriar o clipe usado pelos smoke tests:

```bash
curl -sL -o data/test/bus.jpg \
  https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/assets/bus.jpg
ffmpeg -y -loop 1 -i data/test/bus.jpg -t 5 -r 30 \
  -vf "scale=-2:720,pad=1280:720:(ow-iw)/2:0:color=gray" \
  -c:v libx264 -pix_fmt yuv420p -crf 23 data/test/pipeline_check.mp4
```

É uma imagem em loop — serve para exercitar o pipeline com detecções reais em todo frame,
não para avaliar qualidade. Os vídeos de verdade vão em `data/samples/`.

## Como a faixa acessível é encontrada

O sistema **não usa um polígono desenhado à mão**. Ele localiza o piso tátil direcional no
próprio vídeo e deriva dele a faixa livre de circulação:

```bash
python tools/detect_lane.py data/samples/livre.mp4
python tools/detect_lane.py data/samples/livre.mp4 --ratio 5
```

Quatro passos, só com OpenCV e NumPy:

1. **Fundo** — câmera estática mais mediana temporal de 25 frames dá o corredor sem quem
   estava passando. Objeto parado permanece, e tudo bem: ele é obstáculo, não ruído.
2. **Máscara** — por linha da imagem, o piso é a referência clara e marcamos o que escurece
   em relação a ela. Isso pega o piso tátil **e** os objetos.
3. **Eixo** — o piso tátil é a única estrutura que atravessa toda a profundidade do
   corredor; um objeto ocupa só um trecho. Procuramos a reta de maior cobertura na máscara.
   Como um feixe inteiro de retas dentro da faixa empata, o eixo é a **mediana** do feixe.
4. **Largura** — cresce para os lados a partir do eixo e ajusta `largura = a·y + b`. A
   rejeição de outliers é assimétrica, porque encostar num objeto só infla a largura.

A faixa livre é o piso tátil multiplicado por `lane.free_width_ratio` (padrão 4), simétrico
em torno do eixo — quem se guia pelo piso tátil precisa de folga para o corpo e a bengala.
O valor exato da norma (NBR 9050) deve ser conferido por quem a tenha; o padrão aqui é
escolha de engenharia, não citação de norma.

**Por que isso importa:** o clipe `borda.mp4` foi gravado com a câmera ligeiramente
deslocada, e o detector acompanhou — eixo em `0,468` contra `0,501` dos demais. Um polígono
fixo teria errado a faixa nesse clipe.

**Quando não funciona:** exige câmera estática, piso tátil visivelmente mais escuro que o
piso ao redor e trecho reto. Corredor curvo, piso tátil de cor parecida com o piso ou
câmera em movimento quebram a premissa — aí use `tools/draw_zone.py` e a zona fixa.

## Saída visual

A interface mostra o frame anotado ao vivo. Para gerar um arquivo de vídeo anotado — útil
para a apresentação, e como plano B se a demo ao vivo falhar:

```bash
python tools/render_video.py data/samples/obstruido.mp4
python tools/render_video.py data/samples/obstruido.mp4 --inicio 8 --fim 25
```

O vídeo sai em `outputs/demos/` com a zona, as caixas e o veredito queimados na imagem:

- **zona** verde quando a rota está livre, vermelha quando há barreira confirmada;
- **caixa laranja** = objeto detectado fora da zona; **vermelha** = objeto invadindo;
- **caixa azul** = pessoa, rotulada `não obstrui` — mesmo pisando na faixa;
- **círculo** na base de cada caixa = o ponto inferior central, que é o que a regra testa;
- **barra superior** com `ROTA LIVRE` ou `BARREIRA TEMPORARIA` e o motivo.

Pessoas são detectadas e desenhadas, mas nunca contam como barreira: pedestre em trânsito
não é obstáculo. Apagá-las da tela faria o sistema parecer cego a elas — mostrá-las
rotuladas é o que prova que a distinção é deliberada. Continua valendo a regra do projeto:
é detecção de objeto genérico, sem nenhuma identificação de quem a pessoa é.

> **Privacidade.** O pipeline em operação **não grava vídeo** — só eventos em CSV. Esta
> ferramenta é uma ação deliberada, fora do fluxo normal, e o arquivo gerado contém as
> pessoas que aparecem na gravação original, ainda que o sistema não as identifique.
> Confira quem aparece antes de pôr num slide ou mandar em grupo.

## Como gravar os vídeos

O polígono da zona é **fixo em coordenadas de imagem**. Isso impõe um requisito de gravação:

- **Celular parado.** Apoiado em tripé, muro ou banco. Não caminhar durante a gravação — se
  a câmera se move, a faixa de circulação sai de baixo do polígono e a regra perde o sentido.
- **Orientação horizontal** (paisagem).
- Enquadrar o trecho de passagem de ponta a ponta.
- Salvar em `data/samples/`. Vídeos não são versionados nem persistidos pelo sistema.

Se o vídeo for HEVC/H.265 (padrão do iPhone) e o OpenCV não abrir, converta:

```bash
ffmpeg -i entrada.mov -c:v libx264 -crf 23 saida.mp4
```

## Configuração

Tudo que se ajusta está em [`config/zones.json`](config/zones.json):

| Chave | O que faz |
|---|---|
| `capture.work_width` | largura do frame de trabalho; menor = mais rápido |
| `detection.conf_threshold` | confiança mínima para aceitar uma detecção |
| `detection.target_classes` | classes COCO tratadas como possível obstáculo |
| `detection.ignored_classes` | classes sempre descartadas (`person`, por decisão de projeto) |
| `blockage.confirm_frames` | frames consecutivos para confirmar um bloqueio |
| `blockage.release_frames` | frames consecutivos sem invasão para liberar a rota |
| `zones.<nome>.polygon_norm` | vértices do polígono em coordenadas **normalizadas** (0–1) |

O polígono atual é um trapézio **placeholder**. Substitua pelos pontos do trecho real —
a ferramenta de desenho do polígono é tarefa da frente 2 (`TASKS.md`).

## Estrutura

```
app.py                  interface Streamlit (esqueleto)
.streamlit/config.toml  Streamlit em localhost, sem envio de estatísticas
config/zones.json       zona, classes e limiares
models/                 pesos YOLO pré-treinados (baixados sob demanda)
src/
  config.py             leitura de config/zones.json          [funcional]
  capture.py            leitura de vídeo/webcam               [funcional]
  detector.py           wrapper do YOLO                       [funcional]
  blockage.py           regra espacial e persistência         [stub — frente 2]
  events.py             escrita de outputs/events.csv         [stub — frente 3]
  ui.py                 componentes Streamlit                 [stub — frente 3]
data/samples/           vídeos reais da UNIFEI (não versionados)
data/test/              material de verificação do pipeline
outputs/events.csv      um registro por bloqueio encerrado
tests/                  smoke tests dos três portões
```

## Privacidade

- **Nenhum vídeo ou frame é gravado em disco.** Só as linhas de `outputs/events.csv`.
- **Não há reconhecimento facial nem identificação de pessoas**, em nenhuma forma.
  A classe `person` é ignorada: pedestre em trânsito não é barreira temporária.
- **Telemetria desligada** nas duas bibliotecas que a trazem ativa por padrão: Ultralytics
  (configuração isolada em `.ultralytics/`, sem tocar em `~/.config`) e Streamlit
  (`gatherUsageStats = false`). Nada do projeto sai da máquina.
- O Streamlit escuta apenas em `localhost`, não em todas as interfaces de rede.

## Limitações conhecidas

- **Sem webcam no WSL2.** O WSL2 não expõe `/dev/video*`. A entrada é arquivo de vídeo; o
  caminho de webcam existe em `capture.py` mas não foi exercitado neste ambiente. Para
  câmera ao vivo: rodar em Linux/Windows nativo, ou anexar o dispositivo com `usbipd-win`.
- **Câmera precisa estar parada** — ver "Como gravar os vídeos".
- **Detector pré-treinado em COCO.** Ele conhece cadeira, banco, vaso, bicicleta, carro e
  afins; **não** conhece cone de obra, tapume, placa de sinalização ou entulho — obstáculos
  comuns em campus. Reconhecer esses exigiria treinamento próprio, que está fora do MVP.
- **Sem GPS, mapa do campus, banco de dados ou autenticação.** Por decisão de escopo.

## Documentação do projeto

- [`PROJECT_CONTEXT.md`](PROJECT_CONTEXT.md) — o que o sistema é e as decisões de produto.
- [`AGENTS.md`](AGENTS.md) — regras de código e invariantes, para pessoas e IAs de IDE.
- [`TASKS.md`](TASKS.md) — as quatro frentes de trabalho paralelas.
