# PROJECT_CONTEXT — Caminho Livre — UNIFEI

Memória estável do projeto. Quem chega (pessoa ou IA de IDE) lê este arquivo primeiro.
Mudou uma decisão de produto? Atualize aqui, não só no código.

## Identificação

- **Nome:** Caminho Livre — UNIFEI
- **Tipo:** MVP de visão computacional (hackathon de acessibilidade)
- **Objetivo:** identificar objetos que bloqueiam temporariamente uma faixa de circulação
  acessível no campus e alertar sobre a obstrução.

## Como o sistema funciona

1. **Entrada** — vídeo gravado com celular em vias e caminhos da UNIFEI. A webcam é um
   caminho alternativo previsto no código, mas não é o fluxo principal.
2. **Detecção** — YOLO pré-treinado em COCO (`yolo11n.pt`). **Não há treinamento próprio.**
3. **Zona acessível** — polígono configurado em `config/zones.json`, em coordenadas
   normalizadas (0–1), aplicado sobre o frame de trabalho.
4. **Regra principal** — o **ponto inferior central da bounding box** determina se o objeto
   está dentro da zona de circulação. Esse ponto aproxima onde o objeto toca o chão; o
   centro da caixa faria um objeto alto parecer mais distante do que está.
5. **Persistência temporal** — um bloqueio só é **confirmado** depois de alguns frames
   consecutivos (`blockage.confirm_frames`), e só é liberado depois de alguns frames
   consecutivos sem invasão. Sem isso, uma detecção piscante viraria alerta.
6. **Saída** — status `ROTA LIVRE` ou `BARREIRA TEMPORÁRIA`, mais uma linha em
   `outputs/events.csv` por bloqueio encerrado.

## Protocolo de gravação (requisito, não sugestão)

- **Celular estático**, apoiado em tripé, muro ou banco. Não caminhar durante a gravação.
- **Orientação horizontal** (paisagem).
- Enquadrar o trecho de passagem de ponta a ponta.

O polígono é fixo em coordenadas de imagem. Se a câmera se mover, a faixa de circulação sai
de baixo do polígono e a regra espacial perde o sentido. Clipe gravado caminhando **não deve
entrar em `data/samples/`** — corrigir isso exigiria rastrear a cena, o que está fora do MVP.

## Esquema do evento (`outputs/events.csv`)

| campo | significado |
|---|---|
| `event_id` | sequencial dentro da execução |
| `timestamp` | início do bloqueio, ISO 8601 local |
| `source` | arquivo de vídeo ou `webcam:N` |
| `class_name` | classe COCO do objeto |
| `confidence` | confiança média durante o bloqueio |
| `duration_s` | duração em segundos |
| `frames` | frames em que o bloqueio persistiu |
| `zone` | zona ativa em `config/zones.json` |

## Privacidade e limites éticos

- **Não armazenar vídeo ou frames brutos por padrão.** Só as linhas do CSV são persistidas.
- **Não implementar reconhecimento facial nem identificação de pessoas.** Em nenhuma forma.
- A classe `person` do COCO está em `detection.ignored_classes`: pedestre em trânsito não é
  barreira temporária e geraria falso positivo constante.
- A telemetria do Ultralytics fica desligada (`sync=False` em `.ultralytics/`, local ao
  projeto). Nada sai da máquina.

## Fronteira da stack do MVP

**Dentro:** Python, OpenCV, NumPy, Ultralytics YOLO, Streamlit, Pandas, Pillow.

**Fora, por decisão explícita:** Grounding DINO, SAM, GPS, mapa do campus, banco de dados,
autenticação, treinamento de modelo. Não adicionar tecnologia nova sem necessidade comprovada
para o MVP.

## Ambiente de desenvolvimento verificado

- Ubuntu 24.04 sobre WSL2, Python 3.12.3, ambiente virtual em `.venv/`.
- GPU NVIDIA RTX 5060 Laptop (8 GB, Blackwell `sm_120`) disponível via WSL.
- **Sem webcam:** o WSL2 não expõe `/dev/video*`. Por isso a entrada é arquivo de vídeo.
