# AGENTS — regras operacionais para IA de IDE

Contexto de produto está em `PROJECT_CONTEXT.md`. **Leia aquele arquivo antes deste.**
Aqui ficam só as regras de como mexer no código.

## Invariantes — não quebrar sem combinar com o time

1. **Nenhum treinamento de modelo.** Só pesos pré-treinados COCO (`yolo11n.pt`).
2. **A regra espacial usa o ponto inferior central da bounding box.** Trocar para centro da
   caixa, IoU com o polígono ou máscara de segmentação é mudança de produto, não refatoração.
3. **Bloqueio exige N frames consecutivos.** Não reportar barreira a partir de um frame só.
4. **Não persistir vídeo nem frames.** Só `outputs/events.csv`.
5. **Nunca implementar reconhecimento facial ou identificação de pessoas.**
6. **Não adicionar dependência** fora de OpenCV, NumPy, Ultralytics, Streamlit, Pandas,
   Pillow e PyTorch. Nada de Grounding DINO, SAM, GPS, mapas, banco de dados, autenticação.
   Precisa de algo novo? Levante a questão antes de instalar.
7. **Configuração fica em `config/zones.json`**, não hardcoded no código. Polígonos em
   coordenadas normalizadas (0–1).

## Donos dos arquivos

Quatro pessoas trabalham em paralelo. Antes de editar, confira de quem é o arquivo — ver
`TASKS.md`. Mexer no arquivo de outra frente gera conflito de merge no pior momento.

| Arquivo | Frente |
|---|---|
| `src/capture.py`, `src/detector.py` | 1 — Detecção |
| `src/blockage.py` | 2 — Regra espacial |
| `src/events.py`, `src/ui.py`, `app.py` | 3 — Interface |
| `tests/` | 4 — Testes |
| `config/zones.json`, `src/config.py` | compartilhado — avisar no grupo ao alterar |

## Estado atual do código

`src/capture.py`, `src/detector.py` e `src/config.py` estão **funcionais**.
`src/blockage.py`, `src/events.py` e `src/ui.py` são **stubs**: as assinaturas e os
contratos estão fechados, os corpos levantam `NotImplementedError` com um marcador
`TODO(frente-N)`. Implemente dentro da assinatura existente; mudar assinatura quebra quem
já programou contra ela.

## Convenções

- Python 3.12, `from __future__ import annotations`, type hints nas funções públicas.
- Docstrings e comentários em português; identificadores em inglês.
- Comentário explica **por quê**, não o que a linha faz.
- Frames são `numpy.ndarray` em **BGR** (padrão OpenCV). Converter para RGB só na borda
  da interface, ao exibir.
- Coordenadas de detecção em pixels do **frame de trabalho** (já redimensionado pelo
  `VideoSource`), nunca do frame nativo.
- Sem segredo, token ou chave no repositório. Não há nenhum neste projeto — se surgir a
  necessidade, use variável de ambiente.

## Antes de dar uma tarefa por pronta

- Rodar os smoke tests: `.venv/bin/python tests/smoke_env.py`, `smoke_capture.py`,
  `smoke_inference.py`.
- Não silenciar erro com `try/except` largo para "fazer passar". Se quebrou, reporte.
