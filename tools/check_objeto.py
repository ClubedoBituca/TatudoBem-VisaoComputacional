"""Testa se o YOLO reconhece um objeto, ANTES de gravar o video com ele.

O detector so conhece as 80 classes do COCO. Objeto fora dessa lista - caixa de
papelao, cone, tapume - nao vai ser detectado, ou vai ser confundido com outra
coisa. Vale descobrir isso com uma foto de 10 segundos, nao depois de gravar.

Uso:
    python tools/check_objeto.py foto.jpg
    python tools/check_objeto.py clipe.mp4          # amostra frames ao longo do video

Tire a foto do MESMO angulo, distancia e iluminacao da gravacao. A resposta
muda com a perspectiva: um objeto reconhecido de perto pode sumir a 10 metros.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402

import src  # noqa: E402,F401  (isola a config do Ultralytics no projeto)
from src.config import load_config, resolve_path  # noqa: E402

# Proposital: bem abaixo do limiar de producao. Queremos ver ate o que a rede
# mal cogita, para entender COMO ela enxerga o objeto - nao so o que ela aceita.
CONF_EXPLORATORIA = 0.05
FRAMES_AMOSTRADOS = 12


def classificar(nome: str, alvo: set[str], ignoradas: set[str]) -> str:
    if nome in alvo:
        return "ALVO      "
    if nome in ignoradas:
        return "IGNORADA  "
    return "fora      "


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2

    caminho = Path(sys.argv[1]).resolve()
    if not caminho.exists():
        print(f"Arquivo nao encontrado: {caminho}")
        return 1

    cfg = load_config()["detection"]
    alvo = set(cfg["target_classes"])
    ignoradas = set(cfg["ignored_classes"])

    from ultralytics import YOLO

    from src.detector import disable_telemetry

    disable_telemetry()
    modelo = YOLO(str(resolve_path(cfg["model_path"])))
    nomes = modelo.model.names

    print("=" * 64)
    print(f"O QUE O YOLO VE EM: {caminho.name}")
    print("=" * 64)

    frames: list = []
    if caminho.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        img = cv2.imread(str(caminho))
        if img is None:
            print("Nao foi possivel ler a imagem.")
            return 1
        frames = [img]
    else:
        captura = cv2.VideoCapture(str(caminho))
        if not captura.isOpened():
            print("Nao foi possivel abrir o video. Se for HEVC do iPhone, converta:")
            print("  ffmpeg -i entrada.mov -c:v libx264 -crf 23 saida.mp4")
            return 1
        total = int(captura.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        passo = max(1, total // FRAMES_AMOSTRADOS) if total else 30
        indice = 0
        while len(frames) < FRAMES_AMOSTRADOS:
            ok, frame = captura.read()
            if not ok:
                break
            if indice % passo == 0:
                frames.append(frame)
            indice += 1
        captura.release()
        print(f"Amostrados {len(frames)} frames ao longo do video.")

    melhor: dict[str, float] = {}
    presenca: Counter[str] = Counter()

    for frame in frames:
        resultado = modelo.predict(frame, imgsz=cfg["imgsz"], conf=CONF_EXPLORATORIA, verbose=False)[0]
        vistos_no_frame = set()
        for caixa in resultado.boxes:
            nome = nomes[int(caixa.cls[0])]
            conf = float(caixa.conf[0])
            melhor[nome] = max(melhor.get(nome, 0.0), conf)
            vistos_no_frame.add(nome)
        presenca.update(vistos_no_frame)

    if not melhor:
        print()
        print("NENHUMA deteccao, nem com conf=0.05.")
        print("O objeto esta fora do vocabulario do COCO. Escolha outro - veja a")
        print("lista de classes-alvo em config/zones.json.")
        return 1

    limiar = cfg["conf_threshold"]
    print()
    print(f"{'classe':<18} {'melhor conf':>11}  {'frames':>7}  situacao")
    print("-" * 64)
    for nome, conf in sorted(melhor.items(), key=lambda kv: -kv[1]):
        marca = classificar(nome, alvo, ignoradas)
        aviso = "" if conf >= limiar else "  (abaixo do limiar)"
        print(f"{nome:<18} {conf:>11.3f}  {presenca[nome]:>4}/{len(frames)}  {marca}{aviso}")

    print("-" * 64)
    uteis = [n for n, c in melhor.items() if c >= limiar and n in alvo]
    if uteis:
        print(f"SERVE: detectado como {', '.join(uteis)} acima do limiar de {limiar}.")
        estaveis = [n for n in uteis if presenca[n] == len(frames)]
        if len(frames) > 1 and not estaveis:
            print("ATENCAO: a deteccao nao aparece em todos os frames. Vai piscar -")
            print("aumente confirm_frames ou melhore o enquadramento.")
        return 0

    candidatos = [n for n, c in melhor.items() if c >= limiar and n not in ignoradas]
    if candidatos:
        print(f"NAO SERVE como esta: o objeto e visto como {', '.join(candidatos)},")
        print("que nao esta em detection.target_classes.")
        print("Ou escolha outro objeto, ou adicione essa classe ao config/zones.json")
        print("- sabendo que o rotulo vai aparecer errado na tela.")
    else:
        print("NAO SERVE: nada passa do limiar de confianca. Escolha outro objeto.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
