# Roteiro de apresentação — Caminho Livre (UNIFEI)

Roteiro para a banca. Montado para **8 minutos**, com marcação de corte para 5.
Cada bloco traz o que **falar** e o que **mostrar** na tela.

> **Antes de começar:** Streamlit já aberto em `http://localhost:8501`, vídeo
> `obstruido.mp4` selecionado, primeiro frame na tela. Nunca abra o app na frente da banca.
> Plano B em `outputs/demos/*.mp4` — ver [Plano B](#plano-b-se-a-demo-ao-vivo-falhar).

| # | Bloco | Tempo | Corte para 5 min |
|---|---|---|---|
| 1 | O problema | 1:00 | 0:45 |
| 2 | A proposta | 0:45 | 0:30 |
| 3 | A abordagem de visão computacional | 2:00 | 1:15 |
| 4 | Demonstração | 2:00 | 1:45 |
| 5 | Decisões técnicas e limitações | 1:30 | 0:45 |
| 6 | Trabalhos futuros | 0:45 | 0:30 |

---

## 1. O problema identificado

**Falar:**

> O piso tátil direcional existe para que uma pessoa com deficiência visual possa atravessar
> um espaço sem precisar de ajuda. Ele só cumpre isso enquanto está desimpedido.
>
> Uma bicicleta atravessada, uma cadeira deixada no corredor, uma caixa de entrega no meio da
> passagem. Para quem enxerga, é um desvio de dois passos. Para quem se guia pela faixa, a
> rota inteira se perde — a faixa guia deixa de guiar, e não existe aviso nenhum de que isso
> aconteceu.
>
> E esse é o ponto: **o problema não é o piso tátil mal instalado. É o obstáculo temporário
> sobre um piso tátil que está correto.** Auditoria de acessibilidade é feita uma vez por ano;
> a cadeira aparece numa terça-feira e some na quarta. Nenhum processo hoje enxerga isso.

**Mostrar:** um frame de `obstruido.mp4` sem nenhuma anotação — só o corredor com a cadeira e
a mochila sobre a faixa. Deixe a banca ver a cena crua antes de ver o sistema.

---

## 2. A proposta de solução

**Falar:**

> Construímos o **Caminho Livre**: um sistema que olha um trecho de circulação por vídeo e
> responde uma pergunta só, continuamente — **a rota está livre, ou existe uma barreira
> temporária sobre ela?**
>
> A entrada é um vídeo comum de celular, com a câmera parada. A saída são três coisas: o
> veredito na tela, o vídeo anotado mostrando **por que** o sistema decidiu assim, e um
> registro em CSV de cada barreira, com objeto, horário e duração.
>
> Esse CSV é o que transforma a demonstração em ferramenta de gestão: ele responde *onde* e
> *por quanto tempo* uma rota do campus ficou inacessível.
>
> Uma decisão de escopo desde o começo: **o sistema não identifica pessoas.** Pedestre é
> detectado como objeto genérico e marcado explicitamente como quem *não* obstrui. Sem
> reconhecimento facial, sem rastreio de indivíduo, sem gravação de vídeo em operação.

**Mostrar:** o diagrama de quatro passos do `README.md`, ou a tela do Streamlit parada.

---

## 3. A abordagem de Visão Computacional

> **Este é o bloco técnico. Se a banca for técnica, é aqui que se ganha a apresentação.**

**Falar — a tese:**

> A tentação óbvia é desenhar um polígono sobre a faixa e testar se tem objeto dentro. Nós
> começamos assim e **jogamos fora**: um polígono desenhado vale para um enquadramento só.
> Trocou de corredor, trocou de câmera, tem que desenhar de novo. Isso não é um sistema, é
> uma configuração.
>
> Então o sistema **encontra o piso tátil no próprio vídeo** e deriva o caminho dele.

**Falar — os quatro passos:**

**1. Onde é o caminho.**
> Primeiro, uma **mediana temporal de 25 frames**. Como a câmera é estática, quem passou
> desaparece e sobra o corredor limpo. Objeto parado permanece — e tudo bem, ele é obstáculo.
>
> Sobre esse fundo roda o **`yolo11n_tactile.pt`**, um YOLOv11n de segmentação treinado
> especificamente para piso tátil, publicado pelo projeto GuideTWSI da UMass sob licença MIT,
> com **19.925 imagens anotadas pixel a pixel**. Ele não procura "o que é escuro": aprendeu o
> que é piso tátil.
>
> A máscara vira geometria: para cada linha da imagem, onde a faixa começa e termina. Aí
> ajustamos eixo e largura impondo uma restrição física — **a largura é constante no mundo
> real, então na imagem ela tem que convergir a zero no ponto de fuga.** Sem essa restrição, o
> ajuste sai quase horizontal em cena ruidosa e o polígono vira um retângulo.
>
> A faixa trafegável é o piso tátil **alargado 4 vezes, simétrico em torno do eixo** — quem se
> guia pela faixa precisa de folga para o corpo e para a bengala, e a folga tem que ser igual
> dos dois lados.

**2. O que tem na cena.**
> **YOLO11n** pré-treinado no COCO, 2,6 milhões de parâmetros. Filtramos 14 classes que fazem
> sentido como obstáculo de circulação. Pessoa sai por um canal separado, para ser desenhada
> mas nunca contar como barreira.

**3. Está no caminho?** — *o coração do projeto.*
> O detector diz "existe uma cadeira em tal lugar". Ele não faz ideia do que é uma calçada.
> Quem transforma isso em "esta rota não está acessível" é a nossa geometria.
>
> E ela usa o **ponto inferior central da bounding box**, não o centro. Um armário encostado na
> parede tem o centro da caixa sobre a faixa, mas a base fora dela. Com o ponto de contato com
> o chão, ele não conta. Com o centro, contaria — e o sistema alarmaria sozinho o dia inteiro.

**4. É mesmo uma barreira?**
> Detector pisca. Numa sombra, um vaso vira "mochila" por um frame e some no seguinte. Um
> bloqueio só é confirmado depois de **8 frames consecutivos** — cerca de meio segundo — e só é
> liberado depois de 8 seguidos sem invasão. **Histerese**, para o status não oscilar com um
> objeto exatamente na borda.

**Mostrar:** o diagrama do ponto inferior central. Se der, um frame com o contorno ciano do
piso tátil detectado ao lado da faixa verde derivada dele.

**Se a banca perguntar "e se o modelo de piso tátil falhar?":**
> Existe um método reserva que entra sozinho: uma heurística de luminância, só OpenCV e NumPy,
> que marca o que escurece em relação ao piso da própria linha e acha o eixo pela reta de maior
> cobertura. Os dois métodos produzem exatamente o mesmo objeto de saída — o modelo é uma peça
> substituível, não o alicerce.

---

## 4. Demonstração

> **Roteiro de cliques.** Ensaie uma vez. Tudo abaixo é botão ou seleção, nada de terminal.

| Momento | Ação | O que dizer enquanto roda |
|---|---|---|
| 0:00 | Tela já aberta em `obstruido.mp4` | "Sete clipes gravados no corredor B.2.1, câmera parada, 1280×720." |
| 0:15 | **Processar** | "Ciano é o piso tátil que o modelo achou. Verde é a faixa trafegável derivada dele." |
| 0:30 | Barreira confirma | "Cadeira e mochila com a base dentro da faixa. A zona fica vermelha, e o evento abre no CSV." |
| 0:50 | Apontar o círculo na base | "Esse círculo é o ponto que a regra testa. Dá para conferir a decisão olhando um frame." |
| 1:10 | Trocar para **`pessoas.mp4`** | "Mesma faixa, agora com gente atravessando." |
| 1:30 | Pessoas em azul | "Marcadas em azul, rotuladas 'não obstrui'. O sistema **vê** a pessoa e decide não contá-la — não é cegueira, é decisão de produto." |
| 1:50 | Tabela de eventos | "Um evento, com objeto, horário e duração. É o que vira relatório de rota inacessível." |

**Se sobrar tempo**, mostre `borda.mp4`: objeto encostado na parede, perto da faixa, e o
sistema mantém ROTA LIVRE. É o caso que prova que a regra do ponto de contato está fazendo
trabalho de verdade.

**Resultados medidos** — deixe esta tabela no slide durante a demo:

| clipe | cena | esperado | obtido |
|---|---|---|---|
| `livre.mp4` | passagem desimpedida | ROTA LIVRE | ✅ 0 eventos |
| `obstruido.mp4` | cadeira e mochila na faixa | BARREIRA | ✅ 1 evento, 99% do tempo |
| `borda.mp4` | objeto encostado, sem bloquear | ROTA LIVRE | ✅ 0 eventos |
| `pessoas.mp4` | pedestres atravessando a faixa | ROTA LIVRE | ✅ 0 eventos |

E a precisão da localização do eixo do piso tátil, contra medição manual em `x ≈ 0,498`:

| clipe | heurística | modelo de segmentação |
|---|---|---|
| `obstruido` | erro 0,0030 | **erro 0,0001** |
| `livre` | erro 0,0030 | **erro 0,0006** |
| `objetos_variaveis` | eixo 0,321 — errado | **0,509 — correto** |

---

## 5. Principais decisões técnicas e limitações

> **Regra do bloco:** falar das limitações com a mesma firmeza dos acertos. Banca de hackathon
> desconfia de projeto sem limitação conhecida — e quem já mediu o próprio erro passa mais
> confiança do que quem não olhou.

**As quatro decisões que definiram o projeto:**

1. **Nada de treinar modelo.** Em poucas horas, treino é aposta. Usamos dois pesos
   pré-treinados e investimos o tempo na geometria e na regra — que é onde estava o problema
   de verdade.
2. **A faixa é detectada, não desenhada.** Foi a decisão que mudou o projeto de demonstração
   para sistema. O polígono desenhado continua no código como reserva, e avisa quando entra.
3. **Ponto inferior central, não centro da caixa.** Uma linha de código que decide se o sistema
   é usável ou alarma o tempo todo.
4. **Confirmação por N frames com histerese.** Um frame não é evidência.

**As limitações, ditas na cara:**

- **O vocabulário do detector é o gargalo real.** O COCO tem cadeira, vaso e bicicleta. **Não
  tem cone de obra, tapume, entulho nem caixa de papelão** — que são obstáculos comuns num
  campus. Testamos: uma caixa de papelão no chão não é detectada nem baixando a confiança para
  0,05.
- **Atacamos isso e não resolvemos.** Escrevemos um detector por aparência, sem depender de
  classe (`src/surface.py`). Ele acerta a caixa em `caixas.mp4`, mas dá **3 falsos positivos em
  caminho livre**. Está no repositório, documentado e **desligado do pipeline** — falso alarme
  em acessibilidade destrói a confiança no sistema mais rápido do que a falha que ele evitaria.
- **Um clipe com a faixa errada.** Em `bicicleta_planta.mp4` um vaso grande cobre o piso tátil
  num trecho longo e os dois métodos erram. O sistema cai no polígono fixo, avisando.
- **A câmera precisa estar parada e o trecho, reto.** Corredor curvo ou câmera na mão quebram a
  premissa da mediana temporal e do ajuste de reta.
- **O fator 4 da largura é escolha de engenharia**, não a NBR 9050. Precisa ser conferido por
  quem tenha a norma.
- **Licença.** O peso do piso tátil é MIT. A Ultralytics é **AGPL-3.0**: o repositório público
  cobre a obrigação hoje, mas virar serviço institucional exige revisar o licenciamento.

---

## 6. Trabalhos futuros

Em ordem de retorno sobre esforço:

1. **Anotar um dataset do próprio campus.** É o desbloqueio de maior impacto: resolve cone,
   tapume, entulho e caixa de uma vez. Os sete clipes já gravados são o começo do conjunto, e o
   pipeline não muda — só o peso do detector.
2. **Fechar a detecção de obstrução sem classe conhecida.** Comparar cada frame com a mediana
   temporal do próprio clipe, em vez do modelo paramétrico de piso que está lá hoje. É a
   correção que tira os 3 falsos positivos e liga o `surface.py` ao pipeline.
3. **Robustez da faixa sob oclusão longa.** Unir as máscaras de vários frames em vez de rodar
   o modelo só no fundo mediano — é o que deve resolver o `bicicleta_planta.mp4`.
4. **Câmera ao vivo e alerta.** O caminho de webcam já existe em `capture.py`, sem uso por
   falta de `/dev/video*` no WSL2. Com câmera fixa num corredor, o evento pode virar
   notificação para a manutenção em vez de linha em CSV.
5. **Painel de rotas do campus.** Vários pontos monitorados, com histórico de quanto tempo cada
   rota ficou inacessível. É o que transforma a ferramenta em insumo de decisão para a
   administração — e o CSV já está no formato certo para isso.
6. **Validação com quem usa a faixa.** Nada aqui foi testado com pessoas com deficiência
   visual. O fator de largura, o tempo de confirmação e o que conta como barreira precisam
   passar por elas antes de virar qualquer coisa institucional.

---

## Plano B, se a demo ao vivo falhar

1. Vídeos anotados prontos em `outputs/demos/*.mp4` — rode no player do sistema.
2. Frames estáticos em `outputs/demos/*.png`.
3. Em último caso, `outputs/events.csv` aberto: os eventos registrados são evidência real.

**Não** tente depurar na frente da banca. Passe para o plano B e siga o roteiro.

---

## Perguntas prováveis

**"Por que não usar SAM ou um modelo de fundação?"**
> Escopo e tempo. E honestamente: o problema difícil aqui não era segmentar, era decidir o que
> conta como barreira. Trocar o segmentador é mudar uma função — a arquitetura já prevê isso.

**"Funciona em tempo real?"**
> Medido nesta máquina, num notebook com RTX 5060: **21,7 a 35,3 frames por segundo** de
> processamento ponta a ponta nos quatro clipes — e esse número ainda inclui gravar o vídeo
> anotado em disco, que a operação normal não faz. Roda em tempo real com folga.

**"Por que não rastrear o objeto entre frames?"**
> Porque rastrear objeto e rastrear pessoa usam o mesmo mecanismo, e nós decidimos não rastrear
> pessoa. A histerese resolve o problema do detector piscando sem precisar de identidade.

**"E se a faixa estiver suja, apagada ou mal instalada?"**
> Aí o modelo não acha e o sistema avisa que caiu no polígono reserva. Vale notar que isso
> também é informação útil: uma faixa que o detector não enxerga provavelmente é uma faixa que
> a pessoa também não sente bem.

---

Detalhes técnicos completos em [`README.md`](README.md). Decisões de produto em
[`PROJECT_CONTEXT.md`](PROJECT_CONTEXT.md). O que ficou em aberto, em [`TASKS.md`](TASKS.md).
