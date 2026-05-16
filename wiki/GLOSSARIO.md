# 📖 Glossário de Otimização de IA

Status: 🟢 Em construção
Última atualização: 2026-05-15

Termos técnicos que aparecem no estudo das técnicas de otimização. Cada entrada explica o que é, por que aparece no contexto de Flash Attention e onde se aprofundar.

---

## Sparse Attention

### O que é

Sparse Attention (atenção esparsa) é uma família de técnicas que calcula apenas um **subconjunto** das conexões entre tokens, em vez de todas as N×N conexões. Enquanto a atenção padrão calcula a relevância de cada token para todos os outros, a versão esparsa escolhe quais conexões calcular (ex: só tokens vizinhos, ou um padrão pré-definido).

É como ler um livro e só consultar as palavras no mesmo parágrafo, em vez de comparar cada palavra com todas as outras do livro inteiro.

### Por que aparece no Flash Attention

Flash Attention **não é** sparse attention — ele calcula a atenção exata (todas as conexões). Mas o contexto histórico é: antes de 2022, como a atenção O(N²) era proibitiva, a comunidade tentou resolvê-la com aproximações esparsas. O Flash Attention tornou várias delas obsoletas ao provar que dava para ter atenção exata com O(N) de memória.

### Exemplos

| Técnica | Estratégia | Qualidade |
|---|---|---|
| Sparse Transformer (2019) | Padrão fixo: local + strided | Perde conexões distantes |
| Longformer (2020) | Janela deslizante + tokens globais | Boa para NLP longo |
| BigBird (2020) | Aleatório + local + global | Melhor que sparse fixo |
| Flash Attention (2022) | **Exata**, sem esparsidade | Sem perda de qualidade |

### Para aprofundar
- [Sparse Transformer (OpenAI, 2019)](https://arxiv.org/abs/1904.10509)
- [Longformer (Allen AI, 2020)](https://arxiv.org/abs/2004.05150)

---

## Linformer

### O que é

Linformer (2020) foi uma das primeiras tentativas de reduzir o custo da atenção usando **projeção de baixo rank** (low-rank projection). A ideia: a matriz de atenção N×N pode ser aproximada pelo produto de duas matrizes menores, N×k e k×N, onde k é uma constante pequena (ex: 256). Isso reduz a complexidade de O(N²) para O(N·k).

Intuição: uma foto de 4000×4000 pixels pode ser comprimida para 4000×256 sem perder muita informação, se a informação for redundante (e os autores argumentam que a matriz de atenção é de baixo rank).

### Por que aparece no Flash Attention

O Linformer foi influente por mostrar que a matriz de atenção tem **baixo rank** (muita redundância). Mas ele é uma **aproximação**: descarta informação. O Flash Attention veio depois e mostrou que dava para ser exato e eficiente ao mesmo tempo, aposentando a necessidade de aproximações como o Linformer para a maioria dos casos.

### Limitação principal

Não escala para sequências mais longas que as vistas no treino (o tamanho de k é fixo). Se você treinar com N=512 e depois tentar inferir com N=4096, o Linformer degrada.

### Para aprofundar
- [Linformer (Facebook AI, 2020)](https://arxiv.org/abs/2006.04768)

---

## Performer

### O que é

Performer (2021) usa um truque matemático chamado **FAVOR+** (Fast Attention Via Orthogonal Random features) para aproximar o softmax da atenção. Em vez de calcular `softmax(QK^T)`, ele reescreve o softmax como um produto de kernels e aproxima esses kernels com projeções aleatórias (random features).

A grande vantagem: complexidade **linear** O(N) em vez de quadrática O(N²). Foi o primeiro mecanismo de atenção linear que funcionou bem na prática.

### Por que aparece no Flash Attention

Performer e Flash Attention atacam o mesmo problema de ângulos opostos:
- **Performer:** muda a matemática (aproximação do softmax) → O(N) computação
- **Flash Attention:** muda a engenharia (IO-awareness) → O(N) memória, mas ainda O(N²) computação

O Performer é uma aproximação (perde um pouco de qualidade). O Flash Attention é exato. Para aplicações onde qualidade é crítica, Flash Attention venceu.

### Para aprofundar
- [Performer (Google, 2021)](https://arxiv.org/abs/2009.14794)

---

## HBM vs SRAM — Hierarquia de Memória da GPU

### O que é

GPUs modernas têm uma hierarquia de memória com dois níveis principais:

**HBM (High Bandwidth Memory):**
- Memória principal da GPU (VRAM)
- Grande: 40 GB (A100), 80 GB (H100), 192 GB (B200)
- Lenta: ~1.5-3 TB/s de banda, mas latência de centenas de ciclos
- Todas as matrizes do modelo (pesos, ativações, gradientes) ficam aqui
- Analogia: um galpão do outro lado da rua — cabe tudo, mas cada viagem é demorada

**SRAM (Static RAM / Shared Memory):**
- Memória dentro de cada Streaming Multiprocessor (SM)
- Pequena: ~128-256 KB por SM (H100: 228 KB)
- Rápida: latência de ~20-30 ciclos, banda dezenas de vezes maior que HBM
- Analogia: sua mesa de trabalho — cabe pouco, mas tudo está ao alcance da mão

### Por que aparece no Flash Attention

TODO o Flash Attention gira em torno dessa diferença. O algoritmo ingênuo escreve a matriz N×N na HBM. O Flash Attention processa blocos pequenos na SRAM e **nunca escreve a matriz N×N na HBM**. Cada leitura/escrita evitada da HBM é um ganho direto de velocidade.

**Ordens de magnitude:**
| Memória | Capacidade (H100) | Banda | Latência |
|---|---|---|---|
| HBM (VRAM) | 80 GB | 3.35 TB/s | ~300 ciclos |
| L2 Cache | 50 MB | ~12 TB/s | ~100 ciclos |
| SRAM (Shared Mem) | 228 KB / SM | ~20+ TB/s | ~25 ciclos |
| Registradores | ~64K / SM | Máxima | ~1 ciclo |

### Para aprofundar
- [CUDA C++ Programming Guide — Memory Hierarchy](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#memory-hierarchy)
- [Guia de otimização Hopper](https://docs.nvidia.com/cuda/hopper-tuning-guide/)

---

## Tiles (Ladrilhamento)

### O que é

Tiling (ladrilhamento) é uma técnica de programação onde uma matriz grande é dividida em **blocos menores** (tiles) que cabem na memória rápida (SRAM). Cada bloco é carregado, processado completamente e descartado antes do próximo.

Analogia: você tem que pintar uma parede de 10m × 10m. Em vez de segurar a tinta toda na mão, você pinta um quadrado de 1m × 1m de cada vez. A parede inteira nunca está na sua mão — só o ladrilho atual.

### Por que aparece no Flash Attention

Tiling é a **primeira das duas ideias centrais** do Flash Attention. A matriz de atenção N×N é dividida em tiles de tamanho Br×Bc (ex: 128×128):

```
Matriz N×N:
┌─────────┬─────────┬─────────┐
│ Tile[0,0]│ Tile[0,1]│ Tile[0,2]│  ← 128×128 cada
├─────────┼─────────┼─────────┤
│ Tile[1,0]│ Tile[1,1]│ Tile[1,2]│
├─────────┼─────────┼─────────┤
│ Tile[2,0]│ Tile[2,1]│ Tile[2,2]│
└─────────┴─────────┴─────────┘
```

Cada tile de 128×128 em float16 ocupa 32 KB — cabe na SRAM de 228 KB.

### Tamanho do tile importa

- Muito pequeno: muitas idas à HBM buscar blocos (overhead de loop)
- Muito grande: não cabe na SRAM, cai de volta pra HBM (perde o propósito)
- Ótimo no FA3: Br=128, Bc=128 ou Br=256, Bc=64, dependendo da dimensão d

---

## Warp (CUDA)

### O que é

No modelo de execução CUDA, um **warp** é um grupo de **32 threads** que executam juntas no mesmo Streaming Multiprocessor (SM). Todas as 32 threads de um warp executam a **mesma instrução** ao mesmo tempo (modelo SIMT: Single Instruction, Multiple Threads), mas sobre dados diferentes.

É a unidade fundamental de execução em GPUs NVIDIA. Um SM moderno (H100) pode gerenciar 64 warps simultaneamente (2048 threads por SM).

### Por que aparece no Flash Attention

A evolução do Flash Attention é, em grande parte, sobre melhorar a **divisão de trabalho entre warps**:

| Versão | Como usa warps | Eficiência |
|---|---|---|
| FA1 | Cada warp processa uma linha inteira da matriz | Warps ociosos esperando outros |
| FA2 | Cada warp processa um bloco 128×128 | Menos comunicação entre warps |
| FA3 | Warps diferentes em estágios diferentes do pipeline (WGMMA) | Nenhum warp ocioso |

### Instrução WGMMA (Hopper)

No Hopper (H100), a instrução **WGMMA** (Warp Group Matrix Multiply-Accumulate) permite que um grupo de warps dispare uma multiplicação de matrizes de forma **assíncrona** — o warp não espera o resultado, continua executando outras operações enquanto a multiplicação roda em hardware dedicado (Tensor Core). Isso é o que permite o pipeline do FA3.

### Para aprofundar
- [CUDA C++ Programming Guide — Warps](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#simt-architecture)
- [Hopper Architecture Whitepaper](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)

---

## Online Softmax

### O que é

Online softmax é um algoritmo que calcula o softmax de um vetor **incrementalmente**, processando pedaços do vetor sem nunca precisar ver todos os valores de uma vez.

O softmax padrão precisa de duas passadas sobre os dados: uma para achar o máximo (estabilidade numérica) e outra para calcular exponenciais e soma. O online softmax mantém duas variáveis que são atualizadas a cada bloco:

```
m = máximo corrente (para estabilidade)
l = soma corrente dos exponenciais (para normalização)
```

Quando um novo bloco chega, o algoritmo:
1. Calcula o novo máximo (max entre m antigo e o máximo do bloco)
2. Rescala a soma antiga com `exp(m_velho - m_novo)` — isso põe todos os exponenciais na mesma base
3. Soma os exponenciais do novo bloco

### Por que aparece no Flash Attention

É o que **viabiliza o tiling** no cálculo do softmax da atenção. Sem o online softmax, você precisaria da matriz N×N inteira para calcular o softmax — exatamente o que o Flash Attention evita.

A mágica está na correção `exp(m_velho - m_novo)`: quando o máximo muda, todos os exponenciais calculados antes precisam ser rescalados. Essa correção faz isso de forma numericamente estável, sem overflow ou underflow.

### Exemplo numérico simples

```
Bloco 1: scores = [2, 1, 3]
  m = 3, l = e^(2-3) + e^(1-3) + e^(3-3) = 0.368 + 0.135 + 1 = 1.503

Bloco 2: scores = [5, 0, 4]  ← novo máximo 5 > 3!
  m_novo = 5
  l = 1.503 * e^(3-5) + e^(5-5) + e^(0-5) + e^(4-5)
    = 1.503 * 0.135 + 1 + 0.007 + 0.368
    = 0.203 + 1 + 0.007 + 0.368
    = 1.578
```

O fator `e^(3-5) = 0.135` "encolheu" corretamente os exponenciais do bloco 1 para a nova base.

### Para aprofundar
- [Online Normalizer Calculation for Softmax (Milakov & Gimelshein, 2018)](https://arxiv.org/abs/1805.02867)

---

## Softmax

### O que é

Softmax é uma função que converte um vetor de números quaisquer (positivos, negativos, grandes, pequenos) em **probabilidades que somam 1**. É a função mais usada em redes neurais quando o modelo precisa "escolher" entre várias opções.

Para um vetor z = [z₁, z₂, ..., zₙ]:

$$\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{n} e^{z_j}}$$

**Intuição:** imagine um concurso onde cada candidato recebe uma nota. A softmax pega essas notas e transforma em "chance de ser escolhido" — quem tem nota maior ganha mais chance, mas todos recebem pelo menos um pouco. O denominador garante que todas as chances somem 100%.

**Exemplo:** notas [2, 1, 0] → softmax → [0.67, 0.24, 0.09] (67%, 24%, 9%). O 2 ganhou mais peso porque e² ≈ 7.4 é bem maior que e¹ ≈ 2.7.

### Por que aparece em TODO lugar

- **Atenção:** converte scores QKᵀ em pesos de atenção (probabilidades de "prestar atenção" em cada token)
- **Saída do Transformer:** converte logits em probabilidades sobre o vocabulário para escolher a próxima palavra
- **Estabilidade numérica:** subtrai-se o máximo antes de aplicar exp (evita overflow)

---

## BLEU (Bilingual Evaluation Understudy)

### O que é

BLEU é uma métrica de 0 a 100 que mede a qualidade de tradução automática comparando a tradução do modelo com traduções humanas de referência. Quanto mais alto, melhor. É a métrica padrão usada no paper do Transformer (Vaswani et al., 2017).

**Como funciona (intuição):** conta quantos pedaços de 1, 2, 3 e 4 palavras (n-gramas) da tradução do modelo também aparecem na tradução humana de referência. Penaliza traduções muito curtas (para evitar que o modelo só traduza palavras fáceis).

**Exemplo:** EN→DE BLEU 28.4 do Transformer Big significava que, em 2017, era o melhor resultado já obtido para tradução inglês→alemão.

---

## Notação Big-O — O(n²), O(n), O(1)

### O que é

Big-O é uma notação matemática que descreve **como algo cresce** quando o tamanho da entrada (n) aumenta. Não é um número exato — é a tendência.

| Notação | Significado | Exemplo real |
|---------|------------|-------------|
| O(1) | Tempo/memória constante | Acessar um elemento de array pelo índice |
| O(n) | Cresce linearmente | Somar todos os elementos de uma lista |
| O(n²) | Cresce quadraticamente | Comparar cada elemento com todos os outros (atenção padrão) |
| O(n·d) | Linear em ambas as dimensões | Multiplicar matriz n×d por vetor d |

**Por que aparece na wiki:** o problema central da atenção é O(n²) em memória — se a sequência dobra, a memória quadruplica. Flash Attention reduz isso para O(n). Cada fator de redução é um avanço.

---

## d_model, d_k, d_v, d_ff — A Notação de Dimensões do Transformer

### O que é

No paper do Transformer, todas as dimensões seguem a convenção `d_algumaCoisa`. O `d` vem de **dimension** (dimensão em inglês). Cada `d_` descreve o tamanho de um vetor ou matriz diferente na arquitetura:

| Notação | Significado | Valor (Base) | Onde aparece |
|---------|------------|-------------|-------------|
| **d_model** | Dimension of the **model** — tamanho do vetor de embedding e de todas as camadas ocultas | **512** | Em tudo — embeddings, atenção, FFN, saída |
| **d_k** | Dimension of the **Key** — tamanho do vetor de cada Key por cabeça | **64** (512/8) | Na self-attention: Q·Kᵀ |
| **d_v** | Dimension of the **Value** — tamanho do vetor de cada Value por cabeça | **64** (512/8) | Na self-attention: pesos · V |
| **d_ff** | Dimension of the **Feed-Forward** — tamanho da camada oculta da FFN | **2048** (512×4) | Na Feed-Forward Network |
| **h** | Número de **heads** (cabeças) | **8** | Multi-Head Attention |
| **N** | Número de **camadas** empilhadas | **6** | Encoder e Decoder |

### Intuição

Imagine uma fábrica com uma esteira de 512 cm de largura (d_model = 512). Tudo que passa pela fábrica — embeddings, atenção, FFN — tem exatamente 512 cm de largura. Quando a atenção divide o trabalho em 8 cabeças, cada cabeça recebe uma faixa de 64 cm (d_k = d_v = 512/8 = 64). A FFN expande para 2048 cm temporariamente (d_ff = 512×4) e depois contrai de volta para 512.

**Por que 512?** É um número平衡ado (balanceado): grande o suficiente para capturar significado rico, pequeno o suficiente para caber em GPUs de 2017. Modelos modernos usam valores maiores: GPT-3 usa 12288, LLaMA-7B usa 4096.

**A relação fundamental:** `d_k = d_v = d_model / h`. As dimensões não são números mágicos independentes — todas derivam de d_model e h.

### Por que aparece em TODO lugar

TODA equação do Transformer usa essas notações. Quando você vê `Q ∈ ℝ^(n × d_k)`, significa "Q é uma matriz com n linhas (tokens) e d_k colunas (64 dimensões por Key)". Quando vê `W_Q ∈ ℝ^(d_model × d_k)`, significa "W_Q projeta de 512 dimensões para 64 dimensões". Sem entender essa notação, as equações são ilegíveis.

---

## FP16, BF16, FP32, FP8 — Formatos de Ponto Flutuante

### O que são

São formatos que definem **quantos bits** são usados para representar cada número e como esses bits são distribuídos entre precisão e alcance.

| Formato | Bits | Expoente | Mantissa | Alcance | Precisão |
|---------|------|----------|----------|---------|----------|
| FP32 | 32 | 8 | 23 | ±3.4×10³⁸ | Alta (~7 dígitos) |
| FP16 | 16 | 5 | 10 | ±65,504 | Média (~3 dígitos) |
| BF16 | 16 | 8 | 7 | Igual FP32 | Baixa (~2 dígitos) |
| FP8 (E4M3) | 8 | 4 | 3 | ±448 | Muito baixa (~1 dígito) |

**Intuição (balança de precisão):** imagine medir distâncias com uma régua:
- FP32: régua de 1km com marcação a cada milímetro (muito alcance, muita precisão)
- FP16: régua de 65m com marcação a cada centímetro (alcance menor)
- BF16: régua de 1km com marcação a cada 3cm (mesmo alcance do FP32, menos precisão)
- FP8: régua de 50cm com marcação a cada 3cm (pouco alcance, pouca precisão)

### Por que aparece

- **Mixed Precision (FP16 + FP32):** treina com FP16 (rápido, econômico) mas mantém cópias críticas em FP32 (estável)
- **Flash Attention 3:** usa FP8 no forward para 2× mais velocidade na H100
- **BF16:** preferido em modelos modernos (GPT, LLaMA) porque tem o mesmo alcance do FP32, evitando overflow em gradientes

---

## Gradiente e Backpropagation

### O que é

**Gradiente** é um vetor que aponta a direção de subida mais íngreme de uma função. Em IA, usamos o gradiente da **função de perda** (erro) para saber como ajustar cada peso do modelo.

**Analogia:** você está no topo de uma montanha (erro alto) e quer descer ao vale (erro baixo), mas há neblina (não vê o vale). O gradiente é como tatear o chão com o pé para saber qual direção desce mais. Cada passo na direção oposta ao gradiente te leva mais para baixo.

**Backpropagation** (retropropagação) é o algoritmo que calcula o gradiente para TODOS os pesos do modelo de uma vez, aplicando a regra da cadeia (cálculo) da última camada até a primeira. É o que torna o treinamento de redes profundas possível.

### Por que aparece

Toda técnica de otimização (Flash Attention, Gradient Checkpointing, Mixed Precision) gira em torno de tornar o cálculo de gradientes mais rápido ou mais eficiente em memória durante o backpropagation.

---

## Cross-Entropy (Entropia Cruzada)

### O que é

Função de perda (loss) que mede a diferença entre duas distribuições de probabilidade: a previsão do modelo e o valor real (ground truth). É a função de perda padrão para classificação e modelos de linguagem.

**Intuição:** se o modelo prevê 90% de chance para a palavra correta, a cross-entropy é baixa (≈0.1). Se prevê só 10% de chance, é alta (≈2.3). O treino minimiza essa função para fazer o modelo ficar cada vez mais "certo" nas previsões.

$$\mathcal{L} = -\sum_{i} y_i \log(p_i)$$

Onde yᵢ é a distribuição real (1 para a classe correta, 0 para as outras) e pᵢ é a previsão do modelo.

---

## Perplexity (Perplexidade)

### O que é

Métrica que mede o quão "surpreso" o modelo fica ao ver um texto. É o exponencial da cross-entropy: `perplexity = e^(cross_entropy)`.

**Intuição:** perplexidade 10 significa que, em média, o modelo hesita entre 10 palavras possíveis a cada posição. É como se o modelo estivesse "em dúvida" entre 10 opções. Quanto menor, melhor — perplexidade 1 significa certeza absoluta (nunca acontece na prática com linguagem real).

---

## GEMM (General Matrix Multiply)

### O que é

GEMM é a operação de multiplicação de matrizes que as GPUs executam em hardware dedicado (Tensor Cores). É a operação mais importante em deep learning — atenção, FFN, projeções, tudo vira GEMM.

**Intuição:** é o "motor" da GPU. As GPUs são tão boas em multiplicar matrizes que o gargalo raramente é a conta em si — é mover dados para alimentar o motor (daí Flash Attention).

**Por que importa:** na A100, GEMM atinge 312 TFLOPS (trilhões de operações por segundo), enquanto operações não-matmul (exponencial, divisão) atingem apenas 19.5 TFLOPS — **16× mais lentas**. Otimizar atenção é reduzir não-GEMMs.

---

## Auto-regressivo

### O que é

Modelo auto-regressivo é aquele que gera a saída **um token por vez**, e cada token gerado é adicionado à entrada para gerar o próximo. O GPT é auto-regressivo: ele gera a palavra 1, depois usa a palavra 1 para gerar a 2, depois usa 1 e 2 para gerar a 3, etc.

**Intuição:** é como escrever um texto: cada palavra que você escreve depende de todas as palavras que já escreveu antes. Você não decide a palavra 50 antes de ter escrito as 49 anteriores.

---

## Beam Search

### O que é

Algoritmo de busca usado na **inferência** (geração de texto) que mantém as k melhores continuações parciais em vez de escolher sempre a melhor próxima palavra (greedy). k = beam size.

**Intuição (beam=4):** em vez de sempre escolher A MAIS provável próxima palavra (que pode levar a um beco sem saída), o modelo mantém 4 "hipóteses" simultâneas e no final escolhe a frase completa mais provável entre as 4.

**Exemplo:** "I ate" → beam=4 mantém: "I ate pizza", "I ate the", "I ate a", "I ate some". No próximo passo expande cada uma e mantém as 4 melhores continuações gerais.

---

## Dropout

### O que é

Técnica de regularização que, durante o treino, **zera aleatoriamente** uma fração p dos neurônios (p=0.1 → 10% desligados). Isso força a rede a não depender excessivamente de neurônios específicos, melhorando a generalização.

**Intuição:** é como estudar para uma prova em grupo, mas a cada dia de estudo alguns colegas faltam. Você aprende a não depender de ninguém específico para entender a matéria. Na prova (inferência), todo mundo comparece e o resultado é melhor.

**No Transformer:** dropout é aplicado após cada sub-camada (atenção e FFN), nos embeddings, e nos pesos de atenção. p=0.1 no paper original.

---

## ReLU (Rectified Linear Unit)

### O que é

Função de ativação simples: `ReLU(x) = max(0, x)`. Se x é negativo, vira 0; se é positivo, passa direto.

**Intuição:** um interruptor: ou o neurônio "dispara" (valor positivo passa) ou fica "desligado" (negativo vira zero). É simples mas funciona — a não-linearidade que permite redes profundas aprenderem funções complexas.

**No Transformer:** usada na Feed-Forward Network: `FFN(x) = ReLU(xW₁ + b₁)W₂ + b₂`. Modelos modernos (GPT, LLaMA) substituíram por funções como GELU ou SwiGLU.

---

## Ensemble

### O que é

Técnica de combinar múltiplos modelos treinados independentemente para melhorar a qualidade final. As previsões dos modelos são combinadas (média ou votação).

**Intuição:** pergunte a 5 especialistas e combine as respostas — o resultado é melhor que perguntar a um só, porque os erros de cada um tendem a se cancelar.

**No Transformer:** o paper de 2017 mostra que o Transformer BASE já superava ensembles de RNNs. Ou seja, um Transformer sozinho batia múltiplos modelos RNN combinados. O Transformer BIG com ensemble batia todo mundo.

---

## Label Smoothing

### O que é

Técnica de regularização que "suaviza" os rótulos de treinamento. Em vez de dizer "a resposta correta tem 100% de probabilidade e as erradas 0%", você distribui um pouco de probabilidade para as classes erradas.

Com ε = 0.1 (valor usado no Transformer):
- Classe correta: 1 - ε = 0.9 (90%)
- Classes erradas: ε / (|V|-1) cada (um pouquinho para cada)

**Intuição:** um professor que aceita que às vezes há mais de uma resposta razoável, em vez de dizer "só essa resposta é a certa e o resto é tudo errado". Isso evita overfitting (decorar em vez de aprender).

**Curiosidade:** label smoothing piora a perplexidade (modelo fica menos "certo" de tudo) mas melhora BLEU (traduções ficam melhores). Isso mostra que certeza absoluta nem sempre é melhor.
