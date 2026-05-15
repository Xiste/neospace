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
