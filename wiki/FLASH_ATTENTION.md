# ⚡ Flash Attention

Status: 🟢 Dominado
Última revisão: 2026-05-15

---

## O Problema

Dados Q, K, V ∈ ℝ^(N×d), a atenção padrão calcula:

$$S = QK^T \in \mathbb{R}^{N \times N}, \quad P = \text{softmax}(S), \quad O = PV \in \mathbb{R}^{N \times d}$$

Cada passo escreve e lê da [HBM](GLOSSARIO.md#hbm-vs-sram):
1. S = Q @ K^T → escreve S (N²) na HBM
2. P = softmax(S) → lê S (N²), escreve P (N²)
3. O = P @ V → lê P (N²), escreve O (Nd)

**Total de acessos HBM: Θ(Nd + N²)**. Para N=1024, d=64, 16 cabeças, batch=64: **40.3 GB** de leitura/escrita. É 9x mais que o necessário. O gargalo não é computação — é tráfego de memória.

## Contexto Histórico

Antes do Flash Attention (2022), a comunidade tentava resolver o problema O(N²) com aproximações: [sparse attention](GLOSSARIO.md#sparse-attention) (Sparse Transformer, Longformer, BigBird), [Linformer](GLOSSARIO.md#linformer) (projeção low-rank), [Performer](GLOSSARIO.md#performer) (FAVOR+ com kernels). Todas perdiam qualidade — nenhuma era exata.

**O insight de Rabe & Staats (2021):** "Self-Attention Does Not Need O(n²) Memory" — mostraram que dava para calcular atenção exata sem materializar a matriz N×N, usando tiling. Mas a implementação deles não era IO-aware e não competia em velocidade. O Flash Attention pegou essa ideia e a implementou de forma eficiente em CUDA.

## Intuição Central

A GPU tem dois níveis de memória com velocidades radicalmente diferentes:

| Memória | Capacidade (A100) | Banda | Latência |
|---|---|---|---|
| HBM (VRAM) | 40-80 GB | 1.5-2.0 TB/s | ~300 ciclos |
| SRAM (shared memory) | 192 KB / SM | ~19 TB/s estimado | ~20-30 ciclos |

O algoritmo ingênuo trata a GPU como se só existisse HBM. O Flash Attention usa a SRAM como cache explícito: divide as matrizes em [tiles](GLOSSARIO.md#tiles-ladrilhamento) que cabem na SRAM, processa cada tile completamente e nunca escreve a matriz N×N na HBM.

---

## Paper 1: Flash Attention (Dao et al., NeurIPS 2022 Best Paper)

### Algoritmo 1 — Forward Pass

**Tamanhos de bloco** (derivados do tamanho M da SRAM):

$$B_c = \lceil \frac{M}{4d} \rceil \quad \text{(blocos de K, V)} \quad \quad B_r = \min\left(\lceil \frac{M}{4d} \rceil, d\right) \quad \text{(blocos de Q, O)}$$

$$T_r = \lceil \frac{N}{B_r} \rceil \quad \text{(número de blocos de linha)} \quad \quad T_c = \lceil \frac{N}{B_c} \rceil \quad \text{(número de blocos de coluna)}$$

**Pseudocódigo completo do forward:**

```
Entrada: Q, K, V ∈ ℝ^(N×d) na HBM, blocos B_c, B_r

1. Divide Q em T_r blocos Q_1...Q_{T_r} de tamanho B_r×d
2. Divide K, V em T_c blocos K_1...K_{T_c}, V_1...V_{T_c} de tamanho B_c×d
3. Inicializa O = (0)_{N×d}, ℓ = (0)_N, m = (-∞)_N na HBM
4. Inicializa estado de RNG R para dropout

5. para j = 1 até T_c:                       // outer loop: K,V blocks
6.     carrega K_j, V_j da HBM → SRAM
7.     para i = 1 até T_r:                   // inner loop: Q blocks
8.         carrega Q_i, O_i, ℓ_i, m_i da HBM → SRAM
9.
10.        // Passo 1: scores de atenção para este par de blocos
11.        S_ij = τ · Q_i @ K_j^T            // [B_r × B_c]
12.        S_ij = MASK(S_ij)                  // causal/padding mask
13.
14.        // Passo 2: estatísticas online do softmax
15.        m̃_ij = rowmax(S_ij)                // [B_r,]
16.        P̃_ij = exp(S_ij - m̃_ij)             // [B_r × B_c] pointwise
17.        ℓ̃_ij = rowsum(P̃_ij)                // [B_r,]
18.
19.        // Passo 3: atualiza estatísticas correntes
20.        m_novo = max(m_i, m̃_ij)            // [B_r,]
21.        ℓ_novo = e^(m_i - m_novo)·ℓ_i + e^(m̃_ij - m_novo)·ℓ̃_ij
22.
23.        // Passo 4: dropout (se aplicável)
24.        P̃_ij = dropout(P̃_ij, p_drop)
25.
26.        // Passo 5: acumula output com correção de escala
27.        O_i = diag(ℓ_novo)⁻¹ · (diag(ℓ_i)·e^(m_i - m_novo)·O_i
28.                                  + e^(m̃_ij - m_novo)·P̃_ij @ V_j)
29.
30.        escreve O_i, ℓ_i←ℓ_novo, m_i←m_novo para HBM
31.     fim para
32. fim para
33. retorna O, ℓ, m, R
```

**O truque do online softmax:** a correção `e^(m_velho - m_novo)` rescala todos os exponenciais acumulados para a base do novo máximo, mantendo estabilidade numérica. Sem isso, quando um novo máximo maior aparece, os exponenciais antigos viram zero (underflow) ou os novos estouram (overflow).

### Backward Pass — Recomputation

O backward **não salva** as matrizes S (N×N) e P (N×N). Em vez disso, **recalcula** cada bloco S_ij e P_ij na SRAM usando Q_i, K_j e as estatísticas salvas (ℓ_i, m_i).

**O que é armazenado do forward (O(N)):**
- O (output, N×d)
- ℓ (soma dos exponenciais, N)
- m (máximo por linha, N)
- R (estado do gerador de números aleatórios, O(1))

**O que NÃO é armazenado (economia):**
- S (N×N, ~134 MB para N=4096)
- P (N×N, ~134 MB para N=4096)

**Equações do backward (derivadas analiticamente):**

Dado o gradiente de saída dO:

1. Gradiente de V:
$$dV_j = \sum_i P_{ij}^T \cdot dO_i$$

2. Estatística intermediária D_i (produto escalar por linha, evita reduzir sobre N):
$$D_i = \text{rowsum}(dO_i \circ O_i) \in \mathbb{R}^{B_r}$$

3. Gradiente via Jacobiano do softmax:
$$dS_{ij} = P_{ij} \circ (dP_{ij} - D_i)$$

4. Gradientes de Q e K:
$$dQ_i \mathrel{+}= \tau \cdot dS_{ij} @ K_j$$
$$dK_j \mathrel{+}= \tau \cdot dS_{ij}^T @ Q_i$$

**Pseudocódigo do backward:**

```
Entrada: Q, K, V, O, dO ∈ ℝ^(N×d), L ∈ ℝ^N na HBM, blocos B_c, B_r

1. Computa D = rowsum(dO ∘ O) ∈ ℝ^N   // pointwise multiply, depois rowsum
2. Divide Q em T_r blocos, K, V em T_c blocos
3. Inicializa dQ = (0)_{N×d}, dK = (0)_{N×d}, dV = (0)_{N×d}

4. para j = 1 até T_c:
5.     carrega K_j, V_j → SRAM
6.     dK_j = (0)_{B_c×d}, dV_j = (0)_{B_c×d}
7.     para i = 1 até T_r:
8.         carrega Q_i, O_i, dO_i, dQ_i, L_i, D_i → SRAM
9.
10.        // Recomputa matriz de atenção (NÃO salva do forward!)
11.        S_ij = τ · Q_i @ K_j^T
12.        P_ij = exp(S_ij - L_i)           // usando L = m + log(ℓ)
13.
14.        // dV
15.        dV_j += P_ij^T @ dO_i
16.
17.        // dP e dS via softmax Jacobiano
18.        dP_ij = dO_i @ V_j^T
19.        dS_ij = P_ij ∘ (dP_ij - D_i)
20.
21.        // dQ (acumula, escreve de volta pra HBM)
22.        carrega dQ_i da HBM → SRAM
23.        dQ_i += τ · dS_ij @ K_j
24.        escreve dQ_i → HBM
25.
26.        // dK
27.        dK_j += τ · dS_ij^T @ Q_i
28.     fim para
29.     escreve dK_j, dV_j → HBM
30. fim para
31. retorna dQ, dK, dV
```

### Complexidade de IO

| Métrica | Atenção Padrão | Flash Attention |
|---|---|---|
| Acessos HBM | Θ(Nd + N²) | Θ(N²d²/M) |
| Memória extra | O(N²) | O(N) |
| Exemplo real (GPT-2, N=1024, d=64, A100) | | |
| GFLOPs | 66.6 | 75.2 (+13%) |
| HBM R/W (GB) | 40.3 | 4.4 (-89%) |
| Runtime (ms) | 41.7 | 7.3 (**5.7× mais rápido**) |

**Prova de otimalidade (Proposition 3):** Nenhum algoritmo de atenção exata pode ter menos de Ω(N²d²/M) acessos HBM para todo M ∈ [d, Nd]. O Flash Attention atinge esse limite inferior — é assintoticamente ótimo.

### Block-Sparse Flash Attention

Para uma máscara de esparsidade M ∈ {0,1}^(T_r × T_c), o algoritmo simplesmente pula blocos onde M_ij = 0:

Complexidade: Θ(Nd + N²d²s/M) onde s = fração de blocos não-zero.

---

## Paper 2: Flash Attention 2 (Dao, 2023)

### Por que o FA1 era ineficiente

O FA1 atingia apenas **25-40% do pico teórico de FLOPs/s** na A100. Motivos:
1. **Particionamento de trabalho subótimo entre warps:** usava "split-K" — K e V divididos entre warps, Q compartilhado. Isso forçava sincronização e leituras/escritas extras na shared memory.
2. **FLOPs não-matmul excessivos:** cada FLOP não-matmul é **16× mais caro** que um FLOP matmul na A100 (312 TFLOPS matmul vs 19.5 TFLOPS não-matmul FP32).
3. **Baixa ocupação com sequências longas:** só paralelizava sobre batch × cabeças, não sobre sequence length.

### Três melhorias do FA2

#### 1. Redução de FLOPs não-matmul

**Tweak A — Output sem rescala dupla:** Em vez de rescalar ambos os termos por `diag(ℓ_novo)⁻¹`, mantém uma versão "não-escalada" e só normaliza no final:

$$\tilde{O}^{(j)} = \text{diag}(e^{m^{(j-1)} - m^{(j)}}) \tilde{O}^{(j-1)} + e^{S^{(j)} - m^{(j)}} V^{(j)}$$

Apenas no fim: $$O = \text{diag}(\ell^{(\text{last})})^{-1} \tilde{O}^{(\text{last})}$$

**Tweak B — Usar logsumexp L em vez de m e ℓ separados:** Armazena apenas $$L = m + \log(\ell)$$, reduzindo operações e armazenamento no backward.

#### 2. Paralelismo sobre sequence length

**Forward:** o loop externo (sobre blocos de linha de Q) é embaraçosamente paralelo — cada thread block processa um bloco de linhas, sem comunicação.

**Backward:** paraleliza sobre blocos de coluna (K,V). A única dependência entre colunas é na atualização de dQ, resolvida com **atomic adds**.

Isso aumenta ocupação quando batch size é pequeno (comum em sequências longas).

#### 3. Particionamento entre warps — evita "split-K"

| | FA1 (split-K) | FA2 |
|---|---|---|
| Q | Compartilhado entre warps | **Dividido** entre warps |
| K, V | Divididos entre warps | **Compartilhado** entre warps |
| Sincronização | Warps escrevem na shared memory, sincronizam, somam | **Sem comunicação entre warps** |
| Resultado | Muitas leituras/escritas na shared memory | Cada warp produz sua fatia do output direto |

**Tamanhos de bloco:** {64, 128} × {64, 128}, ajustados manualmente por head dimension.

### Algoritmo 1 — Forward FA2 (completo)

```
Entrada: Q, K, V ∈ ℝ^(N×d) na HBM, blocos B_c, B_r

1. Divide Q em T_r = ⌈N/B_r⌉ blocos
2. Divide K, V em T_c = ⌈N/B_c⌉ blocos
3. Divide O em T_r blocos, L (logsumexp) em T_r blocos

4. para i = 1 até T_r em paralelo:           // PARALELO sobre sequence length!
5.     carrega Q_i → SRAM
6.     O_i = (0)_{B_r×d}, ℓ_i = (0)_{B_r}, m_i = (-∞)_{B_r}
7.
8.     para j = 1 até T_c:
9.         carrega K_j, V_j → SRAM
10.        S_ij = Q_i @ K_j^T                  // [B_r × B_c]
11.
12.        m_novo = max(m_i, rowmax(S_ij))
13.        P̃_ij = exp(S_ij - m_novo)            // pointwise
14.        ℓ_novo = e^(m_i - m_novo)·ℓ_i + rowsum(P̃_ij)
15.
16.        O_i = diag(e^(m_i - m_novo))⁻¹ @ O_i + P̃_ij @ V_j
17.
18.        m_i = m_novo, ℓ_i = ℓ_novo
19.     fim para
20.
21.     O_i = diag(ℓ_i)⁻¹ @ O_i                 // normalização final
22.     L_i = m_i + log(ℓ_i)                    // logsumexp
23.     escreve O_i, L_i → HBM
24. fim para
25. retorna O, L
```

**Máscara causal:** para blocos onde todos os índices de coluna > índices de linha, pula-se a computação inteira. ~1.7-1.8× de speedup adicional.

### Algoritmo 2 — Backward FA2

```
Entrada: Q, K, V, O, dO ∈ ℝ^(N×d), L ∈ ℝ^N na HBM

1. Computa D = rowsum(dO ∘ O)
2. Divide Q em T_r blocos, K, V em T_c blocos
3. Inicializa dQ = 0, dK = 0, dV = 0 na HBM

4. para j = 1 até T_c em paralelo:           // PARALELO sobre colunas!
5.     carrega K_j, V_j → SRAM
6.     dK_j = 0, dV_j = 0
7.
8.     para i = 1 até T_r:
9.         carrega Q_i, O_i, dO_i, dQ_i, L_i, D_i → SRAM
10.        S_ij = Q_i @ K_j^T
11.        P_ij = exp(S_ij - L_i)              // usando logsumexp
12.
13.        dV_j += P_ij^T @ dO_i
14.        dP_ij = dO_i @ V_j^T
15.        dS_ij = P_ij ∘ (dP_ij - D_i)
16.
17.        dQ_i += dS_ij @ K_j                 // atomic add na HBM
18.        escreve dQ_i → HBM
19.
20.        dK_j += dS_ij^T @ Q_i
21.     fim para
22.     escreve dK_j, dV_j → HBM
23. fim para
24. retorna dQ, dK, dV
```

### Resultados FA2

**Benchmark de atenção (A100 80GB SXM4):**

| Head dim | Com causal | Forward TFLOPs/s | % pico | Backward TFLOPs/s | % pico |
|---|---|---|---|---|---|
| 64 | Não | ~230 | 73% | ~195 | 63% |
| 128 | Não | ~210 | 67% | ~180 | 58% |
| 64 | Sim | ~200 | — | ~165 | — |
| 128 | Sim | ~185 | — | ~155 | — |

**Treino end-to-end (8× A100):**

| Modelo | Sem FA | FA1 | FA2 |
|---|---|---|---|
| GPT-2 1.3B, 2K ctx | 142 TF | 189 TF | 196 TF |
| GPT-2 1.3B, 8K ctx | 72 TF | 170 TF | **220 TF** |
| GPT-2 2.7B, 2K ctx | 149 TF | 189 TF | 205 TF |
| GPT-2 2.7B, 8K ctx | 80 TF | 175 TF | **225 TF (72%)** |

**Na H100 (sem instruções especiais):** até 335 TFLOPs/s. O paper já previa que com TMA e FP8 daria mais 1.5-2× — foi exatamente o que o FA3 fez.

---

## Paper 3: Flash Attention 3 (Dao et al., 2024)

### Por que FA2 era ineficiente na H100

O FA2 atingia apenas **35% de utilização** na H100 (vs 70% na A100). As razões:

1. **Instruções de matriz síncronas:** o FA2 usa `mma.sync` da Ampere, que bloqueia a warp. Na H100, `mma.sync` só atinge ~2/3 do throughput máximo dos Tensor Cores.
2. **Sem TMA:** as transferências HBM↔SRAM eram manuais (instruções load/store), consumindo registradores e sem sobreposição com computação.
3. **Sem FP8:** throughput de matmul FP8 é 2× o FP16 (1978 vs 989 TFLOPS na H100 SXM5), mas requer tratamento especial de outliers.
4. **Gargalo da exponencial:** na H100, a unidade de funções especiais (exp, log) tem apenas **3.9 TFLOPS** — 256× menos que matmul FP16. Para head dim 128, há 512× mais FLOPs matmul que exponenciais, então a exp pode consumir **50% do tempo**.

### Três inovações do FA3

#### A. WGMMA + Warp Specialization

**WGMMA (Warp Group Matrix Multiply-Accumulate):** instrução assíncrona exclusiva da Hopper. Substitui `mma.sync`. Um warp group de 4 warps dispara a multiplicação e **continua executando** enquanto ela roda nos Tensor Cores.

**Inter-Warpgroup Pingpong:** dois warpgroups (GEMM1/GEMM2 e softmax) alternam via barreiras `bar.sync`:
- Warpgroup 1 computa GEMMs (GEMM1 da iteração i + GEMM0 da iteração i+1)
- Enquanto isso, Warpgroup 2 executa softmax
- Depois trocam

A softmax acontece "na sombra" dos GEMMs do outro warpgroup.

**Intra-Warpgroup Pipelining:** dentro de um warpgroup, partes do softmax executam enquanto os GEMMs estão em voo. Usa pipeline de 2 estágios com acumuladores separados para GEMM e softmax. Tradeoff: maior pressão de registradores (manter dois acumuladores simultaneamente), mas melhor throughput.

#### B. TMA (Tensor Memory Accelerator)

Hardware dedicado para transferências entre memória global (HBM) e shared memory (SRAM). O TMA:
- Cuida de todo cálculo de índices e predicação de bounds
- Libera registradores (antes usados para endereçamento)
- É assíncrono — permite sobrepor carga de dados com computação
- Permite tiles maiores com mais eficiência

#### C. FP8 com Incoherent Processing

**Throughput FP8 na H100 SXM5:** 1978 TFLOPS (2× FP16).

**Problema dos outliers:** ativações de LLMs contêm features com magnitude muito maior que as demais. Quantizar direto para FP8 causa erro grande nessas features.

**Solução — Incoherent Processing:**
1. Multiplica Q e K por uma matriz ortogonal aleatória (transformada de Hadamard com sinais aleatórios)
2. Isso "espalha" os outliers — a energia fica distribuída uniformemente
3. Custo: O(d log d) por cabeça (vs O(d²) ingênuo), fundido "de graça" com rotary embeddings
4. **2.6× menos erro de quantização** que FP8 baseline

### Resultados FA3

**Benchmark (H100 SXM5, head dim 128, seqlen 8K):**

| Configuração | TFLOPs/s | % Pico FP16 |
|---|---|---|
| FA2 baseline | ~350 | 35% |
| FA3 FP16 | **740** | **75%** |
| FA3 FP8 | **~1200** (1.2 PF) | 61% do pico FP8 |

**Speedup vs FA2:** 1.5-2.0× em FP16.

**Custo da exponencial:** com FP8, a matmul dobra de velocidade mas a exponencial continua igual (3.9 TFLOPS). A razão matmul:exp vai de 512:1 para 1024:1 — a exp domina ainda mais o tempo. O pipelining do FA3 mitiga isso escondendo a exp atrás dos GEMMs.

---

## Comparativo Final

| Característica | FA1 (2022) | FA2 (2023) | FA3 (2024) |
|---|---|---|---|
| **Hardware alvo** | A100 | A100 | H100/B200 |
| **Throughput** | 25-40% do pico | 50-73% do pico | 75% do pico (FP16) |
| **Forward (N=8K)** | baseline | 2× sobre FA1 | 1.5-2× sobre FA2 (FP16) |
| **Instrução matmul** | mma.sync | mma.sync | **wgmma** (assíncrono) |
| **Warp partitioning** | split-K (K,V divididos) | Q dividido, K,V compartilhado | Warp groups com pingpong |
| **Paralelismo** | batch × heads | batch × heads × **seqlen** | batch × heads × seqlen |
| **Precisão** | FP16/BF16 | FP16/BF16 | FP16/BF16 + **FP8** |
| **TMA** | ❌ | ❌ | ✅ |
| **Memória extra** | O(N) | O(N) | O(N) |
| **Logsumexp** | m e ℓ separados | L = m + log(ℓ) | L = m + log(ℓ) |
| **Output scaling** | Rescala a cada iteração | Rescala só no final | Rescala só no final |
| **Causal mask** | Computa tudo | Pula ~metade dos blocos | Pula ~metade dos blocos |

---

## Impacto Prático

### Números concretos de economia de memória

| Comprimento N | Matriz N×N (FP16) | Economia FA |
|---|---|---|
| 1,024 | 2 MB | 10× |
| 2,048 | 8 MB | 10× |
| 4,096 | 34 MB | 20× |
| 8,192 | 134 MB | 20× |
| 32,768 | 2.1 GB | ~50× |
| 65,536 | 8.6 GB | ~100× |

### Treino de LLMs

- GPT-3/OPT tinham contexto de 2-4K tokens
- GPT-4 chegou a 32K (com FA)
- Claude foi a 100K
- Llama 3 variantes chegaram a 1M tokens

### FA4

O repositório já tem `flash-attn-4` (beta), reescrito em **CuTeDSL**, otimizado para Hopper (H100) e Blackwell (B200).

---

## Pré-requisitos

- Estrutura do Transformer (atenção multi-cabeça, Q/K/V)
- Hierarquia de memória da GPU ([HBM vs SRAM](GLOSSARIO.md#hbm-vs-sram))
- Modelo de execução CUDA (threads, [warps](GLOSSARIO.md#warp-cuda), thread blocks, grid)
- [Online softmax](GLOSSARIO.md#online-softmax) e estabilidade numérica
- Operações de álgebra linear (GEMM, matmul)
- Regra da cadeia e backpropagation (para entender o backward)

---

## Conexões

- **[Multi-Query / Grouped Query Attention](GLOSSARIO.md):** reduzem número de cabeças de K,V — FA2 suporta nativamente. Importante para inferência (reduz KV cache)
- **FSDP:** divide parâmetros entre GPUs; combinado com FA permite modelos maiores com contexto mais longo
- **LoRA:** FA acelera forward/backward do modelo base; LoRA reduz parâmetros treináveis
- **Quantização (INT8/FP8):** FA3 incorpora FP8 no forward via incoherent processing
- **[Sparse Attention](GLOSSARIO.md#sparse-attention):** FA1 já tem versão block-sparse. FA é mais usado na prática (exato > aproximado)
- **Flash Decoding:** extensão para inferência — paraleliza sobre sequence length do KV cache
- **Triton:** implementação alternativa do FA (Phil Tillet, OpenAI). FA2 incorporou ideias do Triton (loop order, paralelismo sobre seqlen)

---

## Papers Fundamentais

- **[FlashAttention (2022)](https://arxiv.org/abs/2205.14135)** — Dao, Fu, Ermon, Rudra, Ré. NeurIPS 2022 **Best Paper**. Tiling + recomputation + online softmax.
- **[FlashAttention-2 (2023)](https://arxiv.org/abs/2307.08691)** — Tri Dao. Warp partitioning otimizado, paralelismo sobre seqlen, redução de FLOPs não-matmul.
- **[FlashAttention-3 (2024)](https://arxiv.org/abs/2407.08608)** — Dao et al. WGMMA, TMA, FP8 com incoherent processing. **Este é o paper a implementar em Hopper+.** 
- **[Self-Attention Does Not Need O(n²) Memory](https://arxiv.org/abs/2112.05682)** (2021) — Rabe & Staats. Precursor: provou que tiling funciona para atenção, mas sem implementação IO-aware.

---

## Perguntas de Revisão

1. Calcule exatamente quantos bytes a atenção padrão lê/escreve da HBM para N=4096, d=128, FP16. Compare com Flash Attention (use B_c=B_r=128, M=192KB).
2. Por que `e^(m_velho - m_novo)` é necessário no online softmax e o que aconteceria sem ele?
3. Explique por que recalcular S e P no backward é mais rápido que salvá-los da HBM. Use os números de latência (~300 ciclos HBM vs ~25 ciclos SRAM).
4. Qual a diferença entre o split-K do FA1 e o particionamento do FA2? Por que evitar split-K reduz latência?
5. Por que a função exponencial é o gargalo no FA3 com FP8? (Dica: 989 TFLOPS matmul FP16 vs 3.9 TFLOPS exp na H100)
6. Como o incoherent processing (Hadamard transform) permite FP8 com baixo erro no FA3?
7. Projete: se você tem uma H100 e quer treinar com N=32768, d=128, qual versão do FA usar e por quê?

---

## Recursos Adicionais

- [Código oficial](https://github.com/Dao-AILab/flash-attention) — `flash_attn/csrc/flash_attn_3/`
- [Flash Attention 3 blog post (Tri Dao)](https://tridao.me/blog/2024/flash3/)
- [CUTLASS 3.x (NVIDIA)](https://github.com/NVIDIA/cutlass) — biblioteca usada pelo FA2/FA3
- [Guia de programação CUDA Hopper](https://docs.nvidia.com/cuda/hopper-tuning-guide/)
- [Flash-Attention 4 (CuTeDSL, beta)](https://github.com/Dao-AILab/flash-attention)
