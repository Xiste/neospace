# Pipeline Completa do Transformer — Exemplo Passo a Passo

## O Problema

Você já estudou cada peça separadamente — tokenização, embeddings, self-attention, multi-head, encoder, decoder. Mas como elas se **conectam** de verdade? O que exatamente entra e sai de cada etapa?

Este guia percorre a pipeline INTEIRA com um exemplo concreto: traduzir **"The cat sat"** → **"O gato sentou"**.

---

## Visão Geral da Pipeline

```
"The cat sat"
     │
     ▼
┌──────────────────────────────────────────────────────────────┐
│ 1. TOKENIZAÇÃO                                                │
│    "The cat sat" → ["The", "cat", "sat"] → IDs: [2456, 5432, 8871]  │
└──────────────────────────┬───────────────────────────────────┘
                           ▼
┌──────────────────────────────────────────────────────────────┐
│ 2. EMBEDDINGS + POSITIONAL ENCODING                           │
│    Cada ID vira vetor de 512 dims + seno/cosseno da posição   │
│    Entrada do encoder: matriz 3×512                            │
└──────────────────────────┬───────────────────────────────────┘
                           ▼
┌──────────────────────────────────────────────────────────────┐
│ 3. ENCODER (6 camadas, cada uma com Self-Attention + FFN)     │
│    "sat" descobre que seu sujeito é "The cat"                  │
│    Saída Z_enc: matriz 3×512 (contextualizada)                 │
└──────────────────────────┬───────────────────────────────────┘
                           ▼
┌──────────────────────────────────────────────────────────────┐
│ 4. DECODER — LOOP AUTO-REGRESSIVO                             │
│    Itera até gerar o token de fim "</s>"                       │
│    ┌──────────────────────────────────────────────────────┐   │
│    │ Iteração 1: ["<s>"]           → gera "O"             │   │
│    │ Iteração 2: ["<s>", "O"]      → gera "gato"          │   │
│    │ Iteração 3: ["<s>","O","gato"]→ gera "sentou"        │   │
│    │ Iteração 4: ["<s>","O","gato","sentou"] → "</s>"     │   │
│    └──────────────────────────────────────────────────────┘   │
└──────────────────────────┬───────────────────────────────────┘
                           ▼
                      "O gato sentou"
```

---

## Passo 1: Tokenização

A frase entra como texto bruto. O [tokenizador BPE](TOKENIZACAO.md) quebra em tokens conhecidos:

```
"The cat sat"
     │ BPE tokenizer (vocabulário de 37K tokens)
     ▼
Tokens:  ["The", "cat", "sat"]
IDs:     [ 2456,  5432,  8871 ]
```

**3 tokens** → sequência de comprimento `n_enc = 3`.

A tokenização é pré-processamento em CPU. O que entra na GPU são os IDs inteiros.

---

## Passo 2: Embeddings + Positional Encoding

### 2a. Embedding Lookup

Cada ID busca sua linha na **matriz de embedding E** (37.000 linhas × 512 colunas):

```
ID 2456 ("The")  →  E[2456]  →  vetor de 512 floats (ex: [0.03, -0.12, 0.41, ...])
ID 5432 ("cat")  →  E[5432]  →  vetor de 512 floats
ID 8871 ("sat")  →  E[8871]  →  vetor de 512 floats
```

### 2b. Escala por √d_model

Cada vetor é multiplicado por √512 ≈ 22.6 para equilibrar a magnitude com o positional encoding (±1):

```
Embedding_final = Embedding × √512
```

### 2c. Somar Positional Encoding

Para cada posição `pos` (0, 1, 2), calcula-se PE(pos) — vetor de 512 dimensões com senos e cossenos — e SOMA ao embedding:

```
X_enc[0] = Embedding("The") × √512 + PE(0)   ← posição 0: dimensões baixas oscilam rápido
X_enc[1] = Embedding("cat") × √512 + PE(1)   ← posição 1
X_enc[2] = Embedding("sat") × √512 + PE(2)   ← posição 2
```

**Resultado:** matriz `X_enc` de tamanho **3 × 512** que entra no encoder.

---

## Passo 3: Encoder (6 Camadas)

Cada uma das 6 camadas do encoder faz a mesma coisa (com pesos diferentes):

### Para cada camada:

**Sub-camada 1 — Multi-Head Self-Attention (8 cabeças, SEM máscara):**

```
Q = X_enc · W_Q    (3×512 → 3×64 por cabeça)
K = X_enc · W_K    (3×512 → 3×64 por cabeça)
V = X_enc · W_V    (3×512 → 3×64 por cabeça)

Scores = Q · Kᵀ / √64     → matriz 3×3
Pesos  = softmax(Scores)  → matriz 3×3
Saída  = Pesos · V         → 3×64 por cabeça

Concatena 8 cabeças (3×512) · W_O → 3×512
+ residual (soma X_enc original) + LayerNorm
```

**O que acontece nos pesos de atenção:**
```
         The   cat   sat
The    [0.4   0.3   0.3 ]  ← "The" presta 40% de atenção em si mesma
cat    [0.2   0.5   0.3 ]  ← "cat" presta 50% de atenção em si mesma
sat    [0.5   0.4   0.1 ]  ← "sat" presta 50% de atenção em "The" (sujeito!)
```

"sat" descobre que "The" é seu sujeito — isso é a mágica da atenção.

**Sub-camada 2 — Feed-Forward Network:**

```
FFN(x) = ReLU(x · W₁ + b₁) · W₂ + b₂
         (3×512)·(512×2048) → (3×2048) → ReLU → (3×2048)·(2048×512) → 3×512
+ residual + LayerNorm
```

Cada token é processado independentemente (a FFN não mistura tokens).

**Após 6 camadas:** a saída `Z_enc` é uma matriz **3 × 512** onde cada token está profundamente contextualizado. "sat" sabe que é verbo, que seu sujeito é "The cat", e que está na terceira posição.

**Z_enc é CONSTANTE** — o encoder termina aqui. Essa matriz será usada pelo decoder em TODAS as iterações.

---

## Passo 4: Decoder — Loop Auto-Regressivo

O decoder gera a saída **um token por vez**, e cada token gerado é adicionado à entrada da PRÓXIMA iteração. É isso que se chama **auto-regressivo**.

O decoder começa com um token especial de início: `<s>` (start).

### Estrutura de cada iteração do decoder:

```
Entrada do decoder (m tokens)
     │
     ▼
┌────────────────────────────────────────────┐
│ 1. Embedding + PE (para cada token)         │
├────────────────────────────────────────────┤
│ 2. Masked Multi-Head Self-Attention         │
│    → Cada token só vê tokens passados       │
│    + residual + LayerNorm                   │
├────────────────────────────────────────────┤
│ 3. Multi-Head Cross-Attention               │
│    Q = decoder, K,V = Z_enc (FIXO!)         │
│    → Consulta a frase original              │
│    + residual + LayerNorm                   │
├────────────────────────────────────────────┤
│ 4. FFN (processamento individual)           │
│    + residual + LayerNorm                   │
├────────────────────────────────────────────┤
│ 5. Projeção Linear → Softmax → Próximo token│
│    Linear: 512 → |V| (37K)                  │
│    Softmax: scores → probabilidades          │
│    Escolhe o token mais provável             │
└────────────────────┬───────────────────────┘
                     │
                     ▼
              Próximo token gerado
              (adicionado à entrada da próxima iteração)
```

---

### Iteração 1: Gerando "O"

**Entrada do decoder:** `["<s>"]` — apenas 1 token (ID=1)

```
1. Embedding + PE(0) para "<s>" → vetor 1×512

2. Masked Self-Attention:
   Só tem 1 token, não há o que mascarar
   "<s>" atende a "<s>" (100% em si mesmo)

3. Cross-Attention (a mágica acontece aqui):
   Q("<s>") pergunta: "qual parte da entrada é relevante para começar?"
   K_enc responde: "The" é o mais relevante (é o sujeito da frase)
   V_enc entrega o significado de "The"

   Pesos da cross-attention para "<s>":
   The: 0.60   cat: 0.25   sat: 0.15

4. FFN processa

5. Projeção + Softmax sobre 37K tokens:
   "O"    → 0.23  ← MAIOR probabilidade
   "A"    → 0.08
   "Um"   → 0.05
   ... outros 36.997 tokens ...
```

**Token gerado:** `"O"` (ID=87)

**Auto-alimentação:** `"O"` é concatenado à entrada. Agora o decoder tem `["<s>", "O"]`.

---

### Iteração 2: Gerando "gato"

**Entrada do decoder:** `["<s>", "O"]` — 2 tokens

```
1. Embedding + PE(0) para "<s>", PE(1) para "O" → matriz 2×512

2. Masked Self-Attention:
   "<s>" vê ["<s>"]           ← só passado
   "O"  vê ["<s>", "O"]      ← passado + ele mesmo

   Pesos (mascarados):
          <s>    O
   <s>  [1.0   0.0 ]
   O    [0.3   0.7 ]

3. Cross-Attention:
   Q("O") pergunta: "qual parte da entrada eu represento?"
   K_enc responde: "The"!

   Pesos da cross-attention para "O":
   The: 0.65   cat: 0.20   sat: 0.15

4. FFN processa

5. Projeção + Softmax sobre 37K tokens:
   "gato"  → 0.31  ← MAIOR
   "cão"   → 0.12
   "The"   → 0.07  (não faz sentido gerar inglês aqui)
   ...
```

**Token gerado:** `"gato"` (ID=543)

**Auto-alimentação:** entrada agora é `["<s>", "O", "gato"]`.

---

### Iteração 3: Gerando "sentou"

**Entrada do decoder:** `["<s>", "O", "gato"]` — 3 tokens

```
1. Embedding + PE para os 3 tokens → matriz 3×512

2. Masked Self-Attention:
          <s>    O    gato
   <s>  [1.0   0.0   0.0 ]
   O    [0.2   0.8   0.0 ]
   gato [0.1   0.3   0.6 ]

   "gato" presta atenção em "O" e "<s>" — sabe que "O gato" é o sujeito

3. Cross-Attention:
   Q("gato") pergunta: "qual parte da entrada eu represento?"
   K_enc responde: "cat"!

   Pesos da cross-attention para "gato":
   The: 0.15   cat: 0.70   sat: 0.15

4. FFN processa

5. Projeção + Softmax:
   "sentou" → 0.27  ← MAIOR
   "está"   → 0.10
   "comeu"  → 0.05
   ...
```

**Token gerado:** `"sentou"` (ID=1234)

**Auto-alimentação:** entrada agora é `["<s>", "O", "gato", "sentou"]`.

---

### Iteração 4: Gerando o Fim

**Entrada do decoder:** `["<s>", "O", "gato", "sentou"]` — 4 tokens

```
3. Cross-Attention:
   Q("sentou") pergunta: "qual parte da entrada?"
   K_enc responde: "sat"!

   Pesos:
   The: 0.30   cat: 0.25   sat: 0.45

5. Projeção + Softmax:
   "</s>"   → 0.42  ← MAIOR (token de fim)
   "no"     → 0.08
   "sobre"  → 0.05
   ...
```

**Token gerado:** `"</s>"` (token de fim)

→ **Geração concluída.**

---

## Resumo Visual do Loop Auto-Regressivo

```
TEMPO →
                    Z_enc (FIXO, calculado 1 vez pelo encoder)
                    ┌──────────┬──────────┬──────────┐
                    │ "The"    │ "cat"    │ "sat"    │
                    │ (512dim) │ (512dim) │ (512dim) │
                    └────┬─────┴────┬─────┴────┬─────┘
                         │          │          │
                    K,V  │     K,V  │     K,V  │
                         │          │          │
Iteração 1:  <s> ──Q───The─────────cat────────sat──→ "O"
                         │          │          │
Iteração 2:  <s> ──Q───The─────────cat────────sat──→ "gato"
             O   ──Q───The─────────cat────────sat──
                         │          │          │
Iteração 3:  <s> ──Q───The─────────cat────────sat──→ "sentou"
             O   ──Q───The─────────cat────────sat──
             gato──Q───The─────────cat────────sat──
                         │          │          │
Iteração 4:  <s> ──Q───The─────────cat────────sat──→ "</s>"
             O   ──Q───The─────────cat────────sat──   (fim)
             gato──Q───The─────────cat────────sat──
             sentou─Q──The─────────cat────────sat──
```

**Observe:**
- **Z_enc NUNCA muda** — o encoder rodou 1 vez e acabou
- **Q muda a cada token** — cada palavra gerada faz uma pergunta diferente ao encoder
- **A entrada do decoder CRESCE** a cada iteração — tokens gerados alimentam a próxima iteração (auto-regressivo)
- **A masked self-attention garante** que "gato" não veja "sentou" (futuro) durante o treino

---

## O Que Acontece com os Pesos de Atenção?

A beleza do Transformer está nos **padrões que emergem** naturalmente do treino:

| Token gerado | Atende mais a... | Tipo de relação |
|---|---|---|
| "O" | "The" | Tradução direta (artigo) |
| "gato" | "cat" | Tradução direta (substantivo) |
| "sentou" | "sat" + "The cat" | Tradução + concordância (sujeito) |

A cross-attention aprendeu **alinhamento bilíngue** sem supervisão explícita. Ninguém disse ao modelo "The = O" ou "cat = gato" — ele descobriu sozinho.

---

## Pré-requisitos

TODOS os fundamentos do Transformer (ordem de estudo):
1. [Tokenização](TOKENIZACAO.md)
2. [Embeddings](EMBEDDINGS.md)
3. [Self-Attention](SELF_ATTENTION.md)
4. [Multi-Head Attention](MULTI_HEAD_ATTENTION.md)
5. [Encoder & Decoder — Os 3 Tipos de Atenção](ATENCAO_CODIFICADOR_DECODIFICADOR.md)
6. [Positional Encoding](POSITIONAL_ENCODING.md)
7. [FFN + Residual + LayerNorm](FEED_FORWARD_RESIDUAL_LAYERNORM.md)
8. [Transformer Completo](TRANSFORMER_COMPLETO.md)

---

## Conexões

- **Inferência vs Treino:** no treino, o decoder recebe a saída completa de uma vez (teacher forcing) + máscara. Na inferência, é auto-regressivo como este exemplo
- **KV-cache:** na prática, K e V do decoder são armazenados para não recalcular a cada iteração
- **Beam Search:** em vez de escolher o token mais provável, mantém N hipóteses paralelas

---

## Perguntas de Revisão

1. Quantas vezes o encoder roda para traduzir uma frase? E o decoder?
2. Por que a entrada do decoder cresce a cada iteração? O que acontece com K e V do decoder?
3. Na iteração 3, o que "gato" vê na masked self-attention? E na cross-attention?
4. O que mudaria se a masked self-attention fosse removida do decoder durante a inferência?
5. Se a frase de entrada fosse "The black cat sat quietly", o que mudaria no Z_enc? E no loop do decoder?
