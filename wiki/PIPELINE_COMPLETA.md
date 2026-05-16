# Pipeline Completa do Transformer — Exemplo Passo a Passo

## O Problema

Você já estudou cada peça separadamente — tokenização, embeddings, self-attention, multi-head, encoder, decoder. Mas como elas se **conectam** de verdade? O que exatamente entra e sai de cada etapa?

Este guia percorre a pipeline INTEIRA com um exemplo concreto em português: processar **"O gato dormiu"** no encoder, e o decoder gerar **"no telhado quente"** auto-regressivamente.

---

## Visão Geral da Pipeline

```
"O gato dormiu"
     │
     ▼
┌──────────────────────────────────────────────────────────────┐
│ 1. TOKENIZAÇÃO                                                │
│    "O gato dormiu" → ["O", "gato", "dormiu"] → IDs: [87, 5432, 9103] │
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
│    "dormiu" descobre que seu sujeito é "O gato"               │
│    Saída Z_enc: matriz 3×512 (contextualizada)                 │
└──────────────────────────┬───────────────────────────────────┘
                           ▼
┌──────────────────────────────────────────────────────────────┐
│ 4. DECODER — LOOP AUTO-REGRESSIVO                             │
│    Itera até gerar o token de fim "</s>"                       │
│    ┌──────────────────────────────────────────────────────┐   │
│    │ Iteração 1: ["<s>"]                → gera "no"       │   │
│    │ Iteração 2: ["<s>", "no"]          → gera "telhado"  │   │
│    │ Iteração 3: ["<s>","no","telhado"] → gera "quente"   │   │
│    │ Iteração 4: [...,"quente"]         → gera "</s>"     │   │
│    └──────────────────────────────────────────────────────┘   │
└──────────────────────────┬───────────────────────────────────┘
                           ▼
                      "no telhado quente"
```

---

## Passo 1: Tokenização

A frase entra como texto bruto. O [tokenizador BPE](TOKENIZACAO.md) quebra em tokens conhecidos:

```
"O gato dormiu"
     │ BPE tokenizer (vocabulário de 37K tokens)
     ▼
Tokens:  ["O", "gato", "dormiu"]
IDs:     [ 87,  5432,   9103 ]
```

**3 tokens** → sequência de comprimento `n_enc = 3`.

A tokenização é pré-processamento em CPU. O que entra na GPU são os IDs inteiros.

---

## Passo 2: Embeddings + Positional Encoding

### 2a. Embedding Lookup

Cada ID busca sua linha na **matriz de embedding E** (37.000 linhas × 512 colunas):

```
ID 87   ("O")       →  E[87]    →  vetor de 512 floats (ex: [0.03, -0.12, 0.41, ...])
ID 5432 ("gato")    →  E[5432]  →  vetor de 512 floats
ID 9103 ("dormiu")  →  E[9103]  →  vetor de 512 floats
```

### 2b. Escala por √d_model

Cada vetor é multiplicado por √512 ≈ 22.6 para equilibrar a magnitude com o positional encoding (±1):

```
Embedding_final = Embedding × √512
```

### 2c. Somar Positional Encoding

Para cada posição `pos` (0, 1, 2), calcula-se PE(pos) — vetor de 512 dimensões com senos e cossenos — e SOMA ao embedding:

```
X_enc[0] = Embedding("O")      × √512 + PE(0)   ← posição 0: dimensões baixas oscilam rápido
X_enc[1] = Embedding("gato")   × √512 + PE(1)   ← posição 1
X_enc[2] = Embedding("dormiu") × √512 + PE(2)   ← posição 2
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
          O    gato   dormiu
O      [0.4   0.3    0.3  ]  ← "O" divide atenção entre todos
gato   [0.3   0.5    0.2  ]  ← "gato" foca 50% em si mesmo
dormiu [0.5   0.4    0.1  ]  ← "dormiu" foca 50% em "O gato" (sujeito!)
```

"dormiu" descobre que "O gato" é seu sujeito — isso é a mágica da atenção.

**Sub-camada 2 — Feed-Forward Network:**

```
FFN(x) = ReLU(x · W₁ + b₁) · W₂ + b₂
         (3×512)·(512×2048) → (3×2048) → ReLU → (3×2048)·(2048×512) → 3×512
+ residual + LayerNorm
```

Cada token é processado independentemente (a FFN não mistura tokens).

**Após 6 camadas:** a saída `Z_enc` é uma matriz **3 × 512** onde cada token está profundamente contextualizado. "dormiu" sabe que é verbo, que seu sujeito é "O gato", e que está na terceira posição.

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

### Iteração 1: Gerando "no"

**Entrada do decoder:** `["<s>"]` — apenas 1 token (ID=1)

```
1. Embedding + PE(0) para "<s>" → vetor 1×512

2. Masked Self-Attention:
   Só tem 1 token, não há o que mascarar
   "<s>" atende a "<s>" (100% em si mesmo)

3. Cross-Attention (a mágica acontece aqui):
   Q("<s>") pergunta: "qual parte da entrada é relevante para começar?"
   K_enc responde: "O" e "gato" são os mais relevantes (artigo + sujeito)
   V_enc entrega o significado de "O gato"

   Pesos da cross-attention para "<s>":
   O: 0.45   gato: 0.35   dormiu: 0.20

4. FFN processa

5. Projeção + Softmax sobre 37K tokens:
   "no"      → 0.28  ← MAIOR probabilidade
   "em"      → 0.10
   "sobre"   → 0.06
   ... outros 36.997 tokens ...
```

**Token gerado:** `"no"` (ID=1543)

**Auto-alimentação:** `"no"` é concatenado à entrada. Agora o decoder tem `["<s>", "no"]`.

---

### Iteração 2: Gerando "telhado"

**Entrada do decoder:** `["<s>", "no"]` — 2 tokens

```
1. Embedding + PE(0) para "<s>", PE(1) para "no" → matriz 2×512

2. Masked Self-Attention:
   "<s>" vê ["<s>"]           ← só passado
   "no" vê ["<s>", "no"]     ← passado + ele mesmo

   Pesos (mascarados):
          <s>    no
   <s>  [1.0   0.0 ]
   no   [0.3   0.7 ]

3. Cross-Attention:
   Q("no") pergunta: "qual parte da entrada define ONDE?"
   K_enc responde: "dormiu" (verbo) + "O gato" (sujeito)

   Pesos da cross-attention para "no":
   O: 0.30   gato: 0.30   dormiu: 0.40

4. FFN processa

5. Projeção + Softmax sobre 37K tokens:
   "telhado" → 0.24  ← MAIOR
   "sofá"    → 0.11
   "chão"    → 0.08
   ...
```

**Token gerado:** `"telhado"` (ID=7821)

**Auto-alimentação:** entrada agora é `["<s>", "no", "telhado"]`.

---

### Iteração 3: Gerando "quente"

**Entrada do decoder:** `["<s>", "no", "telhado"]` — 3 tokens

```
1. Embedding + PE para os 3 tokens → matriz 3×512

2. Masked Self-Attention:
          <s>    no   telhado
   <s>   [1.0   0.0   0.0 ]
   no    [0.2   0.8   0.0 ]
   telhado[0.1  0.3   0.6 ]

   "telhado" presta atenção em "no" e "<s>" — contexto do que já foi gerado

3. Cross-Attention:
   Q("telhado") pergunta: "qual característica do local?"
   K_enc responde: contexto geral da cena — "gato dormiu"

   Pesos da cross-attention para "telhado":
   O: 0.25   gato: 0.35   dormiu: 0.40

4. FFN processa

5. Projeção + Softmax:
   "quente"  → 0.22  ← MAIOR
   "frio"    → 0.09
   "escuro"  → 0.07
   ...
```

**Token gerado:** `"quente"` (ID=5612)

**Auto-alimentação:** entrada agora é `["<s>", "no", "telhado", "quente"]`.

---

### Iteração 4: Gerando o Fim

**Entrada do decoder:** `["<s>", "no", "telhado", "quente"]` — 4 tokens

```
3. Cross-Attention:
   Q("quente") pergunta: "já terminei a descrição?"
   Pesos distribuídos sobre todo Z_enc: O: 0.30  gato: 0.30  dormiu: 0.40

5. Projeção + Softmax:
   "</s>"    → 0.38  ← MAIOR (token de fim)
   "e"       → 0.12
   "mas"     → 0.06
   ...
```

**Token gerado:** `"</s>"` (token de fim)

→ **Geração concluída.** O decoder gerou **"no telhado quente"** como continuação de "O gato dormiu".

---

## Resumo Visual do Loop Auto-Regressivo

```
TEMPO →
                    Z_enc (FIXO, calculado 1 vez pelo encoder)
                    ┌──────────┬──────────┬──────────┐
                    │ "O"      │ "gato"   │ "dormiu" │
                    │ (512dim) │ (512dim) │ (512dim) │
                    └────┬─────┴────┬─────┴────┬─────┘
                         │          │          │
                    K,V  │     K,V  │     K,V  │
                         │          │          │
Iteração 1:  <s> ──Q───O─────────gato──────dormiu───→ "no"
                         │          │          │
Iteração 2:  <s> ──Q───O─────────gato──────dormiu───→ "telhado"
             no  ──Q───O─────────gato──────dormiu──
                         │          │          │
Iteração 3:  <s> ──Q───O─────────gato──────dormiu───→ "quente"
             no  ──Q───O─────────gato──────dormiu──
             telhado─Q──O─────────gato──────dormiu──
                         │          │          │
Iteração 4:  <s> ──Q───O─────────gato──────dormiu───→ "</s>"
             no  ──Q───O─────────gato──────dormiu──    (fim)
             telhado─Q──O─────────gato──────dormiu──
             quente──Q──O─────────gato──────dormiu──
```

**Observe:**
- **Z_enc NUNCA muda** — o encoder rodou 1 vez e acabou
- **Q muda a cada token** — cada palavra gerada faz uma pergunta diferente ao encoder
- **A entrada do decoder CRESCE** a cada iteração — tokens gerados alimentam a próxima iteração (auto-regressivo)
- **A masked self-attention garante** que "telhado" não veja "quente" (futuro) durante o treino

---

## O Que Acontece com os Pesos de Atenção?

A beleza do Transformer está nos **padrões que emergem** naturalmente do treino:

| Token gerado | Atende mais a... | Tipo de relação |
|---|---|---|
| "no" | "dormiu" (0.40) + "O gato" (0.60) | Preposição ligada ao verbo e sujeito |
| "telhado" | "dormiu" (0.40) + "gato" (0.30) | Substantivo-lugar ligado ao verbo |
| "quente" | "dormiu" (0.40) + "gato" (0.35) | Adjetivo ligado ao contexto da cena |

A cross-attention aprendeu a **conectar a geração com o contexto relevante** sem supervisão explícita. Ninguém disse ao modelo que "dormiu" pede um lugar ou que "telhado" combina com "gato" — ele descobriu sozinho durante o treino, apenas otimizando a função de perda.

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

1. Quantas vezes o encoder roda para processar "O gato dormiu"? E o decoder?
2. Por que a entrada do decoder cresce a cada iteração? O que acontece com Z_enc?
3. Na iteração 3, o que "telhado" vê na masked self-attention? E na cross-attention?
4. O que mudaria se a masked self-attention fosse removida do decoder durante a inferência?
5. Se a frase de entrada fosse "O gato preto dormiu tranquilamente", o que mudaria no Z_enc? E no loop do decoder?
