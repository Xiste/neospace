# Encoder & Decoder — Os 3 Tipos de Atenção no Transformer

## O Problema

O Transformer precisa de diferentes "regras de visibilidade" para diferentes tarefas: entender a entrada (tudo visível), gerar saída (só passado visível) e conectar entrada-saída (decoder consulta encoder).

Cada lado do Transformer tem um **papel diferente** e, portanto, um **tipo de atenção diferente**.

---

## Visão Geral: Quem Faz o Quê?

```
                    ENCODER                              DECODER
               (entender a entrada)                (gerar a saída)
┌──────────────────────────────────┐    ┌──────────────────────────────────┐
│                                  │    │                                  │
│  6 camadas idênticas             │    │  6 camadas idênticas             │
│  ┌────────────────────────────┐  │    │  ┌────────────────────────────┐  │
│  │ 1. Self-Attention          │  │    │  │ 1. Masked Self-Attention   │  │
│  │    (sem máscara)            │  │    │  │    (máscara triangular)     │  │
│  │    Q,K,V → tudo do encoder  │  │    │  │    Q,K,V → tudo do decoder  │  │
│  │                             │  │    │  │                             │  │
│  │ 2. Feed-Forward (FFN)       │  │    │  │ 2. Cross-Attention          │  │
│  │    + residual + LayerNorm   │  │    │  │    (sem máscara)             │  │
│  └────────────────────────────┘  │    │  │    Q → decoder                │  │
│                                  │    │  │    K,V → encoder              │  │
│                                  │    │  │                             │  │
└──────────────────────────────────┘    │  │ 3. Feed-Forward (FFN)       │  │
         ▲                              │  │    + residual + LayerNorm   │  │
         │                              │  └────────────────────────────┘  │
    ENTRADA                             │                                  │
    "The cat..."                         └──────────────────────────────────┘
                                                   │
                                                   ▼
                                               SAÍDA
                                               "O gato..."
```

**Resumo em uma frase:** o encoder processa a entrada com atenção livre (tudo visível). O decoder gera a saída com atenção restrita (só passado) + consulta ao encoder.

---

## ⬅️ ENCODER — Entender a Entrada

### Papel do Encoder

Ler e entender a frase de entrada completamente, construindo representações ricas onde cada palavra é contextualizada por todas as outras. O encoder processa a frase INTEIRA de uma vez (não é auto-regressivo).

### Estrutura de Uma Camada do Encoder

Cada uma das 6 camadas do encoder tem **2 sub-camadas**:

```
ENTRADA DA CAMADA (n tokens × 512 dims)
│
├─ 1. MULTI-HEAD SELF-ATTENTION (sem máscara)
│      Q = X · W_Q   (do encoder)
│      K = X · W_K   (do encoder)
│      V = X · W_V   (do encoder)
│      → Cada token atende a TODOS os outros
│
│      + residual (soma a entrada)
│      + LayerNorm
│
├─ 2. FEED-FORWARD NETWORK
│      FFN(x) = ReLU(xW₁ + b₁)W₂ + b₂
│      → Cada token processado independentemente
│
│      + residual (soma a entrada)
│      + LayerNorm
│
└─ SAÍDA DA CAMADA (n tokens × 512 dims)
```

### Encoder Self-Attention — A Atenção LIVRE

**Origem de Q, K, V:** todos vêm da saída da camada anterior do encoder (a MESMA fonte).

**Regra:** sem máscara. Todo token vê todo token. É **bidirecional**.

**Função:** cada palavra é enriquecida com o contexto da frase inteira. "is" descobre que seu sujeito é "The cat", não "the fish".

**Visualização da matriz de atenção (n_enc × n_enc):**
```
         O   gato  que  viu  o   cachorro  fugiu
O      [0.2  0.3  0.1  0.1  0.1  0.1      0.1 ]
gato   [0.1  0.2  0.2  0.1  0.1  0.1      0.2 ]
que    [0.1  0.1  0.1  0.3  0.1  0.2      0.1 ]
viu    [0.1  0.1  0.1  0.1  0.1  0.4      0.1 ]
o      [0.1  0.1  0.1  0.1  0.1  0.4      0.1 ]
cachorro[0.1 0.1  0.1  0.2  0.1  0.2      0.2 ]
fugiu  [0.4  0.3  0.1  0.1  0.0  0.0      0.1 ]

TODAS as células são preenchidas. O encoder vê tudo.
```

**Arquitetura encoder-only (exemplos reais):**
- **BERT:** usa APENAS essa atenção. Ideal para tarefas de compreensão (classificação, NER, QA)
- Modelos encoder-only NÃO geram texto — eles representam/entendem texto

---

## ➡️ DECODER — Gerar a Saída

### Papel do Decoder

Gerar a saída **um token por vez** (auto-regressivo), usando duas fontes de informação: o que já foi gerado (via masked self-attention) e a entrada original (via cross-attention).

### Estrutura de Uma Camada do Decoder

Cada uma das 6 camadas do decoder tem **3 sub-camadas** (uma a mais que o encoder):

```
ENTRADA DA CAMADA (m tokens × 512 dims)
│
├─ 1. MASKED MULTI-HEAD SELF-ATTENTION
│      Q = X · W_Q   (do decoder)
│      K = X · W_K   (do decoder)
│      V = X · W_V   (do decoder)
│      + MÁSCARA TRIANGULAR (j > i → −∞)
│      → Cada token só vê tokens anteriores
│
│      + residual + LayerNorm
│
├─ 2. MULTI-HEAD CROSS-ATTENTION
│      Q = X_dec · W_Q     (do decoder — "o que estou gerando?")
│      K = Z_enc · W_K     (do encoder — "o que a entrada contém?")
│      V = Z_enc · W_V     (do encoder — "significado da entrada")
│      SEM máscara — decoder pode ver o encoder INTEIRO
│
│      + residual + LayerNorm
│
├─ 3. FEED-FORWARD NETWORK
│      FFN(x) = ReLU(xW₁ + b₁)W₂ + b₂
│
│      + residual + LayerNorm
│
└─ SAÍDA DA CAMADA (m tokens × 512 dims)
```

**Por que 3 sub-camadas e não 2?** Porque o decoder tem duas fontes de informação: o que já gerou (self) e a entrada original (cross). O encoder só tem uma fonte (a entrada).

---

### Decoder Sub-Camada 1: Masked Self-Attention

**Origem de Q, K, V:** todos vêm do decoder (a MESMA fonte, como no encoder).

**Regra:** máscara triangular. Token na posição `i` só pode atender a `j ≤ i`.

**Por que a máscara existe:**
- **Durante o treino:** o decoder recebe a saída completa de uma vez (teacher forcing). Sem a máscara, o token 3 "colaria" vendo a resposta correta do token 4
- **Durante a inferência:** tokens futuros simplesmente não existem ainda — a máscara é natural

**Visualização da máscara (█ = visível, ░ = bloqueado):**
```
         O   gato  comeu  peixe
O      [██   ░░    ░░     ░░ ]
gato   [██   ██    ░░     ░░ ]
comeu  [██   ██    ██     ░░ ]
peixe  [██   ██    ██     ██ ]
```

**Como se aplica:** antes do [softmax](GLOSSARIO.md#softmax), coloca-se `−∞` nas posições com ░░. `softmax(−∞) = 0` → peso zero.

---

### Decoder Sub-Camada 2: Cross-Attention

**Origem de Q, K, V:**
- **Q** = `X_dec · W_Q` → vem do decoder ("o que estou tentando gerar agora?")
- **K** = `Z_enc · W_K` → vem da saída FINAL do encoder ("índice da entrada")
- **V** = `Z_enc · W_V` → vem da saída FINAL do encoder ("significado da entrada")

**Regra:** sem máscara. O decoder pode (e DEVE) ver a entrada INTEIRA.

**Função:** alinhar a geração com a informação relevante da entrada. É a PONTE entre entender e gerar.

**Walkthrough: "I love you" → "Eu amo você":**
```
Passo 1: decoder gera "Eu"
  Q_dec("Eu") consulta K_enc → "I" é o mais relevante
  V_enc entrega o significado de "I"

Passo 2: decoder gera "amo"
  Q_dec("amo") consulta K_enc → "love" é o mais relevante
  V_enc entrega o significado de "love"

Passo 3: decoder gera "você"
  Q_dec("você") consulta K_enc → "you" é o mais relevante
  V_enc entrega o significado de "you"

A cada passo, Q muda (pergunta diferente), mas K,V do encoder
são FIXOS (a frase original não muda — o encoder já processou tudo).
```

**Por que K e V são fixos?** Porque o encoder já processou a entrada INTEIRA uma única vez antes do decoder começar. A saída `Z_enc` é constante durante toda a geração.

**Arquitetura decoder-only (exemplos reais):**
- **GPT:** remove o encoder e a cross-attention. Só tem a sub-camada 1 (masked self-attention)
- Modelos decoder-only geram texto diretamente a partir do prompt, sem "consultar" uma entrada separada
- O prompt do usuário é tratado como os primeiros tokens da sequência do decoder

---

## Comparativo Final: Encoder vs Decoder

| Característica | ENCODER | DECODER |
|---|---|---|
| **Função** | Entender a entrada | Gerar a saída |
| **Camadas** | 6 | 6 |
| **Sub-camadas por camada** | **2** (Self-Attn + FFN) | **3** (Masked Self + Cross + FFN) |
| **Tipo 1** | Self-Attention (sem máscara) | Masked Self-Attention (triangular) |
| **Tipo 2** | — | Cross-Attention (Q=dec, K,V=enc) |
| **Q, K, V vêm de** | Mesma fonte (encoder) | Self: decoder. Cross: Q=dec, K,V=enc |
| **Visibilidade** | Tudo visível (bidirecional) | Self: só passado. Cross: tudo visível |
| **Processamento** | 1 vez (frase inteira) | N vezes (auto-regressivo, 1 token por vez) |
| **Exemplos** | BERT, ViT | GPT (decoder-only), Transformer original |

---

## Matemática

### Encoder (Sub-Camada 1)
$$\text{SelfAttention}(X_{enc}) = \text{softmax}\!\left(\frac{X_{enc}W_Q \cdot (X_{enc}W_K)^T}{\sqrt{d_k}}\right) X_{enc}W_V$$

### Decoder (Sub-Camada 1 — Masked Self)
$$\text{MaskedSelfAttention}(X_{dec}) = \text{softmax}\!\left(\frac{X_{dec}W_Q \cdot (X_{dec}W_K)^T}{\sqrt{d_k}} + M\right) X_{dec}W_V$$
$$M_{ij} = \begin{cases} 0 & \text{se } j \leq i \\ -\infty & \text{se } j > i \end{cases}$$

### Decoder (Sub-Camada 2 — Cross)
$$\text{CrossAttention}(X_{dec}, Z_{enc}) = \text{softmax}\!\left(\frac{X_{dec}W_Q \cdot (Z_{enc}W_K)^T}{\sqrt{d_k}}\right) Z_{enc}W_V$$

**Importante:** $Z_{enc}$ é a saída FINAL do encoder (após 6 camadas) e é **constante** durante toda a geração. $X_{dec}$ muda a cada token gerado.

---

## Pré-requisitos

[Self-Attention](SELF_ATTENTION.md) — entender Q, K, V e a fórmula base `softmax(QK^T/√d_k)V`
[Multi-Head Attention](MULTI_HEAD_ATTENTION.md) — todos os 3 tipos usam Multi-Head (8 cabeças) internamente

---

## Conexões

**Este tópico desbloqueia:**
[Multi-Query Attention (MQA)](MQA_GQA.md) — otimização da masked self-attention do decoder para inferência
[Grouped Query Attention (GQA)](MQA_GQA.md) — evolução do MQA usada no LLaMA 2/3

**Conceitos relacionados:**
- **Teacher forcing:** técnica de treino que alimenta o decoder com a saída correta — razão pela qual a máscara é necessária
- **KV-cache:** durante a inferência, K e V de tokens passados do decoder são reutilizados

---

## Papers Fundamentais

- Vaswani et al. (2017) — *Attention Is All You Need* (Seções 3.1 e 3.2.3)
- Bahdanau et al. (2014) — *Neural Machine Translation by Jointly Learning to Align and Translate*

---

## Perguntas de Revisão

1. Quantas sub-camadas tem o encoder? E o decoder? Por que são diferentes?
2. Qual a diferença ENTRE o encoder self-attention e o decoder masked self-attention?
3. Na cross-attention, de onde vêm Q, K e V? Por que K e V são fixos?
4. O GPT (decoder-only) remove quais partes do Transformer original? Ele tem cross-attention?
5. Se você quisesse um modelo para analisar sentimentos (e não gerar texto), usaria encoder-only ou decoder-only? Justifique.
