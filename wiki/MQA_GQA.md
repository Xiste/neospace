# Multi-Query Attention (MQA) & Grouped Query Attention (GQA)

## O Problema

Durante a **inferência** (geração de texto), o decoder armazena os vetores K e V de todos os tokens já gerados em uma **KV-cache**. Com 8 cabeças de atenção, isso significa `8 × 2 × n × 64` floats por camada. Em modelos grandes com contexto longo (ex: 32 camadas, 4096 tokens), a KV-cache pode ocupar **dezenas de GB** — mais que os próprios pesos do modelo.

**O problema não é o treino — é a inferência.** Durante o treino, o teacher forcing processa tudo em paralelo e a KV-cache não existe. Mas na inferência, cada token gerado expande a KV-cache, e ela precisa ficar na VRAM.

## Contexto Histórico

- **Multi-Head Attention (2017):** 8 cabeças independentes de Q, K e V. Cada cabeça tem seus próprios K e V → KV-cache de `h × 2 × n × d_k` floats
- **MQA (Shazeer, 2019):** notou que os K e V entre cabeças são redundantes. Propôs compartilhar UMA K,V entre todas as cabeças de Q → KV-cache reduz em `h×`
- **GQA (Ainslie et al., 2023):** percebeu que MQA é agressivo demais para modelos muito grandes. Propôs meio-termo: agrupar cabeças em G grupos → KV-cache reduz em `h/G ×`

## Intuição Central

**Analogia da biblioteca:** 8 pesquisadores (Q₁...Q₈) querem fazer perguntas a uma coleção de livros.

- **Multi-Head:** cada pesquisador tem seu PRÓPRIO índice (K) e sua PRÓPRIA cópia dos livros (V). 8 índices, 8 cópias → muito espaço na estante.
- **MQA:** todos os 8 pesquisadores compartilham o MESMO índice (K) e a MESMA cópia dos livros (V). Cada um faz perguntas diferentes, mas consulta a mesma fonte → 1/8 do espaço.
- **GQA:** pesquisadores são organizados em 4 duplas. Cada dupla compartilha um índice e uma cópia → 1/4 do espaço. Melhor que MQA porque pesquisadores diferentes precisam de ângulos diferentes de consulta.

## Como Funciona

### Multi-Query Attention (MQA)

```
Estrutura:
  W_Q₁, W_Q₂, ..., W_Qₕ  (h projeções de Query, como no Multi-Head)
  W_K                    (UMA projeção de Key, compartilhada)
  W_V                    (UMA projeção de Value, compartilhada)

Para cada cabeça i:
  Q_i = X · W_Q_i
  K   = X · W_K          (mesmo K para TODAS as cabeças)
  V   = X · W_V          (mesmo V para TODAS as cabeças)
  head_i = Attention(Q_i, K, V)

Saída = Concat(head₁, ..., headₕ) · W_O
```

**Tamanho da KV-cache (por camada):** `2 × n × d_k` (vs `h × 2 × n × d_k` do Multi-Head)

### Grouped Query Attention (GQA)

```
Estrutura com G grupos (ex: G=4, h=8):
  Grupo 1: Q₁, Q₂ → K₁, V₁
  Grupo 2: Q₃, Q₄ → K₂, V₂
  Grupo 3: Q₅, Q₆ → K₃, V₃
  Grupo 4: Q₇, Q₈ → K₄, V₄

Cada grupo tem h/G = 2 cabeças de Q compartilhando a mesma K,V
```

**Tamanho da KV-cache (por camada):** `G × 2 × n × d_k` (vs `h × 2 × n × d_k` do Multi-Head)

## Impacto Prático

| Configuração | h | G | KV-cache por token (64-dim) | Redução |
|---|---|---|---|---|
| Multi-Head | 8 | — | 8 × 2 × 64 = 1024 floats | 1× |
| MQA | 8 | 1 | 1 × 2 × 64 = 128 floats | **8×** |
| GQA (LLaMA 2) | 32 | 8 | 8 × 2 × 128 = 2048 floats | **4×** (vs 32 cabeças) |

**Modelos reais:**
- **PaLM (540B):** usa MQA — essencial para inferência com contexto de 2048 tokens
- **LLaMA 2 70B:** usa GQA com G=4 (h=32 cabeças no total, 8 por grupo)
- **LLaMA 3:** mantém GQA
- **Mistral:** usa GQA com sliding window

## Pré-requisitos

- [Self-Attention](SELF_ATTENTION.md)
- [Multi-Head Attention](MULTI_HEAD_ATTENTION.md)
- [Os 3 Tipos de Atenção](ATENCAO_CODIFICADOR_DECODIFICADOR.md)

## Conexões

- **KV-cache:** MQA e GQA só fazem sentido por causa da KV-cache na inferência. Sem KV-cache, não há vantagem em compartilhar K,V
- **Flash Attention:** FA + MQA/GQA é a combinação usada em produção — FA reduz memória O(N²) da atenção, MQA/GQA reduz memória O(N) da KV-cache
- **Flash Decoding:** extensão do Flash Attention otimizada para inferência com KV-cache

## Papers Fundamentais

- Shazeer (2019) — *Fast Transformer Decoding: One Write-Head is All You Need* — paper original do MQA
- Ainslie et al. (2023) — *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints* — paper do GQA
- Touvron et al. (2023) — *LLaMA 2: Open Foundation and Fine-Tuned Chat Models* — implementação de referência do GQA em produção

## Perguntas de Revisão

1. Por que MQA e GQA só importam na inferência, não no treino?
2. Qual a diferença entre MQA e GQA? Desenhe os dois esquemas.
3. Se h=16 e G=4, qual o fator de redução da KV-cache? E se G=1?
4. Por que GQA foi adotado no LLaMA 2 em vez de MQA puro?
5. Se você estivesse projetando um modelo para rodar em celular (pouca memória), usaria MQA ou GQA? Justifique.

## Recursos Adicionais
- [Fast Transformer Decoding (MQA paper)](https://arxiv.org/abs/1911.02150)
- [GQA: Training Generalized Multi-Query Transformer Models](https://arxiv.org/abs/2305.13245)
- [KV Cache explained — Hugging Face](https://huggingface.co/blog/kv-cache)
