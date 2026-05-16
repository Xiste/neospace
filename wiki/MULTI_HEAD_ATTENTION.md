# Multi-Head Attention — 8 Atenções em Paralelo

## O Problema
Uma única função de atenção precisa capturar relações sintáticas, semânticas, posicionais e de correferência simultaneamente nos mesmos 64 números. Isso força o modelo a fazer uma "média" que perde nuances.

## Contexto Histórico
- **Atenção de Bahdanau (2014) e Luong (2015):** uma única função de atenção
- **Transformer (2017):** propõe h=8 atenções em paralelo, cada uma com seus próprios W_Q, W_K, W_V
- **Linha A (Tabela 3):** cabeça única perde 0.9 [BLEU](GLOSSARIO.md#bleu-bilingual-evaluation-understudy); cabeças demais também pioram → ponto ótimo em h=8

## Intuição Central
Em vez de UM perito analisando a cena do crime, chame 8 especialistas: um de digitais, um de balística, um de DNA, um de documentos... Cada um olha a mesma cena com ferramentas diferentes. Depois, junte todos os laudos.

**Analogia:** Detetives especializados — cada cabeça aprende um "tipo de pergunta" diferente sobre a sequência.

## Como Funciona
1. Para cada cabeça i (1 a 8): projeta X com W_Q_i, W_K_i, W_V_i (512→64)
2. Cada cabeça computa Attention(Q_i, K_i, V_i) → saída n×64
3. Concatena as 8 saídas → n×512
4. Projeta com W_O (512×512) → n×512

O custo total é similar a 1 cabeça com dimensionalidade total: 8 × O(n²×64) = O(n²×512).

## Matemática

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O$$

$$\text{head}_i = \text{Attention}(QW^Q_i, KW^K_i, VW^V_i)$$

- h = 8 cabeças
- [d_k](GLOSSARIO.md#d_model-d_k-d_v-d_ff--a-notação-de-dimensões-do-transformer) = [d_v](GLOSSARIO.md#d_model-d_k-d_v-d_ff--a-notação-de-dimensões-do-transformer) = [d_model](GLOSSARIO.md#d_model-d_k-d_v-d_ff--a-notação-de-dimensões-do-transformer)/h = 512/8 = 64
- W_Q_i, W_K_i ∈ ℝ^{512×64}, W_V_i ∈ ℝ^{512×64}
- W_O ∈ ℝ^{512×512}

## Impacto Prático
- Cabeças descobrem padrões interpretáveis sozinhas: sintaxe, correferência, adjacência
- Exemplo real: "its" atende fortemente a "The Law" — o modelo aprendeu correferência sem supervisão explícita
- [Dropout](GLOSSARIO.md#dropout) (p=0.1) nos pesos de atenção evita dependência excessiva de uma única cabeça

## Pré-requisitos
- [Self-Attention](SELF_ATTENTION.md) — cada cabeça é uma self-attention independente

## Conexões
- **Masked Multi-Head Attention:** decoder usa máscara para impedir atenção ao futuro
- **Cross-Attention:** decoder atende à saída do encoder
- **Multi-Query Attention (MQA) / Grouped Query Attention (GQA):** variações modernas que compartilham K e V entre cabeças para reduzir KV-cache

## Papers Fundamentais
- Vaswani et al. (2017) — *Attention Is All You Need* (Seção 3.2.2)
- Voita et al. (2019) — *Analyzing Multi-Head Self-Attention* (interpretabilidade de cabeças)

## Perguntas de Revisão
1. Por que usar 8 cabeças em vez de 1?
2. Explique por que o custo de 8 cabeças (d_k=64) é similar a 1 cabeça (d_k=512).
3. Qual o papel da matriz W_O?
4. O que uma cabeça com atenção sempre uniforme indicaria?
5. Por que cabeça única e cabeças demais são ambas piores que h=8?

## Recursos Adicionais
- [Illustrated Transformer — Jay Alammar](https://jalammar.github.io/illustrated-transformer/)
- [Analyzing Multi-Head Self-Attention — Voita et al.](https://arxiv.org/abs/1905.09418)
