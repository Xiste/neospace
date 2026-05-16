# Transformer — Arquitetura Completa e Treinamento (Attention Is All You Need)

## O Problema
Como substituir completamente RNNs e CNNs em tradução automática por um mecanismo puramente baseado em atenção, alcançando melhor qualidade em menos tempo de treino?

## Contexto Histórico
- **Pré-2017:** estado da arte em tradução usava [ensembles](GLOSSARIO.md#ensemble) de RNNs com atenção. Treino levava semanas. Melhor resultado EN→DE: ~26 [BLEU](GLOSSARIO.md#bleu-bilingual-evaluation-understudy).
- **Transformer (Vaswani et al., 2017):** primeiro modelo de transdução puramente baseado em atenção. 28.4 [BLEU](GLOSSARIO.md#bleu-bilingual-evaluation-understudy) em EN→DE (modelo único), 41.0 [BLEU](GLOSSARIO.md#bleu-bilingual-evaluation-understudy) em EN→FR, treinado em 3.5 dias com 8 GPUs P100.

## Arquitetura Completa

```
ENTRADA → Tokenização → Embedding + PE → ENCODER (6×) → DECODER (6×) → Linear + Softmax → SAÍDA
```

**Encoder (6 camadas):**
Cada camada = Self-Attention (Multi-Head, 8) + FFN (512→2048→512), ambos com residual + LayerNorm (pós-norm)

**Decoder (6 camadas):**
Cada camada = Masked Self-Attention (Multi-Head, 8) + Cross-Attention + FFN, todos com residual + LayerNorm

## Hiperparâmetros (Modelo Base)

| Parâmetro | Valor |
|-----------|-------|
| d_model | 512 |
| d_ff | 2048 |
| h (cabeças) | 8 |
| d_k, d_v | 64 |
| N (camadas) | 6 encoder + 6 decoder |
| P_drop | 0.1 |
| Label smoothing | 0.1 |
| Batch | ~25K tokens |
| Optimizer | Adam (β₁=0.9, β₂=0.98, ε=10⁻⁹) |
| Warmup | 4000 steps |
| Training steps | 100K (base) / 300K (large) |

## Treinamento

**Função de perda:** [Cross-entropy](GLOSSARIO.md#cross-entropy-entropia-cruzada) sobre o vocabulário, com [label smoothing](GLOSSARIO.md#label-smoothing) (0.1).

**Learning rate com warmup:**
$$lr = d_{model}^{-0.5} \cdot \min(step\_num^{-0.5}, step\_num \cdot warmup\_steps^{-1.5})$$

- Warmup (0→4000 steps): lr aumenta linearmente → estabiliza gradientes no início
- Decaimento (4000+ steps): lr ∝ 1/√step

**Regularização:**
- [Dropout](GLOSSARIO.md#dropout) (p=0.1): após cada sub-camada, nos embeddings, nos pesos de atenção
- [Label smoothing](GLOSSARIO.md#label-smoothing) (ε=0.1): evita overfitting, melhora [BLEU](GLOSSARIO.md#bleu-bilingual-evaluation-understudy) apesar de piorar [perplexidade](GLOSSARIO.md#perplexity-perplexidade)

## Inferência
- [Beam search](GLOSSARIO.md#beam-search) com tamanho 4
- Penalidade de comprimento α=0.6
- Encoder roda 1 vez, decoder roda N vezes ([auto-regressivo](GLOSSARIO.md#auto-regressivo))

## Distribuição de Parâmetros (~65M total)

| Componente | Parâmetros | % |
|-----------|-----------|-----|
| Embeddings | 38M | 58% |
| Atenção (12 camadas) | 12.6M | 19% |
| FFN (12 camadas) | 25.2M | 39% |
| Outros | 1M | 1.5% |

## Resultados

| Dataset | Transformer Base | Transformer Big | Estado da Arte (2017) |
|---------|-----------------|-----------------|----------------------|
| EN→DE | 27.3 BLEU | 28.4 BLEU | ~26 BLEU (ensembles) |
| EN→FR | 38.1 BLEU | 41.0 BLEU | ~40 BLEU (ensembles) |

## Componentes Individuais (Ordem de Estudo)

1. [Tokenização (BPE)](TOKENIZACAO.md) — texto → inteiros
2. [Embeddings](EMBEDDINGS.md) — inteiros → vetores densos
3. [Self-Attention](SELF_ATTENTION.md) — Q·Kᵀ/√d_k · softmax · V
4. [Multi-Head Attention](MULTI_HEAD_ATTENTION.md) — 8 atenções em paralelo
5. [Atenção Encoder/Decoder](ATENCAO_CODIFICADOR_DECODIFICADOR.md) — 3 tipos de atenção
6. [Positional Encoding](POSITIONAL_ENCODING.md) — senos e cossenos
7. [Feed-Forward + Residual + LayerNorm](FEED_FORWARD_RESIDUAL_LAYERNORM.md)

## Pré-requisitos Para Estudar o Transformer
- Estrutura básica de redes neurais (camadas, ativação, backprop)
- Conceitos de álgebra linear (multiplicação de matrizes, softmax)
- Tokenização e embeddings (ou estudar junto)

## Conexões
- **Flash Attention:** resolve O(n²) em memória na atenção
- **LoRA:** fine-tuning eficiente congelando pesos e treinando adaptadores de baixa ordem
- **Quantização:** reduz precisão dos pesos (FP16→INT8→INT4) para economia de memória
- **Paralelismo (DDP, FSDP, TP, PP):** distribui treino entre múltiplas GPUs

## Papers Fundamentais
- Vaswani et al. (2017) — *Attention Is All You Need*
- He et al. (2015) — *Deep Residual Learning for Image Recognition*
- Ba et al. (2016) — *Layer Normalization*
- Sennrich et al. (2016) — *Neural Machine Translation of Rare Words with Subword Units*

## Perguntas de Revisão
1. Quantas camadas tem o encoder e decoder do Transformer base? E dimensões?
2. Por que o treino usa warmup nos primeiros 4000 passos?
3. O que é label smoothing e por que ele melhora BLEU apesar de piorar perplexidade?
4. Durante a inferência, quantas vezes o encoder roda? E o decoder? Por quê?
5. Dobrando d_model para 1024 e d_ff para 4096, quantos parâmetros teria? Quanto mais lento seria?

## Recursos Adicionais
- [Illustrated Transformer — Jay Alammar](https://jalammar.github.io/illustrated-transformer/)
- [The Annotated Transformer — Harvard NLP](http://nlp.seas.harvard.edu/annotated-transformer/)
- [Attention Is All You Need — Paper Original](https://arxiv.org/abs/1706.03762)
