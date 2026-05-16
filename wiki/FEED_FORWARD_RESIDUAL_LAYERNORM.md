# Feed-Forward, Conexões Residuais e LayerNorm

## O Problema
Como empilhar muitas camadas sem sofrer com gradiente evanescente, e como cada token processa individualmente a informação após a atenção?

## Contexto Histórico
- **Pré-2015:** redes profundas eram instáveis — [gradiente](GLOSSARIO.md#gradiente-e-backpropagation) sumia nas primeiras camadas (gradiente evanescente)
- **ResNet (He et al., 2015):** introduz conexões residuais (skip connections) — F(x) + x
- **BatchNorm (Ioffe & Szegedy, 2015):** normaliza ativações, mas depende do batch
- **Transformer:** adota residual + LayerNorm (independe do batch, ideal para sequências)

## Intuição Central

**Feed-Forward:** Após a reunião de equipe (atenção), cada token processa as ideias sozinho (FFN). Expande para 2048 dims (rascunho), comprime para 512 (resumo final).

**Conexão Residual:** Em vez de reescrever o texto do zero, faça pequenas edições. `saída = SubLayer(x) + x`. Se a camada não tiver nada a acrescentar, pode zerar sua saída e o sinal original passa intacto.

**LayerNorm:** Equalizador de áudio — ajusta cada token para média 0 e desvio padrão 1 nas 512 dimensões. Estabiliza o treino e permite taxas de aprendizado mais altas.

## Como Funciona

**Ordem no Transformer original (pós-norm):**
```
x → SubLayer(x) → Dropout → + x (residual) → LayerNorm → saída
```

**Feed-Forward Network:**
$$FFN(x) = \text{[ReLU](GLOSSARIO.md#relu-rectified-linear-unit)}(xW_1 + b_1)W_2 + b_2$$
- W₁: 512×2048, W₂: 2048×512 ([d_model](GLOSSARIO.md#d_model-d_k-d_v-d_ff--a-notação-de-dimensões-do-transformer)=512, [d_ff](GLOSSARIO.md#d_model-d_k-d_v-d_ff--a-notação-de-dimensões-do-transformer)=2048)
- Expansão 4×, depois contração

**LayerNorm:**
$$LayerNorm(x) = \gamma \cdot \frac{x - \mu}{\sigma} + \beta$$
Onde μ e σ são calculados por token (não por batch). γ e β são aprendidos.

## Impacto Prático

| Componente | Parâmetros |
|-----------|-----------|
| Multi-Head Attention | ~1.05M |
| FFN (2 camadas) | ~2.1M |
| 2 LayerNorms | ~2K |
| **Total por camada** | **~3.15M** |

Modelo base (12 camadas) + embeddings ≈ **~65M parâmetros**

## Pré-requisitos
- [Self-Attention](SELF_ATTENTION.md)
- [Multi-Head Attention](MULTI_HEAD_ATTENTION.md)
- Conceitos básicos: camada linear, ReLU

## Conexões
- **Pré-norm vs. pós-norm:** modelos modernos (GPT, LLaMA) usam LayerNorm ANTES da SubLayer
- **SwiGLU:** variante moderna da FFN com ativação diferente
- **Gradient Checkpointing:** técnica que explora camadas residuais para economizar memória
- **Deep Transformers:** ViT e modelos com 100+ camadas só são possíveis graças a essas técnicas

## Papers Fundamentais
- He et al. (2015) — *Deep Residual Learning for Image Recognition* (ResNet)
- Ba et al. (2016) — *Layer Normalization*
- Vaswani et al. (2017) — *Attention Is All You Need* (Seção 3.1, 3.3)

## Perguntas de Revisão
1. Qual a função da conexão residual?
2. Por que a FFN expande para 2048 e depois contrai?
3. Qual a diferença entre BatchNorm e LayerNorm?
4. Quantos parâmetros tem o modelo base (~65M)?
5. Por que modelos modernos preferem pré-norm a pós-norm?

## Recursos Adicionais
- [ResNet Explained — Towards Data Science](https://towardsdatascience.com/residual-blocks-in-deep-learning-80d09e74e87e)
- [Layer Normalization Paper](https://arxiv.org/abs/1607.06450)
