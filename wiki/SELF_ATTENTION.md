# Self-Attention — O Coração do Transformer

## O Problema
Como permitir que cada token de uma sequência "olhe" diretamente para todos os outros tokens em uma única operação, sem os problemas de caminho longo e não-paralelizável das RNNs?

## Contexto Histórico
- **RNN (pré-2017):** cada token via apenas o estado oculto do anterior. Informação de tokens distantes se degradava. Impossível paralelizar.
- **Atenção de Bahdanau (2014):** atenção entre encoder e decoder, mas ainda com RNN processando sequencialmente.
- **Transformer (2017):** remove RNNs completamente. Cada token atende a todos os outros em O(1) passos sequenciais.

## Intuição Central
Imagine uma sala de aula onde cada aluno (token) tem uma **Query** ("o que eu quero saber?"), uma **Key** ("o que eu ofereço?") e um **Value** ("qual é minha informação?"). Cada aluno compara sua Query com as Keys de todos, e faz uma média ponderada dos Values de quem for mais compatível.

**Analogia:** Busca em banco de dados — Query é sua pergunta, Key é o índice, Value é o conteúdo retornado. Mas aqui tudo é "soft" (ponderado e diferenciável).

## Como Funciona
1. **Projeção:** X (n×512) × W_Q, W_K, W_V → Q, K, V (cada n×64)
2. **Scores:** Q × Kᵀ → matriz n×n (quanto cada token presta atenção em cada outro)
3. **Escala:** divide por √[d_k](GLOSSARIO.md#d_model-d_k-d_v-d_ff--a-notação-de-dimensões-do-transformer) = 8 (mantém variância ~1 para [softmax](GLOSSARIO.md#softmax) saudável)
4. **[Softmax](GLOSSARIO.md#softmax):** normaliza cada linha para soma = 1
5. **Saída:** multiplica pesos de atenção pelos Values

## Matemática

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

- Q = X·W_Q, K = X·W_K, V = X·W_V
- W_Q, W_K ∈ ℝ^{[d_model](GLOSSARIO.md#d_model-d_k-d_v-d_ff--a-notação-de-dimensões-do-transformer) × [d_k](GLOSSARIO.md#d_model-d_k-d_v-d_ff--a-notação-de-dimensões-do-transformer)}, W_V ∈ ℝ^{[d_model](GLOSSARIO.md#d_model-d_k-d_v-d_ff--a-notação-de-dimensões-do-transformer) × [d_v](GLOSSARIO.md#d_model-d_k-d_v-d_ff--a-notação-de-dimensões-do-transformer)}
- [d_k](GLOSSARIO.md#d_model-d_k-d_v-d_ff--a-notação-de-dimensões-do-transformer) = [d_v](GLOSSARIO.md#d_model-d_k-d_v-d_ff--a-notação-de-dimensões-do-transformer) = 64 (para h=8 cabeças)
- Complexidade: [O(n²·d)](GLOSSARIO.md#notação-big-o--on²-on-o1) — dominada pelo produto QKᵀ

O fator 1/√d_k é crucial: sem ele, produtos escalares grandes saturam o softmax, matando o gradiente.

## Impacto Prático
| Seq Len | Matriz de Atenção | Memória (float32) |
|---------|-------------------|-------------------|
| 512 | 262K elementos | ~1 MB |
| 2048 | 4.2M elementos | ~17 MB |
| 8192 | 67M elementos | ~268 MB |
| 32K | 1B elementos | ~4 GB |

Isso explica por que contexto longo é o principal gargalo. Flash Attention resolve esse O(n²) em memória.

## Pré-requisitos
- [Embeddings](EMBEDDINGS.md) — o X de entrada da atenção é a saída dos embeddings
- Produto escalar e softmax

## Conexões
- **Multi-Head Attention:** 8 atenções em paralelo, cada uma com seu próprio W_Q, W_K, W_V
- **Masked Self-Attention:** decoder usa máscara para não ver tokens futuros
- **Cross-Attention:** decoder atende à saída do encoder
- **Flash Attention:** otimização de memória para atenção em sequências longas

## Papers Fundamentais
- Vaswani et al. (2017) — *Attention Is All You Need* (Seção 3.2.1)
- Bahdanau et al. (2014) — *Neural Machine Translation by Jointly Learning to Align and Translate* (atenção original)

## Perguntas de Revisão
1. O que significam Query, Key e Value? Dê a analogia com suas palavras.
2. Por que dividimos QKᵀ por √d_k? O que acontece se não dividirmos?
3. Por que a complexidade é O(n²·d)? Qual parte domina?
4. O que significa um peso de atenção 0.95 na posição (i=3, j=7)?
5. Por que self-attention captura dependências longas melhor que RNN?

## Recursos Adicionais
- [Illustrated Transformer — Jay Alammar](https://jalammar.github.io/illustrated-transformer/)
- [Attention Mechanisms — Lilian Weng](https://lilianweng.github.io/posts/2018-06-24-attention/)
