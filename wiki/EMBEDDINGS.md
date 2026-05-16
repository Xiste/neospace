# Embeddings — De Inteiros a Vetores com Significado

## O Problema
Como representar tokens (inteiros) como vetores densos que capturem significado semântico, permitindo que palavras similares tenham representações similares?

## Contexto Histórico
- **One-hot encoding (pré-2013):** vocabulário de 37K = vetores de 37K dimensões com um único 1. Espaço enorme, distâncias sem sentido entre palavras.
- **Word2Vec (Mikolov et al., 2013):** mostrou que vetores densos de ~300 dimensões capturam relações semânticas como `rei - homem + mulher ≈ rainha`.
- **Transformer (2017):** usa embeddings aprendidos de 512 dimensões, com weight tying e fator de escala √d_model.

## Intuição Central
Em vez de listar todas as palavras do mundo (one-hot), descreva uma palavra por características abstratas: tom emocional, tipo gramatical, domínio semântico. 512 dimensões bastam para capturar esses aspectos.

**Analogia:** Ficha de biblioteca com 512 campos (gênero, tema, época, dificuldade...) vs. uma prateleira exclusiva por livro.

## Como Funciona
1. Token ID `t` indexa a linha `t` da matriz de embedding `E` (|V| × d_model)
2. O vetor resultante (512 dims) é multiplicado por √d_model ≈ 22.6
3. Isso equilibra a magnitude com o positional encoding (±1)

A matriz E é inicializada aleatoriamente (N(0, 1/d_model)) e **aprendida** durante o treino.

## Matemática
- **Matriz de embedding:** E ∈ ℝ^{|V| × d_model}
- **Lookup:** embedding(token) = E[t_id] × √d_model
- **Inicialização:** E_ij ~ N(0, 1/d_model)
- **Weight tying:** mesma E usada no encoder, decoder e projeção pré-softmax

## Impacto Prático
- **Parâmetros:** 37K × 512 = ~19M floats ≈ 76 MB (float32)
- **GPT-3:** 50K × 12.288 = 614M parâmetros ≈ 2.5 GB só nos embeddings
- Lookup é O(1) por token — operação extremamente rápida na GPU

## Pré-requisitos
- [Tokenização](TOKENIZACAO.md) — cada token é um ID inteiro antes do embedding

## Conexões
- **Positional Encoding:** soma-se ao embedding para injetar informação de posição
- **Self-Attention:** opera sobre os vetores de embedding
- **Weight tying:** reduz parâmetros e força consistência entrada-saída

## Papers Fundamentais
- Mikolov et al. (2013) — *Efficient Estimation of Word Representations in Vector Space* (Word2Vec)
- Vaswani et al. (2017) — *Attention Is All You Need* (Seção 3.4)

## Perguntas de Revisão
1. O que é um embedding e por que ele é melhor que one-hot encoding?
2. Por que o Transformer multiplica os embeddings por √d_model?
3. O que é weight tying e por que ele é usado?
4. Quantos parâmetros tem a matriz de embedding para |V|=37K e d_model=512?
5. Por que a inicialização dos embeddings importa?

## Recursos Adicionais
- [Word2Vec Tutorial — TensorFlow](https://www.tensorflow.org/text/tutorials/word2vec)
- [Embeddings — Jay Alammar](https://jalammar.github.io/illustrated-word2vec/)
