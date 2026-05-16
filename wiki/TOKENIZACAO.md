# Tokenização (Byte-Pair Encoding & Subword Tokenization)

## O Problema
Como representar texto para uma rede neural de forma eficiente, sem vocabulários gigantes e sem perder palavras raras ou desconhecidas?

## Contexto Histórico
- **Word-level:** cada palavra = um ID. Vocabulário enorme, palavras desconhecidas viram `<UNK>`.
- **Character-level:** cada caractere = um ID. Vocabulário pequeno, mas sequências ficam longas demais.
- **Solução (2016):** Subword tokenization (BPE de Sennrich et al.) — meio-termo que quebra palavras em pedaços frequentes.

## Intuição Central
Se você conhece os pedaços "gat" + "inho", entende "gatinho" mesmo sem nunca ter visto essa palavra. O BPE extrai esses pedaços automaticamente do corpus.

**Analogia:** LEGO — word-level são peças únicas rígidas, character-level são pinos minúsculos, subword são peças de tamanho certo que se encaixam.

## Como Funciona
1. Começa com todas as palavras quebradas em caracteres
2. Conta todos os pares de tokens adjacentes no corpus
3. Funde o par mais frequente em um novo token
4. Repete até atingir o tamanho de vocabulário desejado (ex: 37.000)

Exemplo: "gato gato gatos" → `g a t o` → funde "ga" → funde "gat" → vocabulário final inclui "gat", permitindo tokenizar "gatos" como `gat o s`.

## Matemática
Algoritmo puramente estatístico, sem redes neurais. A cada iteração k:
- Conta frequência f(a,b) de cada par adjacente
- Seleciona par com max f(a,b)
- Substitui todas as ocorrências desse par pelo novo token
- |V| aumenta em 1 por iteração

## Impacto Prático
- **Transformer:** 37K tokens (EN→DE), 32K tokens (EN→FR)
- **GPT-3:** ~50K tokens
- **LLaMA:** ~32K tokens
- Tokenização é pré-processamento em CPU, não afeta tempo de GPU

## Pré-requisitos
Nenhum.

## Conexões
- **Embeddings:** cada token (ID inteiro) é convertido em vetor denso
- **Vocabulário × d_model:** define o tamanho da matriz de embedding
- **Weight tying:** Transformer compartilha pesos entre embedding de entrada e saída

## Papers Fundamentais
- Sennrich et al. (2016) — *Neural Machine Translation of Rare Words with Subword Units*

## Perguntas de Revisão
1. Por que não usamos palavras inteiras como tokens?
2. Qual a diferença entre tokenização word-level, character-level e subword?
3. Explique o algoritmo BPE com suas próprias palavras.
4. Por que "gatinho" e "gatos" compartilham tokens no BPE?
5. O que acontece se o vocabulário for muito pequeno? E muito grande?

## Recursos Adicionais
- [Byte Pair Encoding — Lei Mao](https://leimao.github.io/blog/Byte-Pair-Encoding/)
- [Tokenization in LLMs — Hugging Face](https://huggingface.co/docs/transformers/tokenizer_summary)
