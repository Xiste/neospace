# Positional Encoding — Como o Modelo Sabe a Ordem das Palavras?

## O Problema
A self-attention trata a sequência como um conjunto não-ordenado de tokens. Sem informação explícita de posição, "A ama B" e "B ama A" produzem representações idênticas.

## Contexto Histórico
- **RNNs:** posição implícita no processamento sequencial — cada token é processado em ordem
- **CNNs (ConvS2S, ByteNet):** posição implícita via janela de convolução limitada
- **Transformer (2017):** precisa injetar posição explicitamente, já que removeu RNNs e CNNs

## Intuição Central
Como codificar o compasso 42 para 100 cantores? Em vez de gritar "42!" (inteiro sem relação natural com 41), use um **código binário contínuo** baseado em ondas de frequências diferentes.

**Analogia do relógio:** ponteiro dos segundos (alta frequência), minutos (média), horas (baixa). A combinação única das posições dos 3 ponteiros codifica qualquer horário do dia. O PE faz o mesmo com 256 "ponteiros" (pares seno/cosseno).

## Como Funciona
1. Para cada posição `pos` e cada dimensão `i`, calcula-se:
   - Dimensões pares (2i): sen(pos / 10000^(2i/d_model))
   - Dimensões ímpares (2i+1): cos(pos / 10000^(2i/d_model))
2. O vetor PE(pos) de 512 dimensões é somado ao embedding do token
3. Isso acontece uma única vez, antes da primeira camada

As frequências formam progressão geométrica: dimensões baixas oscilam rápido (período ~2π), dimensões altas oscilam devagar (período ~20000π).

## Matemática

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$
$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

- pos = 0, 1, 2, ..., n-1
- i = 0, 1, ..., d_model/2 - 1
- Valores sempre em [-1, 1]
- PE(pos+k) é função linear de PE(pos) → facilita aprendizado de posições relativas

## Impacto Prático
- PE senoidal vs. aprendido: resultados quase idênticos (Tabela 3, linha E)
- Modelos modernos (GPT, LLaMA): preferem PE aprendido ou RoPE
- Sem fator √d_model nos embeddings, o PE (±1) dominaria o sinal semântico

## Pré-requisitos
- [Embeddings](EMBEDDINGS.md) — o PE é somado ao embedding
- Seno e cosseno (matemática básica)

## Conexões
- **RoPE (Rotary Position Embedding):** evolução usada em LLaMA, Mistral — aplica rotação nos Q e K em vez de somar
- **ALiBi:** alternativa sem PE explícito — adiciona viés linear negativo na atenção
- **Embedding × √d_model:** equilibra magnitude com PE

## Papers Fundamentais
- Vaswani et al. (2017) — *Attention Is All You Need* (Seção 3.5)
- Su et al. (2021) — *RoFormer: Enhanced Transformer with Rotary Position Embedding* (RoPE)

## Perguntas de Revisão
1. Por que o Transformer precisa de positional encoding?
2. Por que usar senos/cossenos de diferentes frequências em vez do número da posição?
3. Por que somar o PE ao embedding e não concatenar?
4. O que significa que PE(pos+k) é função linear de PE(pos)? Por que isso importa?
5. Por que as frequências formam progressão geométrica?

## Recursos Adicionais
- [Illustrated Transformer — Jay Alammar](https://jalammar.github.io/illustrated-transformer/)
- [Positional Encoding Explained — Amirhossein Kazemnejad](https://kazemnejad.com/blog/transformer_architecture_positional_encoding/)
