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
   - Dimensões pares (2i): sen(pos / 10000^(2i/[d_model](GLOSSARIO.md#d_model-d_k-d_v-d_ff--a-notação-de-dimensões-do-transformer)))
   - Dimensões ímpares (2i+1): cos(pos / 10000^(2i/[d_model](GLOSSARIO.md#d_model-d_k-d_v-d_ff--a-notação-de-dimensões-do-transformer)))
2. O vetor PE(pos) de [d_model](GLOSSARIO.md#d_model-d_k-d_v-d_ff--a-notação-de-dimensões-do-transformer)=512 dimensões é somado ao embedding do token
3. Isso acontece uma única vez, antes da primeira camada

As frequências formam progressão geométrica: dimensões baixas oscilam rápido (período ~2π), dimensões altas oscilam devagar (período ~20000π).

**Por que seno e cosseno?** Porque a trigonometria tem uma propriedade especial: `sen(a+b)` e `cos(a+b)` podem ser reescritos como combinações lineares de `sen(a)`, `cos(a)`, `sen(b)` e `cos(b)`. Isso não vale para outras funções como tangente ou sigmoid. Essa propriedade é o que permite ao modelo aprender **distâncias relativas** entre tokens (explicado na seção Matemática abaixo).

## Matemática

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$
$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

- pos = 0, 1, 2, ..., n-1
- i = 0, 1, ..., d_model/2 - 1
- Valores sempre em [-1, 1]
- PE(pos+k) é função linear de PE(pos) → facilita aprendizado de posições relativas

### Por que "PE(pos+k) é linear em PE(pos)" importa?

Essa é a propriedade mais importante do positional encoding senoidal. Sem ela, o modelo só entenderia posições absolutas ("sou o token 7"). Com ela, o modelo pode aprender posições relativas ("estou a 3 posições do token X"), independente de onde está na sequência.

**De onde vem essa propriedade:**

Lembra das identidades trigonométricas do ensino médio?

$$\sin(a + b) = \sin(a)\cos(b) + \cos(a)\sin(b)$$
$$\cos(a + b) = \cos(a)\cos(b) - \sin(a)\sin(b)$$

Agora olha o que acontece com o PE na posição `pos + k`. Para uma dimensão par 2i:

$$PE(pos+k, 2i) = \sin\left(\frac{pos + k}{10000^{2i/d}}\right) = \sin\left(\frac{pos}{10000^{2i/d}} + \frac{k}{10000^{2i/d}}\right)$$

Aplicando a identidade:

$$= \underbrace{\sin\left(\frac{pos}{10000^{2i/d}}\right)}_{PE(pos,2i)} \cdot \cos\left(\frac{k}{10000^{2i/d}}\right) + \underbrace{\cos\left(\frac{pos}{10000^{2i/d}}\right)}_{PE(pos,2i+1)} \cdot \sin\left(\frac{k}{10000^{2i/d}}\right)$$

Ou seja: **PE(pos+k) pode ser escrito como combinação linear de PE(pos)**, e os coeficientes (sen(k/...) e cos(k/...)) **dependem apenas de k** (a distância), não de `pos`.

**Traduzindo:** se o modelo aprende uma transformação linear que mapeia PE(pos) → "atenção ao token pos+3", essa mesma transformação funciona em QUALQUER posição: funciona para pos=5, pos=100, pos=5000. Isso é o que permite generalizar padrões de distância ao longo de toda a sequência.

**Analogia da régua:** se você tem uma régua com marcações em cm de 0 a 100, cada marcação é independente (números absolutos). Mas com PE senoidal, em vez de números, você tem um "código" que se repete com padrão conhecido: o deslocamento de +3 é sempre a mesma transformação linear, esteja você no cm 5 ou no cm 80. É como ter uma régua onde a distância entre marcações é codificada diretamente na forma como você representa cada posição.

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
