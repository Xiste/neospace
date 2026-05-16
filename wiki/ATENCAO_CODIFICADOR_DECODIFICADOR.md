# Os 3 Tipos de Atenção no Transformer

## O Problema

O Transformer precisa de diferentes "regras de visibilidade" para diferentes tarefas: entender a entrada (tudo visível), gerar saída (só passado visível) e conectar entrada-saída (decoder consulta encoder).

## Contexto Histórico

**Seq2Seq com RNN (2014):** encoder e decoder processavam sequencialmente. Toda a informação da entrada era comprimida em um único vetor de contexto fixo. Para frases longas, informação se perdia.

**Atenção de Bahdanau (2014):** o decoder podia "visitar" cada posição do encoder, mas o encoder em si ainda era uma RNN (processamento sequencial, sem paralelismo).

**Transformer (2017):** unificou tudo sob o mesmo mecanismo matemático de atenção, mas com 3 configurações diferentes para funções diferentes. Todos os 3 tipos usam **Multi-Head Attention** (8 cabeças) — a diferença está apenas na origem de Q/K/V e na presença de máscara.

## Intuição Central — O Tradutor Humano

Um tradutor humano usa 3 operações mentais ao traduzir "The cat that ate the fish is full" → "O gato que comeu o peixe está cheio":

1. **Ler e entender a frase original (Encoder Self-Attention):** conecta todas as palavras entre si. "is" se conecta a "The cat" (e não a "the fish") — sujeito da oração. Para entender, você olha LIVREMENTE para todas as palavras.

2. **Escrever a tradução em ordem (Decoder Masked Self-Attention):** começa "O gato que comeu o peixe está..." — quando vai escrever "cheio", só pode olhar o que JÁ escreveu. Não pode ver o futuro (a próxima palavra que ainda nem foi gerada).

3. **Consultar o original enquanto escreve (Cross-Attention):** ao escrever "cheio", revisita o original: "full" → "cheio". Verifica "is full", "The cat" para garantir concordância de gênero e número.

## Os 3 Tipos em Detalhe

### 1. Encoder Self-Attention (sem máscara)

**Origem:** Q, K e V vêm da mesma fonte — a camada anterior do encoder.

**Regra:** cada token atende a TODOS os outros tokens. Sem restrições.

**Função:** construir representações ricas e contextualizadas da entrada. Cada palavra é enriquecida com informação de todas as outras.

**Exemplo:** na frase "O gato que viu o cachorro fugiu", o encoder Self-Attention permite que "fugiu" olhe para "O gato" (e não para "o cachorro") porque a atenção captura a estrutura sintática.

```
Matriz de atenção (todas as células preenchidas):
         O   gato  que  viu  o   cachorro  fugiu
O      [0.2  0.3  0.1  0.1  0.1  0.1      0.1 ]
gato   [0.1  0.2  0.2  0.1  0.1  0.1      0.2 ]
que    [0.1  0.1  0.1  0.3  0.1  0.2      0.1 ]
viu    [0.1  0.1  0.1  0.1  0.1  0.4      0.1 ]
o      [0.1  0.1  0.1  0.1  0.1  0.4      0.1 ]
cachorro[0.1 0.1  0.1  0.2  0.1  0.2      0.2 ]
fugiu  [0.4  0.3  0.1  0.1  0.0  0.0      0.1 ]
```

### 2. Decoder Masked Self-Attention

**Origem:** Q, K e V vêm da mesma fonte — a camada anterior do decoder.

**Regra:** máscara triangular. Token na posição `i` só pode atender a tokens nas posições `≤ i` (passado e presente, nunca futuro).

**Por que a máscara é necessária:** durante o treino, o decoder recebe a sequência de saída COMPLETA de uma vez (teacher forcing). Sem a máscara, o token na posição 3 "colaria" olhando a resposta correta na posição 4. A máscara força o modelo a prever cada token usando apenas o que já foi gerado.

**Visualização da máscara (█ = visível, ░ = bloqueado):**
```
         O   gato  comeu  peixe
O      [██   ░░    ░░     ░░ ]
gato   [██   ██    ░░     ░░ ]
comeu  [██   ██    ██     ░░ ]
peixe  [██   ██    ██     ██ ]
```

**O que acontece na prática:** aplica-se `−∞` nas posições bloqueadas ANTES do [softmax](GLOSSARIO.md#softmax). Como `softmax(−∞) = 0`, as posições futuras recebem peso zero.

### 3. Cross-Attention (Encoder-Decoder)

**Origem:** 
- **Q** (Query) vem do **decoder** — "o que estou tentando gerar agora?"
- **K e V** vêm do **encoder** — "o que a frase original contém?"

**Regra:** sem máscara. O decoder pode (e deve) ver a entrada INTEIRA.

**Função:** alinhar cada token sendo gerado com a informação relevante da entrada. É a ponte entre entender (encoder) e gerar (decoder).

**Exemplo concreto:** ao traduzir "I love you" → "Eu amo você":

```
Passo 1: decoder gera "Eu"
  Q("Eu") consulta K_enc → descobre que "I" é o mais relevante
  V_enc entrega o significado de "I"

Passo 2: decoder gera "amo"  
  Q("amo") consulta K_enc → descobre que "love" é o mais relevante
  V_enc entrega o significado de "love"

Passo 3: decoder gera "você"
  Q("você") consulta K_enc → descobre que "you" é o mais relevante
  V_enc entrega o significado de "you"
```

## Matemática

Os 3 tipos usam exatamente a mesma fórmula base — a diferença está nos inputs e na máscara:

**Encoder Self-Attention:**
$$\text{Attention}(X_{enc}W_Q,\; X_{enc}W_K,\; X_{enc}W_V)$$

**Decoder Masked Self-Attention:**
$$\text{Attention}(X_{dec}W_Q,\; X_{dec}W_K,\; X_{dec}W_V + M)$$
$$M_{ij} = \begin{cases} 0 & \text{se } j \leq i \\ -\infty & \text{se } j > i \end{cases}$$

**Cross-Attention:**
$$\text{Attention}(X_{dec}W_Q,\; Z_{enc}W_K,\; Z_{enc}W_V)$$

Onde:
- $X_{enc}$ = saída da camada anterior do encoder, $X_{dec}$ = saída da camada anterior do decoder
- $Z_{enc}$ = saída FINAL do encoder (após todas as 6 camadas)
- $M$ = máscara triangular com $-\infty$ nas posições futuras
- Todos usam **Multi-Head** (8 cabeças), cada cabeça com seu próprio $W_Q, W_K, W_V$

## Impacto Prático

| Tipo | Q vem de | K,V vêm de | Máscara | Custo | Usa Multi-Head? |
|------|---------|-----------|---------|-------|-----------------|
| Encoder Self | Encoder | Encoder | Não | O(n_enc²) | Sim (8 cabeças) |
| Decoder Masked Self | Decoder | Decoder | Triangular (j ≤ i) | O(n_dec²) | Sim (8 cabeças) |
| Cross-Attention | Decoder | Encoder | Não | O(n_enc × n_dec) | Sim (8 cabeças) |

**Arquiteturas modernas (variantes):**
- **GPT:** decoder-only — usa APENAS Masked Self-Attention (sem encoder, sem cross-attention)
- **BERT:** encoder-only — usa APENAS Self-Attention sem máscara (bidirectional)
- **T5:** encoder-decoder completo — usa os 3 tipos como o Transformer original

## Pré-requisitos

[Self-Attention](SELF_ATTENTION.md) — entender Q, K, V e a fórmula base `softmax(QK^T/√d_k)V`
[Multi-Head Attention](MULTI_HEAD_ATTENTION.md) — os 3 tipos usam Multi-Head (8 cabeças) internamente

## Conexões

**Este tópico desbloqueia:**
[Multi-Query Attention (MQA)](MQA_GQA.md) — otimização da masked self-attention para inferência
[Grouped Query Attention (GQA)](MQA_GQA.md) — evolução do MQA usada no LLaMA 2/3

**Conceitos relacionados:**
- **Teacher forcing:** técnica de treino que alimenta o decoder com a saída correta — razão pela qual a máscara é necessária
- **KV-cache:** durante a inferência, K e V de tokens passados do decoder são reutilizados para não recalcular

## Papers Fundamentais

- Vaswani et al. (2017) — *Attention Is All You Need* (Seções 3.1 e 3.2.3)
- Bahdanau et al. (2014) — *Neural Machine Translation by Jointly Learning to Align and Translate*

## Perguntas de Revisão

1. Por que o decoder precisa de máscara na self-attention? O que aconteceria no treino sem ela?
2. Qual a diferença entre self-attention e cross-attention em termos de origem de Q, K, V?
3. Por que a cross-attention NÃO tem máscara? (Dica: pense no tradutor consultando o original)
4. GPT (decoder-only) usa quais tipos de atenção dos 3? E BERT (encoder-only)?
5. O que aconteceria se a cross-attention usasse Q do encoder em vez do decoder? A tradução ainda funcionaria?
