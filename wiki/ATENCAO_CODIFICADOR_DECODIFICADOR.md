# Os 3 Tipos de Atenção no Transformer

## O Problema
O Transformer precisa de diferentes "regras de visibilidade" para diferentes tarefas: entender a entrada (tudo visível), gerar saída (só passado visível) e conectar entrada-saída (decoder consulta encoder).

## Contexto Histórico
- **Seq2Seq com RNN (2014):** encoder e decoder processavam sequencialmente. Informação comprimida em vetor de contexto fixo.
- **Atenção de Bahdanau (2014):** decoder podia "visitar" cada posição do encoder, mas encoder ainda era RNN.
- **Transformer (2017):** 3 configurações de atenção sob o mesmo mecanismo matemático.

## Intuição Central
Um tradutor humano usa 3 operações mentais:
1. **Ler a frase original:** conecta todas as palavras entre si para entender o significado completo
2. **Escrever a tradução:** cada palavra nova só considera o que já foi escrito
3. **Consultar o original:** enquanto escreve, revisita a frase fonte para garantir fidelidade

## Os 3 Tipos

### 1. Encoder Self-Attention (sem máscara)
- Q, K, V vêm da mesma fonte (camada anterior do encoder)
- Cada token atende a TODOS os outros tokens
- Função: construir representações ricas contextualizadas da entrada

### 2. Decoder Masked Self-Attention
- Q, K, V vêm da mesma fonte (camada anterior do decoder)
- Máscara triangular: token na posição i só atende a tokens nas posições ≤ i
- Função: preservar a propriedade auto-regressiva (cada token é gerado um por um)

### 3. Cross-Attention (Encoder-Decoder)
- Q vem do decoder, K e V vêm do encoder
- Sem máscara: decoder pode ver toda a entrada
- Função: alinhar a geração do decoder com a informação da entrada

## Matemática

**Encoder Self-Attention:**
$$\text{Attention}(X_{enc}W_Q, X_{enc}W_K, X_{enc}W_V)$$

**Decoder Masked Self-Attention:**
$$\text{Attention}(X_{dec}W_Q, X_{dec}W_K, X_{dec}W_V + M)$$
Onde M tem −∞ nas posições futuras (j > i).

**Cross-Attention:**
$$\text{Attention}(X_{dec}W_Q, Z_{enc}W_K, Z_{enc}W_V)$$

## Impacto Prático
| Tipo | Formato | Máscara | Custo |
|------|---------|---------|-------|
| Encoder Self | n_enc × n_enc | Não | O(n_enc²) |
| Decoder Self | n_dec × n_dec | Triangular | O(n_dec²) |
| Cross | n_dec × n_enc | Não | O(n_enc × n_dec) |

## Pré-requisitos
- [Self-Attention](SELF_ATTENTION.md)
- [Multi-Head Attention](MULTI_HEAD_ATTENTION.md)

## Conexões
- **GPT:** decoder-only (apenas masked self-attention, sem cross-attention)
- **BERT:** encoder-only (apenas self-attention sem máscara)
- **KV-cache:** otimização de inferência que armazena K e V já computados no decoder

## Papers Fundamentais
- Vaswani et al. (2017) — *Attention Is All You Need* (Seções 3.1 e 3.2.3)

## Perguntas de Revisão
1. Por que o decoder precisa de máscara na self-attention?
2. Qual a diferença entre self-attention e cross-attention?
3. Por que a cross-attention NÃO tem máscara?
4. GPT (decoder-only) usa quais tipos de atenção? E BERT (encoder-only)?
5. O que aconteceria se a cross-attention usasse Q do encoder em vez do decoder?
