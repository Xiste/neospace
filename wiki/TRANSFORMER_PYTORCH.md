# Transformer do Zero em PyTorch — Guia Passo a Passo

## Sobre Este Guia

Este guia assume que você **nunca usou PyTorch**. Cada função, cada parâmetro, cada `dim=` e cada `.view()` será explicado. Você vai desde "o que é um tensor" até ter um Transformer completo capaz de gerar texto.

O código aqui é feito para ser executado no **Google Colab** (gratuito, GPU T4) ou em qualquer computador com Python 3.8+.

**Pré-requisitos da trilha Neospace:** todos os 8 fundamentos até [Transformer Completo](TRANSFORMER_COMPLETO.md).

---

## Parte 0 — O Que É PyTorch?

PyTorch é uma biblioteca Python que faz duas coisas:

**1. Tensores (= arrays multidimensionais que rodam na GPU):**
```python
import torch
x = torch.tensor([1.0, 2.0, 3.0])  # vetor de 3 números
M = torch.randn(3, 512)            # matriz 3×512 com números aleatórios
```

Um **tensor** é como uma lista Python, mas com superpoderes: pode ter dezenas de dimensões, rodar na GPU, e calcular gradientes automaticamente.

**2. Autograd (= derivadas automáticas):**
Quando você faz `y = f(x)`, o PyTorch consegue calcular `dy/dx` sozinho. Isso é o que permite treinar redes neurais: você define a rede, calcula o erro, e o PyTorch calcula como ajustar cada peso.

**3. nn.Module (= bloco de construção):**
Toda rede neural em PyTorch herda de `nn.Module`. É uma classe Python com dois métodos principais:
```python
class MeuModelo(nn.Module):
    def __init__(self):    # CONSTRUTOR: define as peças (camadas)
        super().__init__()
        self.linear = nn.Linear(512, 512)  # uma camada linear

    def forward(self, x):  # FORWARD: como os dados fluem pelas peças
        return self.linear(x)
```

**Regra de ouro:** `__init__` define QUAIS camadas existem. `forward` define COMO os dados passam por elas.

---

## Parte 1 — Configurando o Ambiente

```python
import torch
import torch.nn as nn           # camadas prontas (Linear, LayerNorm, etc.)
import torch.nn.functional as F  # funções sem peso (softmax, relu, etc.)
import math                      # sqrt, sin, cos, etc.

# Verifica se tem GPU disponível (no Colab: Ambiente > Acelerador > GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Rodando em: {device}")
```

**O que cada import faz:**
- `torch` — o framework principal (tensores, operações)
- `torch.nn` (`nn`) — camadas que TEM parâmetros aprendíveis (pesos)
- `torch.nn.functional` (`F`) — funções matemáticas que NÃO têm parâmetros (softmax, relu)
- `torch.device` — diz se o código roda na CPU ou GPU

---

## Parte 2 — Hiperparâmetros (Iguais ao Paper)

```python
d_model = 512      # dimensão do embedding e de todas as camadas ocultas
d_ff = 2048         # dimensão interna da FFN (4× d_model)
h = 8               # número de cabeças de atenção
d_k = d_model // h  # dimensão de cada Key por cabeça (512/8 = 64)
d_v = d_model // h  # dimensão de cada Value por cabeça (512/8 = 64)
N_encoder = 6       # número de camadas do encoder
N_decoder = 6       # número de camadas do decoder
vocab_size = 37000  # tamanho do vocabulário BPE
max_len = 512       # comprimento máximo de sequência
dropout_rate = 0.1  # probabilidade de dropout
```

**Por que `//` e não `/`?** `//` é divisão inteira em Python. `512 / 8 = 64.0` (float), `512 // 8 = 64` (int). Dimensões precisam ser inteiras.

---

## Parte 3 — Embeddings

```python
class Embeddings(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        # nn.Embedding é uma tabela de lookup: ID → vetor
        # vocab_size linhas, d_model colunas
        self.embed = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        # x tem shape (batch_size, seq_len)
        # Ex: (64, 50) = 64 frases, cada uma com 50 tokens
        embed = self.embed(x)  # (batch, seq_len, d_model)
        return embed * math.sqrt(self.d_model)  # × √512 ≈ 22.6
```

**Linha por linha:**
- `nn.Embedding(vocab_size, d_model)` — cria uma matriz 37.000 × 512. Quando você passa um ID, retorna a linha correspondente. É exatamente a "ficha de biblioteca" do conceito de [Embeddings](EMBEDDINGS.md).
- `self.embed(x)` — `x` é uma matriz de IDs (batch × seq_len). O resultado é uma matriz (batch × seq_len × 512), onde cada ID virou um vetor de 512 dimensões.
- `* math.sqrt(self.d_model)` — multiplica por √512 ≈ 22.6 para equilibrar com o positional encoding (±1).

**O que é "batch"?** GPUs processam várias frases ao mesmo tempo. Um batch de 64 frases de 50 tokens cada é mais eficiente que processar 1 frase por vez. `batch_size` é a primeira dimensão de todo tensor no PyTorch.

---

## Parte 4 — Positional Encoding

```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        # Cria matriz (max_len × d_model) com os senos/cossenos
        pe = torch.zeros(max_len, d_model)  # placeholder: max_len linhas, 512 colunas

        # pos: vetor coluna [0, 1, 2, ..., max_len-1] com shape (max_len, 1)
        pos = torch.arange(0, max_len).unsqueeze(1).float()

        # denominador: 10000^(2i/d_model) para cada dimensão i
        # torch.arange(0, d_model, 2) → [0, 2, 4, ..., 510] (só pares)
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(pos * div)   # dimensões pares: seno
        pe[:, 1::2] = torch.cos(pos * div)   # dimensões ímpares: cosseno

        # Registra como buffer (não é peso treinável, mas faz parte do modelo)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        # self.pe[:, :x.size(1)] corta o PE no tamanho certo da sequência
        return x + self.pe[:, :x.size(1), :]
```

**Linha por linha:**
- `torch.zeros(max_len, d_model)` — cria uma matriz de zeros. `max_len=512` linhas (posições), `d_model=512` colunas (dimensões).
- `torch.arange(0, max_len)` — como `range(0, 512)` do Python, mas retorna um tensor: `[0, 1, 2, ..., 511]`.
- `.unsqueeze(1)` — adiciona uma dimensão extra na posição 1. Transforma `(512,)` em `(512, 1)`. Necessário para o broadcasting multiplicar corretamente com `div`.
- `torch.arange(0, d_model, 2)` — `[0, 2, 4, ..., 510]`, só os índices pares (256 valores).
- `torch.exp(...)` — calcula `e^x` para cada elemento. Aqui implementa `10000^(2i/d)` usando a identidade `a^b = e^(b·ln(a))`.
- `pe[:, 0::2]` — indexing avançado: "todas as linhas, colunas pares começando em 0, passo 2".
- `pe[:, 1::2]` — "todas as linhas, colunas ímpares começando em 1, passo 2".
- `register_buffer` — registra `pe` como parte do modelo, mas NÃO como peso treinável. Quando você salva o modelo, o PE é salvo junto. Quando move para GPU, o PE vai junto.
- `self.pe[:, :x.size(1), :]` — corta o PE para o comprimento da sequência atual. Se a frase tem 50 tokens, pega só as 50 primeiras linhas.

---

## Parte 5 — Scaled Dot-Product Attention (Uma Cabeça)

```python
def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q: (batch, heads, seq_len, d_k)   — "o que eu quero saber?"
    K: (batch, heads, seq_len, d_k)   — "o que eu ofereço?"
    V: (batch, heads, seq_len, d_v)   — "minha informação"
    mask: (batch, 1, seq_len, seq_len) ou None
    """
    d_k = Q.size(-1)  # último valor do shape de Q (= 64)
    
    # Passo 1: scores = Q · K^T / √d_k
    # Q: (batch, heads, seq_q, d_k)
    # K.transpose(-2, -1): (batch, heads, d_k, seq_k)
    # resultado: (batch, heads, seq_q, seq_k)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    
    # Passo 2: máscara (se existir)
    if mask is not None:
        # mask tem -∞ nas posições proibidas, 0 nas permitidas
        scores = scores.masked_fill(mask == 0, float('-inf'))
    
    # Passo 3: softmax
    attn_weights = F.softmax(scores, dim=-1)
    
    # Passo 4: média ponderada dos Values
    # attn_weights: (batch, heads, seq_q, seq_k)
    # V: (batch, heads, seq_k, d_v)
    # resultado: (batch, heads, seq_q, d_v)
    output = torch.matmul(attn_weights, V)
    
    return output, attn_weights
```

**Linha por linha:**
- `.size(-1)` — retorna o tamanho da ÚLTIMA dimensão. Para um tensor `(2, 8, 50, 64)`, `.size(-1)` retorna `64`.
- `.transpose(-2, -1)` — troca as DUAS ÚLTIMAS dimensões. `(2, 8, 50, 64)` → `(2, 8, 64, 50)`. Isso é a transposição K^T.
- `torch.matmul(A, B)` — multiplicação de matrizes. Se A e B são batches (3D ou 4D), multiplica a última dimensão de A pela penúltima de B, tratando as dimensões anteriores como batch.
- `math.sqrt(d_k)` — √64 = 8. Escala que mantém variância ~1.
- `scores.masked_fill(mask == 0, float('-inf'))` — onde a máscara for 0, coloca `-∞`. `softmax(-∞) = 0`, então essas posições viram peso 0.
- `F.softmax(scores, dim=-1)` — aplica softmax na ÚLTIMA dimensão. `dim=-1` significa "normaliza cada linha da última dimensão para soma = 1". Para uma matriz `(2, 8, 50, 50)`, cada uma das 50 colunas das últimas 2 dimensões é normalizada independentemente.
- `float('-inf')` — representa "menos infinito" em Python. `e^(-∞) = 0`.

---

## Parte 6 — Multi-Head Attention

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, h, dropout=0.1):
        super().__init__()
        self.h = h  # número de cabeças (8)
        self.d_k = d_model // h  # 64
        self.d_v = d_model // h  # 64
        
        # Projeções lineares: 512 → 512 (para Q, K, V juntas)
        self.W_Q = nn.Linear(d_model, d_model)  # 512 → 512
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)  # projeção de saída
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)
        
        # Passo 1: projeções lineares
        Q = self.W_Q(Q)  # (batch, seq_len, d_model)
        K = self.W_K(K)
        V = self.W_V(V)
        
        # Passo 2: divide em cabeças
        # (batch, seq_len, d_model) → (batch, seq_len, h, d_k)
        Q = Q.view(batch_size, -1, self.h, self.d_k)
        K = K.view(batch_size, -1, self.h, self.d_k)
        V = V.view(batch_size, -1, self.h, self.d_v)
        
        # Transpõe para (batch, h, seq_len, d_k)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # Passo 3: atenção (chama a função da Parte 5)
        attn_output, attn_weights = scaled_dot_product_attention(Q, K, V, mask)
        # attn_output: (batch, h, seq_len, d_v)
        
        # Passo 4: junta as cabeças
        # (batch, h, seq_len, d_v) → (batch, seq_len, h, d_v)
        attn_output = attn_output.transpose(1, 2)
        # → (batch, seq_len, h * d_v) = (batch, seq_len, 512)
        attn_output = attn_output.contiguous().view(batch_size, -1, self.h * self.d_v)
        
        # Passo 5: projeção final W_O
        return self.W_O(attn_output)
```

**Linha por linha:**
- `nn.Linear(d_model, d_model)` — camada totalmente conectada: `y = xW + b`. Cria uma matriz de pesos W (512×512) e um vetor de bias b (512). TOTALMENTE aprendível.
- `.view(batch_size, -1, self.h, self.d_k)` — reorganiza o tensor sem copiar dados. `-1` significa "calcula essa dimensão automaticamente". `(64, 50, 512)` → `(64, 50, 8, 64)`.
- `.transpose(1, 2)` — troca as dimensões 1 (seq_len) e 2 (h). `(64, 50, 8, 64)` → `(64, 8, 50, 64)`. Agora a dimensão das cabeças está na posição correta para a função de atenção.
- `.contiguous()` — garante que o tensor está em ordem contínua na memória antes do `.view()`. Necessário após `.transpose()`.
- `.view(batch_size, -1, self.h * self.d_v)` — achata as cabeças de volta: `(64, 50, 8, 64)` → `(64, 50, 512)`.

**Por que juntar Q, K, V em matrizes 512→512 em vez de 512→64?** Fazer 512→512 e depois dividir em 8 fatias de 64 é matematicamente idêntico a fazer 8 projeções separadas 512→64, mas é mais eficiente na GPU (uma multiplicação grande em vez de 8 pequenas).

---

## Parte 7 — Feed-Forward Network

```python
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.W1 = nn.Linear(d_model, d_ff)   # 512 → 2048
        self.W2 = nn.Linear(d_ff, d_model)   # 2048 → 512
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = self.W1(x)           # (batch, seq_len, d_ff) = (batch, seq_len, 2048)
        x = F.relu(x)            # ReLU: max(0, x) — zera negativos
        x = self.dropout(x)      # 10% dos neurônios zerados aleatoriamente (só no treino)
        x = self.W2(x)           # (batch, seq_len, d_model) = (batch, seq_len, 512)
        return x
```

**Linha por linha:**
- `nn.Linear(d_model, d_ff)` — cria pesos W₁ (512×2048) e bias b₁ (2048).
- `F.relu(x)` — `max(0, x)`. Se o valor é negativo, vira 0. Se positivo, passa direto. É o que torna a rede NÃO-linear — sem isso, 100 camadas lineares seria equivalente a 1 camada linear (matematicamente: `W₁(W₂(W₃x)) = (W₁W₂W₃)x`, que é só outra matriz).
- `nn.Dropout(dropout)` — durante o treino, zera aleatoriamente 10% dos valores e multiplica os 90% restantes por 1/(1-0.1). Durante a inferência (eval), não faz nada.
- A FFN é aplicada **identicamente** a cada token, mas de forma independente. O `nn.Linear` opera na última dimensão de `x`: cada token (vetor 512-dim) passa pela mesma transformação.

---

## Parte 8 — Encoder Layer (1 Camada)

```python
class EncoderLayer(nn.Module):
    def __init__(self, d_model, h, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, h, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)  # normaliza APÓS atenção + residual
        self.norm2 = nn.LayerNorm(d_model)  # normaliza APÓS FFN + residual
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Sub-camada 1: Self-Attention + Residual + LayerNorm
        attn_out = self.self_attn(x, x, x, mask)  # Q, K, V = x (mesma fonte)
        x = x + self.dropout(attn_out)             # residual: soma entrada
        x = self.norm1(x)                          # LayerNorm: normaliza
        
        # Sub-camada 2: FFN + Residual + LayerNorm
        ffn_out = self.ffn(x)
        x = x + self.dropout(ffn_out)              # residual: soma entrada
        x = self.norm2(x)                          # LayerNorm: normaliza
        
        return x
```

**Linha por linha:**
- `nn.LayerNorm(d_model)` — normaliza CADA token independentemente: calcula média µ e desvio σ das 512 dimensões daquele token, depois faz `(x-µ)/σ * γ + β`. γ e β são aprendidos (permitem "desfazer" a normalização se necessário).
- `self.self_attn(x, x, x, mask)` — Q, K e V vêm do mesmo lugar `x` (self-attention). No encoder, todos os tokens podem ver todos os outros.
- `x + self.dropout(attn_out)` — conexão residual: soma a saída da atenção com a entrada original. Se a atenção não tiver nada útil a acrescentar, o sinal original passa intacto. Isso resolve o problema do gradiente evanescente.
- **Ordem pós-norm:** atenção → residual → LayerNorm. O Transformer original usa esta ordem. Modelos modernos invertem (pré-norm: LayerNorm → atenção → residual).

---

## Parte 9 — Decoder Layer (1 Camada)

```python
class DecoderLayer(nn.Module):
    def __init__(self, d_model, h, d_ff, dropout=0.1):
        super().__init__()
        self.masked_attn = MultiHeadAttention(d_model, h, dropout)  # self com máscara
        self.cross_attn = MultiHeadAttention(d_model, h, dropout)   # cross: Q=dec, K,V=enc
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        # Sub-camada 1: Masked Self-Attention + Residual + LayerNorm
        # Q, K, V = x (decoder), máscara triangular para esconder futuro
        attn_out = self.masked_attn(x, x, x, tgt_mask)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)
        
        # Sub-camada 2: Cross-Attention + Residual + LayerNorm
        # Q = decoder, K e V = encoder (Z_enc, a saída do encoder!)
        cross_out = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = x + self.dropout(cross_out)
        x = self.norm2(x)
        
        # Sub-camada 3: FFN + Residual + LayerNorm
        ffn_out = self.ffn(x)
        x = x + self.dropout(ffn_out)
        x = self.norm3(x)
        
        return x
```

**Por que 3 sub-camadas e não 2?** O decoder tem duas fontes de informação: o que já gerou (masked self-attention) e a entrada original (cross-attention). O encoder só tem uma fonte (a entrada).

---

## Parte 10 — Criando as Máscaras

```python
def create_padding_mask(seq, pad_token=0):
    """
    Impede que a atenção preste atenção em tokens <PAD>.
    seq: (batch, seq_len) — IDs dos tokens
    Retorna: (batch, 1, 1, seq_len) — True onde NÃO é padding
    """
    # seq != pad_token → True para tokens reais, False para <PAD>
    return (seq != pad_token).unsqueeze(1).unsqueeze(2)


def create_look_ahead_mask(seq_len, device):
    """
    Máscara triangular: token i só vê tokens j ≤ i (passado e presente).
    Usa torch.tril (triangular inferior = lower triangular).
    Retorna: (1, 1, seq_len, seq_len)
    """
    # torch.ones(seq_len, seq_len) → matriz quadrada de 1s
    # torch.tril → mantém só a parte triangular INFERIOR
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
```

**Linha por linha:**
- `seq != pad_token` — comparação elemento a elemento. Retorna uma máscara booleana: True onde há tokens reais, False onde há padding.
- `.unsqueeze(1).unsqueeze(2)` — adiciona dimensões para broadcasting com a matriz de atenção: `(batch, seq_len)` → `(batch, 1, 1, seq_len)`.
- `torch.tril(...)` — "triangle lower". Mantém os valores na diagonal e abaixo, zera acima. Ex: `torch.tril(torch.ones(3,3))` = `[[1,0,0],[1,1,0],[1,1,1]]`.
- `device=device` — cria o tensor diretamente na GPU, evitando transferência posterior.

**Como as máscaras se combinam:** você faz `mask = padding_mask & look_ahead_mask`. O resultado é True só onde o token NÃO é padding E está em posição ≤ i.

---

## Parte 11 — Encoder (Stack de 6 Camadas)

```python
class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, h, d_ff, N, max_len, dropout=0.1):
        super().__init__()
        self.embed = Embeddings(vocab_size, d_model)
        self.pe = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, h, d_ff, dropout) for _ in range(N)
        ])
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, src, src_mask=None):
        # src: (batch, src_seq_len) — IDs dos tokens de entrada
        x = self.embed(src)      # (batch, src_seq_len, d_model)
        x = self.pe(x)           # soma positional encoding
        x = self.dropout(x)
        
        for layer in self.layers:
            x = layer(x, src_mask)
        
        return x  # Z_enc: (batch, src_seq_len, d_model)
```

**Linha por linha:**
- `nn.ModuleList([...])` — lista Python que o PyTorch sabe que contém módulos. Diferente de `nn.Sequential`, permite acessar camadas individuais (`self.layers[2]`). Necessário para o loop `for`.
- `for _ in range(N)` — `_` é convenção Python para "não vou usar essa variável". Cria 6 camadas idênticas (com pesos diferentes, cada uma inicializada aleatoriamente).
- O encoder processa a frase INTEIRA de uma vez — não é auto-regressivo.

---

## Parte 12 — Decoder (Stack de 6 Camadas)

```python
class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, h, d_ff, N, max_len, dropout=0.1):
        super().__init__()
        self.embed = Embeddings(vocab_size, d_model)
        self.pe = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, h, d_ff, dropout) for _ in range(N)
        ])
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, tgt, enc_output, src_mask=None, tgt_mask=None):
        # tgt: (batch, tgt_seq_len) — IDs dos tokens de saída
        # enc_output: (batch, src_seq_len, d_model) — saída Z_enc do encoder
        x = self.embed(tgt)
        x = self.pe(x)
        x = self.dropout(x)
        
        for layer in self.layers:
            x = layer(x, enc_output, src_mask, tgt_mask)
        
        return x  # (batch, tgt_seq_len, d_model)
```

---

## Parte 13 — O Transformer Completo

```python
class Transformer(nn.Module):
    def __init__(self, vocab_size=37000, d_model=512, h=8, d_ff=2048,
                 N_enc=6, N_dec=6, max_len=512, dropout=0.1):
        super().__init__()
        self.encoder = Encoder(vocab_size, d_model, h, d_ff, N_enc, max_len, dropout)
        self.decoder = Decoder(vocab_size, d_model, h, d_ff, N_dec, max_len, dropout)
        
        # Projeção final: vetor 512-dim → scores sobre o vocabulário (37K)
        self.proj = nn.Linear(d_model, vocab_size)
        
        # Weight tying: compartilha pesos do embedding com a projeção final
        self.proj.weight = self.encoder.embed.embed.weight
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """
        src: (batch, src_seq_len) — frase de entrada tokenizada
        tgt: (batch, tgt_seq_len) — frase de saída (shiftada, ver Parte 14)
        """
        enc_output = self.encoder(src, src_mask)
        dec_output = self.decoder(tgt, enc_output, src_mask, tgt_mask)
        logits = self.proj(dec_output)  # (batch, tgt_seq_len, vocab_size)
        return logits
```

**Linha por linha:**
- `nn.Linear(d_model, vocab_size)` — projeção final: mapeia cada vetor de 512 dimensões para 37.000 scores (um para cada token do vocabulário).
- `self.proj.weight = self.encoder.embed.embed.weight` — **weight tying**: compartilha a mesma matriz de pesos entre o embedding de entrada e a projeção de saída. Economiza ~19M parâmetros e força o modelo a usar o mesmo espaço vetorial para representar entrada e saída. Isso só funciona porque `vocab_size × d_model` = `d_model × vocab_size` (a matriz é transposta automaticamente pelo `nn.Linear`).
- `logits` — scores brutos (antes do softmax). Cada posição da sequência de saída tem 37.000 scores. O maior score indica o token mais provável.

---

## Parte 14 — Preparando os Dados (Teacher Forcing)

Durante o TREINO, o decoder recebe a frase de saída COMPLETA, mas **deslocada em 1**:

```python
# Exemplo: traduzir "O gato dormiu" → "The cat slept"
src = ["O", "gato", "dormiu"]           # entrada do encoder
tgt = ["<s>", "The", "cat", "slept"]    # entrada do decoder (shiftada)
labels = ["The", "cat", "slept", "</s>"] # o que o modelo deve prever

# O decoder vê "<s>" e prevê "The"
# O decoder vê "<s> The" e prevê "cat"
# O decoder vê "<s> The cat" e prevê "slept"
# O decoder vê "<s> The cat slept" e prevê "</s>"
```

**A máscara triangular garante** que, mesmo recebendo a frase completa, o modelo não "cole" — na posição 2, ele só vê posições 0, 1, 2 (não vê "slept" ainda).

---

## Parte 15 — Inferência (Geração Auto-Regressiva)

Durante a INFERÊNCIA, geramos um token por vez:

```python
@torch.no_grad()  # desliga o cálculo de gradientes (não estamos treinando)
def generate(model, src, max_new_tokens=100, start_token=1, end_token=2,
             temperature=1.0, device='cpu'):
    """
    Gera texto auto-regressivamente.
    
    model: Transformer já treinado
    src: (1, src_seq_len) — frase de entrada com batch=1
    max_new_tokens: máximo de tokens a gerar
    start_token: ID do token <s>
    end_token: ID do token </s>
    temperature: controla criatividade (1.0 = padrão)
    """
    model.eval()  # modo inferência (desliga dropout)
    
    # Passo 1: encoder processa a entrada UMA vez
    src_mask = create_padding_mask(src, pad_token=0).to(device)
    enc_output = model.encoder(src, src_mask)  # Z_enc: constante
    
    # Passo 2: inicializa o decoder com <s>
    generated = torch.tensor([[start_token]], device=device)  # (1, 1)
    
    for step in range(max_new_tokens):
        # Cria máscara triangular para o decoder
        seq_len = generated.size(1)
        tgt_mask = create_look_ahead_mask(seq_len, device)
        
        # Decoder processa tudo gerado até agora
        dec_output = model.decoder(generated, enc_output, src_mask, tgt_mask)
        
        # Pega só o ÚLTIMO token da saída (é auto-regressivo)
        last_token_logits = model.proj(dec_output[:, -1, :])  # (1, vocab_size)
        
        # Aplica temperatura e softmax
        last_token_logits = last_token_logits / temperature
        probs = F.softmax(last_token_logits, dim=-1)  # (1, vocab_size)
        
        # Escolhe o token mais provável (greedy)
        next_token = torch.argmax(probs, dim=-1, keepdim=True)  # (1, 1)
        
        # Concatena com os tokens já gerados (auto-alimentação!)
        generated = torch.cat([generated, next_token], dim=-1)
        
        # Se gerou </s>, para
        if next_token.item() == end_token:
            break
    
    return generated.squeeze(0).tolist()  # lista de IDs
```

**Linha por linha:**
- `@torch.no_grad()` — decorator que desliga o rastreamento de gradientes. Durante a inferência, não precisamos calcular derivadas, economizando memória e tempo.
- `model.eval()` — coloca o modelo em modo de inferência. Isso faz o dropout e batch normalization se comportarem corretamente (dropout não zera nada, batchnorm usa estatísticas móveis).
- `dec_output[:, -1, :]` — indexing: "todas as batches, ÚLTIMO token, todas as dimensões". Pega só o vetor do último token gerado para prever o próximo.
- `torch.argmax(probs, dim=-1, keepdim=True)` — retorna o ÍNDICE do maior valor. `dim=-1` = na última dimensão (vocabulário). `keepdim=True` mantém a dimensão: `(1,)` em vez de `()`.
- `torch.cat([generated, next_token], dim=-1)` — concatena o novo token ao final da sequência. `dim=-1` = concatena na última dimensão (seq_len). Isso é a AUTO-ALIMENTAÇÃO.
- `.item()` — converte um tensor de 1 elemento para um valor Python. `tensor([2])` → `2`.
- `.squeeze(0).tolist()` — remove a dimensão do batch e converte para lista Python: `[[1, 87, 543, 5612]]` → `[1, 87, 543, 5612]`.

---

## Parte 16 — Exemplo de Forward Pass

```python
# Cria o modelo (sem treinar — pesos aleatórios)
model = Transformer(
    vocab_size=100,  # reduzido para teste (em produção seria 37K)
    d_model=512,
    h=8,
    d_ff=2048,
    N_enc=6,
    N_dec=6,
    max_len=512,
    dropout=0.1
).to(device)

# Dados de exemplo (batch de 2 frases)
src = torch.tensor([[5, 12, 8, 1, 0, 0],     # "O gato dormiu bem <PAD> <PAD>"
                    [5, 20, 15, 1, 3, 0]])    # "A menina leu o livro <PAD>"
src = src.to(device)

tgt = torch.tensor([[1, 10, 7, 15, 2, 0],     # "<s> The cat slept </s> <PAD>"
                    [1, 8, 18, 22, 2, 0]])    # "<s> The girl read the book </s> <PAD>"
tgt = tgt.to(device)

# Cria máscaras
src_mask = create_padding_mask(src, pad_token=0).to(device)
tgt_mask = create_look_ahead_mask(tgt.size(1), device)
# Combina as máscaras do decoder: triangular E padding
tgt_mask = tgt_mask & create_padding_mask(tgt, pad_token=0).to(device)

# Forward pass
logits = model(src, tgt[:, :-1], src_mask, tgt_mask[:, :, :-1, :-1])
# Nota: tgt[:, :-1] remove o último token (shift para teacher forcing)
# logits: (2, 5, 100) → 2 batches, 5 tokens, 100 scores por token

print(f"Shape da saída: {logits.shape}")
# Output: Shape da saída: torch.Size([2, 5, 100])

# Parâmetros totais
total_params = sum(p.numel() for p in model.parameters())
print(f"Parâmetros totais: {total_params:,}")
# Output: ~65M (com vocab_size=37K)
```

---

## Parte 17 — Estrutura Final do Código

Juntando tudo, a hierarquia de classes fica:

```
Transformer
├── Encoder (6 camadas)
│   ├── Embeddings (|V| × 512, lookup)
│   ├── PositionalEncoding (senos/cossenos)
│   └── EncoderLayer × 6
│       ├── MultiHeadAttention (8 cabeças, Q=K=V=encoder)
│       │   ├── W_Q, W_K, W_V (Linear 512→512)
│       │   ├── scaled_dot_product_attention (Q·K^T/√d_k, softmax, ·V)
│       │   └── W_O (Linear 512→512)
│       └── FeedForward
│           ├── W₁ (Linear 512→2048) + ReLU
│           └── W₂ (Linear 2048→512)
│
├── Decoder (6 camadas)
│   ├── Embeddings (mesma matriz do encoder via weight tying)
│   ├── PositionalEncoding
│   └── DecoderLayer × 6
│       ├── MultiHeadAttention (masked, 8 cabeças, Q=K=V=decoder)
│       ├── MultiHeadAttention (cross, Q=decoder, K=V=encoder)
│       └── FeedForward
│
└── Projeção (Linear 512 → |V|, pesos compartilhados com Embeddings)
```

---

## Perguntas de Revisão

1. Qual a diferença entre `nn.Linear` e `F.softmax` em termos de parâmetros aprendíveis?
2. Por que usamos `.transpose(-2, -1)` para calcular K^T em vez de criar uma nova matriz?
3. O que acontece com o `nn.Dropout` quando chamamos `model.eval()`?
4. Na função `generate()`, por que o encoder só roda UMA vez e o decoder N vezes?
5. Por que `tgt[:, :-1]` remove o último token no teacher forcing?
6. Explique com suas palavras o que `.view()`, `.transpose()` e `.unsqueeze()` fazem.

---

## Recursos Adicionais

- [The Annotated Transformer — Harvard NLP](http://nlp.seas.harvard.edu/annotated-transformer/) — implementação de referência comentada linha a linha
- [PyTorch Documentation](https://pytorch.org/docs/stable/) — documentação oficial
- [Colab: Transformer do Zero](https://colab.research.google.com/) — abra um notebook e cole este código
