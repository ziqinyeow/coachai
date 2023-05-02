import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedForward(nn.Module):
    """a simple linear layer followed by a non-linearity"""
    
    def __init__(self, n_embd, dropout = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd), # projection layer
            nn.Dropout(dropout) # dropout residual connections
        )
    
    def forward(self, x):
        return self.net(x)

class PositionalEncoding(nn.Module):
    """simple paper implementation of pe: +add position embeddings, can replace with nn.Embeddings(block_size, n_embd)"""
    def __init__(self, block_size, n_embd):
        super().__init__()

        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
        pe = torch.zeros(block_size, n_embd)
        position = torch.arange(0, block_size, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, n_embd, 2).float() * (-torch.log(10000.0) / n_embd))
        pe[:, 0::2] = torch.sin(position * div_term) # even index
        pe[:, 1::2] = torch.cos(position * div_term) # odd index
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x
    
class Head(nn.Module):
    def __init__(self, block_size, n_embd, head_size, dropout):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x): # x -> idx that is being embedded
        B, T, C = x.shape # B: batch size, T: sequence/timestep length, C: d_model/n_embd

        k = self.key(x)
        q = self.query(x) # (B, T, C)

        wei = q @ k.transpose(-2, -1) * C ** -0.5 # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = nn.functional.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        
        v = self.value(x) # (B, T, head_size)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, block_size, n_embd, num_heads, head_size, dropout):
        super().__init__()
        self.heads = nn.ModuleList([Head(block_size, n_embd, head_size, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out

class Block(nn.Module):
    """Transformer encoder block: communication followed by computation"""
    
    def __init__(self, block_size, n_embd, n_head, dropout):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(block_size, n_embd, n_head, head_size, dropout)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    
    def forward(self, x):
        x = x + self.sa(self.ln1(x)) # residual connections
        x = x + self.ffwd(self.ln2(x))
        return x

class GenerativeTransformer(nn.Module):
    """Generative Transformer
    inputs
        - vocab_size: number of vocab tokens/vocab len
        - block_size: seq len
        - n_embd: d_model
        - n_head: number of heads
        - n_layer: number of MultiHeadAttention Block
    """
    
    def __init__(self, vocab_size, block_size, n_embd, n_head, n_layer, dropout = 0.2):
        super().__init__()
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        # self.position_embedding_table = PositionalEncoding(block_size, n_embd) # can replace this layer 
        self.blocks = nn.Sequential(*[Block(block_size, n_embd, n_head, dropout) for _ in range(n_layer)])            
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
    
    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        # idx and targets are both (B, T) tensors of integers
        tok_emb = self.token_embedding_table(idx) # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T)) # (T, C) # add device=device
        x = tok_emb + pos_emb # (B, T, C)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        """normal sampling technique with multinomial sampling"""
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -self.block_size:]
            
            # get the predictions
            logits, loss = self(idx_cond)
            
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            
            # apply softmax to get prob
            probs = F.softmax(logits, dim=-1) # (B, C)
            
            # sample from the distributions
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            
            # append sampled index to the running sequence
            idx = torch.cat([idx, idx_next], dim=1) # (B, T + 1)
        
        return idx
    
    def params(self):
        return f'{sum(p.numel() for p in super().parameters())/1e6} M parameters'