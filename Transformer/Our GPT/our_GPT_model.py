# --------- #tag *Using same imports as nanoGPT* ---------
import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__() # Starts by running init of nn.Module
        self.weight = nn.Parameter(torch.ones(ndim)) # nn.Parameter() seems to be required for parameters
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
        
    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5) 
    

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Embedding length is multiple of n_head since we split Q K and V by the number of heads instead of having oen Q K and V per embedding vector
        assert config.n_embd % config.n_head == 0
        self.c_att_w_q = nn.Linear(config.n_embd, config.n_embd, bias = config.bias) # Not sure how we choose the bias later
        self.c_att_w_k = nn.Linear(config.n_embd, config.n_embd, bias = config.bias)
        self.c_att_w_v = nn.Linear(config.n_embd, config.n_embd, bias = config.bias)
        
        
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        # Some setting fields
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            # Creates a lower triangular matrix accessible in self.bias
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))
    
    
    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        q = self.c_att_w_q(x)
        k = self.c_att_w_k(x)
        v = self.c_att_w_v(x)
        
        # Reshape into 4d from 3d with number of heads as one dimension, for later attention calculations
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        
        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash: # I hope this wasn't what needed the triton package
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) # First part before softmax
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf')) # Implement the masking
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att) # If dropout is used on the attention output
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        y = self.resid_dropout(self.c_proj(y)) ##tag #TODO475 Why do we use c_proj (output projection) ? 
        return y
    
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc  = nn.Linear(config.n_embd, 4*config.n_embd, bias = config.bias) ##tag #TODO81c Why these particular sizes for input and output
        self.gelu = nn.GELU() ##tag #TODOf10 What is GELU
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias) ##tag #TODOa86 Why another projection right after and without activation
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
    

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x)) # Layer normalize, then do self-attention, then have some residual
        x = x + self.mlp(self.ln_2(x)) # Layer normalize again, then do MLP, then have some residual again
        return x
    

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None # Require a vocab_size parameter
        assert config.block_size is not None #Require a block_size parameter 
        self.config = config
        
        self.transformer
