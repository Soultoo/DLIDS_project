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
        #print('')
        #print('x.size() (Note: In forward):')
        #print(x.size())
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
        y = self.resid_dropout(self.c_proj(y)) 
        return y
    
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc  = nn.Linear(config.n_embd, 4*config.n_embd, bias = config.bias) 
        self.gelu = nn.GELU() 
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias) 
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
    

@dataclass # These are the default parameters of the model itself
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    pretrained_wte: bool = False
    finetune_wte: bool = False


class GPT(nn.Module):
    def __init__(self, config, predef_wte = None, from_checkpoint=False):
        super().__init__()
        assert config.vocab_size is not None # Require a vocab_size parameter
        assert config.block_size is not None #Require a block_size parameter 
        
        print('')
        print('config.pretrained_wte (Note: Please be true):')
        print(config.pretrained_wte)
        
        if (config.pretrained_wte and not from_checkpoint):
            assert predef_wte is not None
            wte_to_set = nn.Embedding.from_pretrained(predef_wte)
            wte_to_set.weight.requires_grad = False # Default is no fine-tuning of wte
            if config.finetune_wte:
                print("Pre-trained weights will be fine-tuned")
                wte_to_set.weight.requires_grad = True
            
        else:
            wte_to_set = nn.Embedding(config.vocab_size, config.n_embd)
        
        self.config = config
        
        
        
        
        self.transformer = nn.ModuleDict(dict( ##tag #TODOb3a Typ wte = glove. Måste kolla så att alla dimensioner stämmer.
            wte = wte_to_set,
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        #self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying 
        # ^comenting out should be enough to disable weight tying?^
        
        # init all weights
        self.apply(self._init_weights) # .apply applies method to all submodules
        # apply special scaled init to the residual projections, per GPT-2 paper
        #for pn, p in self.named_parameters(): 
        #    if pn.endswith('c_proj.weight'):
        #        torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
        # ^This has been removed in our implementation to align the GPT more with the RNN and LSTM^
                
        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))
        for name, param in self.named_parameters():
            print(name, param.numel())
            print(param.size())
        print('')
        print('self.config.vocab_size (Note: Modellens vocab size?):')
        print(self.config.vocab_size)
        
    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        ##tag #TODOa83 Are they referring to the output projection weights? Why would those not be trained? Do they mean if they were pretrained? Is symmetry usually what happens?
        # I think the point might be that the embeddings are usually not considered part of the model
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module): # Keep this initialization in mind in case it doesn't align with other models
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            
    
    
    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss
    

    def crop_block_size(self, block_size): # I don't think we'll be using this
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]
                
    
    # DID NOT copy over from_pretrained since that's for loading GPT2 weights only
    
    
    
    
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type): 
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        # CHANGED TO ADAM vanilla, since we are not aiming to use weight decay and want consistency between models
        fused_available = 'fused' in inspect.signature(torch.optim.Adam).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.Adam(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused Adam: {use_fused}")

        return optimizer
    
    
    def estimate_mfu(self, fwdbwd_per_iter, dt): # This seems to be an evaluation metric, can leave it here
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu
        
        
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None): 
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

