### Todos

#tag #TODO716 Read about the different hyper-parameters
#tag #TODOf84 What is cosine similarity
#tag #TODOe0f And what is BERT score
#tag #TODO5f5 What is BLEU score
#tag #TODO566 What's the whole "Flop" stuff in estimate_mfu()




##### Building GPT model

#tag #TODOf3b Try to implement something similar to model.py in pytorch
  - Follow this: "class Block(nn.Module):"
    - Check its calls because everything is in that order (its forward() method)
#tag #TODOd82 I don't get the @-syntax (Decorators?)
#tag #TODO248 Copy the generate-method







### Learned stuff

##### [Model.py](nanoGPT/model.py)
- #tag #GLOSS150 *nn.Parameter()*
  - Initiates a new parameter (used )
- #tag #GLOSS01a *self.c_attn*:
  - According to GPT one matrix mult is done instead of 3 separate ones for Q, K and V
    - "Then split() is used to separate them."
- #tag #GLOSS958 *Weight tying*:
  - self.transformer.wte.weight = self.lm_head.weight
    - [Weight tying link from model.py](https://paperswithcode.com/method/weight-tying)
    - Acc. to GPT:
      -   The input embedding matrix and output projection matrix share weights.
      -   Helps reduce the number of parameters and improves performance.
      -   Takeaway: Weight tying is a common trick in language modeling to reduce parameters and enforce symmetry between input and output vocabularies.
      -   lm_head is the output projection to logits, while self.transformer.wte is the token embedding
- #tag #GLOSS7c4 *Triangular mask*   
  - Makes sense if you assume that there is one time step per axis
  - Implemented as lower triangular 
    - "# causal mask to ensure that attention is only applied to the left in the input sequence"
- #tag #GLOSS9e0 *top_k:*
  - Picks the top k results to sample from instead of whole token distribution
- #tag #GLOSS0a3 *x = x + self.attn(self.ln_1(x))*
  - This row (and the row after) apply layer normalization before the layer, not between layer and activation
- #tag #GLOSSf36 *Dropout*
  - Input tensor to any layer has a small chance of having some elements zerod
    - Creates regulraziation because it introduces noise
    - Probability of around 10% acc to GPT? Might not be true tho
    - (Disabled during inference)
- #tag #GLOSS143 *Temperature*
  - Temperature between 0 and 1 usually
    - Divide logits (values pre softmax) by temperature
    - Since softmax uses e**xi in nom. and denom. this makes the distribution more skewed
- #tag #GLOSS918 *Weight decay*
  - Used instead of L2 reg when using ADAM
  - W = W - lr * lam * W (After gradient update)
    - in other words we literally shrink the weight size by a small factor every time step 
      - Thereby name of weight decay
- #tag #GLOSS685 *Embeddings*
  - For number of tokens in vocabulary V and embed vector dimensionality d:
    - Embedding matrix is Vxd
    - Just do lookup with token index 
    - similarities and disimilarities in meaning are learned during training
- #tag #GLOSS826 *Class GPT*
  - Is the main nn.module which contains a nn.moduleDict for all submodules
  - I'm not finding the usage of GPT.forward() anywhere so I assume it's something that we overwrite from nn.module, that's run when model.eval() is used
- #tag #GLOSSaa2 *torcharray.shape vs torcharray.size*
  - They are equivalent
- #tag #GLOSSaac *How multi-head attention works in practice*
  - In model.py there isnt one set of WQ WK and WV matrices per embedding vector. 
    - Instead, the output of these is split by the number heads -> each head processes a sub-part of the embedding vector
- #tag #GLOSS038 *.view()*
  - The better version of .reshape in pytorch
- #tag #GLOSSfa5 *BPE / Byte-pair encoding*
  - The stuff Sullivan mentioned where most used combinations of text becomes tokens
- #tag #GLOSS256 *Why 4x embedding size and back again in MLP*
  - Just standard acc to GPT
    - But also to get higher dim and then get non-linearity with act-function
- #tag #GLOSS7af *Projection weights*
  - 
- #tag #GLOSS3bb *Relu vs Gelu*
  - Gelu has a a bit more nonlinearity: Small negative inputs become small negative outputs, but bigger negative values are still zero
    - Appearantly very important for transformer models or deep networks because a lot of info gets annihilated by relu otherwise
    - So performs better, but takes a bit longer to train










##### Other:


asd