trial = 8

max_iters = 1000
out_dir = 'baseline_model_checkpoints_our_loader-shakespeare'
eval_interval = 100 # keep frequent because we'll overfit
eval_iters = 20
log_interval = 20 # don't print too too often

always_save_checkpoint = True 
init_from = 'scratch' # 'scratch' or 'resume'

wandb_log = False # override via command line if you like
wandb_project = 'train_nanogpt_shakespeare'
wandb_run_name = 'baseline_GPT_model'

# --------- #tag *Our data loader* ---------
stride = 1
level = 'char'
tokenization = 'nltk_shakespeare'
traverse = 'once'
embedding_dim = None  # Will be set based on vocabulary size
emb_dim_is_token_dim = True 

# File paths
#train_file = 'Data/train_shakespeare_full_corpus.txt' 
train_file = '../../Data/train_shakespeare_full_corpus.txt'
#val_file = 'Data/val_shakespeare_full_corpus.txt'
val_file = '../../Data/val_shakespeare_full_corpus.txt'

dataset = 'train_nanogpt_shakespeare' ##tag #TODO370 Get rid of to avoid confusion later
gradient_accumulation_steps = 1
batch_size = 64



# --------- #tag *Some of the model parameters (others are inferred)* ---------
block_size = 50 # context of up to 256 previous tokens
n_layer = 8
n_head = 8
n_embd = 384
dropout = 0.0
bias = False # do we use bias inside LayerNorm and Linear layers?

# --------- #tag *Adam and opt. params* ---------
learning_rate = 1e-2 # NOTE this is max LR if using decay schedule
beta1 = 0.9
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

# --------- #tag *LR decay* ---------
decay_lr = True # whether to decay the learning rate
lr_decay_iters = max_iters # make equal to max_iters usually
min_lr = learning_rate / 100 # learning_rate / 10 usually

# --------- #tag *Other that we probably want to set to not have effect* ---------
weight_decay = 0.0
grad_clip = 0.0 # clip gradients at this value, or disable if == 0.0
warmup_iters = 0

# --------- #tag *Device* ---------

device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
compile = False # use PyTorch 2.0 to compile the model to be faster


smooth_loss_factor = 0.9 # 0.01 of new losses get added

# on macbook also add
# device = 'cpu'  # run on cpu only
# compile = False # do not torch compile the model
