trial = 125

max_iters = 400
out_dir = 'baseline_model_checkpoints_our_loader-shakespeare'
eval_interval = 200 # keep frequent because we'll overfit
eval_interval = 40 # keep frequent because we'll overfit
eval_iters = 20
log_interval = 40 # don't print too too often
log_interval = 20 # don't print too too often

always_save_checkpoint = True 
init_from = 'scratch' # 'scratch' or 'resume'

wandb_log = False # override via command line if you like
wandb_project = 'train_nanogpt_shakespeare'
wandb_run_name = 'baseline_GPT_model'

# --------- #tag *Our data loader* ---------
stride = 1
level = 'word'
tokenization = 'BPE' #'nltk_shakespeare' or 'BPE'
traverse = 'once'
emb_dim_is_token_dim = False

# File paths
#train_file = 'Data/train_shakespeare_full_corpus.txt' 
train_file = '../../Data/train_shakespeare_full_corpus.txt'
#val_file = 'Data/val_shakespeare_full_corpus.txt'
val_file = '../../Data/val_shakespeare_full_corpus.txt'

dataset = '' #We don't want stuff from other dataset to bleed in here
gradient_accumulation_steps = 1
batch_size = 20 #(Was 64 before)



# --------- #tag *Some of the model parameters (others are inferred)* ---------
block_size = 67 # context of up to 256 previous tokens
n_layer = 2
n_head = 5
n_embd = 384 # Will be changed anyway by our trainer
dropout = 0.15
bias = False # do we use bias inside LayerNorm and Linear layers?

# --------- #tag *Adam and opt. params* ---------
learning_rate = 0.01742513522746501 # NOTE this is max LR if using decay schedule
beta1 = 0.9
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

# --------- #tag *LR decay* ---------
decay_lr = True # whether to decay the learning rate
lr_decay_iters = max_iters # make equal to max_iters usually
min_lr = learning_rate / 10 # learning_rate / 10 usually

# --------- #tag *Other that we probably want to set to not have effect* ---------
weight_decay = 0.00391387325090902
grad_clip = 0.0 # clip gradients at this value, or disable if == 0.0
warmup_iters = 0

# --------- #tag *Device* ---------

device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
compile = False # use PyTorch 2.0 to compile the model to be faster

# --------- #tag *Other settings* ---------
smooth_loss_factor = 0.0 # 0.5 of new losses get added, 0.0 to only keep new

# --------- #tag *Final stretch* ---------
pretrained_wte = True
finetune_wte = True

# on macbook also add
# device = 'cpu'  # run on cpu only
# compile = False # do not torch compile the model
