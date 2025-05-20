import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from Utils.data_handling import create_DataLoader, Vocabulary
from Utils.embeddings_Loader import load_glove_embeddings
from LSTM.LSTM import LSTM, train_lstm
import random
import numpy as np
import os

def performExperimentLSTM(dim_hidden=256, n_layers=2, tokenization_level='char', tokenization_type='nltk_shakespeare', embedding_type='one-hot',
                          fine_tune_embedding=False, seq_length=50, init_lr=0.001, min_lr=0.0001,
                          trial=1, experiment_dir='./Baseline_LSTM', log_file='training_log_BaselineLSTM.txt'):
    # ================ Hyper-parameters ================ #
    #--- Fixed Hyperparameters --- #
    seq_length = seq_length # Length of sequence fed into network
    dim_hidden = dim_hidden # Dimension of hidden nodes
    n_layers = n_layers # Number of layers in (stacked) LSTM
    dropout = 0.15 # Determines dropout rate (might become nuisance parameter later)
    persistent_hidden_state = True # Stays fixed for all experiments
    stride = seq_length # Stays fixed for all experiments
    traverse = 'once' # Stays fixed for all experiments, as recommended in data_handling documentation
    embedding_type = embedding_type # or 'GloVe'
    fine_tune_embedding = fine_tune_embedding
    embedding_dim = None # Is fixed through embedding type later, will play a role if we train embedding layer OR use prettrained embeddings
    tokenization_level = tokenization_level # could alternatively be 'word' (only applicable for 2nd experiment)
    tokenization_type = tokenization_type # if level = 'word' => choose that to be 'nltk_shakespeare' (or later BPE)

    learning_rate_decay = 'cosine'

    batch_size = 20 # Works relatively well with the shakespeare data, default buckets and stride = seq_length
    lam = 0 # L2-Regularization parameter (is always going to be 0)
    shuffle = True # Shuffle data between epochs (always true for us)
    optimizer_algo = 'ADAM' # TODO: TEST THAT AGAINST SGD

    #--- Scientific Parameters---#

    #--- Nuisance parameters ---#
    init_lr = init_lr # initial learning rate at beginning of the decay
    min_lr = min_lr # minimum learning rate at the end of the decay

    #--- other parameters ---#
    save = False # Save or not save the model
    n_epochs = 5 # Is fixed over all experiments

    # ====================== Data ===================== #
    current_dir = os.getcwd()
    print(f"You are in: {current_dir}")
    training_file = os.path.join(current_dir, 'Data', 'train_shakespeare_full_corpus.txt')
    training_file = os.path.abspath(training_file)  # resolves to full path

    val_file = os.path.join(current_dir, 'Data', 'val_shakespeare_full_corpus.txt')
    val_file = os.path.abspath(val_file)  # resolves to full path

    test_file = os.path.join(current_dir, 'Data', 'test_shakespeare_full_corpus.txt')
    test_file = os.path.abspath(test_file)  # resolves to full path

    # Set up experiment folder
    experiment_dir = experiment_dir
    log_file = log_file

    # Set up embeddings folder
    if embedding_type == 'glove':
        if tokenization_type == 'BPE':
            embedding_file = os.path.join(current_dir, 'Data', 'Glove_vectors_BPE.txt')
        else:
            embedding_file = os.path.join(current_dir, 'Data', 'Glove_vectors.txt')

    # ==================== RANDOM FIXING ==================== #
    # Reproducibility
    # Read a bit more here -- https://pytorch.org/docs/stable/notes/randomness.html
    random.seed(5719)
    np.random.seed(5719)
    torch.manual_seed(5719)
    #torch.use_deterministic_algorithms(True)

    # ==================== DATA PREP ==================== #
    if persistent_hidden_state:
        advanced_batching = True
    else:
        advanced_batching = False

    train_dataloader, train_dataset, vocab = create_DataLoader(filename=training_file, batch_size=batch_size,
                      seq_Length=seq_length, shuffle=shuffle, stride=stride,
                      level=tokenization_level, tokenization=tokenization_type,
                      vocab=None, record_tokens=True, advanced_batching=advanced_batching, boundaries=None,
                      traverse='once')

    val_dataloader, val_dataset, vocab = create_DataLoader(filename=val_file, batch_size=batch_size,
                      seq_Length=seq_length, shuffle=False, stride=stride,
                      level=tokenization_level, tokenization=tokenization_type,
                      vocab=vocab, record_tokens=False, advanced_batching=advanced_batching, boundaries=None,
                      traverse='once')

    # Note if it works correctly the dataset will contain the correct hidden states
    # for each play in each epoch. The play id is its index in data.samples

    # Prepare the hidden states:
    if persistent_hidden_state:
        hidden_states = torch.zeros(train_dataset.n_plays, n_layers, dim_hidden) # size: (n_plays, n_layers, hidden_dim)
        hidden_states_val = torch.zeros(val_dataset.n_plays, n_layers, dim_hidden) # size: (n_plays, n_layers, hidden_dim)
        cell_states = torch.zeros(train_dataset.n_plays, n_layers, dim_hidden) # size: (n_plays, n_layers, hidden_dim)
        cell_states_val = torch.zeros(val_dataset.n_plays, n_layers, dim_hidden) # size: (n_plays, n_layers, hidden_dim)

    # Extract vocabulary size:
    vocab_size = vocab.vocab_size

    # Create embedding
    if embedding_type == 'one-hot':
        # Create one_hot embedding
        embedding = torch.eye(vocab_size)
        embedding_dim = vocab_size
    elif embedding_type == 'glove':
        #embedding = torch.zeros((vocab_size, embedding_dim))
        embedding_dim, embedding = load_glove_embeddings(embedding_file=embedding_file, vocab=vocab)
    else:
        raise NotImplementedError('Invalid embedding_type given')

    # ==================== TRAINING ==================== #
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Running on", device)

    model = LSTM(vocab_size=vocab_size, embedding_dim=embedding_dim,
                 hidden_size=dim_hidden, output_size=vocab_size, num_layers=n_layers,
                 dropout=dropout, use_pretrained_embedding=True, pretrained_weights=embedding, persistent_hidden_state=persistent_hidden_state, fine_tune_embedding=fine_tune_embedding)

    # Create optimizer
    if optimizer_algo == 'ADAM':
        optimizer = torch.optim.Adam(model.parameters(), lr=init_lr)
    elif optimizer_algo == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=init_lr)

    # Create the scheduler
    if learning_rate_decay == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=min_lr)
    elif learning_rate_decay == 'lin-decay':
        raise NotImplementedError('Linear decay is not implemented yet.')

    # Train the model
    history, model = train_lstm(model, train_dataloader, val_dataloader, optimizer, persistent_hidden_state=True, hidden_state=hidden_states, hidden_state_val=hidden_states_val, cell_state=cell_states, cell_state_val=cell_states_val,
                device=device, num_epochs=n_epochs, print_every=100, val_every_n_steps=500, scheduler=scheduler, experiment_dir=experiment_dir, log_file=log_file, 
                trial=trial)

    return model, history, vocab, train_dataset, val_dataset


if __name__ == '__main__':
    performExperimentLSTM(tokenization_level='word', embedding_type='glove', fine_tune_embedding=True, tokenization_type='BPE')
    performExperimentLSTM(tokenization_level='char', embedding_type='one-hot')