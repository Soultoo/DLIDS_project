import torch
from Utils.data_handling import create_DataLoader, Vocabulary
from LSTM.LSTM import LSTM, train_lstm, generate_text
from Utils.metrics import compute_bleu_score
import random
import numpy as np
import os

def performExperimentLSTM():
    # ================ Hyper-parameters ================ #
    #--- Fixed Hyperparameters --- #
    seq_length = 50 # Length of sequence fed into network
    dim_hidden = 256 # Dimension of hidden nodes
    n_layers = 2 # Number of layers in (stacked) LSTM
    dropout = 0.15 # Determines dropout rate (might become nuisance parameter later)
    persistent_hidden_state = True # Stays fixed for all experiments
    stride = seq_length # Stays fixed for all experiments
    traverse = 'once' # Stays fixed for all experiments, as recommended in data_handling documentation
    embedding_type = 'one-hot' # or 'GloVe'
    embedding_dim = None # Is fixed through embedding type later, will play a role if we train embedding layer OR use prettrained embeddings
    tokenization_level = 'char' # could alternatively be 'word' (only applicable for 2nd experiment)
    tokenization_type = None # if level = 'word' => choose that to be 'nltk_shakespeare' (or later BPE)

    learning_rate_decay = 'cosine'

    batch_size = 20 # Works relatively well with the shakespeare data, default buckets and stride = seq_length
    lam = 0 # L2-Regularization parameter (is always going to be 0)
    shuffle = True # Shuffle data between epochs (always true for us)
    optimizer_algo = 'ADAM' # TODO: TEST THAT AGAINST SGD

    #--- Scientific Parameters---#

    #--- Nuisance parameters ---#
    learning_rate = 0.001
    min_lr = 0.0001

    #--- other parameters ---#
    save = False # Save or not save the model
    n_epochs = 1 # Is fixed over all experiments

    # ====================== Data ===================== #
    current_dir = os.getcwd()
    print(f"You are in: {current_dir}")
    training_file = os.path.join(current_dir, 'Data', 'train_shakespeare_full_corpus.txt')
    training_file = os.path.abspath(training_file)  # resolves to full path

    val_file = os.path.join(current_dir, 'Data', 'val_shakespeare_full_corpus.txt')
    val_file = os.path.abspath(val_file)  # resolves to full path

    test_file = os.path.join(current_dir, 'Data', 'test_shakespeare_full_corpus.txt')
    test_file = os.path.abspath(test_file)  # resolves to full path

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

    train_dataloader, dataset, vocab = create_DataLoader(filename=training_file, batch_size=batch_size,
                      seq_Length=seq_length, shuffle=shuffle, stride=stride,
                      level=tokenization_level, tokenization=tokenization_type,
                      vocab=None, record_tokens=True, advanced_batching=advanced_batching, boundaries=None,
                      traverse='once')

    val_dataloader, _, _ = create_DataLoader(filename=val_file, batch_size=batch_size,
                      seq_Length=seq_length, shuffle=False, stride=stride,
                      level=tokenization_level, tokenization=tokenization_type,
                      vocab=vocab, record_tokens=False, advanced_batching=advanced_batching, boundaries=None,
                      traverse='once')

    test_dataloader, _, _ = create_DataLoader(filename=test_file, batch_size=batch_size,
                      seq_Length=seq_length, shuffle=False, stride=stride,
                      level=tokenization_level, tokenization=tokenization_type,
                      vocab=vocab, record_tokens=False, advanced_batching=advanced_batching, boundaries=None,
                      traverse='once')

    # Note if it works correctly the dataset will contain the correct hidden states
    # for each play in each epoch. The play id is its index in data.samples

    # Prepare the hidden states:
    if persistent_hidden_state:
        hidden_states = torch.zeros(dataset.n_plays, n_layers, dim_hidden) # size: (n_plays, n_layers, hidden_dim)
        hidden_states_val = torch.zeros(dataset.n_plays, n_layers, dim_hidden) # size: (n_plays, n_layers, hidden_dim)

    # Extract vocabulary size:
    vocab_size = vocab.vocab_size

    # Create embedding
    if embedding_type == 'one-hot':
        # Create one_hot embedding
        embedding = torch.eye(vocab_size)
        embedding_dim = vocab_size
    elif embedding_type == 'GloVe':
        # TODO: LOAD GLOVE EMBEDDINGS
        embedding = torch.zeros((vocab_size, embedding_dim))
        raise NotImplementedError('GloVe Embeddings are missing so far')
    else:
        raise NotImplementedError('Invalid embedding_type given')

    # ==================== TRAINING ==================== #
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Running on", device)

    model = LSTM(vocab_size=vocab_size, embedding_dim=embedding_dim,
                 hidden_size=dim_hidden, output_size=vocab_size, num_layers=n_layers,
                 dropout=dropout, use_pretrained_embedding=True, pretrained_weights=embedding, persistent_hidden_state=persistent_hidden_state)

    # Create optimizer
    if optimizer_algo == 'ADAM':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_algo == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # Create the scheduler
    if learning_rate_decay == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=min_lr)
    elif learning_rate_decay == 'lin-decay':
        raise NotImplementedError('Linear decay is not implemented yet.')
    else:
        scheduler = None

    # Train the model
    if persistent_hidden_state:
        history, model = train_lstm(model, train_dataloader, val_dataloader, optimizer, persistent_hidden_state=persistent_hidden_state, hidden_state=hidden_states, hidden_state_val=hidden_states_val, device=device, num_epochs=n_epochs, print_every=100, scheduler=scheduler)
    else:
        history, model = train_lstm(model, train_dataloader, val_dataloader, optimizer, device=device, num_epochs=n_epochs, print_every=100, scheduler=scheduler)

    # ==================== TEXT GENERATION ==================== #
    # Read the validation data
    with open(val_file, 'r') as f:
        validation_data = f.read()

    # Extract a snippet from the validation data
    snippet = validation_data[:500]  # Extract the first 500 characters as a snippet
    start_str = ' '.join(snippet.split()[:10])  # Use the first 10 words as the starting string
    reference = ' '.join(snippet.split()[10:20])  # Use the next 10 words as the reference

    generated_text = generate_text(model, start_str, length=100, vocab=vocab, device=device, temperature=0.7)
    print(f"Generated Text: {generated_text}")

    # ==================== EVALUATION ==================== #
    # Compute BLEU score
    hypotheses = [generated_text.split()]  # List of generated sequences
    references = [[reference.split()]]  # List of lists of ground truth sequences
    bleu_score = compute_bleu_score(hypotheses, references)
    print(f"BLEU Score: {bleu_score:.4f}")

if __name__ == '__main__':
    performExperimentLSTM()
