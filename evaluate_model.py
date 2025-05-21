# from Utils.metrics import compute_bleu_score


import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
import os
from torchtext.data.metrics import bleu_score
import random
import numpy as np

from Utils.data_handling import create_DataLoader, Vocabulary
from Utils.embeddings_Loader import load_glove_embeddings
from RNN.RNN import RNN
from LSTM.LSTM import LSTM
from Utils.generateFormModels import sample_model_indefinitely, generate_text
from Utils.shakespeare_parser import ENDTOKEN


def eval_model_bleu(model_path, model_type='RNN', sampling_strategy='nucleus', top_p=0.5, temperature = 1):
    '''Retrieves the BLEU score on the test sonnets, only works for advanced batching on!!'''
    # get model and the data_structures
    model, vocab, test_dataset = get_model(model_path, model_type)

    # Right now i only want to evaluate Sonnets
    sonnet_marker_id = vocab.lookup_id("<<SONNETS>>")
    end_id = vocab.lookup_id(ENDTOKEN)

    # NOTE: THIS LINE WOULD BREAK WITH BPE tokenization!!!
    linebreak_marker_id = vocab.lookup_id("\n")

    complete_sonnets = [] # complete sonnets
    seeding_sonnets = [] # gather only the first line of each sonnet
    reference_sonnets = [] # gather only the last lines of each sonnet
    for play in test_dataset.samples:
        # Check if its a sonnet
        if play[0][0] == sonnet_marker_id:
            # Gather the play into a 1D list
            complete_play = [token for sample in play for token in sample]
            complete_sonnets.append(complete_play)
            # Find where the sonnet first line ends, which is always after the 5th linebreak
            idx_start = nth_index(complete_play, linebreak_marker_id, 4)
            seeding_play = complete_play[:idx_start+1] # to include linebreak
            seeding_sonnets.append(seeding_play)
            reference_play = complete_play[idx_start+2:]
            reference_sonnets.append(reference_play)

    # Turn the reference_sonnets list already back to strings
    reference_sonnets_token = [[vocab.lookup_token(id) for id in sonnet] for sonnet in reference_sonnets]

    # Put model in evaluation mode
    model.eval()
    # Check device
    device = next(model.parameters()).device

    bleu_scores_list = []
    with torch.no_grad():
        # For each sonnet get the id list of the generated sonnet
        for i, sonnet_seed in enumerate(seeding_sonnets):
            generated_ids_sonnet = sample_model_indefinitely(model, sonnet_seed, max_length=200, end_id=end_id, sampling_strategy=sampling_strategy, top_p=top_p, temperature=temperature, device=device)

            # Turn the ids back into strings 
            generated_token = [vocab.lookup_token(id) for id in generated_ids_sonnet]

            # Calculate bleu_score
            score = bleu_score(generated_token, reference_sonnets_token[i])
            bleu_scores_list.append(score)
    
    # Calculate mean bleu score
    bleu_avg = sum(bleu_scores_list) / len(bleu_scores_list)

    return bleu_avg
        




def get_model(model_path, model_type='RNN'):
    if model_type == 'RNN':
        # Hardcode the model that was the best here, its not nice but it will do
        # Trial 8, i.e. the model with h=384 and n_layers = 2 performed best
        # Set experiment hyerparameters
        trial = 8
        # Define scientic parameter
        dim_hidden = 384

        # Define nuisance parameters
        init_lr = 0.0008473975597328964

        # seq_length = 50
        seq_length = 95


        # Define general parameters
        min_lr = 0.001
        experiment_dir = './Exp1_RNN'
        log_file = 'training_log_Exp_1_RNN.txt'

        model, vocab, test_dataset = reload_RNN(dim_hidden = dim_hidden, n_layers= 2, tokenization_level='char', embedding_type ='one-hot', 
                         fine_tune_embedding = False, seq_length = seq_length, init_lr= init_lr, min_lr = min_lr, 
                         trial=trial, experiment_dir = experiment_dir, log_file = log_file)
        
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    return model, vocab, test_dataset




def nth_index(lst, value, n):
    count = 0
    for i, x in enumerate(lst):
        if x == value:
            count += 1
            if count == n:
                return i
    return -1  # or raise an exception if not found



def reload_RNN(dim_hidden = 256, n_layers= 2, tokenization_level='char',tokenization_type='nltk_shakespeare', embedding_type ='one-hot', 
                         fine_tune_embedding = False, seq_length = 50, init_lr= 0.001, min_lr = 0.0001, 
                         trial=1, experiment_dir = './Baseline_RNN', log_file = 'training_log_BaselineRNN.txt'):
    '''This funtion reloads the RNN, in its current form this function is the complete overkill, but I am too tired to clean that up...'''
    # ================ Hyper-parameters ================ #
    #--- Fixed Hyperparameters --- # 
    seq_length = seq_length # Length of sequence fed into network
    dim_hidden = dim_hidden # Dimension of hidden nodes
    activation_func = 'tanh' # Activation function (stays fixed for all experiments) ('tanh' or 'relu')
    n_layers = n_layers # Number of layers in (stacked) RNN
    dropout = 0.15 # Determines dropout rate (might become nuisance parameter later)
    persistent_hidden_state = True # Stays fixed for all experiments
    stride = seq_length # Stays fixed for all experiments
    traverse = 'once' # Stays fixed for all experiments, as recommended in data_handling documentation
    embedding_type = embedding_type # or 'glove'
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
    init_lr = init_lr  # initial learning rate at beginning of the decay
    min_lr = min_lr # minimum learning rate at the end of the decay

    #--- other paratemers ---#
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
                      level=tokenization_level,tokenization=tokenization_type,
                      vocab=None,record_tokens=True,advanced_batching=advanced_batching,boundaries=None, 
                      traverse='once')
    
    val_dataloader, val_dataset, vocab = create_DataLoader(filename=val_file, batch_size=batch_size, 
                      seq_Length=seq_length, shuffle=shuffle, stride=stride,
                      level=tokenization_level,tokenization=tokenization_type,
                      vocab=vocab,record_tokens=False, advanced_batching=advanced_batching,boundaries=None, 
                      traverse='once')

    test_dataloader, test_dataset, vocab = create_DataLoader(filename=test_file, batch_size=batch_size, 
                      seq_Length=seq_length, shuffle=shuffle, stride=stride,
                      level=tokenization_level,tokenization=tokenization_type,
                      vocab=vocab,record_tokens=False, advanced_batching=advanced_batching,boundaries=None, 
                      traverse='once')
    
    # Note if it works correctly the dataset will contain the correct hidden states 
    # for each play in each epoch. The play id is its index in data.samples

    # Prepare the hidden states:
    if persistent_hidden_state:
        hidden_states = torch.zeros(train_dataset.n_plays,n_layers,dim_hidden) # size: (n_plays, n_layers, hidden_dim)
        hidden_states_val = torch.zeros(val_dataset.n_plays,n_layers,dim_hidden)
    
    # Extract vocabulary size:
    vocab_size = vocab.vocab_size

    # Create embedding
    if embedding_type == 'one-hot':
        # Create one_hot embedding
        embedding = torch.eye(vocab_size)
        embedding_dim = vocab_size
    elif embedding_type == 'glove':
        # TODO: LOAD GLOVE EMBEDDINGS
        # embedding = torch.zeros(vocab_size, embedding_dim)
        embedding_dim, embedding = load_glove_embeddings(embedding_file=embedding_file, vocab=vocab)
    else:
        raise NotImplementedError('Invalid embedding_type given')
    
    


    # ==================== TRAINING ==================== #
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Running on", device)

    model = RNN(vocab_size=vocab_size, embedding_dim=embedding_dim, 
                hidden_size=dim_hidden, output_size=vocab_size, num_layers=n_layers, 
                activation_function=activation_func, dropout_rate=dropout, 
                use_pretrained_embedding=True, pretrained_weights=embedding, persistent_hidden_state=True, fine_tune_embedding=fine_tune_embedding)
    
    # Create optimizer
    if optimizer_algo == 'ADAM':
        optimizer = torch.optim.Adam(model.parameters(), lr=init_lr)
    elif optimizer_algo == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=init_lr)

    # Create the scheduler
    # Estimate how many update steps there are. NOTE: This is an upper bound as we set drop_last to true, the likely update steps will be shorter
    # total_update_steps = n_epochs*train_dataset.n_samples // batch_size
    if learning_rate_decay == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=min_lr)
    elif learning_rate_decay == 'lin-decay':
        raise NotImplementedError('Linear decay is not implemented yet.')
    
    return model, vocab, test_dataset





if __name__ == '__main__':
    model_path = 'Exp1_RNN/chekpoints/trial_8/model_best_epoch_4.pt'


    model, vocab, test_dataset = get_model(model_path=model_path, model_type='RNN')

    text = generate_text(model, start_str='<<SONNETS>>\n13\n\nO that you were your self, but love you are\n', length=2000,
                         vocab=vocab,device='cpu',temperature=1,tokenization_level='char', tokenization_type='char', 
                         sampling_strategy='temperature')
    print(text)
    eval_model_bleu(model_path=model_path, model_type='RNN')

