import numpy as np
import torch
from Utils.data_handling import Vocabulary


from Utils.data_handling import SPECIAL_TOKENS
from Utils.data_handling import PADDING_SYMBOL
from Utils.data_handling import UNKNOWN_SYMBOL

def load_glove_embeddings(embedding_file: str, vocab: Vocabulary) :
    """
    Reads pre-made embeddings from a file, adds the words to the dictionary and assigns all tokens
    that are not in the glove embedding a random embedding vector (note, this will change the vocabulary dimension)

    Inputs:
        embedding_file: 'Filepath to to txt file'
        vocab: Goes from a pre-determined vocabulary and tries to embedd this. Additionally adds all words to the embedding that are in the embedding file but
        not in the vocabulary

    Returns:
        embedding_dim: dimension of the embedding
        embedding: torch tensor of size (vocab_size, emebedding_dim)
    """
    # Get the current size of the vocabulary and init their embedding
    N = vocab.vocab_size
    embeddings = [0]*N

    whitespace = False
    line_break = False
    with open(embedding_file, 'r', encoding='utf-8') as f:
        text_list = f.readlines()
        for i, line in enumerate(text_list):
            if line[0] == ' ' and not line_break:
                whitespace = True
            elif line[0] == '\n':
                line_break = True
                continue

            data = line.split()

            if whitespace:
                data = [' '] + data
                whitespace = False
            elif line_break:
                data = ['\n'] + data
                line_break = False
            
            token = data[0]
            if token not in SPECIAL_TOKENS:
                token = data[0].lower()
            
            # If token from glove embedding is not in vocabulary, add it
            if not vocab.token_exist(token):
                vocab.add_token(token)
                embeddings.append(0)
            
            vec = [float(x) for x in data[1:]]
            embeddings[vocab.lookup_id(token)] = vec
            if i ==0:
                D = len(vec)
            assert len(vec) == D ;f'Missed special token at iteration {i}'
        # Just get the dimension of the embedding from the last vector we processed (does not matter from where)
        D = len(vec)
    # Add a '0' embedding for the padding symbol, as the learned padding symbol is not really the same as the padding we use
    embeddings[vocab.lookup_id(PADDING_SYMBOL)] = [0]*D
    # Add a '-1' embedding for the unknown symbol as that should have not occured in the training set
    embeddings[vocab.lookup_id(UNKNOWN_SYMBOL)] = [-1]*D
    # Check if there are words that did not have a ready-made Glove embedding
    # For these words, add a random vector
    for token, id in vocab.token2id.items():
        if embeddings[id] == 0 :
            embeddings[id] = (np.random.random(D)-0.5).tolist()

    embeddings = torch.tensor(embeddings) # dim # (vocab_size, D) so (vocab_size, embedding_dim)
    return D, embeddings






def main():
    test_vocab = Vocabulary()
    test_vocab.add_token('SULLIVAN')
    D, embeddings = load_glove_embeddings('Data/Glove_vectors.txt', test_vocab)

    return -1

if __name__ == '__main__' :
    main()    