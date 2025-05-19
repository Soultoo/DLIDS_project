import os
import math
import random
import nltk
import numpy as np
import numpy.random as rand
import os.path
import argparse
from collections import defaultdict
from transformers import GPT2Tokenizer

from Utils.data_handling import Vocabulary, ShakespeareTokenizer

from Utils.data_handling import PADDING_SYMBOL
from Utils.data_handling import SPECIAL_TOKENS

"""
Python implementation of the Glove training algorithm  inspired by DD2417 - Language Engineering
"""
class Glove:
    def __init__( self, continue_training, left_window_size, right_window_size ) :
        self.left_window_size = left_window_size
        self.right_window_size = right_window_size

        # Save the vocabulary that you want to train the vector on
        self.vocab = Vocabulary()
        
        # Mapping from focus words to neighbours to counts (called X 
        # to be consistent with the notation in the Glove paper).
        self.X = defaultdict(lambda: defaultdict(int))
        
        # Mapping from word IDs to (focus) word vectors. (called w_vector 
        # to be consistent with the notation in the Glove paper).
        self.w_vector = defaultdict(lambda: None)

        # Mapping from word IDs to (context) word vectors (called w_tilde_vector
        # to be consistent with the notation in the Glove paper)
        self.w_tilde_vector = defaultdict(lambda: None)


        # Total number of tokens processed
        self.tokens_processed = 0

        # Dimension of word vectors.
        self.dimension = 50

        # Cutoff for gradient descent.
        self.epsilon = 0.01

        # Initial learning rate.
        self.learning_rate = 0.05

        # The number of times we can tolerate that loss increases
        self.patience = 5
    
        # Padding at the beginning and end of the token stream
        self.pad_word = PADDING_SYMBOL

        # Temporary file used for storing the model
        self.temp_file = "Data/temp__.txt"
    
        # Possibly continue training from pretrained vectors 
        if continue_training and os.path.exists(self.temp_file):
            self.read_temp_file( self.temp_file )
        
        ## Added for debugging
        seed = 42
        self.rng = np.random.default_rng(seed=seed) # plus change in get_word_id from random.rand to rng.rand in 2 instances
        # plus in my trainig class to smaple via choice()

        self.init_params()
    


    #------------------------------------------------------------
    #
    #  Methods for processing all files and computing all counts
    #

    def init_params(self):
        for id, token in enumerate(self.vocab.id2token):
                w = self.rng.random(self.dimension)-0.5
                self.w_vector[id] = w
                w_tilde = self.rng.random(self.dimension)-0.5
                self.w_tilde_vector[id] = w_tilde


    def get_word_id( self, token ) :
        """ 
        Returns the word ID for a given word. If the word has not
        been encountered before, the necessary data structures for
        that word are initialized.
        We use this a wrapper around the vocabularies addtoken() function as we need to initialize a few more structures
        in case the token has never been seen before
        """
        if token not in SPECIAL_TOKENS:
            token = token.lower()
        if self.vocab.token_exist(token):
            return self.vocab.add_token(token)
    
        else : 
            # This word has never been encountered before. Add the word Init all necessary
            # data structures.

            id = self.vocab.add_token(token)

            # Initialize arrays with random numbers in [-0.5,0.5].
            w = self.rng.random(self.dimension)-0.5
            self.w_vector[id] = w
            w_tilde = self.rng.random(self.dimension)-0.5
            self.w_tilde_vector[id] = w_tilde
            return id


    def update_counts( self, focus_word, context ) :
        """
        Updates counts based on the local context window.
        """
        focus_word_id = self.get_word_id( focus_word )
        all_context_words = self.X[focus_word_id]
        if all_context_words == None :
            all_context_words = defaultdict(int)
            self.X[focus_word_id] = all_context_words
        for idx in context :
            count = all_context_words[idx]
            if count == None :
                count = 0
            all_context_words[idx] = count+1


    def get_context(self, i):
        """
        Returns the context of token no i as a list of word indices.
        
        :param      i:     Index of the focus word in the list of tokens
        :type       i:     int
        """

        # USE YOUR CODE FROM THE RANDOM INDEXING TASK
        # NOTE: The list self.tokens holds all the tokenized words in the correct order
        # Creating the context is thus as easy as accessing them in the self tokens and getting their ids
        context = []

        # Problem: The first left_windowsize tokens (i=0,1) and the last right_window_size tokens will get 
        # either a smaller context OR we need to use the padding token (which is what i do here)

        # Left context
        left_context_value = i-self.left_window_size

        # Check if padding is needed
        if left_context_value < 0:
            # Add padding to word so that we get the total
            while left_context_value < 0:
                id = self.get_word_id(self.pad_word)
                context.append(id)
                left_context_value += 1

        # Add rest of left context
        for token in self.tokens[left_context_value:i]:
            id = self.get_word_id(token)
            context.append(id)

        # Right context
        right_context_value = i+1 + self.right_window_size
    
        # Add right context (if right context needs padding, the list slice wil just be shorter.
        # As slicing allows indeces to be over the lenght of the list)
        for token in self.tokens[i+1:right_context_value]:
            id = self.get_word_id(token)
            context.append(id)

        # Check if padding is needed
        if right_context_value > len(self.tokens):
            # Add padding to the words contex
            while right_context_value > len(self.tokens):
                id = self.get_word_id(self.pad_word)
                context.append(id)
                right_context_value -= 1

        return context


    def process_files( self, file_or_dir, tokenization='nltk_shakespeare' ) :
        """
        This function recursively processes all files in a directory.
        
        Each file is tokenized and the tokens are put in the list
        self.tokens. Then each token is processed through the methods
        'get_context' and 'update_counts' above.
        """
        if os.path.isdir( file_or_dir ) :
            for root,dirs,files in os.walk( file_or_dir ) :
                for file in files :
                    self.process_files( os.path.join(root, file ))
        else :
            print( file_or_dir )
            with open( file_or_dir, mode='r', encoding='utf-8', errors='ignore' ) as stream:
                text = stream.read()

            if tokenization == 'nltk_word':
                try :
                    nltk.word_tokenize("hi there.")
                except LookupError:
                    nltk.download('punkt')
                self.tokenize = nltk.word_tokenize
            elif tokenization == 'nltk_shakespeare':
                self.tokenize = ShakespeareTokenizer().tokenize
            elif tokenization == 'BPE':
                # Check if we already created our adapted GPT-2 BPE tokenizer
                if os.path.isdir("./custom_gpt2_tokenizer"):
                    tokenizer = GPT2Tokenizer.from_pretrained("./custom_gpt2_tokenizer")
                else:
                    # Load pre-trained GPT-2 tokenizer
                    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

                    # Add special tokens
                    special_tokens_dict = {"additional_special_tokens": SPECIAL_TOKENS}
                    tokenizer.add_special_tokens(special_tokens_dict)

                    # Save for later use
                    tokenizer.save_pretrained("./custom_gpt2_tokenizer")

                self.tokenize = tokenizer.tokenize

            
            self.tokens = self.tokenize(text) # tokenizes the text and return a list of tokens
            
            for i, token in enumerate(self.tokens) :
                self.tokens_processed += 1
                context = self.get_context(i)
                self.update_counts(token, context)
                if self.tokens_processed % 10000 == 0 :
                    print( 'Processed', "{:,}".format(self.tokens_processed), 'tokens' )

        
    #
    #  End of methods for processing all files and computing all counts
    #
    #------------------------------------------------------------

    #------------------------------------------------------------
    #
    #   Loss function, gradient descent, etc.
    #

    def f( self, count ) :
        """
        The "f" function from the Glove article
        """
        if count<100 :
            ratio = count/100.0
            return math.pow( ratio, 0.75 )
        return 1.0
    

    def loss( self ) :
        """
        Returns the total loss, computed from all the vectors.
        """
        loss = 0
        # REPLACE WITH YOUR CODE 
        for i in self.X:
            for j in self.X[i]:
                loss = loss + self.f(self.X[i][j])*(self.w_vector[i].T @ self.w_tilde_vector[j] - np.log(self.X[i][j]))**2
        
        loss = loss/2
        return loss

    

    def compute_gradient(self, i, j) :
        """
        Computes the gradient of the loss function w.r.t. w_vector[i] and
        w.r.t w_tilde_vector[j]
        
        Returns wi_vector_grad, wj_tilde_vector_grad
        """

        # REPLACE WITH YOUR CODE 
        wi_vector_grad = self.f(self.X[i][j])*( self.w_vector[i].T @ self.w_tilde_vector[j] -np.log(self.X[i][j])) * self.w_tilde_vector[j]
        
        wj_tilde_vector_grad = self.f(self.X[i][j])*( self.w_vector[i].T @ self.w_tilde_vector[j] -np.log(self.X[i][j])) * self.w_vector[i]
        return wi_vector_grad, wj_tilde_vector_grad
        
        
    def train(self) :
        """
        Trains the vectors using stochastic gradient descent
        """
        iterations = 0

        # YOUR CODE HERE
        # With the way self.X[i][j] is populated, i think it does not have any entry which keys will return zero
        # i.e. the list generated by nonzero_pairs = [(i, j) for i in self.X for j in self.X[i] if self.X[i][j] > 0]
        # will return all the keys that are stored in X_ij
        # => Implies that we can sample from i and j directly as X_ij will always have a value > 0 
        # IF it exists. 

        # Calculate sampling distribution 
        row_weights = [sum(self.X[i].values()) for i in self.X] # TODO: DAS KANN IRRGENDWIE NICHT
        row_ids = list(self.X.keys())
        row_probs = [row_weights[i] / sum(row_weights) for i in range(len(row_ids))]

        prev_losses = []
        
        while ( self.patience > 0 ) :

            # YOUR CODE HERE
            # TODO: PRÜFE DAS ALLE BITTE DA KANN MAN NOCH EINIGES VERBESSERN
            # 1. Sample i based on co-occurrence frequency
            i_idx = self.rng.choice(range(len(row_ids)), p=row_probs)# [0]
            i = row_ids[i_idx]

            # Sample j such that X[i][j] > 0
            j_candidates = list(self.X[i].keys())
            if not j_candidates:
                continue
            j = self.rng.choice(j_candidates)

            # 2. Compute gradients
            wi_grad, wj_grad = self.compute_gradient(i, j)

            # 3. Update vectors
            self.w_vector[i] -= self.learning_rate * wi_grad
            self.w_tilde_vector[j] -= self.learning_rate * wj_grad

            iterations += 1

            # 4. Evaluate loss
            if iterations % 1000000 == 0:
                current_loss = self.loss()
                print(f"Iteration {iterations}, loss = {current_loss:.4f}")
                prev_losses.append(current_loss)
                if len(prev_losses) > 1 and current_loss > prev_losses[-2]:
                    self.patience -= 1
                else:
                    self.patience = 5  # Reset if not increasing


            if iterations%1000000 == 0 :
                self.write_word_vectors_to_file( self.outputfile )
                self.write_temp_file( self.temp_file )
                self.learning_rate *= 0.99
                

    #
    #  End of loss function, gradient descent, etc.
    #
    #-------------------------------------------------------

    #-------------------------------------------------------
    #
    #  I/O
    #

    def write_word_vectors_to_file( self, filename ) :
        """
        Writes the vectors to file. These are the vectors you would
        export and use in another application.
        """
        with open(filename, 'w') as f:
            for idx, token in enumerate(self.vocab.id2token) :
                f.write('{} '.format( token ))
                for i in self.w_vector[idx] :
                    f.write('{} '.format( i ))
                f.write( '\n' )
        f.close()


    def write_temp_file( self, filename ) :
        """
        Saves the state of the computation to file, so that
        training can be resumed later.
        """
        with open(filename, 'w') as f:
            f.write('{} '.format( self.learning_rate ))
            f.write( '\n' )
            for idx, token in enumerate(self.vocab.id2token) :
                f.write('{} '.format( token ))
                for i in list(self.w_vector[idx]) :
                    f.write('{} '.format( i ))
                for i in list(self.w_tilde_vector[idx]) :
                    f.write('{} '.format( i ))
                f.write( '\n' )
        f.close()


    def read_temp_file(self, fname):
        """
        Reads the partially trained model from file, so
        that training can be resumed.
        """

        with open(fname) as f:
            self.learning_rate = float(f.readline())
            for line in f:
                data = line.split()
                w = data[0]
                i = self.vocab.add_token(w)

                vec = np.array([float(x) for x in data[1:self.dimension+1]])
                self.w_vector[i] = vec
                vec = np.array([float(x) for x in data[self.dimension+1:]])
                self.w_tilde_vector[i] = vec

        f.close()
        self.dimension = len( self.w_vector[0] )

        
    #
    #  End of I/O
    #
    #-------------------------------------------------------

       
def main() :

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Glove trainer')
    parser.add_argument('--file', '-f', type=str,  default='Data/train_shakespeare_full_corpus.txt', help='The files used in the training.')
    parser.add_argument('--output', '-o', type=str, default='Data/Glove_vectors.txt', help='The file where the vectors are stored.')
    parser.add_argument('--left_window_size', '-lws', type=int, default='2', help='Left context window size')
    parser.add_argument('--right_window_size', '-rws', type=int, default='2', help='Right context window size')
    parser.add_argument('--continue_training', '-c', action='store_true', default=False, help='Continues training from where it was left off.')

    arguments = parser.parse_args()  
    
    glove = Glove(arguments.continue_training, arguments.left_window_size, arguments.right_window_size)
    glove.outputfile = arguments.output
    glove.process_files( arguments.file )
    print( 'Processed', "{:,}".format(glove.tokens_processed), 'tokens' )
    print( 'Found', glove.vocab.vocab_size, 'unique words' )
    glove.train()

        
if __name__ == '__main__' :
    main()    

