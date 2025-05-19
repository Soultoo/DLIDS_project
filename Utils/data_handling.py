import os
import torch
import nltk
from torch.utils.data import Dataset, DataLoader
from nltk.tokenize import RegexpTokenizer
import random
import math
import itertools
from collections import defaultdict
# from transformers import GPT2Tokenizer


# Import global variables
from Utils.shakespeare_parser import ENDTOKEN, SECTION_MARKER

# For debugging comment line above out and put instead:
#from shakespeare_parser import ENDTOKEN, SECTION_MARKER


## Define globale variables

# The padding symbol will be used to ensure that all tensors in a batch have equal length.
# This only really applied to the end of the processing string (or the end of a piece if the tokenizer can recognize this
# i.e. when tokenizer is either character level or ShakespeareTokenizer) when the leftover tokens are not enough
# to fill the whole window size
PADDING_SYMBOL = ' ' # just a whitespace
UNKNOWN_SYMBOL = "<<UNK>>"
# Collect all special tokens for BPE 
SPECIAL_TOKENS = [ENDTOKEN, PADDING_SYMBOL, UNKNOWN_SYMBOL] + list(SECTION_MARKER)
# Max number of words to be predicted if <END> symbol is not reached
MAX_PREDICTIONS = 20

# Set the boundaries for the buckets. These are my best guess estimates of what could be good buckets
# Basically you get 2 Buckets, one for Sonnets and one for plays
# With these buckets I would recommend the 'once' traverse strategy. 
# For a 'balanced strategy I woud set CHAR_BUCKETS = (2500, 200000)
WORD_BUCKETS = (200, 50000)
CHAR_BUCKETS = (2000, 200000)

class ShakespeareTokenizer:
    '''Custom tokenizer. I built it because nltk.word_tokenizer() excludes newline
    tokens, and we need them for training!! Problem is that RegexpTokenizer does not
    respect punctuations so I had to get tat too and then lastly i realized
     that shakespeare has a lot of weird old words, that should probably be left
      as one token so the model can learn it. Plus i want the section markers << >> to be untouched'''
    def __init__(self):
        # Regex to match:
        # - individual punctuation marks ->  [^\w\s] part
        # - newlines ->   \n part
        # - section markers (<<...>>) ->  <<\w+>> part
        # - words with optional apostrophe-based contractions (e.g., 'tis, o'er) -> (?:[A-Za-z]+(?:['’][a-z]+)?) part
        # => (?:[A-Za-z]+(?:['’][a-z]+)?) : (?:) = () but doesn't keep it in memory for more effiency
        #  and () makes this a logical unit where regex knows to match that
        # [A-Za-z] matches any combination of capital and small letters
        # (?:['’][a-z]+)?): captures an addtional apostrophe and letter (otptional through the ? at the end)
        self.tokenizer = RegexpTokenizer(r"(?:[A-Za-z]+(?:['’][a-z]+)?)|<<\w+>>|[^\w\s]|\n")

    def tokenize(self, text):
        return self.tokenizer.tokenize(text)


class CharTokenizer:
    '''Custom char tokenizer, as list would tokenize the <<>> markers too and we do not want that'''
    def __init__(self):
        self.tokenizer = RegexpTokenizer(r"<<\w+>>|.")

    def tokenize(self, text):
        return self.tokenizer.tokenize(text)


# Create a Vocabulary Class that holds the token2id and id2token dicts and handels their retrieval in the Datasets
class Vocabulary():
    def __init__(self):
        # Create word2id and id2token dicts/lists
        # Mappings between symbols and integers, and vice versa.
        self.token2id = {}
        self.id2token = []

        self.vocab_size = 0

        # The padding symbol will be used to ensure that all tensors in a batch
        # have equal length. (i.e. the padding symbol is only used if a sample is not long enough 
        # (mostly for advanced batching where padding gets added before the ENDTOKEN of each play))
        self.token2id[PADDING_SYMBOL] = len(self.id2token)
        self.id2token.append(PADDING_SYMBOL)
        self.vocab_size += 1

        self.token2id[ENDTOKEN] = 1
        self.id2token.append(ENDTOKEN)
        self.vocab_size += 1

        for markerToken in SECTION_MARKER:
            self.token2id[markerToken] = len(self.id2token)
            self.id2token.append(markerToken)
            self.vocab_size += 1
        
        # Also add the token that describes that a token (in the validation and test sets) is not in the vocabulary
        self.token2id[UNKNOWN_SYMBOL] = len(self.id2token)
        self.id2token.append(UNKNOWN_SYMBOL)
        self.vocab_size += 1

        

    def add_token(self, token):
        '''Adds the token to the vocabulary and returns its (newly) assigned ID. If the token is already added
        this function will just return its ID without adding the token to the vocabulary again'''
        if token not in self.token2id:
            token_id = len(self.id2token)
            self.token2id[token] = token_id
            self.id2token.append(token)
            self.vocab_size += 1
        
        return self.token2id[token]
    
    def lookup_id(self, token):
        '''Checks if the token is in the vocabulary and returns the ID for the UNKNOWN_SYMBOL in case the token was not found.
        Crucially, it does NOT add the token to the vocabulary!'''
        if token in self.token2id:
            return self.token2id[token]
        else:
            return self.token2id[UNKNOWN_SYMBOL]
    
    def lookup_token(self, id):
        '''Returns the token to the corresponding ID, if the ID is not found the UNKNOWN_SYMBOL is returned'''
        if id in self.id2token:
            return self.id2token[id]
        else:
            return UNKNOWN_SYMBOL
        
    def transform_ID_2_String(self, id_list):
        '''Transforms a list of ids back into its original string using the vocabulary seen here'''
        text_list = []
        for id in id_list:
            text_list.append(self.lookup_token(id))
        
        text = "".join(text_list)
        return text

    def token_exist(self, token):
        '''Checks if token exists in vocabulary and returns either True or False'''
        if token in self.token2id:
            return True
        else: 
            return False



# Create custom dataclass
class ShakespeareDataset(Dataset):
    def __init__(self,filename, vocab: Vocabulary, seq_length, level='char', tokenization='nltk_shakespeare', record_tokens = False, advanced_batching=False, stride=1):
        ''' Constructs a data set based on the txt-file given to it
        Inputs: \\
            filname: Location and name of the respective txt file \\
            vocab: Is a Vocabulary object that should be used to convert/lookup ids and tokens (and to which in 
            the case of training the tokens should get added to) \\
            seq_length: Determines the lenght of the sequences that are fed to the model (i.e. window_size)
            => NOTE: Whether the sequence is a sequence of characters or words is determined in level
            level: expects'word' or 'char' -> Determines the level of embedding, i.e. 
            whether the data is encoded character or word wise \\
            tokenization: IF word-level is used you MUST specifiy 'nltk_shakespeare, 'nltk_word' or 'BPE' for tokenization \\
            record_tokens: (Boolean) Determines whether the tokesn in the file are added to the vocabulary or not 
            NOTE: YOU SHOULD ONLY ADD TOKENS TO THE VOCABULARY IN THE CASE OF TRAINING!! \\
            advanced_batching: (Boolean) If true, the Data is used for advanced batching, which means that instead of 
            serving samples as if they were indepedent, the Dataset keeps track of the different plays and serves them 
            individually. NOTE: Advanced batching should be probably used in the case of state persistent RNNs/LSTMs
            stride: (Integer) Choose how much the individual samples overlap, \\
            e.g. with a stride=1 (default) on text = ['hello', 'pretty', 'world', 'where', 'are', 'you' '?'] and 
            a seq_length = 2 the samples would look like this: sample1 =  ['hello', 'pretty'], sample2 = ['pretty', 'world']
            with their labels ALWAYS being the sample shifted by 1 (i.e. stride does not affect the labels!!) \\
            '''
        
        self.vocab = vocab
        self.seq_length = seq_length
        self.tokenization = tokenization
        self.level = level
        self.record_tokens = record_tokens
        self.advanced_batching = advanced_batching
        self.stride = stride
        

        # This will save the IDs that are used in the batch (for advanced indexing)
        # Its going to be empty if advanced batchin is disabled as we do not differentiate between plays
        self.batch_play_ids = []

        # Set tokenization tpye
        # Word level embedding
        if level=='word':
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
                raise NotImplementedError("Advanced batching not implemented yet.")
        # Character level embedding
        else:
            self.tokenize = CharTokenizer().tokenize # This as a function will split the string into characters except for the <<...>> markers, they are handled as one character too!!

        # Tokenize the text
        with open(filename,'r', encoding='utf-8') as f:
            text = f.read() # Read whole document as string \n and all 

        tokenized_text = self.tokenize(text) # tokenizes the text and return a list of tokens

        # Transform the text into respective IDs
        id_tokenized_text = []
        for token in tokenized_text:
            if self.level == 'word':
            # Only use lower-case words in case of word level tokenization due to data constraints
                token = token.lower()
            if self.record_tokens:
                id = vocab.add_token(token)
                id_tokenized_text.append(id)
            else:
                id = vocab.lookup_id(token)
                id_tokenized_text.append(id)
        
        
        padding_id = self.vocab.token2id[PADDING_SYMBOL]
        endtoken_id = self.vocab.token2id[ENDTOKEN]
        section_marker_ids =[]
        for markerToken in SECTION_MARKER:
            section_marker_ids.append(vocab.lookup_id(markerToken))



        # Store all samples and labels each just in a single list. 
        # WITHOUT: Advanced batching: samples and labels are a 2D list dim: (n_samples, seq_length)
        # WITH: Advanced batching: samples and labels are a 3D list dim: (n_plays, n_samples, seq_length)
        self.samples, self.labels = self.create_data(id_tokenized_text,padding_id=padding_id, endtoken_id=endtoken_id, 
                                                     section_marker_ids=section_marker_ids, seq_length=self.seq_length, 
                                                     stride=self.stride, advanced_batching=self.advanced_batching)
        
        if advanced_batching:
            self.n_plays = len(self.samples)
            self.n_samples = sum([len(play) for play in self.samples])
        else:
            self.n_plays = None # Is not defined here as we did not differentiate between plays
            self.n_samples = len(self.samples)

        # To get to the real samples in advanced indexing, you will need "read_indices", for each play exactly one. These
        # read indices will act like bookmarks, so that the dataset knows that when it is supposed to give a sample from play
        # x, what the last sample was it retrieved, and so it knows which is the next sample to retrieve
        # Also initiate a length dict, that saves how long each play is so that we know when to start from the beginning
        if self.advanced_batching:
            self.read_indices = {}
            self.play_length = {}
            for play_id in range(self.n_plays):
                self.read_indices[play_id] = 0
                self.play_length[play_id] = len(self.samples[play_id])

        

            


    def create_samples(self, id_text, padding_id, endtoken_id, seq_length,stride):
        """
        Creates the samples and labels based on the text that was transformed into IDs
        You can adjust the stride over the stride setting. 
        Inputs: \\
        id_text: Is the tokenized and ID-transformed text \\
        seq_length: the sequence length for the model \\
        padding_id: The id of the padding symbol in the Vocabulary you use \\
        endtoken_id: The ID of the ENDSYMBOL you use in the vocabulary \\
        stride: int, step size between sequences (stride=1 = full overlap; stride=seq_length = no overlap) \\

        Return:
        - samples: A list of list containing intergers, each list in the list is a list of input sequences
        - labels:  A list of list containing intergers, each list in the list is a list of label sequences
        """
        samples = []
        labels = []
        end_token = id_text[-1]
        max_index = len(id_text) - 1  # Exclude end token for shifting (as slicing excludes index boundary, its really the last index)

        i = 0
        while i + seq_length <= max_index:
            x_seq = id_text[i:i + seq_length]
            y_seq = id_text[i + 1:i + seq_length + 1]
            samples.append(x_seq)
            labels.append(y_seq)
            i += stride

        # Handle remaining tokens near the end (potential padding case)
        remaining_tokens = id_text[i:max_index]  # Don't include the end token in input
        if remaining_tokens:
            x_seq = remaining_tokens
            y_seq = id_text[i + 1:]  # This will include the end token

            # Pad if too short
            pad_len = seq_length - len(x_seq)
            if pad_len > 0:
                x_seq = x_seq + [padding_id] * pad_len
                # Check if the last token is really the ID from the endtoken and if it is put the padding in between.
                # If it is not, the end token has not been recognized by the tokenizer and so we apply the padding afterwards
                # as to not interfere with the tokenization
                if end_token == endtoken_id: # ENDSYMBOL IS RECOGNIZED
                    y_seq = y_seq[:-1] + [padding_id] * pad_len  + [end_token]
                else:
                    y_seq = y_seq + [padding_id] * pad_len


            samples.append(x_seq)
            labels.append(y_seq)

        return samples, labels
    

    def create_data(self, id_text, padding_id, endtoken_id, section_marker_ids, seq_length, stride, advanced_batching):

        # Traverse the text and create samples play by play
        plays = [] # List of list containing a play in each list
        current_play = []
        start_idx = 0
        end_idx = 0
        # Checks if play has started, so that we can append the rest to plays
        # also checksif another play has been started (is there to catch mistakes in the parsing, 
        # when there is no ENDTOKEN to end a play before another section marker marks the beginning of a new play)
        play_started = False 
        
        for token_id in id_text:
            # Check if we have a new play
            if token_id in section_marker_ids and not play_started:
                play_started = True
                current_play = [token_id]
            elif token_id in section_marker_ids and play_started: # Should never happen
                raise ValueError('A play was started before another one ended. Mistakes in parsing are likely')
            # Play has ended => Save the play in its own list
            elif token_id == endtoken_id and play_started:
                current_play.append(token_id)
                plays.append(current_play)
                play_started = False
            elif play_started: # Ensures that token in between plays (even tho they should not exist) are not recorded anymore
                current_play.append(token_id)

        # Iterate through the plays to create samples
        samples_list = []
        labels_list = []
        for play_text in plays:
            samples, labels = self.create_samples(play_text,padding_id, endtoken_id, seq_length, stride) # These are 2D lists

            # If advanced batching is enable the 2D lists get accumulated to a 3D list dim: (n_play, n_samples, seq_length)
            if advanced_batching:
                samples_list.append(samples)
                labels_list.append(labels)
            # If not all samples should be collected into a single 2D list of dim (n_samples, seq_length)
            else:
                samples_list.extend(samples)
                labels_list.extend(labels)
        
        return samples_list, labels_list



    def __len__(self):
        if self.advanced_batching:
            return self.n_plays
        else:
            return self.n_samples

    def __getitem__(self, idx):

        if self.advanced_batching: # Here is idx an index of the first dim of samples, i.e. its an index for a play not an indivual sample
            sample = self.samples[idx][self.read_indices[idx]]
            label = self.labels[idx][self.read_indices[idx]]
            # Lastly, increment the read_index of that play. If that increment would lead us to be outside of the play reset the read index to 0
            self.read_indices[idx] += 1
            if self.read_indices[idx] == self.play_length[idx]:
                self.read_indices[idx] = 0 # Skip to the beginning of the play

            return sample, label # Is again just two lists of lenght seq_length containing the data point
        
        else: 
            return self.samples[idx], self.labels[idx]
        
        

def collate_fn(batch):
    '''Takes a batch and preperares them to be in the right format to be given as samples, labels
    Input: 
        batch: is a list of tuples, each tuple is a (sample, label) pair. Each sample is a list and each label is a list
    Output: 2 torch tensors, one for all samples one for all labels
        sample_tensor and label_tensor have size (n_batch,seq_length)
    '''
    samples, labels = zip(*batch)

    samples_tensor = torch.tensor(samples)
    labels_tensor = torch.tensor(labels)

    return samples_tensor, labels_tensor




def create_play_buckets(dataset, boundaries):
    """ Function creates buckets full of play indices. In each bucket we collect plays of similar length.
    Inputs:
        dataset: The custom Pytorch dataset (NOTE: We make use of the  exposure of  .samples as a 3D structure!!).
        boundaries: List of upper bounds for buckets (e.g. [50, 150, 500])

    Returns:
        buckets: dict mapping bucket_id to list of play indices.
    """
    buckets = defaultdict(list)
    for play_idx, play_samples in enumerate(dataset.samples): #play_samples is a list of samples which are also lists
        n_samples_in_play = len(play_samples)
        for i, boundary in enumerate(boundaries):
            if n_samples_in_play <= boundary:
                buckets[i].append(play_idx)
                break
        else:
            # Goes into last bucket if exceeds all boundaries
            print('A play exceeds the most upper boundary that was given and was put into the last bucket.')
            buckets[len(boundaries)].append(play_idx)
    return buckets

class BucketedPlaySampler(torch.utils.data.Sampler):
    def __init__(self, dataset, play_indices, batch_size, max_samples_per_play=None, drop_last=True):
        """ This is a CustomBatch Sampler that samples from a specific set of plays determined by play_indices
        Input:
            dataset (ShakespeareDataset): The dataset instance. Must implement __len__ and indexing by play index.
            play_indices: (list): The play_indices are the subset of play indices for a specific bucket, so that we only sample from that bucket with that sampler
            batch_size (int): Number of unique plays per batch.
            max_samples_per_play (int or dict): How many times we can sample from each play in one epoch. 
                => Its chosen outside but i would recommend choosing this in some form that guarantees that the whole play
                is actually traversed (so that max_samples_per_play >= n_samples in play)
                Can be an int (applied to all), or a dict {play_idx: max_samples}.
            drop_last (bool): Whether to drop the (last) incomplete batch. (NOTE: Depending on the size of the bucket 
                incomplete batches could start wayyy before the first batch and you could potentially loose a lot of data if the 
                buckets are too big. Nevertheless when aiming for batch_size consistency thats the trade off we have to do...)
        """

        self.dataset = dataset
        self.play_indices = play_indices
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.sample_counts = self.init_sample_counts(max_samples_per_play) #{idx: max_samples_per_play for idx in play_indices}

    def init_sample_counts(self, max_samples_per_play):
        # Either fixed multiplier or per-play count
        if max_samples_per_play is None:
            raise ValueError("You must specify max_samples_per_play to avoid infinite epochs.")
        elif isinstance(max_samples_per_play, int):
            return {idx: max_samples_per_play for idx in self.play_indices}
        elif isinstance(max_samples_per_play, dict):
            return max_samples_per_play
        else:
            raise TypeError("max_samples_per_play must be int or dict.")

    def __iter__(self):
        # Create mutable copy to track remaining samples
        remaining = self.sample_counts.copy()
        # This will create an initial list where each play is included ONCE IF they can be included
        # => This is done to avoid that a play can occur more than once in a batch
        available_plays = [i for i in remaining if remaining[i] > 0]
        # This will collect the order of the plays (it will be a 1D list instead of a 2D list through which we traverse by slicing when interating)
        batch = []
        while available_plays:
            # Shuffle at each batch construction
            random.shuffle(available_plays)
            # Select play
            selected = available_plays[:self.batch_size]
            batch.extend(selected)
            # Go through the selected plays to reduce their counter with which they can occur
            for idx in selected:
                remaining[idx] -= 1
                if remaining[idx] <= 0:
                    available_plays.remove(idx)

        # Create batch-wise iterator
        for i in range(0, len(batch), self.batch_size):
            b = batch[i:i + self.batch_size]
            # TRICK (THIS IS NOT NICE CODING, but as a shortcut works haha.
            # Write the information which play_ids have been chosen back to the dataset
            self.dataset.batch_play_ids = b
            if len(b) < self.batch_size and self.drop_last:
                continue
            yield b

    def __len__(self):
        # Estimate total number of full batches from that bucket
        total = sum(self.sample_counts.values()) # How many samples are we having in total?
        # Thats of course just an approximation because it could stop sooner with drop last
        return total // self.batch_size if self.drop_last else math.ceil(total / self.batch_size)





class UnifiedBucketLoader:
    def __init__(self, bucket_loaders, bucket_weights=None):
        """ This class manages the different dataloaders that have been initialized with the bucket samplers
        and chooses from which dataloader to sample from. This is done, so that we later still only have one 
        data loader and the whole logic behid multiple dataloaders is something we can abstract from.
        As such the Class has to be a generator class itself, so that it can produce iterators on demand
        (to be more precise to access other iterators on deman while acting like an iterator itself...)
        Inputs:
            bucket_loaders (dict): {bucket_id: DataLoader}, i.e. it contains a CustomDataLoader for each bucket
            bucket_weights (dict or None): {bucket_id: weight} to control sampling probability. 
                                           If None, uniform sampling.
        """
        # Load information about buckets
        self.bucket_ids = list(bucket_loaders.keys())
        self.bucket_loaders = bucket_loaders

        

        if bucket_weights is None:
            self.bucket_weights = [1.0] * len(self.bucket_ids)
        else:
            self.bucket_weights = [bucket_weights.get(bid, 0.0) for bid in self.bucket_ids]

    # Implement __iter__ method so that we can use it as a "normal" data loader
    # __iter__ gets called internally when we init a loop with for x in Y
    # For us to iterate more than once over the object, we are implementing __iter__ to be a generator function, see more here
    # https://realpython.com/python-iterators-iterables/#exploring-alternative-ways-to-write-__iter__-in-iterables
    # https://realpython.com/introduction-to-python-generators/#understanding-the-python-yield-statement
    def __iter__(self):
        # Collect all the iterators of the buckets => Idea is to put every Dataloader in this artifical loop
        # in which we decide which loop move forward manually
        # NOTE: Initilize it new for every epoch (do NOT safe it to self.), as we need fresh iterators of every loader 
        # for every new epoch (no matter if the old iterator was exhausted or not)
        bucket_iters = {bid: iter(loader) for bid, loader in self.bucket_loaders.items()} 

        # Copy it into active ids and weights per epoch so that we can delete them during the epoch without
        # deleting the original object, as we still need them for the next epochs, when __iter__ gets called again
        active_ids = self.bucket_ids.copy()
        weights = self.bucket_weights.copy()

        while active_ids:
            bucket_id = random.choices(active_ids, weights=weights, k=1)[0]
            # This is a bit complext but essentially, the generator function should grab the next item
            # from the iterator of on of the original dataloader. If that dataloader is exhausted
            # it will raise a StopIteration error, which we will have to catch by hand as we do not use
            # the data loaders in a real loop but rather go through them manually. (HINT: I could have  done it 
            # with a default value in next() as well but then i would need to save the next() result in a variable
            # first e.g. batch = next(bucket_iters[bucket_id], None) and then ask if batch is None: (...Do what is now
            # in the except statement). It is the same I think, but this saves a bit of space and is clean too, I think )
            try:
                yield next(bucket_iters[bucket_id]) # Yields StopIteration if iterator of that bucket is exhausted otherwise yields next item and preserves state of function
            except StopIteration:
                # Remove exhausted bucket from this epoch state
                idx = active_ids.index(bucket_id)
                del bucket_iters[bucket_id]
                del active_ids[idx]
                del weights[idx]


    # __next__ method is used for the iterator in the list.
    def __next__(self):
        # If all iterators of all dataloader terminated we can terminate as well
        if not self.iters:
            raise StopIteration

        # Try to draw a batch from one of the available bucket iterators as long as there are iterators available
        while self.iters:
            # Choose bucket
            bucket_id = random.choices(self.bucket_ids, weights=self.weights, k=1)[0]
            # Try iterating that bucket 
            try:
                return next(self.iters[bucket_id])
            except StopIteration: # If it does not work, it means the iterator is done => Remove exhausted iterator
                # Seach for index of hte bucke_id => Can't be sure that is the same as the ID as buckets was a dict and that is unordered
                idx = self.bucket_ids.index(bucket_id)
                del self.iters[bucket_id]
                del self.bucket_ids[idx]
                del self.weights[idx]

        raise StopIteration




# Create custom dataloader
def create_DataLoader(filename, batch_size, seq_Length, shuffle=True, stride=1, level='char', tokenization='nltk_shakespeare', vocab=None, record_tokens=False, advanced_batching=False, boundaries=None, traverse='once' ):
    '''Create a DataSet, DataLoader and Vocabulary based on the hyperparameters given to it.
    
    Inputs:
        filename: (str) File location (txt file)\\
        seq_length: (int) length of the sequence that gets processed by the model 
            (== number of hidden nodes in RNN, LSTM, and window_size of transformers) \\
        batch_size: (int) Specifies how many samples are given at each iteration (NOTE: NOT THE SAME THINF AS seq_length!)
        shuffle: (Boolean) Specifies if the data should be shuffled between epochs (Note that when advanced batching is 
            activated setting shuffle DOES NOTHING. Instead the data within the plays will always be traversed 
            sequentially, but the order in which plays are shown to the model during training will be shuffled) \\
        stride: (int) Determines how far the samples are overlapping, e.g. with a stride=1 (default) on 
            text = ['hello', 'pretty', 'world', 'where', 'are', 'you' '?'] and 
            a seq_length = 2 the samples would look like this: 
            sample1 =  ['hello', 'pretty'], sample2 = ['pretty', 'world']  \\
        level: (str) ['char' or 'word'] Whether the tokenization is done on the character or word level \\
        tokenization: (str) ['nltk_shakespeare, 'nltk_word' or 'BPE'] Only used IF level = 'word', uses different 
            tokenization algorithms
        vocabulary: (Vocabulary or None): Lets so give the DataSets a previously made vocabulary (usefull for 
            valdiation and testing => NOTE: then record_token should be set to False) \\
        record_tokens: (Boolean): Determines whether the new found tokens should be added to the vocabulary 
            (ONLY ADD TOKENS DURING TRAINING) NOTE: If no vocabulary was given record_tokens is automatically 
            set to true no matter what you specify \\
        advanced_batching: (Boolean): Determines, if you want to use advanced batching or not 
            (needed for persistent state RNNs, and LSTMs). Advanced batching splits up the corpus into 
            the individual plays and makes the Dataloader iterate over them separately!!
            NOTE: advanced_batching REQUIRES EITHER chararcter level tokenization OR nltk_shakespeare tokenization to work correctly!!
            NOTE: THe batch size in advanced batching is responsible for how well the advanced batching can work. Too large batch sizes 
            reduce the ability to split the plays into buckets where each bucket holds plays of similar length. Consequently the buckets
            have to be larger, which means that shorter plays are going to be oversampled a lot (might introduce bias into the data)
            Thus, with advanced batching, choose a adequate batch_size. Experimenting with it has shown that a batch size of 
            20 seems to work well with the shakespeare cropus, if DEFAULT buckets are used. \\
        boundaries: These are the upper boundaries to determine which play goes into which bucket for sampling. If None is given, the standard
            boundaries found by experimenting with the dataset are used (tbh I would just keep them that way... its only if you want to experiment
            with the batch size where you have to maybe edit the boundaries to allow bigger buckets) \\
        traverse: (str) ['once', 'balanced', 'partial']: If set to 'once' each play gets traversed exaclty once per epoch. 
            If set to balanced, plays are repeated as often as they fit into the largest play in the same bucket.
            If set to partial, plays within the same bucket are repeated until the biggest play in the bucket is traversed once,
            i.e. shorter plays do not have to end, they are likely to end in the middle of the play.
            If boundaries are set to None (== Default buckets) I recommend setting traverse to 'once'
        '''

    # If no vocabulary was given instantiate a new one
    if vocab is None:
        vocab = Vocabulary()
        record_tokens = True
    
    # Create data set
    dataset = ShakespeareDataset(filename=filename, vocab=vocab,seq_length=seq_Length, 
                                 level=level,tokenization=tokenization, record_tokens=record_tokens, 
                                 advanced_batching=advanced_batching, stride=stride)
    

    # Create DataLoader
    if not advanced_batching:
        # A simple dataloader is enough
        dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle,collate_fn=collate_fn)
    else:
        # With adavanced batching we have to put in a bit more work...
        # 1. Create the buckets of the play
        if boundaries is None: # Create the boundaries based on my own experimenting 
        # Step 1: Create buckets
            if level == 'word':
                boundaries = [*WORD_BUCKETS] # Transform it from a tuple into a list
            else:
                boundaries = [*CHAR_BUCKETS] # Divide into short and long plays
        buckets = create_play_buckets(dataset, boundaries)

        # 2. Compute bucket weights by total sample volume, i.e. buckets that have more samples in them get chosen more often
        # this is done to achieve a balance where all types of plays are continously seen during the training of one epoch
        bucket_weights = {}
        for bucket_id, play_indices in buckets.items():
            total_samples = sum(len(dataset.samples[play_idx]) for play_idx in play_indices)
            bucket_weights[bucket_id] = total_samples

        total_weight = sum(bucket_weights.values())
        bucket_weights = {k: v / total_weight for k, v in bucket_weights.items()}

        # 3. Create per-bucket loaders, i.e. create one Data loader for each bucket, which we will feed to the Unified Dataloader
        bucket_loaders = {}
        for bucket_id, play_indices in buckets.items():
            # TODO: You could make this modular, to allow people to choose how much to traverse to each (or maybe to steer that in between epochs)
            if traverse == 'once':
                # This ensures that each play is traversed exactly once
                max_samples_per_play = {play_idx: len(dataset.samples[play_idx]) for play_idx in play_indices} 

            elif traverse == 'balanced': # Traverse in  a balanced way
                max_len = max(len(dataset.samples[i]) for i in play_indices)
                max_samples_per_play = {idx: math.ceil(max_len / len(play_samples)) * len(play_samples) for idx, play_samples in enumerate(dataset.samples)}

            elif traverse == 'partial': # Traverse in a as long as largest play in bucket is playing
                max_len = max(len(dataset.samples[i]) for i in play_indices)
                max_samples_per_play = {play_idx: max_len for play_idx in play_indices}
            else:
                raise ValueError('Wrong traverse argument was given')

            # TODO: Maybe allow different batch sizes for different buckets...
            sampler = BucketedPlaySampler(dataset=dataset,play_indices=play_indices, batch_size=batch_size, 
                                          max_samples_per_play=max_samples_per_play,drop_last=True)
            loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_fn)
            bucket_loaders[bucket_id] = loader

        #4. Create the  unified data loader with automatic weights
        unified_loader = UnifiedBucketLoader(bucket_loaders, bucket_weights)


    
    if advanced_batching:
        return unified_loader, dataset, vocab
    else:
        return dataloader, dataset, vocab
    


# For debugging purposes
if __name__ == '__main__':
    dataloader, dataset, vocab = create_DataLoader('Data/train_shakespeare_full_corpus.txt', batch_size=10, seq_Length=20,shuffle=True, 
                      stride=1, level='word', tokenization='nltk_shakespeare', record_tokens=True, 
                      advanced_batching=True, traverse='once')
    
    # Just print the first 10 samples
    for epoch in range(3):
        print(f'Epoch:{epoch}')
        for i, (x, y) in enumerate(dataloader):
            if i <= 3:
                print(x)
                print(y)
            else:
                break
    
