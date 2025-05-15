import torch
import nltk
from torch.utils.data import Dataset, DataLoader
from nltk.tokenize import RegexpTokenizer


# Import global variables
from shakespeare_parser import ENDTOKEN, SECTION_MARKER


## Define globale variables

# The padding symbol will be used to ensure that all tensors in a batch have equal length.
# This only really applied to the end of the processing string (or the end of a piece if the tokenizer can recognize this
# i.e. when tokenizer is either character level or ShakespeareTokenizer) when the leftover tokens are not enough
# to fill the whole window size
PADDING_SYMBOL = ' ' # just a whitespace
UNKNOWN_SYMBOL = "<<UNK>>"

# Max number of words to be predicted if <END> symbol is not reached
MAX_PREDICTIONS = 20


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

        # The padding symbol will be used to ensure that all tensors in a batch
        # have equal length.
        self.token2id[PADDING_SYMBOL] = 0
        self.id2token.append(PADDING_SYMBOL)

        self.token2id[ENDTOKEN] = 1
        self.id2token.append(ENDTOKEN)

        for i, markerToken in enumerate(SECTION_MARKER):
            self.token2id[markerToken] = len(self.id2token)
            self.id2token.append(markerToken)
        
        # Also add the token that describes that a token (in the validation and test sets) is not in the vocabulary
        self.token2id[UNKNOWN_SYMBOL] = len(self.id2token)
        self.id2token.append(UNKNOWN_SYMBOL)

        # Also add the Padding symbol to the vocabulary (the padding symbol is only used if a sample is not long enough 
        # (mostly for advanced batching where padding gets added before the ENDTOKEN of each play))
        self.token2id[PADDING_SYMBOL] = len(self.id2token)
        self.id2token.append(PADDING_SYMBOL)

    def add_token(self, token):
        '''Adds the token to the vocabulary and returns its (newly) assigned ID. If the token is already added
        this function will just return its ID without adding the token to the vocabulary again'''
        if token not in self.token2id:
            token_id = len(self.id2token)
            self.token2id[token] = token_id
            self.id2token.append(token)
    
    def lookup_token(self, token):
        '''Checks if the token is in the vocabulary and returns the ID for the UNKNOWN_SYMBOL in case the token was not found.
        Crucially, it does NOT add the token to the vocabulary!'''
        if token in self.token2id:
            return self.token2id[token]
        else:
            return self.token2id[UNKNOWN_SYMBOL]
    
    def lookup_id(self, id):
        '''Returns the token to the corresponding ID, if the ID is not found the UNKNOWN_SYMBOL is returned'''
        if id in self.id2token:
            return self.id2token[id]
        else:
            return UNKNOWN_SYMBOL


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
        self.record_tokenns = record_tokens
        self.advanced_batching = advanced_batching
        self.stride = stride

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
                pass # NOT IMPLEMENTED YET

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
            if record_tokens:
                id = vocab.add_token(token)
                id_tokenized_text.append(id)
            else:
                id = vocab.lookup_id(token)
                id_tokenized_text.append(id)
        
        
        self.total_samples = 0
        if not advanced_batching:
            # Store all samples and labels each just in a single list.
            self.samples = [] # Is just a list of samples
            self.labels = [] # Is just a list of labels

            self.samples, self.labels = self.create_samples(id_tokenized_text, seq_length=self.seq_length, stride=self.stride)
            self.total_samples = len(self.samples)
            



        else: # Use advanced batching
            # Is a list of the different shakespeare pieces (if the can be extracted). Each piece is again a list
            # which contains again lists of size seq_length, i.e. each piece has been cut into lists of seq_length.
            # Note: The last list can be smaller than the sequence length (and is thus padded with the padding symbol
            # to make that list seq_legnth long!!) 
            # => This is a 3D list ([idx1][idx2][idx3] = [play][no. of sample in that place][actual sample of the play]
            # IF that ENDTOKEN was found, the endtoken is not put into samples but only into labels!!
            samples = []
            labels = [] # Are the samples with a single offset
            idx1 = -1 # Start at -1 as we increment it immeaditly in the for loop
            idx2 = 0
            idx3 = 0
            total_token = 0
            total_samples = 0
            for token in tokenized_text:
                # Check if token is the start of a new play
                if token in SECTION_MARKER:
                    idx1 += 1
                    idx2 = 0
                    idx3 = 0
                    samples[idx1] = []
                    samples[idx1].append()

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



        

        
        

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        pass
        
        

def collate_fn_normal(batch):
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

def collate_fn_advanced():

    # Here
    # if token[-1] == ENDTOKEN
    # DANN WURDE DAS ENDTOKEN RICHTIG ERKANNT UND PADDING KANN MANN dann vor dem endtoken machen
    # Überleg dir noch wie du das auf char ebene machst mabye checke dass alle token länge 1 haben und dann 
    # kannst du das padding token als ID(!!) vor dem token len(ENDTOKEN) einfügen
    # len(token[-1]) == 1 and token[-1]
    # NEE USE EINE CUSTOM CALLABLE CLASS UM DAS ZU MACHEN UND SPEICHERE DA DIE SACHEN EINFACH DRIN AB!!
    # DANN KANNST DU EINFACH CHECKEN AUF WELCHEM LEVEL WIR SIND UND DAS DEMENTSPRECHEN MACHEN
    print('WARNING: ADVANCED BATCHING IS NOT YET IMPLEMENTED!!')

# Create custom dataloader

def create_DataLoader(filename, batch_size, seq_Length, shuffle=True, stride=1, level='char', tokenization='nltk_shakespeare', vocab=None, record_tokens=False, advanced_batching=False):
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
        '''

    # If no vocabulary was given instantiate a new one
    if vocab is None:
        vocab = Vocabulary()
        record_tokens = True
    
    # Create data set
    dataset = ShakespeareDataset(filename=filename, vocab=vocab,seq_length=seq_Length, 
                                 level=level,tokenization=tokenization, record_tokens=record_tokens, 
                                 advanced_batching=advanced_batching, stride=stride)
    
    if advanced_batching:
        collate_fn = collate_fn_advanced
    else:
        collate_fn = collate_fn_normal
    # Create DataLoader
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle,collate_fn=collate_fn)

    return dataloader, dataset, vocab
    
