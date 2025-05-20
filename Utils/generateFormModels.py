import torch
import os 
import nltk
from transformers import GPT2Tokenizer

from Utils.data_handling import Vocabulary, ShakespeareTokenizer, CharTokenizer


def generate_text(model, start_str, length, vocab: Vocabulary, device='cpu', temperature=1.0, tokenization_level = 'char', tokenization_type='nltk_shakespeare'):
    model.eval()
    model.to(device)

    # NOTE: THIS IS NOT NEEDED AND ALSO GOES AGAINST PROJECT AS WE DEFINE AN UNKWOWN_SYMBOL ALREADY
    # PLUS: NEVER ACCESS token2id and id2token directly as i said!!
    # Ensure <UNK> token exists
    # unk_token = "<UNK>"
    # if unk_token not in vocab.token2id:
    #     vocab.token2id[unk_token] = len(vocab.token2id)
    #     vocab.id2token.append(unk_token)
    # unk_id = vocab.token2id[unk_token]

    # Encode start string
    # AGAIN DONT DO IT DIRECTLY!! USE vocab function
    # input_ids = [vocab.token2id.get(token, unk_id) for token in start_str]
    # generated_ids = input_ids[:]
    # PLUS tokenize string first
    if tokenization_level=='word':
        if tokenization_type == 'nltk_word':
            try :
                nltk.word_tokenize("hi there.")
            except LookupError:
                nltk.download('punkt')
            tokenize = nltk.word_tokenize
        elif tokenization_type == 'nltk_shakespeare':
            tokenize = ShakespeareTokenizer().tokenize
        elif tokenization_type == 'BPE':
            # Check if we already created our adapted GPT-2 BPE tokenizer
            if os.path.isdir("./custom_gpt2_tokenizer"):
                tokenizer = GPT2Tokenizer.from_pretrained("./custom_gpt2_tokenizer")
            else:
                # Load pre-trained GPT-2 tokenizer
                tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

                # Add special tokens
                special_tokens_dict = {"additional_special_tokens": SPECIAL_TOKENS_BPE}
                tokenizer.add_special_tokens(special_tokens_dict)

                # Save for later use
                tokenizer.save_pretrained("./custom_gpt2_tokenizer")

            tokenize = tokenizer.tokenize
    # Character level embedding
    else:
        tokenize = CharTokenizer().tokenize # This as a function will split the string into characters except for the <<...>> markers, they are handled as one character too!!

    tokenized_start_str = tokenize(start_str)

    input_ids = [vocab.lookup_id(token) for token in tokenized_start_str]

    # Get the inputs for the model
    input_tensor = torch.tensor(input_ids).unsqueeze(0) # dim: (1,seq_length)
    hidden = torch.zeros(model.n_layers_RNN,1,model.hidden_dim) #  (n_layers, 1, hidden_dim)


    # Only use the last token as input to speed things up to change later but now for test
    # input_tensor = torch.tensor([[input_ids[-1]]], dtype=torch.long, device=device)


    generated_ids = []
    for _ in range(length):
        with torch.no_grad():
            # logits: (1, seq_length, vocab_size), 
            # hidden: (n_layers, 1, hidden_dim), 
            # output: (1, seq_length, hidden_dim)
            logits, hidden, _ = model(input_tensor, hidden) 
        
        logits.squeeze() # (seq_length, vocab_size)

        #Only interested in next word => For that
        logits_next = logits[-1,:] # (vocab_size)
        # hidden_next = hidden  # (n_layers, 1, hidden_dim)

        # Sample from the logits
        probs = torch.softmax(logits / temperature, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1).item()
        generated_ids.append(next_id)

        # Next input is the last generated token

        input_tensor = torch.tensor([[next_id]], dtype=torch.long, device=device)

    # Decode the token IDs
    tokenization_vocab = 'char' if tokenization_level=='char' else tokenization_type
    return vocab.transform_ID_2_String(generated_ids, tokenization=tokenization_vocab)
    # return ''.join(vocab.id2token[i] if i < len(vocab.id2token) else unk_token for i in generated_ids)
