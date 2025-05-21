import os
import sys
import torch

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
from generateFromModelsGPT import generate_text
from Utils.data_handling import create_DataLoader, Vocabulary

from our_GPT_model import GPTConfig, GPT

device = 'cuda'


out_dir = 'baseline_model_checkpoints_our_loader-shakespeare/checkpointtrial_110'
exec(open('configurator.py').read()) # Overwrite previous variables?

ckpt_path = os.path.join(out_dir, 'ckpt_40.pt')
print(ckpt_path)
checkpoint = torch.load(ckpt_path, map_location=device)
gptconf = GPTConfig(**checkpoint['model_args'])
model = GPT(gptconf, from_checkpoint=True)


train_file = '../../Data/train_shakespeare_full_corpus.txt'
batch_size = 20
seq_length = 67
stride = 1
level = 'char' # word or char
tokenization = 'nltk_shakespeare' #'nltk_shakespeare' or 'BPE'
traverse = 'once'


#vocab = Vocabulary() # Is this unironically enough
_, _, vocab = create_DataLoader(train_file, batch_size, seq_length, shuffle=True, stride=stride,
                                        level=level, tokenization=tokenization, record_tokens=True,
                                        advanced_batching=True, traverse=traverse)



print('')
print('vocab (Note: ):')
print(vocab)

print('')
print('vocab.vocab_size (Note: ):')
print(vocab.vocab_size)


start_str = ''

generated_text = generate_text(model, start_str, length=100, vocab=vocab, device=device, temperature=0.7)
print(f"Generated Text: {generated_text}")





