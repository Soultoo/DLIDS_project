import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from Utils.embeddings_Loader import load_glove_embeddings
from Utils.data_handling import create_DataLoader

training_file = '../../Data/train_shakespeare_full_corpus.txt'
batch_size = 20
seq_length = 50
shuffle = True
stride = seq_length # Proabbly
tokenization_level = 'word'
tokenization_type = 'BPE' ##tag #TODO601 Check how to change this to BPE
advanced_batching = True


train_dataloader, train_dataset, vocab = create_DataLoader(filename=training_file, batch_size=batch_size, 
                      seq_Length=seq_length, shuffle=shuffle, stride=stride,
                      level=tokenization_level,tokenization=tokenization_type,
                      vocab=None,record_tokens=True,advanced_batching=advanced_batching,boundaries=None, 
                      traverse='once')

embedding_type = 'glove'
current_dir = os.getcwd()

root_path = '../..'

if embedding_type == 'glove':
        embedding_file = os.path.join(root_path, 'Data', 'Glove_vectors.txt')

embedding_dim, embedding = load_glove_embeddings(embedding_file=embedding_file, vocab=vocab)

print('')
print('embedding_dim (Note: ):')
print(embedding_dim)

print('')
print('embedding (Note: ):')
print(embedding)

print('')
print('embedding.size() (Note: ):')
print(embedding.size())

