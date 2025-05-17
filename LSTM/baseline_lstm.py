import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from LSTM import LSTM, train_lstm, evaluate_lstm
from Utils.data_handling import create_DataLoader, Vocabulary

def main():
    print("Starting the script...")

    # Hyperparameters
    seq_length = 50
    hidden_size = [256, 256]
    num_layers = 2
    dropout = 0.15
    learning_rate = 0.001
    num_epochs = 10
    batch_size = 32
    stride = 1
    level = 'char'
    tokenization = 'nltk_shakespeare'
    traverse = 'once'
    embedding_dim = None  # Will be set based on vocabulary size

    print("Hyperparameters and configuration set.")

    # File paths
    train_file = 'Data/train_shakespeare_full_corpus.txt'
    val_file = 'Data/val_shakespeare_full_corpus.txt'

    # Create vocabulary
    vocab = Vocabulary()
    print("Vocabulary created.")

    # Create data loaders
    print("Creating data loaders...")
    train_loader, _, _ = create_DataLoader(train_file, batch_size, seq_length, shuffle=True, stride=stride,
                                           level=level, tokenization=tokenization, vocab=vocab, record_tokens=True,
                                           advanced_batching=True, traverse=traverse)
    val_loader, _, _ = create_DataLoader(val_file, batch_size, seq_length, shuffle=False, stride=stride,
                                         level=level, tokenization=tokenization, vocab=vocab, record_tokens=False,
                                         advanced_batching=True, traverse=traverse)
    print("Training data loader created.")
    print("Validation data loader created.")

    # Set embedding dimension based on vocabulary size
    embedding_dim = vocab.vocab_size
    print(f"Embedding dimension set to vocabulary size: {embedding_dim}")

    # Initialize model
    model = LSTM(vocab_size=embedding_dim, embedding_dim=embedding_dim, hidden_size=hidden_size[0],
                 output_size=embedding_dim, num_layers=num_layers, dropout=dropout)
    print("Model initialized.")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    print("Loss function and optimizer set.")

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Train the model
    print("Starting training...")
    history = train_lstm(model, train_loader, val_loader, optimizer, device=device, num_epochs=num_epochs)

    # Evaluate the model
    print("Evaluating the model...")
    evaluate_lstm(model, val_loader, device=device)

if __name__ == '__main__':
    # Add the root directory of your project to the Python path
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    main()
