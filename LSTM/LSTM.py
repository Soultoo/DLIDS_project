import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import os

class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, output_size, num_layers=1, dropout=0.0, use_pretrained_embedding=False, pretrained_weights=None, persistent_hidden_state=False):
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_size
        self.output_dim = output_size
        self.n_layers = num_layers
        self.dropout_rate = dropout
        self.persistent_hidden_state = persistent_hidden_state

        if use_pretrained_embedding:
            if pretrained_weights is None:
                raise ValueError("You must provide pretrained_weights if use_pretrained_embedding is True.")
            self.embedding = nn.Embedding.from_pretrained(pretrained_weights, freeze=True)
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, X, hidden=None):
        '''Implements the forward pass of the model
        Inputs:
            X: (tensor) size (n_batch, seq_length) containing integers indices corresponding to tokens
            hidden: (tensor) size: (n_layers, n_batch, hidden_dim)'''
        # Apply embedding
        X = self.embedding(X)  # dim: (n_batch, seq_length, embedding_dim)

        output, hidden = self.lstm(X, hidden)  # output: (n_batch, seq_length, hidden_dim), hidden: (n_layers, n_batch, hidden_dim)
        logits = self.fc(output)  # (n_batch, seq_length, vocab_size)
        return logits, hidden, output

def train_lstm(model, dataloader_train, dataloader_val, optimizer, device='cpu', num_epochs=10,
               print_every=100, val_every_n_steps=500, scheduler=None, checkpoint_dir='./checkpoints',
               log_file='training_log.txt'):
    # Create directories and set up logging with tensorboard
    os.makedirs(checkpoint_dir, exist_ok=True)
    writer = SummaryWriter(log_dir='./runs/my_experiment')

    # Create history dicts to visualize the loss curves later
    history = {
        'train_loss': [],  # List of (step, loss)
        'val_loss': [],  # List of (step, loss)
    }

    best_val_loss = float('inf')

    # Also log the loss in txt files to use them even after the script has run
    with open(log_file, 'w') as f:
        f.write('epoch,train_loss,val_loss\n')

    model.to(device)
    criterion = nn.CrossEntropyLoss()

    global_step = 0  # Collect update steps over all epochs for logging
    for epoch in range(num_epochs):

        # Put model into training mode at the beginning of each epoch after evaluating after each epoch
        model.train()
        running_loss = 0.0  # This is the loss averaged within batches and added
        total_loss = 0.0  # This is the loss really just summed up without any averaging done
        total_samples = 0  # Used to average the total_loss above

        for i, (inputs, targets) in enumerate(dataloader_train):
            global_step += 1
            batch_size = inputs.size(0)
            inputs, targets = inputs.to(device), targets.to(device)  # input: (n_batch, seq_length), (n_batch, seq_length)

            optimizer.zero_grad()
            logits, _, _ = model(inputs)  # logits: (n_batch, seq_length, vocab_size)
            # You need to flatten the arrays
            # logits: (n_batch, seq_length, vocab_size) -> (n_batch*seq_length, vocab_size)
            # targets: (n_batch, seq_length) -> (n_batch*seq_length)
            loss = criterion(logits.reshape(-1, logits.shape[2]), targets.reshape(-1))

            loss.backward()

            # Log gradient size for stability analysis
            for name, param in model.named_parameters():
                if param.grad is not None:
                    writer.add_histogram(f'gradients/{name}', param.grad, global_step)

            total_norm = 0
            for param in model.parameters():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            writer.add_scalar('gradients/grad_norm', total_norm, global_step)

            optimizer.step()

            # Log learning rate too
            for i, param_group in enumerate(optimizer.param_groups):
                writer.add_scalar(f'learning_rate/group_{i}', param_group['lr'], global_step)

            loss_item = loss.item()
            running_loss += loss_item  # This
            total_loss += loss_item * batch_size
            total_samples += batch_size

            if global_step % print_every == 0:
                print(f"Epoch [{epoch}/{num_epochs}], Step [{global_step}], Training Loss: {running_loss / print_every:.4f}")
                running_loss = 0.0

            if global_step % val_every_n_steps == 0:
                # Online validation (during epochs)
                model.eval()
                val_loss_total = 0.0
                val_samples = 0
                with torch.no_grad():
                    for val_inputs, val_targets in dataloader_val:
                        val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)
                        val_logits, _, _ = model(val_inputs)
                        val_loss = criterion(val_logits.reshape(-1, val_logits.shape[2]), val_targets.reshape(-1))
                        val_loss_total += val_loss.item() * val_inputs.size(0)
                        val_samples += val_inputs.size(0)
                # Average over all batch averages as the loss of a batch is done via 1/n_batch sum(loss for samples in batch)
                # but you want 1/n_samples *sum(loss of all samples)
                val_loss_avg = val_loss_total / val_samples
                train_loss_avg = total_loss / total_samples

                # Save print and log losses
                writer.add_scalar('Loss/train', train_loss_avg, global_step)
                writer.add_scalar('Loss/val', val_loss_avg, global_step)
                history['train_loss'].append((global_step, train_loss_avg))
                history['val_loss'].append((global_step, val_loss_avg))

                print(f"[Step {global_step}] Train Loss: {train_loss_avg:.4f} | Inline Val Loss: {val_loss_avg:.4f}")

                with open(log_file, 'a') as f:
                    f.write(f"{epoch},{global_step},{train_loss_avg:.6f},{val_loss_avg:.6f}\n")

                # Put model back into train mode
                model.train()

        # Full-epoch validation at the end of each epoch
        model.eval()
        val_loss_total = 0.0
        val_samples = 0

        # No gradients needed for evaluation here haha
        with torch.no_grad():
            for val_inputs, val_targets in dataloader_val:
                val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)
                val_logits, _, _ = model(val_inputs)
                val_loss = criterion(val_logits.reshape(-1, val_logits.shape[2]), val_targets.reshape(-1))
                val_loss_total += val_loss.item() * val_inputs.size(0)
                val_samples += val_inputs.size(0)

        val_loss_avg = val_loss_total / val_samples
        train_loss_avg = total_loss / total_samples

        # Save print and log losses
        writer.add_scalar('Loss/train', train_loss_avg, global_step + 0.01) # + 0.01 as to avoid double writing for the same update step between online and after epoch validation
        writer.add_scalar('Loss/val', val_loss_avg, global_step + 0.01)
        history['train_loss'].append((global_step, train_loss_avg))
        history['val_loss'].append((global_step, val_loss_avg))

        with open(log_file, 'a') as f:
            f.write(f"{epoch},{global_step},{train_loss_avg:.6f},{val_loss_avg:.6f}\n")

        print(f"[Epoch {epoch+1}] Train Loss: {train_loss_avg:.4f} | Val Loss: {val_loss_avg:.4f}")

        # Save models
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"model_epoch_{epoch}.pt"))
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, "model_latest.pt"))

        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"model_best_epoch_{epoch}.pt"))

        if scheduler is not None:
            scheduler.step()

    print("Training finished.")
    # Close writer
    writer.close()
    return history

def evaluate_lstm(model, dataloader, device='cpu'):
    model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, _, _ = model(inputs)
            loss = criterion(outputs.reshape(-1, outputs.shape[2]), targets.reshape(-1))
            total_loss += loss.item()

            _, predicted = torch.max(outputs, 2)
            total_correct += (predicted == targets).sum().item()
            total_samples += targets.size(0) * targets.size(1)

    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / total_samples * 100
    print(f"Eval Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    return avg_loss, accuracy



def generate_text(model, start_str, length, vocab, device='cpu', temperature=1.0):
    model.eval()
    model.to(device)

    UNKNOWN_SYMBOL = "<UNK>"

    if UNKNOWN_SYMBOL not in vocab.token2id:
        vocab.token2id[UNKNOWN_SYMBOL] = len(vocab.token2id)
        vocab.id2token.append(UNKNOWN_SYMBOL)

    input_tokens = [vocab.token2id.get(token, vocab.token2id[UNKNOWN_SYMBOL]) for token in start_str]
    input_tensor = torch.tensor(input_tokens, dtype=torch.long).unsqueeze(0).to(device)

    input_tensor = torch.clamp(input_tensor, 0, len(vocab.token2id) - 1)

    generated_tokens = input_tokens.copy()

    hidden = None
    for _ in range(length):
        with torch.no_grad():
            try:
                logits, hidden, _ = model(input_tensor, hidden)
            except Exception as e:
                print(f"Error during model forward pass: {e}")
                print(f"Input tensor shape: {input_tensor.shape}")
                print(f"Input tensor: {input_tensor}")
                raise

        last_token_logits = logits[0, -1, :] / temperature
        probabilities = torch.softmax(last_token_logits, dim=-1)
        predicted_token = torch.multinomial(probabilities, num_samples=1).item()

        predicted_token = min(predicted_token, len(vocab.id2token) - 1)

        generated_tokens.append(predicted_token)
        input_tensor = torch.cat([input_tensor, torch.tensor([[predicted_token]], dtype=torch.long).to(device)], dim=1)

        input_tensor = torch.clamp(input_tensor, 0, len(vocab.token2id) - 1)

    generated_text = ''.join([vocab.id2token[token] if token < len(vocab.id2token) else UNKNOWN_SYMBOL for token in generated_tokens])

    return generated_text
