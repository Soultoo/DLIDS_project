import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import os

class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, output_size, num_layers=1, dropout=0.0, use_pretrained_embedding=False, pretrained_weights=None, persistent_hidden_state=False, fine_tune_embedding=False):
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
            if fine_tune_embedding:
                self.embedding = nn.Embedding.from_pretrained(pretrained_weights, freeze=False) 
            else:
                self.embedding = nn.Embedding.from_pretrained(pretrained_weights, freeze=True) 
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim)

        if self.persistent_hidden_state:
            self.lstm1 = nn.LSTM(embedding_dim, hidden_size, num_layers=1, batch_first=True)
            self.dropout = nn.Dropout(dropout)
            self.lstm_subsequent = nn.LSTM(hidden_size, hidden_size, num_layers=1, batch_first=True)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, batch_first=True, dropout=dropout)

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, X, hidden=None, cell=None):
        '''Implements the forward pass of the model. NOTE DUE TO HOW THE LSTMs are implemented in Pytorch you can only use a stride of seq_lenght on it when you use persistent hidden states
        Inputs:
            X: (tensor) size (n_batch, seq_length) containing integers indices corresponding to tokens
            hidden: (tensor) size: (n_layers, n_batch, hidden_dim)
            cell: (tensor) size: (n_layers, n_batch, hidden_dim), Cell state of the last sample:'''
        n_batch = X.shape[0]
        # Apply embedding
        X = self.embedding(X)  # dim: (n_batch, seq_length, embedding_dim)

        if self.persistent_hidden_state:
            hout_list = []
            cout_list = []
            hidden_states1 = hidden[0].unsqueeze(0)  # dim (1,n_batch, hidden_dim)
            cell_states1 = cell[0].unsqueeze(0) # dim (1,n_batch, hidden_dim)
            output, (hout1, cout1) = self.lstm1(X, (hidden_states1, cell_states1))  # output: (n_batch, seq_length, hidden_dim), hidden: (1, n_batch, hidden_dim)
            hout_list.append(hout1)
            cout_list.append(cout1)

            for layer in range(1, self.n_layers):
                # Retrieve hidden states for that layer
                hidden_temp = hidden[layer].unsqueeze(0)  # dim (1,n_batch, hidden_dim)
                cell_temp = cell[layer].unsqueeze(0) # dim (1,n_batch, hidden_dim)
                # Apply dropout
                output_temp = self.dropout(output)  # dim (n_batch, seq_length, hidden_dim)
                # Apply LSTM
                output_temp, (hout_temp, cout_temp) = self.lstm_subsequent(output_temp, (hidden_temp, cell_temp))  # output: (n_batch, seq_length, hidden_dim), hidden: (n_layers, n_batch, hidden_dim)
                hout_list.append(hout_temp)
                cout_list.append(cout_temp)

            logits = self.fc(output_temp)  # (n_batch, seq_length, vocab_size)
            hidden_out = torch.cat(hout_list, dim=0)  # shape: (n_layers, n_batch, hidden_dim)
            cell_out = torch.cat(cout_list, dim=0) # shape: (n_layers, n_batch, hidden_dim)
            return logits, hidden_out, cell_out, output_temp  # (n_batch, seq_length, vocab_size), (n_layers, n_batch, hidden_dim), (n_batch, seq_length, hidden_dim)
        else:
            output, (hidden, cell) = self.lstm(X, hidden)  # output: (n_batch, seq_length, hidden_dim), hidden: (n_layers, n_batch, hidden_dim)
            logits = self.fc(output)  # (n_batch, seq_length, vocab_size)
            return logits, hidden, cell, output

def train_lstm(model, dataloader_train, dataloader_val, optimizer, persistent_hidden_state=True, hidden_state=None, cell_state=None, hidden_state_val=None, cell_state_val=None, device='cpu', num_epochs=10,
              print_every=100, val_every_n_steps=500, scheduler=None, experiment_dir='./Baseline_LSTM', 
              log_file='training_log.txt', trial=1, resume_training_epoch=0, resume_checkpoint_file=None):

    ###--- Read out data and prepare data structures for logging -- ###
    # Little sanity checks, i.e. if we use a persistent hidden state the dataset
    # across the buckets must be the same and we extract information from them and read out data
    if persistent_hidden_state:
        # Move hidden_state and cell_state to appropriate device
        hidden_state = hidden_state.to(device)
        hidden_state_val = hidden_state_val.to(device)
        cell_state = cell_state.to(device)
        cell_state_val = cell_state_val.to(device)

        # Save original hidden and cell state to reset after each epoch
        hidden_state_og = hidden_state.clone().to(device)
        hidden_state_val_og = hidden_state_val.clone().to(device)
        cell_state_og = cell_state.clone().to(device)
        cell_state_val_og = cell_state_val.clone().to(device)

        same_dataset = True
        # Set the first dataset as the reference dataset
        original_dataset = dataloader_train.bucket_loaders[0].dataset
        for loader_b in dataloader_train.bucket_loaders.values():
            if loader_b.dataset is not original_dataset:
                same_dataset = False
                break
        # Good read out the values needed for a persistent hidden state
        # NOTE: Not needed right now, as we can only ever use the last hidden and cell state of the LSTM, 
        # maybe in the future, recode the LSTM so that it works with arbitrary strides and then you need this again 
        # see RNN for how it works
        if same_dataset:
            pass
        #     hidden_pos = original_dataset.stride - 1
        else:
            raise ValueError("You must specify a dataloader, which has the same dataset for all buckets if you use a persistent hidden state. You most likely have attempted to use UnifiedBucketLoader, BucketedSampler or any other internal Bucket function explicitly. Use create_dataset with persistent_hidden_state set to true instead.")
    if persistent_hidden_state and (hidden_state is None or hidden_state_val is None):
        raise ValueError("You must specify a hidden_state tensor size (n_plays, n_layers, hidden_dim) if you want to use a persistent hidden state")

    # Create directories needed for logging
    checkpoint_dir = os.path.join(experiment_dir, 'checkpoints', f'trial_{trial}')
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Also log the loss in txt files to use them even after the script has run
    log_file_dir = os.path.join(experiment_dir, 'log_files', f'trial_{trial}')
    os.makedirs(log_file_dir, exist_ok=True)
    log_file_path = os.path.join(log_file_dir, log_file)

    tensorboard_dir = os.path.join(experiment_dir, 'runs', f'trial_{trial}')

    # Initialize global values
    best_val_loss = float('inf')
    best_epoch = 0
    global_step = 0  # Collect update steps over all epochs for logging
    # Create history dicts to visualize the loss curves later
    history = {
        'train_loss': [],  # List of (step, loss)
        'val_loss': [],  # List of (step, loss)
        'train_acc': [],  # List of (step, accuracy)
        'val_acc': [],  # List of (step, accuracy)
    }

    ###---- Checkpoint logic ----###
    # Load checkpoint if resuming
    if resume_training_epoch > 0 and resume_checkpoint_file is not None and os.path.exists(resume_checkpoint_file):
        # Load checkpoint, which contains all the information needed
        checkpoint = torch.load(resume_checkpoint_file, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        history = checkpoint.get('history', history)
        global_step = checkpoint.get('global_step', 0)
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"Resumed training from epoch {resume_training_epoch} and global step {global_step}")
    elif resume_training_epoch > 0 and (resume_checkpoint_file is None or not os.path.exists(resume_checkpoint_file)):
        raise FileNotFoundError(f"No checkpoint found at {resume_checkpoint_file}, but you wanted to resume training at epoch {resume_training_epoch}")

    # Reload log file
    if resume_training_epoch > 0:
        # Clean log_file.txt of entries beyond the current global_step (by just parsing the lines smaller than global step)
        new_lines = []
        with open(log_file_path, 'r') as f:
            for line in f:
                if line.startswith("epoch"):  # header
                    new_lines.append(line)
                else:
                    parts = line.strip().split(',')
                    if len(parts) >= 2 and int(parts[1]) < global_step:
                        new_lines.append(line + '\n')
        with open(log_file_path, 'w') as f:
            f.writelines(new_lines)
    else:
        # Create logging file manually
        with open(log_file_path, 'w') as f:
            f.write('epoch,global_step,train_loss,val_loss,train_acc,val_acc\n')

    # Create tensorboard with optional purging for resuming the model
    if resume_training_epoch > 0:
        writer = SummaryWriter(log_dir=tensorboard_dir, purge_step=global_step)
    else:
        writer = SummaryWriter(log_dir=tensorboard_dir)

    ###--- Start real training ----###
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(resume_training_epoch, num_epochs):
        # Put model into training mode at the beginning of each epoch after evaluating after each epoch
        model.train()
        running_loss = 0.0  # This is the loss averaged within batches and added
        total_loss = 0.0  # This is the loss really just summed up without any averaging done
        total_samples = 0  # Used to average the total_loss above
        total_correct = 0  # Used for accuracy calculations

        # Reset hidden and cell state
        hidden_state = hidden_state_og.clone().to(device)
        cell_state = cell_state_og.clone().to(device)
        hidden_state_val = hidden_state_val_og.clone().to(device)
        cell_state_val = cell_state_val_og.clone().to(device)
        for i, (inputs, targets) in enumerate(dataloader_train):
            batch_size = inputs.size(0)
            inputs, targets = inputs.to(device), targets.to(device)  # input: (n_batch, seq_length), (n_batch, seq_length)

            optimizer.zero_grad()
            if persistent_hidden_state:
                batch_play_ids = original_dataset.batch_play_ids
                # Convert to tensor and move to accurate device
                batch_play_ids = torch.tensor(batch_play_ids, device=hidden_state.device)
                batch_hidden_state = hidden_state[batch_play_ids]  # dim: (n_batch, n_layers, hidden_dim)
                batch_cell_state = cell_state[batch_play_ids] #dim: (n_batch, n_layers, hidden_dim)
                # Reshape to correct shape of (n_layers, n_batch, hidden_dim)
                batch_hidden_state = batch_hidden_state.permute(1, 0, 2).contiguous()  # dim: (n_layers, n_batch, hidden_dim)
                batch_cell_state = batch_cell_state.permute(1, 0, 2).contiguous()  # dim: (n_layers, n_batch, hidden_dim)
                # logits: (n_batch, seq_length, vocab_size), hidden: (n_layers, n_batch, hidden_dim), output: (n_batch, seq_length, hidden_dim)
                logits, hidden, cell, output = model(inputs, batch_hidden_state, batch_cell_state)
                # Update the hidden_state vector
                hidden_state[batch_play_ids] = hidden.detach().permute(1, 0, 2)  # Make sure that hidden does not carry any gradients to save it => Use detach
                # Update the cell_state vector
                cell_state[batch_play_ids] = cell.detach().permute(1, 0, 2)  # Make sure that hidden does not carry any gradients to save it => Use detach
            else:
                logits, _, _ = model(inputs)  # logits: (n_batch, seq_length, vocab_size), output: (n_batch, seq_length, hidden_dim), hidden: (n_layers, n_batch, hidden_dim)

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

            # Calculate and log loss for that batch
            loss_item = loss.item()
            running_loss += loss_item
            total_loss += loss_item * batch_size
            # Log accuracy too
            _, predicted = torch.max(logits, 2)  # dim (n_batch, seq_length)
            total_correct += (predicted == targets).sum().item()
            total_samples += batch_size

            if global_step % print_every == 0:
                print(f"Epoch [{epoch}/{num_epochs}], Step [{global_step}], Training Loss: {running_loss / print_every:.4f}, Training Accuracy: {total_correct/total_samples:.4f}")
                running_loss = 0.0

            if global_step % val_every_n_steps == 0:
                # Online validation (during epochs)
                model.eval()
                val_loss_total = 0.0
                val_samples = 0
                val_correct = 0.0
                with torch.no_grad():
                    for val_inputs, val_targets in dataloader_val:
                        val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)
                        if persistent_hidden_state:
                            # Use validation dataset and not the original training dataset (here called original_dataset)
                            batch_play_ids = dataloader_val.bucket_loaders[0].dataset.batch_play_ids  # list of play indices in the batch
                            # Convert to tensor and move to accurate device
                            batch_play_ids = torch.tensor(batch_play_ids, device=device)
                            # Extract hidden state for batch
                            batch_hidden_state = hidden_state_val[batch_play_ids]  # (n_batch, n_layers, hidden_dim)
                            batch_hidden_state = batch_hidden_state.permute(1, 0, 2).contiguous()  # (layers, batch, hidden_dim)
                            batch_cell_state = cell_state_val[batch_play_ids] #dim: (n_batch, n_layers, hidden_dim)
                            batch_cell_state = batch_cell_state.permute(1, 0, 2).contiguous()  # (layers, batch, hidden_dim)
                            
                            # batch_hidden_state = batch_hidden_state.to(device)
                            val_logits, hidden, cell, output = model(val_inputs, batch_hidden_state, batch_cell_state)

                            # Detach and store back to global val hidden state
                            hidden_state_val[batch_play_ids] = hidden.detach().permute(1, 0, 2)
                            cell_state_val[batch_play_ids] = cell.detach().permute(1, 0, 2)
                        else:
                            val_logits, _, _ = model(val_inputs)

                        val_loss = criterion(val_logits.reshape(-1, val_logits.shape[2]), val_targets.reshape(-1))
                        val_loss_total += val_loss.item() * val_inputs.size(0)  # un-average the loss
                        # Log accuracy too
                        _, val_predicted = torch.max(logits, 2)  # dim (n_batch, seq_length)
                        val_correct += (val_predicted == val_targets).sum().item()
                        # Collect validation samples
                        val_samples += val_inputs.size(0)  # collect the amount of samples in the batch
                # Average over all batches as the loss of a batch is done via 1/n_batch sum(loss for samples in batch) and we * n_batch
                # but you want 1/n_samples *sum(loss of all samples)
                val_loss_avg = val_loss_total / val_samples
                train_loss_avg = total_loss / total_samples
                train_acc_avg = total_correct / total_samples
                val_acc_avg = val_correct / val_samples

                # Save print and log losses
                writer.add_scalar('Loss/train', train_loss_avg, global_step)
                writer.add_scalar('Loss/val', val_loss_avg, global_step)
                history['train_loss'].append((global_step, train_loss_avg))
                history['val_loss'].append((global_step, val_loss_avg))

                # Save print and log accuracy
                writer.add_scalar('Accuracy/train', train_acc_avg, global_step)
                writer.add_scalar('Accuracy/val', val_acc_avg, global_step)
                history['train_acc'].append((global_step, train_acc_avg))
                history['val_acc'].append((global_step, val_acc_avg))

                print(f"[Step {global_step}] Train Loss: {train_loss_avg:.4f} | Online Val Loss: {val_loss_avg:.4f} | Train Acc: {train_acc_avg:.4f} | Val Acc: {val_acc_avg:.4f}")

                with open(log_file_path, 'a') as f:
                    f.write(f"{epoch},{global_step},{train_loss_avg:.6f},{val_loss_avg:.6f},{train_acc_avg:.6f},{val_acc_avg:.6f}\n")

                # Put model back into train mode
                model.train()
            # Increment the step count
            global_step += 1

        # Full-epoch validation at the end of each epoch
        model.eval()
        val_loss_total = 0.0
        val_samples = 0
        val_correct = 0

        # No gradients needed for evaluation here
        with torch.no_grad():
            for val_inputs, val_targets in dataloader_val:
                val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)
                if persistent_hidden_state:
                    batch_play_ids = dataloader_val.bucket_loaders[0].dataset.batch_play_ids  # list of play indices in the batch
                    # Convert to tensor and move to accurate device
                    batch_play_ids = torch.tensor(batch_play_ids, device=device)
                    # Extract hidden state for batch
                    batch_hidden_state = hidden_state_val[batch_play_ids]  # (n_batch, n_layers, hidden_dim)
                    batch_hidden_state = batch_hidden_state.permute(1, 0, 2).contiguous()  # (layers, batch, hidden_dim)
                    batch_cell_state = cell_state_val[batch_play_ids] #dim: (n_batch, n_layers, hidden_dim)
                    batch_cell_state = batch_cell_state.permute(1, 0, 2).contiguous()  # (layers, batch, hidden_dim)   
                    # batch_hidden_state.to(device)
                    val_logits, hidden, cell, output = model(val_inputs, batch_hidden_state, batch_cell_state)

                    # Detach and store back to global val hidden state
                    hidden_state_val[batch_play_ids] = hidden.detach().permute(1, 0, 2)
                    cell_state_val[batch_play_ids] = cell.detach().permute(1, 0, 2)
                else:
                    val_logits, _, _ = model(val_inputs)
                
                val_loss = criterion(val_logits.reshape(-1, val_logits.shape[2]), val_targets.reshape(-1))
                val_loss_total += val_loss.item() * val_inputs.size(0)
                # Log accuracy too
                _, val_predicted = torch.max(logits, 2)  # dim (n_batch, seq_length)
                val_correct += (val_predicted == val_targets).sum().item()
                # Collect validation samples
                val_samples += val_inputs.size(0)

        val_loss_avg = val_loss_total / val_samples
        train_loss_avg = total_loss / total_samples
        train_acc_avg = total_correct / total_samples
        val_acc_avg = val_correct / val_samples

        # Save print and log losses
        writer.add_scalar('Loss/train', train_loss_avg, global_step + 0.01)  # + 0.01 as to avoid double writing for the same update step between online and after epoch validation
        writer.add_scalar('Loss/val', val_loss_avg, global_step + 0.01)
        history['train_loss'].append((global_step, train_loss_avg))
        history['val_loss'].append((global_step, val_loss_avg))

        # Save print and log accuracy
        writer.add_scalar('Accuracy/train', train_acc_avg, global_step)
        writer.add_scalar('Accuracy/val', val_acc_avg, global_step)
        history['train_acc'].append((global_step, train_acc_avg))
        history['val_acc'].append((global_step, val_acc_avg))

        with open(log_file_path, 'a') as f:
            f.write(f"{epoch},{global_step},{train_loss_avg:.6f},{val_loss_avg:.6f},{train_acc_avg:.6f},{val_acc_avg:.6f}\n")

        print(f"[Epoch {epoch}] Train Loss: {train_loss_avg:.4f} | Val Loss: {val_loss_avg:.4f} | Train Acc: {train_acc_avg:.4f} | Val Acc: {val_acc_avg:.4f}")

        # Save models
        # Save full checkpoint
        full_state = {
            'epoch': epoch,
            'global_step': global_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'history': history,
            'best_val_loss': best_val_loss
        }

        torch.save(full_state, os.path.join(checkpoint_dir, f"model_epoch_{epoch}.pt"))
        torch.save(full_state, os.path.join(checkpoint_dir, "model_latest.pt"))

        # Save best model if applicable
        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            best_epoch = epoch
            torch.save(full_state, os.path.join(checkpoint_dir, f"model_best_epoch_{best_epoch}.pt"))

        if scheduler is not None:
            scheduler.step()

    print("Training finished.")
    # Close writer
    writer.flush()
    writer.close()

    # Load best model and return it
    checkpoint = torch.load(os.path.join(checkpoint_dir, f"model_best_epoch_{epoch}.pt"), map_location='cuda')
    model.load_state_dict(checkpoint['model_state_dict'])

    return history, model

def evaluate_lstm(model, dataloader, device='cpu'):
    model = model.to(device)
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