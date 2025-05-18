import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import os

class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, output_size, num_layers=1, activation_function='tanh', dropout=0.0, use_pretrained_embedding=False, pretrained_weights=None, persistent_hidden_state=False):
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_size
        self.output_dim = output_size
        self.n_layers_RNN = num_layers
        self.dropout_rate = dropout
        self.persistent_hidden_state = persistent_hidden_state

        if use_pretrained_embedding:
            if pretrained_weights is None:
                raise ValueError("You must provide pretrained_weights if use_pretrained_embedding is True.")
            self.embedding = nn.Embedding.from_pretrained(pretrained_weights, freeze=True) 
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.rnn = nn.RNN(embedding_dim, hidden_size, num_layers, nonlinearity=activation_function, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, X, hidden=None):
        '''Implements the forward pass of the model
        Inputs:
            x: (tensor) size (n_batch, seq_length) containing integers indeces corresponding to tokens
            hidden: (tensor) size: (n_layers, n_batch, hidden_dim)'''
        #Apply emebding
        X = self.embedding(X) # dim: (n_batch, seq_length, embedding_dim)

        output, hidden = self.rnn(X, hidden) # output: (n_batch, seq_length, hidden_dim) , hidden: (n_layers, n_batch, hidden_dim)
        logits = self.fc(output)  # (n_batch, seq_length, vocab_size)
        return logits, hidden, output 




def train_rnn(model, dataloader_train, dataloader_val, optimizer, device='cpu', num_epochs=10, 
              print_every=100, val_every_n_steps=500, scheduler=None, experiment_dir = './Baseline_RNN', 
              log_file='training_log.txt', trial=1, resume_training_epoch=0, resume_checkpoint_file=None):

    # Create directories needed for logging
    checkpoint_dir = os.path.join(experiment_dir, 'chekpoints', f'trial{trial}')
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Also log the loss in txt files to use them even after the script has run
    log_file_dir = os.path.join(experiment_dir, 'log_files')
    os.makedirs(log_file_dir, exist_ok=True)
    log_file_path = os.path.join(log_file_dir, log_file)

    tensorboard_dir = os.path.join(experiment_dir, 'runs',f'trail{trial}')

    # Initialize globale values
    best_val_loss = float('inf')
    global_step = 0  # Collect update steps over all epochs for logging
    # Create history dicts to visualize the loss curves later
    history = {
        'train_loss': [], # List of (step, loss)
        'val_loss': [], # List of (step, loss)

    }

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
    

    # Load logging file manually (and purge update steps if resumting)
    with open(log_file_path, 'w') as f:
        f.write('epoch,global_step,train_loss,val_loss\n')

    # Create tensorboard with optional purging for resuming the model
    if resume_training_epoch > 0:
        writer = SummaryWriter(log_dir=tensorboard_dir, purge_step=global_step)
    else:
        writer = SummaryWriter(log_dir=tensorboard_dir)

    

    

    model.to(device)
    criterion = nn.CrossEntropyLoss()

    
    for epoch in range(resume_training_epoch, num_epochs):

        # Put model into training mode at the beginning of each epoch after evaluating after each epoch
        model.train()
        running_loss = 0.0 # This is the loss averaged within batches and added
        total_loss = 0.0 # This is the loss really just summed up without any averaging done
        total_samples = 0 # Used to average the total_loss above

        for i, (inputs, targets) in enumerate(dataloader_train):
            
            batch_size = inputs.size(0)
            inputs, targets = inputs.to(device), targets.to(device) # input: (n_batch, seq_length), (n_batch, seq_length) 

            optimizer.zero_grad()
            logits, _, _= model(inputs) # logits: (n_batch, seq_length, vocab_size)
            # You need to flatten the arrays  
            # logits: (n_batch, seq_length, vocab_size) -> (n_batch*seq_length, vocab_size)
            # targets: (n_batch, seq_length) -> (n_batch*seq_length)
            loss = criterion(logits.reshape(-1, logits.shape[2]), targets.reshape(-1,))

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
            running_loss += loss_item  
            total_loss += loss_item * batch_size
            total_samples += batch_size

            if global_step % print_every == 0:
                print(f"Epoch [{epoch}/{num_epochs}], Step [{global_step}], Training Loss: {running_loss / print_every:.4f}")
                running_loss = 0.0

            if global_step % val_every_n_steps == 0:
                # Online validation (during epoochs)
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

                print(f"[Step {global_step}] Train Loss: {train_loss_avg:.4f} | Online Val Loss: {val_loss_avg:.4f}")

                with open(log_file, 'a') as f:
                    f.write(f"{epoch},{global_step},{train_loss_avg:.6f},{val_loss_avg:.6f}\n")

                # Put model back into train mode 
                model.train()
            # Increment the step count
            global_step += 1

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

        print(f"[Epoch {epoch}] Train Loss: {train_loss_avg:.4f} | Val Loss: {val_loss_avg:.4f}")

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
            torch.save(full_state, os.path.join(checkpoint_dir, f"model_best_epoch_{epoch}.pt"))

        if scheduler is not None:
            scheduler.step()


    print("Training finished.")
    # Close writer
    writer.close()
    return history


def evaluate_rnn(model, dataloader, device='cpu'):
    model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, _ = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == targets).sum().item()
            total_samples += targets.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / total_samples * 100
    print(f"Eval Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    return avg_loss, accuracy