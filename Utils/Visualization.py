# FIle for collecting visualization results

import matplotlib.pyplot as plt
import os
import csv

# Root path from where the experiments start
ROOT_DIR = os.path.join(os.path.dirname(__file__), '..', 'Experiments', 'RNN') #change that to LSTMs later

# You can change this depending on how you want to set experiment_type
EXPERIMENT_TYPE_MAPPING = {
    "Exp_1": 1,
    "Exp_2": 2,
    "Exp_3": 3
}

def VisualizeLoss(loss):
    '''Visualize the loss curves
    loss: List of tuples (iteration, loss)'''

    x_loss , y_loss = zip(*loss)

    fig, ax = plt.subplots(1, figsize=(12, 5))


    ax.plot(x_loss, y_loss, label="training loss (smoothed)")
    ax.legend()  # Add legend

    ax.set_xlabel("Update steps")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")

    filename = f"Loss_smoothed.png"
        
    plt.savefig(filename, bbox_inches='tight')


def VisualizeLoss(loss_Train1, loss_Val1, acc_Train1, acc_Val1, loss_Train2, loss_Val2, acc_Train2, acc_Val2, experiment_type = 1,filename = f"Loss_acc_smoothed.png"):
    '''Visualize the loss curves
    loss: List of tuples (iteration, loss)
    experiment_type = 1 or 2 or 3'''
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12, 5))
    x_loss_Train1 , y_loss_Train1 = zip(*loss_Train1) # to unpack the tuples (x,y) in the list
    x_loss_Val1 , y_loss_Val1 = zip(*loss_Val1) # to unpack the tuples (x,y) in the list
    x_loss_Train2 , y_loss_Train2 = zip(*loss_Train2) # to unpack the tuples (x,y) in the list
    x_loss_Val2 , y_loss_Val2 = zip(*loss_Val2) # to unpack the tuples (x,y) in the list

    if experiment_type == 1:
        ax1.plot(x_loss_Train1, y_loss_Train1, label="training h=128")
        ax1.plot(x_loss_Val1, y_loss_Val1, label="validation h=128")
        ax1.plot(x_loss_Train2, y_loss_Train2, label="training h=384")
        ax1.plot(x_loss_Val2, y_loss_Val2, label="validation h=384")
    elif experiment_type == 2:
        ax1.plot(x_loss_Train1, y_loss_Train1, label="training n_layers=3")
        ax1.plot(x_loss_Val1, y_loss_Val1, label="validation n_layers=3")
        ax1.plot(x_loss_Train2, y_loss_Train2, label="training n_layers=4")
        ax1.plot(x_loss_Val2, y_loss_Val2, label="validation n_layers=4")
    elif experiment_type == 3:
        ax1.plot(x_loss_Train1, y_loss_Train1, label="training word-tokenization ")
        ax1.plot(x_loss_Val1, y_loss_Val1, label="validation word-tokenization")
        ax1.plot(x_loss_Train2, y_loss_Train2, label="training BPE encoding")
        ax1.plot(x_loss_Val2, y_loss_Val2, label="validation BPE encoding")
    else:    
        ax1.plot(x_loss_Train1, y_loss_Train1, label="training 1")
        ax1.plot(x_loss_Val1, y_loss_Val1, label="validation 1")
        ax1.plot(x_loss_Train2, y_loss_Train2, label="training 2")
        ax1.plot(x_loss_Val2, y_loss_Val2, label="validation 2")
    ax1.set_xlabel("Update steps")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training and Validation Loss")
    ax1.legend()  # Add legend


    x_acc_Train1 , y_acc_Train1 = zip(*acc_Train1) # to unpack the tuples (x,y) in the list
    x_acc_Val1 , y_acc_Val1 = zip(*acc_Val1) # to unpack the tuples (x,y) in the list
    x_acc_Train2 , y_acc_Train2 = zip(*acc_Train2) # to unpack the tuples (x,y) in the list
    x_acc_Val2 , y_acc_Val2 = zip(*acc_Val2) # to unpack the tuples (x,y) in the list

    if experiment_type == 1:
        ax2.plot(x_acc_Train1, y_acc_Train1, label="training h=128")
        ax2.plot(x_acc_Val1, y_acc_Val1, label="validation h=128")
        ax2.plot(x_acc_Train2, y_acc_Train2, label="training h=384")
        ax2.plot(x_acc_Val2, y_acc_Val2, label="validation h=384")
    elif experiment_type == 2:
        ax2.plot(x_acc_Train1, y_acc_Train1, label="training n_layers=3")
        ax2.plot(x_acc_Val1, y_acc_Val1, label="validation n_layers=3")
        ax2.plot(x_acc_Train2, y_acc_Train2, label="training n_layers=4")
        ax2.plot(x_acc_Val2, y_acc_Val2, label="validation n_layers=4")
    elif experiment_type == 3:
        ax2.plot(x_acc_Train1, y_acc_Train1, label="training word-tokenization ")
        ax2.plot(x_acc_Val1, y_acc_Val1, label="validation word-tokenization")
        ax2.plot(x_acc_Train2, y_acc_Train2, label="training BPE encoding")
        ax2.plot(x_acc_Val2, y_acc_Val2, label="validation BPE encoding")
    else:    
        ax2.plot(x_acc_Train1, y_acc_Train1, label="training 1")
        ax2.plot(x_acc_Val1, y_acc_Val1, label="validation 1")
        ax2.plot(x_acc_Train2, y_acc_Train2, label="training 2")
        ax2.plot(x_acc_Val2, y_acc_Val2, label="validation 2")
    
    ax2.set_xlabel("Update steps")
    ax2.set_ylabel("Accuracy in %")
    ax2.set_title("Training and Validation Accuracy")
    ax2.set_ylim([0,100])
    ax2.legend()  # Add legend

    #filename = f"Loss_smoothed.png"
        
    plt.savefig(filename, bbox_inches='tight')


def read_log_file(file_path):
    loss_train, loss_val = [], []
    acc_train, acc_val = [], []

    with open(file_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            step = int(row['global_step'])
            loss_train.append((step, float(row['train_loss'])))
            loss_val.append((step, float(row['val_loss'])))
            acc_train.append((step, float(row['train_acc'])))
            acc_val.append((step, float(row['val_acc'])))
    
    return loss_train, loss_val, acc_train, acc_val

def main():
    for exp_folder in sorted(os.listdir(ROOT_DIR)):
        exp_path = os.path.join(ROOT_DIR, exp_folder)
        if not os.path.isdir(exp_path):
            continue

        log_files = [f for f in os.listdir(exp_path) if f.endswith('.txt')]
        if len(log_files) != 2:
            print(f"[Warning] Skipping {exp_folder} - expected 2 log files, found {len(log_files)}")
            continue

        log_files.sort()  # Consistent order for 1 and 2

        file1_path = os.path.join(exp_path, log_files[0])
        file2_path = os.path.join(exp_path, log_files[1])

        # Read data
        loss_Train1, loss_Val1, acc_Train1, acc_Val1 = read_log_file(file1_path)
        loss_Train2, loss_Val2, acc_Train2, acc_Val2 = read_log_file(file2_path)

        experiment_type = EXPERIMENT_TYPE_MAPPING.get(exp_folder, 0)
        output_file = os.path.join(exp_path, 'Loss_acc_smoothed.png')

        print(f"[Info] Processing {exp_folder} -> Saving to {output_file}")
        
        VisualizeLoss(
            loss_Train1, loss_Val1, acc_Train1, acc_Val1,
            loss_Train2, loss_Val2, acc_Train2, acc_Val2,
            experiment_type=experiment_type,
            filename=output_file
        )

if __name__ == "__main__":
    main()