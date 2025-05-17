# FIle for collecting visualization results

import matplotlib.pyplot as plt

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