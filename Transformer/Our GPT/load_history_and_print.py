import json
import matplotlib.pyplot as plt


with open('baseline_model_checkpoints_our_loader-shakespeare/checkpointtrial_53/history.json', 'r') as f:
    history = json.load(f)
    
    


steps, train_losses = zip(*history['train_loss'])
_, val_losses = zip(*history['val_loss'])

plt.plot(steps, train_losses, label='Train Loss')
plt.plot(steps, val_losses, label='Val Loss')
plt.xlabel("Step")
plt.ylabel("Loss")
plt.legend()
plt.title("Training and Validation Loss")
plt.show()