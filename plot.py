import matplotlib.pyplot as plt

def plot_training(history):
    epochs = list(range(len(history['train_loss'])))

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Val Loss')
    plt.legend()
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.xlim(left=0)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['val_acc'], label='Val Accuracy')
    plt.legend()
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.xlim(left=0)

    plt.tight_layout()
    plt.savefig("training_plot.png")
