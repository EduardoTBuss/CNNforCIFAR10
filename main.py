import os
import tensorflow as tf
from data import load_data
from model import build_model
from train import train_model
from plot import plot_loss, plot_accuracy, show_cifar10_predictions
from config import EPOCHS

def main():
    
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"\n\n[+] GPU detectada: {gpus}\n\n")
    else:
        print("\n\n[!] Nenhuma GPU detectada. Usando CPU.\n\n")


    train_ds, test_ds, ds_info = load_data()
    class_names = ds_info.features['label'].names



    model = build_model()


    history = train_model(
        model, 
        train_ds,
        test_ds,
        EPOCHS,

    )


    loss, acc = model.evaluate(test_ds)
    print(f"\n🔍 Avaliação no conjunto de teste:")
    print(f"Loss: {loss:.4f}")
    print(f"Acurácia: {acc*100:.2f}%\n")

    plot_loss(history)
    plot_accuracy(history)
    show_cifar10_predictions(model, test_ds, class_names, num_images=16)

if __name__ == "__main__":
    main()
