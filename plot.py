import os
import matplotlib.pyplot as plt
import tensorflow as tf
from config import OUTPUT_DIR

def plot_loss(history, filename="loss.png"):
    plt.figure(figsize=(8,5))
    plt.plot(history.history['loss'], label='Treino')
    plt.plot(history.history['val_loss'], label='Validação')
    plt.title('Curva de Loss por Época')
    plt.xlabel('Época')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[+] Loss salvo em {path}")

def plot_accuracy(history, filename="accuracy.png"):
    plt.figure(figsize=(8,5))
    plt.plot(history.history['accuracy'], label='Treino')
    plt.plot(history.history['val_accuracy'], label='Validação')
    plt.title('Curva de Acurácia por Época')
    plt.xlabel('Época')
    plt.ylabel('Acurácia')
    plt.legend()
    plt.grid(True)
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[+] Acurácia salva em {path}")

def show_cifar10_predictions(model, test_data, class_names, num_images=16, filename="predictions.png"):
    images, labels = next(iter(test_data))
    preds = model.predict(images)
    n = int(num_images**0.5)

    plt.figure(figsize=(n*2.5, n*2.5))
    for i in range(num_images):
        plt.subplot(n, n, i+1)
        plt.imshow(images[i])
        pred_label = class_names[tf.argmax(preds[i]).numpy()]
        true_label = class_names[labels[i].numpy()]
        confidence = tf.nn.softmax(preds[i]).numpy().max()
        plt.title(f"P:{pred_label}\nT:{true_label}\n{confidence:.2f}", fontsize=8)
        plt.axis('off')
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[+] Previsões salvas em {path}")
