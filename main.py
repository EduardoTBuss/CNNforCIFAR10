import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

IMG_SIZE = 64
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCA = 30


def normalize ( image , label):
    image = tf.image.resize(image , [IMG_SIZE , IMG_SIZE])
    image = image / 255.0
    return image , label

(ds_train, ds_test), ds_info = tfds.load('cifar10', split=['train[:90%]', 'train[90%:]'],  as_supervised = True, with_info = True)


train = ds_train.map(normalize).shuffle(1000).batch(BATCH_SIZE).prefetch(1)
test = ds_test.map(normalize).batch(1000).prefetch(1)


model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3) , activation = "relu" , padding='same' , input_shape = (IMG_SIZE , IMG_SIZE , 3)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D( 2 , 2),
    tf.keras.layers.Dropout(0.2),
    
    tf.keras.layers.Conv2D(32 , (3, 3) , activation = "relu" ,padding='same' ),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D( 2 , 2),
    tf.keras.layers.Dropout(0.25),
    
    tf.keras.layers.Conv2D(64 , (3, 3) , activation = "relu" ,padding='same' ),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D( 2 , 2),
    tf.keras.layers.Dropout(0.3),
    
    tf.keras.layers.Conv2D(128 , (3, 3) , activation = "relu" , padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D( 2 , 2),
    tf.keras.layers.Dropout(0.3),
    
    tf.keras.layers.Conv2D(256 , (3, 3) , activation = "relu" ,padding='same' ),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D( 2 , 2),
    tf.keras.layers.Dropout(0.4),
    
    tf.keras.layers.Flatten(),
    
    tf.keras.layers.Dense( 256 , activation = "relu"),
    tf.keras.layers.Dropout(0.6),
    tf.keras.layers.Dense(10 , activation = "softmax")
])


model.compile(
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, decay = 0.0001),
    loss = "sparse_categorical_crossentropy",
    metrics = ["accuracy"]
)


history = model.fit(
    train,
    epochs = EPOCA,                 
    validation_data = test,       
    verbose = 1
)

loss, accuracy = model.evaluate(test)
print(f"\n🔍 Avaliação no conjunto de teste:")
print(f"Loss: {loss:.4f}")
print(f"Acurácia: {accuracy*100:.2f}%")


plt.figure(figsize=(8,5))
plt.plot(history.history['loss'], label='treino')
plt.plot(history.history['val_loss'], label='validação')
plt.title('Curva de Loss por Época')
plt.xlabel('Época')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

class_names = ds_info.features['label'].names

for images, labels in test.take(1):
    preds = model.predict(images)
    plt.figure(figsize=(15,5))
    for i in range(10):
        plt.subplot(2, 5, i+1)
        plt.imshow(images[i])
        pred_label = class_names[tf.argmax(preds[i]).numpy()]
        true_label = class_names[labels[i].numpy()]
        confidence = tf.nn.softmax(preds[i]).numpy().max()
        plt.title(f"P:{pred_label}\nT:{true_label}\nConf:{confidence:.2f}", fontsize=8)
        plt.axis('off')
        
    plt.tight_layout()
    plt.show()
    break


