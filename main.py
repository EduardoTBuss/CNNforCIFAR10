import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

IMG_SIZE = 32

def normalize ( image , label):
    image = tf.image.resize(image , [IMG_SIZE , IMG_SIZE])
    image = image / 255.0
    return image , label

(ds_train, ds_test), ds_info = tfds.load('cifar10', split=['train[:80%]', 'train[80%:]'],  as_supervised = True, with_info = True)

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomTranslation(0.1, 0.1),
])


train = ds_train.map(normalize).map(lambda x, y: (data_augmentation(x), y)).shuffle(1000).batch(128).prefetch(1)
test = ds_test.map(normalize).shuffle(1000).batch(128).prefetch(1)


model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3) , activation = "relu" , input_shape = (IMG_SIZE , IMG_SIZE , 3)),
    
    tf.keras.layers.Conv2D(32 , (3, 3) , activation = "relu"),
    tf.keras.layers.MaxPooling2D( 2 , 2),
    
    tf.keras.layers.Conv2D(64 , (3, 3) , activation = "relu" ),
    tf.keras.layers.MaxPooling2D( 2 , 2),
    
    tf.keras.layers.Conv2D(128 , (3, 3) , activation = "relu"),
    tf.keras.layers.MaxPooling2D( 2 , 2),
    
    tf.keras.layers.Flatten(),
    
    tf.keras.layers.Dense( 128 , activation = "relu"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10)
])


model.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001),
    loss = "sparse_categorical_crossentropy",
    metrics = ["accuracy"]
)


history = model.fit(
    train,
    epochs = 200,                 
    validation_data=test,       
    verbose=1
)

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


