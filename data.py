import os
import tensorflow as tf
import tensorflow_datasets as tfds
from config import IMG_SIZE, BATCH_SIZE, OUTPUT_DIR




def normalize(image, label):

    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = image / 255.0
    return image, label

def load_data():
    
    (ds_train, ds_test), ds_info = tfds.load(
        'cifar10',
        split=['train[:90%]', 'train[90%:]'],
        as_supervised=True,
        with_info=True
    )

    train = (
        ds_train
        .map(normalize)
        .shuffle(1000)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )
    test = (
        ds_test
        .map(normalize)
        .batch(1000)
        .prefetch(tf.data.AUTOTUNE)
    )

    
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    return train, test, ds_info
