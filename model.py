import tensorflow as tf
from config import LEARNING_RATE

def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation="relu", padding="same", input_shape=(32 ,32 ,3)),
        tf.keras.layers.MaxPooling2D(2,2),

        tf.keras.layers.Conv2D(64, (3,3), activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D(2,2),

        tf.keras.layers.Conv2D(128, (3,3), activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D(2,2),

        tf.keras.layers.Conv2D(256, (3,3), activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D(2,2),
        
        tf.keras.layers.Conv2D(512, (3,3), activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D(2,2),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10 , activation = "softmax")
    ])

   

    model.compile(
        optimizer = tf.keras.optimizers.Adam(
            learning_rate = LEARNING_RATE
        ),
        loss = "sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model
