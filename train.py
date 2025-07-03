import tensorflow as tf

def train_model(model, train_data, val_data, epochs,):

    history = model.fit(
        train_data,
        validation_data = val_data,
        epochs = epochs,
        callbacks = tf.keras.callbacks.ReduceLROnPlateau(monitor = "val_loss", factor = 0.4, patience = 7 , min_lr = 0.000001),
        verbose = 1
    )

    return history
