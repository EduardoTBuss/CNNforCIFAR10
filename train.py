import tensorflow as tf

def train_model(model, train_data, val_data, epochs,):

    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs,
        verbose=1
    )

    return history
