import tensorflow as tf


def build_model(shape):

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Lambda(lambda x: x/127.5-1.0, input_shape=shape))
    model.add(tf.keras.layers.Cropping2D(cropping=((50, 20), (0, 0)), input_shape=shape))
    model.add(tf.keras.layers.Conv2D(24, (5, 5), strides=(2, 2), activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Conv2D(36, (5, 5), strides=(2, 2), activation='relu'))
    model.add(tf.keras.layers.Dropout(0.4))
    model.add(tf.keras.layers.Conv2D(48, (5, 5), strides=(2, 2), activation='relu'))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Conv2D(64, (5, 5), activation='relu'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(100, kernel_regularizer=tf.keras.regularizers.l2(0.003), activation='relu'))
    model.add(tf.keras.layers.Dense(50, kernel_regularizer=tf.keras.regularizers.l2(0.002), activation='relu'))
    model.add(tf.keras.layers.Dense(10, kernel_regularizer=tf.keras.regularizers.l2(0.001), activation='relu'))
    model.add(tf.keras.layers.Dense(1))
    model.summary()

    return model


def build_callbacks():
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint('model.h5',
                                           monitor='val_loss',
                                           save_best_only=True,
                                           mode='auto'),
        tf.keras.callbacks.TensorBoard(log_dir='./logs')
    ]
    return callbacks


if __name__ == '__main__':
    input_shape = (160, 320, 3)
    build_model(input_shape)
    build_callbacks()
