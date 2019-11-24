import tensorflow as tf


def build_model(input_shape):

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Lambda(lambda x: x/127.5-1.0, input_shape=input_shape))
    model.add(tf.keras.layers.Conv2D(24, (5, 5), strides=(2, 2), activation='relu', input_shape=input_shape))
    model.add(tf.keras.layers.Conv2D(36, (5, 5), strides=(2, 2), activation='relu'))
    model.add(tf.keras.layers.Conv2D(48, (5, 5), strides=(2, 2), activation='relu'))
    model.add(tf.keras.layers.Conv2D(64, (5, 5), activation='relu'))
    model.add(tf.keras.layers.Conv2D(64, (5, 5), activation='relu'))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(100, activation='relu'))
    model.add(tf.keras.layers.Dense(50, activation='relu'))
    model.add(tf.keras.layers.Dense(10, activation='relu'))
    model.add(tf.keras.layers.Dense(1))
    model.summary()

    return model


if __name__ == '__main__':
    input_shape = (160, 320, 3)
    build_model(input_shape)
