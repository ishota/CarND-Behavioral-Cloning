import numpy as np


def generate_batch(x_train, y_train, image_shape, batch_size=64):
    images = np.zeros((batch_size, image_shape.shape[0], image_shape.shape[1], image_shape.shape[2]), dtype=np.float32)
    steers = np.zeros((batch_size), dtype=np.float32)
    for i in range(batch_size):
        random_index = np.random.randint(0, len(x_train))
        random_image = np.random.randint(0, len(x_train[0]))


if __name__ == '__main__':
    pass
