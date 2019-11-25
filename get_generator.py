import os
import numpy as np
import matplotlib.image as mping


def preprocess_image():
    pass


def generate_batch(data_dir, x_train, y_train, image_shape, batch_size=64):
    images = np.zeros((batch_size, image_shape[0], image_shape[1], image_shape[2]), dtype=np.float32)
    steers = np.zeros(batch_size, dtype=np.float32)
    while True:
        i = 0
        for index in np.random.permutation(x_train.shape[0]):
            image = mping.imread((os.path.join(data_dir, x_train[index, 0].strip())))
            steering_angle = y_train[index]
            images[i] = image
            steers[i] = steering_angle
            i += 1
            if i == batch_size:
                break
        yield images, steers


if __name__ == '__main__':
    pass
