import os
import cv2
import numpy as np
import matplotlib.image as mping
import tensorflow as tf


def processed_data(image, steering):
    if np.random.rand() < 0.2:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    if np.random.rand() < 0.3:
        image = cv2.flip(image, 1)
        steering = -steering
    if np.random.rand() < 0.2:
        image = cv2.GaussianBlur(image, (5, 5), 0)
        
    return image, steering


def generate_batch(data_dir, x_train, y_train, image_shape, batch_size=64):
    images = np.zeros((batch_size, image_shape[0], image_shape[1], image_shape[2]), dtype=np.float32)
    steers = np.zeros(batch_size, dtype=np.float32)
    while True:
        i = 0
        for index in np.random.permutation(x_train.shape[0]):
            image = mping.imread((os.path.join(data_dir, x_train[index, 0].strip())))
            steering_angle = y_train[index]
            
            if np.abs(steering_angle) > 0.1:
                images[i], steers[i] = processed_data(image, steering_angle)
                i += 1
            else:
                if np.random.rand() < 0.3:
                    images[i], steers[i] = processed_data(image, steering_angle)
                    i += 1
                
            if i == batch_size:
                break
                
        yield images, steers


if __name__ == '__main__':
    pass
