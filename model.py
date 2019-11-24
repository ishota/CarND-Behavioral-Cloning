import os
from build_model import *
from inspect import currentframe
from keras.models import Sequential
from keras.layers import Lambda, Conv2D, Dropout, Flatten, Dense


def train_model():
    pass


def input_param(names, value):
    print('{} = {}'.format(names, value))
    return value


def main():
    pass


if __name__ == '__main__':
    print("=" * 20)
    print('Parameters')
    data_dir = input_param('data direction', 'carnd_p3' + os.sep + 'data')
    input_shape = input_param('input shape', (160, 320, 3))
    valid_frac = input_param('proportion of validation', 0.2)
    print("=" * 20)

    print("Build model")
    build_model(input_shape)

    main()
