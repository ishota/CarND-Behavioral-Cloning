from load_data import *
from build_model import *
from get_generater import *
from inspect import currentframe
from keras.models import Sequential
from keras.layers import Lambda, Conv2D, Dropout, Flatten, Dense


def train_model():
    pass


def input_param(names, value):
    """
    :param names:
    :param value:
    :return:
    """
    print('{} = {}'.format(names, value))
    return value


if __name__ == '__main__':
    print("=" * 20)
    print('Parameters')
    data_dir = input_param('data direction', 'carnd_p3' + os.sep + 'data')
    csv_name = input_param('csv name', 'driving_log.csv')
    input_shape = input_param('input shape', (160, 320, 3))
    valid_frac = input_param('proportion of validation', 0.2)
    print("=" * 20)

    x_train, y_train, x_valid, y_valid = create_data_set(data_dir, csv_name, valid_frac)

    build_model(input_shape)

    generate_batch(x_train, y_train, input_shape)



