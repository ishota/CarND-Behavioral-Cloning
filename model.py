from load_data import *
from build_model import *
from get_generator import *
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


def main():
    print("=" * 20)
    print('Parameters')
    data_dir = input_param('data direction', 'carnd_p3' + os.sep + 'data')
    csv_name = input_param('csv name', 'driving_log.csv')
    input_shape = input_param('input shape', (160, 320, 3))
    valid_frac = input_param('proportion of validation', 0.2)
    print("=" * 20)

    x_train, y_train, x_valid, y_valid = create_data_set(data_dir, csv_name, valid_frac)

    model = build_model(input_shape)

    model.compile(loss='mse', optimizer='adam')

    model.fit_generator(generate_batch(data_dir, x_train, y_train, input_shape, batch_size=64),
                        steps_per_epoch=2,
                        epochs=1,
                        verbose=1,
                        validation_data=generate_batch(data_dir, x_valid, y_valid, input_shape, batch_size=64),
                        validation_steps=1)


if __name__ == '__main__':
    main()
