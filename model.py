from load_data import *
from build_model import *
from get_generator import *


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
    print("-" * 20)
    data_dir     = input_param('data direction', 'carnd_p3' + os.sep + 'data')
    csv_name     = input_param('csv name', 'driving_log.csv')
    input_shape  = input_param('input shape', (160, 320, 3))
    valid_frac   = input_param('proportion of validation', 0.2)
    model_loss   = input_param('loss function', 'mse')
    model_opti   = input_param('model optimizer', 'adam')
    batch_size   = input_param('batch size', 32)
    per_epoch    = input_param('steps per epoch', 100)
    train_epochs = input_param('training epochs', 10)
    valid_steps  = input_param('valid per steps', 20)
    print("=" * 20)

    x_train, y_train, x_valid, y_valid = create_data_set(data_dir, csv_name, valid_frac)

    model = build_model(input_shape)
    model_callbacks = build_callbacks()
    model.compile(loss=model_loss, optimizer=model_opti)
    model.fit_generator(generate_batch(data_dir, x_train, y_train, input_shape, batch_size=batch_size),
                        steps_per_epoch=per_epoch,
                        epochs=train_epochs,
                        verbose=1,
                        callbacks=[n for n in model_callbacks],
                        validation_data=generate_batch(data_dir, x_valid, y_valid, input_shape, batch_size=batch_size),
                        validation_steps=valid_steps)


if __name__ == '__main__':
    main()
