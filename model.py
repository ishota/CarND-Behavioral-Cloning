from load_data import *
from build_model import *
from get_generator import *


def input_param(name, value):
    """
    Display input name and value for checking.
    :param name: parameter name.
    :param value: parameter value.
    :return: value.
    """
    print('{} = {}'.format(name, value))
    return value


def main():
    # Set parameters.
    print("=" * 20)
    print('Parameters')
    print("-" * 20)
    data_dir     = input_param('data direction', os.sep + 'opt' + os.sep + 'carnd_p3' + os.sep + 'data')
    csv_name     = input_param('csv name', 'driving_log.csv')
    input_shape  = input_param('input shape', (160, 320, 3))
    valid_frac   = input_param('proportion of validation', 0.2)
    model_loss   = input_param('loss function', 'mse')
    model_opti   = input_param('model optimizer', 'adam')
    batch_size   = input_param('batch size', 100)
    per_epoch    = input_param('steps per epoch', 300)
    train_epochs = input_param('training epochs', 20)
    valid_steps  = input_param('valid per steps', 50)
    print("=" * 20)

    # Create data set from data directory.
    x_train, y_train, x_valid, y_valid = create_data_set(data_dir, csv_name, valid_frac)

    # Build and compile model.
    model = build_model(input_shape)
    model.compile(loss=model_loss, optimizer=model_opti)

    # Define callbacks.
    model_callbacks = build_callbacks()

    # Learn using python generator.
    model.fit_generator(generate_batch(data_dir, x_train, y_train, input_shape, batch_size=batch_size),
                        steps_per_epoch=per_epoch,
                        epochs=train_epochs,
                        verbose=1,
                        callbacks=[n for n in model_callbacks],
                        validation_data=generate_batch(data_dir, x_valid, y_valid, input_shape, batch_size=batch_size),
                        validation_steps=valid_steps)


if __name__ == '__main__':
    main()
