import pandas as pd
import os
from sklearn.model_selection import train_test_split


def create_data_set(data_dir, csv_name, valid_size_frac):
    """

    :param data_dir:
    :param csv_name:
    :param valid_size_frac:
    :return:
    """

    data_set = pd.read_csv(os.path.join(data_dir, csv_name))

    x = data_set[['center', 'left', 'right']].values
    y = data_set['steering'].values

    x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=valid_size_frac)

    return x_train, y_train, x_valid, y_valid


if __name__ == '__main__':
    data_dir = 'carnd_p3' + os.sep + 'data'
    csv_name = 'driving_log.csv'
    valid_size_frac = 0.2
    create_data_set(data_dir, csv_name, valid_size_frac)
