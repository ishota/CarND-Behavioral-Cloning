import pandas as pd
import os
from sklearn.model_selection import train_test_split


def create_data(data_dir, csv_name, valid_size_frac):
    """
    Load
    :return: x_train, y_train, x_valid, y_valid
    """

    dat_aset = pd.read_csv(os.path.join(data_dir, csv_name))

    x = dat_aset[['center', 'left', 'right']].values
    y = dat_aset['steering'].values

    x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=valid_size_frac)

    return x_train, y_train, x_valid, y_valid


if __name__ == '__main__':
    data_dir = 'carnd_p3' + os.sep + 'data'
    csv_name = 'driving_log.csv'
    valid_size_frac = 0.2
    create_data(data_dir, csv_name, valid_size_frac)
