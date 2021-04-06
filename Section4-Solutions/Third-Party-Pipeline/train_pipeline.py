import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
import joblib

from pipeline import titanic_pipe
import utils as ut


def run_training():
    """Train the model."""
    # read training data
    config = ut.read_config_file('config.yaml')
    path = config[0]['Paths'].get('directory')
    filename = config[0]['Paths'].get('data_filename')
    extension = config[0]['Paths'].get('data_extension')
    cols = config[2]['Feature_Groups'].get('data_columns')
    target = config[2]['Feature_Groups'].get('target')
    data = ut.load_data(
    path=path, filename=filename, extension=extension, cols=cols
    )
    # divide train and test
    data[target] = data[target].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
    data.drop(target, axis = 1),
    data[target],
    test_size = 0.2,
    random_state = 0
    )
    # fit pipeline
    titanic_pipe.fit(X_train, y_train)
    # save pipeline
    joblib.dump(titanic_pipe, config[0]['Paths'].get('output_model_path'))

if __name__ == '__main__':
    run_training()
