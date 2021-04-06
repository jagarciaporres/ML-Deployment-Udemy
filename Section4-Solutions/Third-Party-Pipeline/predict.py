import pandas as pd

import joblib
import utils as ut

def make_prediction(input_data):

    # load pipeline and make predictions
    config = ut.read_config_file('config.yaml')
    _titanic_pipe = joblib.load(
    filename=config[0]['Paths'].get('output_model_path')
    )
    # return predictions
    results = _titanic_pipe.predict(input_data)
    return results

if __name__ == '__main__':

    # test pipeline
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    import utils as ut
    config = ut.read_config_file('config.yaml')
    path = config[0]['Paths'].get('directory')
    filename = config[0]['Paths'].get('data_filename')
    extension = config[0]['Paths'].get('data_extension')
    cols = config[2]['Feature_Groups'].get('data_columns')
    target = config[2]['Feature_Groups'].get('target')
    data = ut.load_data(
    path=path, filename=filename, extension=extension, cols=cols
    )
    data[target] = data[target].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        data.drop(target, axis=1),
        data[target],
        test_size=0.2,
        random_state=0)  # we are setting the seed here

    pred = make_prediction(X_test)

    # determine the accuracy
    print('test accuracy: {}'.format(accuracy_score(y_test, pred)))
    print()
