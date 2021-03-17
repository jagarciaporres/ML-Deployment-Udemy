import preprocessing_functions as pf
import yaml
# =========== scoring pipeline =========

# impute categorical variables
def predict(data):
    config_file = pf.read_config_file('config.yaml')

    # extract first letter from cabin
    data['cabin'] = pf.extract_cabin_letter(data, 'cabin')

    # impute NA categorical
    for var in config_file[2]['Feature_Groups'].get('categorical_vars'):
        data[var] = pf.impute_na(data, var, 'Missing')

    # impute NA numerical
    medians = config_file[1]['Parameters'].get('imputation_dict')
    for var in config_file[2]['Feature_Groups'].get('numerical_to_impute'):
        data = pf.add_missing_indicator(data, var)
        data[var] = pf.impute_na(data, var, medians.get(var))

    # Group rare labels
    for var in config_file[2]['Feature_Groups'].get('categorical_vars'):
        data[var] = pf.remove_rare_labels(data, var, config_file[1]['Parameters'].get('frequent_labels'))

    # encode variables
    for var in config_file[2]['Feature_Groups'].get('categorical_vars'):
        data = pf.encode_categorical(data, var)

    # check all dummies were added
    data = pf.check_dummy_variables(data, config_file[1]['Parameters'].get('dummy_variables'))

    # scale variables
    data = pf.scale_features(data, config_file[0]['Paths'].get('output_scaler_path'))

    # make predictions
    predictions = pf.predict(data, config_file[0]['Paths'].get('output_model_path'))

    return predictions

# ======================================

# small test that scripts are working ok

if __name__ == '__main__':

    from sklearn.metrics import accuracy_score
    import preprocessing_functions as pf
    import yaml
    import warnings
    warnings.simplefilter(action='ignore')

    # Load data
    config_file = pf.read_config_file('config.yaml')
    path = config_file[0]['Paths'].get('directory')
    data_filename = config_file[0]['Paths'].get('data_filename')
    extension = config_file[0]['Paths'].get('data_extension')
    cols = config_file[2]['Feature_Groups'].get('data_columns')
    data = pf.load_data(path, data_filename, extension, cols)

    X_train, X_test, y_train, y_test = pf.divide_train_test(data,
                                                            config_file[2]['Feature_Groups'].get('target'))

    pred = predict(X_test)

    # evaluate
    # if your code reprodues the notebook, your output should be:
    # test accuracy: 0.6832
    y_test = y_test.astype(int)
    print('test accuracy: {}'.format(accuracy_score(y_test, pred)))
    print()
