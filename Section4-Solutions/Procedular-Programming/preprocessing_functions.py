import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import yaml
import joblib


# Individual pre-processing and training functions
# ================================================
def read_config_file(filename):
    path = os.getcwd()
    with open(path + "/" + filename, 'r') as yamlfile:
        data = yaml.load(yamlfile, Loader=yaml.FullLoader)
    return data

def load_data(path, filename, extension, cols):
    # Function loads data for training
    try:
        dataset = pd.read_csv(path + filename + extension)
    except:
        dataset = np.load(path + filename + extension, allow_pickle = True)
        dataset = pd.DataFrame(dataset, columns=cols)
    return dataset

def divide_train_test(df, target):
    # Function divides data set in train and test
    X_train, X_test, y_train, y_test = train_test_split(
    df.drop(target, axis = 1), df[target],
    test_size = 0.2, random_state = 0
    )
    print("Training set shape: " + str(X_train.shape))
    print("Test set shape: " + str(X_test.shape))
    return X_train, X_test, y_train, y_test



def extract_cabin_letter(df, var):
    # captures the first letter
    df = df.copy()
    return df[var].str[0]



def add_missing_indicator(df, var):
    # function adds a binary missing value indicator
    df[var+'_NA'] = np.where(df[var].isnull(), 1, 0)
    return df



def impute_na(df, var, replacement = 'Missing'):
    # function replaces NA by value entered by user
    # or by string Missing (default behaviour)
    return df[var].fillna(replacement)


def remove_rare_labels(df, var, freq_list):
    # groups labels that are not in the frequent list into the umbrella
    # group Rare
    return np.where(df[var].isin(freq_list[var]), df[var], 'Rare')

def encode_categorical(df, var):
    # adds ohe variables and removes original categorical variable
    df = df.copy()
    df = pd.concat(
    [df, pd.get_dummies(df[var], prefix = var, drop_first = True)], axis = 1
    )
    df.drop(labels = [var], axis = 1, inplace = True)
    return df

def check_dummy_variables(df, dummy_list):

    # check that all missing variables where added when encoding, otherwise
    # add the ones that are missing
    missing_cols = [
    col for col in dummy_list if col in dummy_list
    and col not in df.columns
    ]
    if len(missing_cols) == 0:
        print("All variables were added")
    else:
        df[missing_cols] = 0
    return df

def train_scaler(df, output_path):
    # train and save scaler
    scaler = StandardScaler()
    scaler.fit(df)
    joblib.dump(scaler, output_path)
    return scaler



def scale_features(df, output_path):
    # load scaler and transform data
    scaler = joblib.load(output_path)
    return scaler.transform(df)



def train_model(df, target, output_path):
    # train and save model
    model = LogisticRegression(C=0.0005, random_state=0)
    model.fit(df, target)
    joblib.dump(model, output_path)
    return None


def predict(df, model):
    # load model and get predictions
    model = joblib.load(model)
    return model.predict(df)
