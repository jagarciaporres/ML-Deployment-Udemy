import yaml
import os
import pandas as pd
import numpy as np

def read_config_file(filename):
    path = os.getcwd()
    with open(path + "/" + filename, 'r') as yamlfile:
        data = yaml.load(yamlfile, Loader = yaml.FullLoader)
    return data

def load_data(path, filename, extension, cols):
    try:
        dataset = pd.read_csv(path + filename + extension)
    except:
        dataset = np.load(path + filename + extension, allow_pickle = True)
        dataset = pd.DataFrame(dataset, columns = cols)
    return dataset
