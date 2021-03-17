import preprocessing_functions as pf
import yaml
# ================================================
# TRAINING STEP - IMPORTANT TO PERPETUATE THE MODEL

# Load data
config_file = pf.read_config_file('config.yaml')
path = config_file[0]['Paths'].get('directory')
data_filename = config_file[0]['Paths'].get('data_filename')
extension = config_file[0]['Paths'].get('data_extension')
cols = config_file[2]['Feature_Groups'].get('data_columns')
df = pf.load_data(path, data_filename, extension, cols)

# divide data set
target = config_file[2]['Feature_Groups'].get('target')
X_train, X_test, y_train, y_test = pf.divide_train_test(df, target)

# get first letter from cabin variable
X_train['cabin'] = pf.extract_cabin_letter(X_train, 'cabin')

# impute categorical variables
cat_vars = config_file[2]['Feature_Groups'].get('categorical_vars')
num_vars = config_file[2]['Feature_Groups'].get('numerical_to_impute')
for var in cat_vars:
    X_train[var] = pf.impute_na(X_train, var, 'Missing')

# impute numerical variables
medians = config_file[1]['Parameters'].get('imputation_dict')
for var in num_vars:
    X_train = pf.add_missing_indicator(X_train, var)
    X_train[var] = pf.impute_na(X_train, var, medians.get(var))

## Group rare labels
frequent_list = config_file[1]['Parameters'].get('frequent_labels')
for var in cat_vars:
    X_train[var] = pf.remove_rare_labels(X_train, var, frequent_list)

# encode categorical variables
dummies = config_file[1]['Parameters'].get('dummy_variables')
for var in cat_vars:
    X_train = pf.encode_categorical(X_train, var)

# check all dummies were added
X_train = pf.check_dummy_variables(X_train, dummies)

# train scaler and save
output_path = config_file[0]['Paths'].get('output_scaler_path')
output_model_path = config_file[0]['Paths'].get('output_model_path')
scaler = pf.train_scaler(X_train, output_path)

# scale train set
X_train = scaler.transform(X_train)
y_train = y_train.astype(int)
# train model and save
pf.train_model(X_train,
               y_train,
               output_model_path
)
print('Finished training')
