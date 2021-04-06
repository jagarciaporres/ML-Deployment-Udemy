from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import preprocessors as pp
import utils as ut

config = ut.read_config_file('config.yaml')

titanic_pipe = Pipeline(
    # complete with the list of steps from the preprocessors file
    # and the list of variables from the config
    [
        ('categorical_imputer',
        pp.CategoricalImputer
        (variables=config[2]['Feature_Groups'].get('categorical_vars'))
        ),

        ('missing_indicator',
        pp.MissingIndicator(
        variables=config[2]['Feature_Groups'].get('numerical_to_impute'))
        ),

        ('numerical_imputer',
        pp.NumericalImputer(variables =
        config[2]['Feature_Groups'].get('numerical_to_impute'))
        ),

        ('cabin_variable',
        pp.ExtractFirstLetter(
        variables=config[2]['Feature_Groups'].get('categorical_vars')[1])
        ),

        ('rare_label_encoder',
        pp.RareLabelCategoricalEncoder(
        tol=0.05,
        variables=config[2]['Feature_Groups'].get('categorical_vars'))
        ),

        ('categorical_encoder',
        pp.CategoricalEncoder(
        variables=config[2]['Feature_Groups'].get('categorical_vars'))
        ),

        ('scaler', StandardScaler()),

        ('linear_model', LogisticRegression(C=0.0005, random_state=0))
    ]
)
