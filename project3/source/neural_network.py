import utils
import numpy as np
import pandas as pd


def make_design_matrix(aux_df, current_df, temp_df):
    ''''''
    X = aux_df.copy()
    X = X.join(current_df**2, how='outer')
    
    X = X.join(temp_df, how='outer')
    X = X.dropna()
    
    y = X[temp_df.columns]
    X = X.drop(columns=temp_df.columns)

    return X, y
