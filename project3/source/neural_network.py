import utils
import numpy as np
import pandas as pd


def make_design_matrix(aux_df, current_df, temp_df, tempX_df, tempY_df):
    ''''''
    X = aux_df.copy()
    X = X.join(current_df**2, how='outer')
    
    X = X.join(temp_df, how='outer')
    X = X.join(tempX_df, how='outer')
    X = X.join(tempY_df, how='outer')
    X = X.dropna()
    
    y = X[temp_df.columns]
    X = X.drop(columns=temp_df.columns)

    return X, y


def prep_current_model_dataset(aux_df, current_df, temp_df):
    '''Preps dataset for the current-based RNN model. Args are the exogenous signals aux_df and current_df and the target temp_df. All rows containing np.NaN are dropped from the dataset. Returns pd.DataFrames (X,y), where X contains the features and y contains the target temperatures.'''
    X = aux_df.copy()
    X = X.join(current_df, how='outer')  # Loading
    X['I2'] = current_df**2  # Power dissipation
    # X['r'] = X['I2']*temp_df.iloc[:,0].shift(periods=1)  # Power in temperature-dependent resistance

    X = X.join(temp_df, how='outer')    
    X = X.dropna()
    
    y = X[temp_df.columns]
    X = X.drop(columns=temp_df.columns)

    return X, y
