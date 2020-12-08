import utils
import numpy as np
import pandas as pd


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


def prep_current_temp_model_dataset(aux_df, current_df, temp_df, tempX_df, tempY_df, shift_max):
    '''Preps dataset for the current-temperature-based RNN model. Args are the exogenous signals aux_df and current_df and the target temp_df, in addition to exogenous signals tempX_df and tempY_df, which are the temperatures of the other transformer coils. All rows containing np.NaN are dropped from the dataset. Returns pd.DataFrames (X,y), where X contains the feautures and y contains the target temperatures.'''
    X = aux_df.copy()
    X['aux_off'] = 1 - aux_df

    X = X.join(current_df, how='outer')  # Loading
    X['I2'] = current_df**2  # Power dissipation
    # X['r'] = X['I2']*temp_df.iloc[:,0].shift(periods=1)  # Power in temperature-dependent resistance
    
    X = X.join(tempX_df, how='outer')
    X = X.join(tempY_df, how='outer')
    X = X.join(temp_df, how='outer')
    X = X.dropna()
    
    y = X[temp_df.columns]
    X = X.drop(columns=temp_df.columns)

    return X, y


def reshape_seq_to_vec(X, y, sample_size):
    ''''Reshapes X to shape [num_samples-sample_size, sample_size, features] and y to [num_samples-sample_size, 1, 1] for input into sequence-to-vector RNN with 1 output value.'''
    n, p = X.shape
    features = np.empty((n-sample_size, sample_size, p))
    targets = np.empty((n-sample_size, 1, 1))
    for i in range(n-sample_size):
        features[i,:,:] = X[i:i+sample_size,:]
        targets[i,0,0] = y[i+sample_size,0]
    return features, targets


def reshape_seq_to_seq(X, y, sample_size):
    ''''Reshapes X to shape [num_samples-sample_size, sample_size, features] and y to [num_samples-sample_size, sample_size, 1] for input into sequence-to-sequence RNN with 1 output value.'''
    n, p = X.shape
    features = np.empty((n-sample_size, sample_size, p))
    targets = np.empty((n-sample_size, sample_size, 1))
    for i in range(n-sample_size):
        features[i,:,:] = X[i:i+sample_size,:]
        targets[i,:,0] = y[i:i+sample_size,0]
    return features, targets

