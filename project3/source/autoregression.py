'''Methods for autoregression with exogenous signals (ARX modelling).'''

import utils
import numpy as np
import pandas as pd

def shifts(df, shift_max, prefix='temp__'):
    '''Adds to df its first column shifted by 1,2,...,shift_max forwards in time. The new columns are named "<prefix__><shift>" by default, as required by evaluate(). Returns the modified pd.DataFrame.'''
    result_df = df.copy()
    col_name = df.columns[-1]

    for shift in range(1,shift_max+1):
        result_df[prefix + str(shift)] = df.shift(periods=shift)

    return result_df


def prep_current_model_dataset(aux_df, current_df, temp_df, shift_max):
    '''Preps dataset for the current-based autoregression model. Args are the exogenous signals aux_df and current_df and the target temp_df. All rows containing np.NaN are dropped from the dataset. Returns pd.DataFrames (X,y) where X contains shift_max shifts in the temperatures for autoregression and y contains the target temperatures.'''
    X = aux_df.copy()
    X['aux_off'] = 1 - aux_df

    X = X.join(current_df, how='outer')  # Loading
    X['I2'] = current_df**2  # Power dissipation
    # X['r'] = X['I2']*temp_df.iloc[:,0].shift(periods=1)  # Power in temperature-dependent resistance
    
    X = X.join(shifts(temp_df, shift_max), how='outer')
    X = X.dropna()
    
    y = X[temp_df.columns]
    X = X.drop(columns=temp_df.columns)

    return X, y



def prep_current_temp_model_dataset(aux_df, current_df, temp_df, tempX_df, tempY_df, shift_max):
    '''Preps dataset for the current-temperature-based autoregression model. Args are the exogenous signals aux_df and current_df and the target temp_df, in addition to exogenous signals tempX_df and tempY_df, which are the temperatures of the other transformer coils. All rows containing np.NaN are dropped from the dataset. Returns pd.DataFrames (X,y) where X contains shift_max shifts in the temperatures for autoregression and y contains the target temperatures.'''
    X = aux_df.copy()
    X['aux_off'] = 1 - aux_df

    X = X.join(current_df, how='outer')  # Loading
    X['I2'] = current_df**2  # Power dissipation
    # X['r'] = X['I2']*temp_df.iloc[:,0].shift(periods=1)  # Power in temperature-dependent resistance
    
    X = X.join(tempX_df, how='outer')
    X = X.join(tempY_df, how='outer')
    X = X.join(shifts(temp_df, shift_max), how='outer')
    X = X.dropna()
    
    y = X[temp_df.columns]
    X = X.drop(columns=temp_df.columns)

    return X, y


def ols(X, y):
    '''Performs Ordinary Least Squares fitting of design matrix X to targets y. Returns np.ndarray with coefficient estimates. '''
    coeffs = X.copy()
    coeffs = X[:1]
    coeffs.iloc[:,:] = (np.linalg.pinv(X.to_numpy()) @ y.to_numpy()).T
    return coeffs


def evaluate(coeffs, X, init_temp, start=None, end=None):
    '''Evaluates model defined by the coeffs np.ndarray and the exogenous signals in X, the design matrix pd.DataFrame, with the initial temperature init_temp. Optional limitation to (start,end) time delta. Returns predicted temperature time series as pd.DataFrame.'''
    X = X[start:end]
    indexes = X.index
    n, p = X.shape
    shift_max = len(X.filter(regex='temp__\\d').columns)

    temps = np.ndarray((n,1))
    coeffs = coeffs.to_numpy().T
    X = X.to_numpy()

    temps[0,0] = init_temp
    X[:,-1*shift_max:] = init_temp

    for i in range(1, n):
        k = 0
        for j in range(p-1, p-shift_max, -1):
            k = j
            X[i,j] = X[i-1,j-1]
        X[i,k-1] = temps[i-1,0]
        temps[i] = X[i,:] @ coeffs

    temp_df = pd.DataFrame(temps, columns=['temp'])
    temp_df.index = indexes
    return temp_df
