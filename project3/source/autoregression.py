'''Methods for autoregression with exogenous signals (ARX modelling).'''

import utils
import numpy as np
import pandas as pd

def shifts(df, shift_max):
    '''Adds to df its first column shifted by 1,2,...,shift_max forwards in time. The new columns are named "<first_col_name><shift>". Returns the modified pd.DataFrame.'''
    result_df = df.copy()
    col_name = df.columns[-1]

    for shift in range(1,shift_max+1):
        result_df[col_name + str(shift)] = df.shift(periods=shift)

    return result_df


def make_design_matrix(aux_df, current_df, temp_df, shift_max):
    '''Makes design matrix from the exogenous signals in aux_df and current_df and the target temp_df, with shift_max shifts in the temperatures. All rows containing np.NaN are dropped from the design matrix. Returns pd.DataFrames with design matrix and aligned targets (X, y).'''
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


def ols(X, y, start=None, end=None):
    '''Performs Ordinary Least Squares fitting of design matrix X to targets y in time delta (start,end). Returns np.ndarray with coefficient estimates. '''
    X = X[start:end]
    y = y[start:end]
    coeffs = np.linalg.pinv(X) @ y
    return coeffs.to_numpy()


def evaluate(coeffs, X, init_temp, start=None, end=None):
    '''Evaluates model defined by the coeffs np.ndarray and the exogenous signals in X, the design matrix pd.DataFrame, with the initial temperature init_temp. Optional limitation to (start,end) time delta. Returns predicted temperature time series as pd.DataFrame.'''
    X = X[start:end]
    p = len(coeffs)
    t_mem = len(X.filter(regex='temp*').columns)

    temp_df = X.copy()
    temp_df = temp_df.drop(columns=X.columns)
    temp_df['temp'] = init_temp

    X = X.copy()
    X.iloc[:,-1*t_mem:] = init_temp

    for i in range(1, temp_df.size):
        k = 0
        for j in range(p-1, p-t_mem, -1):
            k = j
            X.iloc[i,j] = X.iloc[i-1,j-1]
        X.iloc[i,k-1] = temp_df.iloc[i-1,0]
        temp_df.iloc[i,0] = X.iloc[i] @ coeffs
        
    return temp_df
