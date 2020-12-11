'''Methods for autoregression with exogenous signals (ARX modelling).'''

import utils
import numpy as np
import pandas as pd
import sklearn as skl


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
    X['intercept'] = 1
    X = X.join(current_df, how='outer')  # Loading
    X['I2'] = current_df**2  # Power dissipation

    # X['I2_inv'] = (1 + current_df**3)*current_df 
    # X['r'] = X['I2']*temp_df.iloc[:,0].shift(periods=1)  # Power in temperature-dependent resistance
    
    X = X.join(shifts(temp_df, shift_max), how='outer')
    X = X.dropna()
    
    y = X[temp_df.columns]
    X = X.drop(columns=temp_df.columns)

    return X, y



def prep_current_temp_model_dataset(aux_df, current_df, temp_df, tempX_df, tempY_df, shift_max):
    '''Preps dataset for the current-temperature-based autoregression model. Args are the exogenous signals aux_df and current_df and the target temp_df, in addition to exogenous signals tempX_df and tempY_df, which are the temperatures of the other transformer coils. All rows containing np.NaN are dropped from the dataset. Returns pd.DataFrames (X,y) where X contains shift_max shifts in the temperatures for autoregression and y contains the target temperatures.'''
    X = aux_df.copy()
    X['intercept'] = 1
    
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


def scale(*dfs, cols=None, with_std=True, with_mean=True):
    '''Wrapper for skl.preprocessing.StandardScaler. Does fit() on the first pd.DataFrame in dfs, and transform() on all. Scaling is limited to the columns indicated in cols.'''
    scaler = skl.preprocessing.StandardScaler(with_std=with_std, with_mean=with_mean)
    scaler = skl.preprocessing.MinMaxScaler()
    
    dfs = [df for df in dfs]
    df = dfs[0]
    cols = cols if cols else df.columns
    cols_idx = [df.columns.get_loc(col) for col in cols]
    
    scaler.fit(df[cols])
    for i, df in enumerate(dfs):
        a = df.to_numpy()
        a[:, cols_idx] = scaler.transform(df[cols])
        a[:, cols_idx] += 1
        dfs[i] = pd.DataFrame(data=a, index=df.index, columns=df.columns)
        
    return dfs



def ols(X, y):
    '''Performs Ordinary Least Squares fitting of design matrix X to targets y. Both inputs are np.DataFrames. Returns np.DataFrame with coefficient estimates indexed by X.columns.'''
    coeffs = np.linalg.pinv(X.to_numpy()) @ y.to_numpy()
    return pd.DataFrame(data=coeffs.T, columns=X.columns)


def evaluate(coeffs, X, init_temp):
    '''Evaluates model defined by the coeffs and the exogenous signals in X, the design matrix, with the initial temperature init_temp. Returns predicted temperature time series.'''
    columns = X.columns
    indexes = X.index
    n, p = X.shape
    shift_max = len(X.filter(regex='temp__\\d').columns)

    temps = np.ndarray((n,1))
    coeffs = coeffs.to_numpy().T  # into shape (p,1) 
    X = X.to_numpy()

    if shift_max > 0:
        temps[0] = init_temp
        X[:,-1*shift_max:] = init_temp
        for i in range(1, n):
            k = 0
            for j in range(p-1, p-shift_max, -1):
                k = j
                X[i,j] = X[i-1,j-1]
            X[i,k-1] = temps[i-1]
            temps[i] = X[i] @ coeffs

    else:
        for i in range(n):
            temps[i] = X[i] @ coeffs 
        
    return pd.DataFrame(data=temps, columns=['temp'], index=indexes)
