import numpy as np
import pandas as pd
import sklearn as skl


def parse_file(fname, col_name=None):
    '''Parses fname csv formatted file. "Stamp" parsed as np.datetime64[m] objects and set as index in returned pd.DataFrame. Time sorted in ascending order. Any duplicates are dropped. Optional rename of "Value" to col_name.'''
    df = pd.read_csv(fname, parse_dates=['Stamp'], index_col=0)
    df = df.sort_index()
    df.index = df.index.values.astype('datetime64[m]')
    df = df[~df.index.duplicated(keep='first')]
    if col_name:
        df = df.rename(columns={'Value':col_name})
    return df


def align(*dfs):
    '''Aligns dfs pd.DataFrames and drops any rows with missing values.'''
    aligned = dfs[0].copy()
    for df in dfs[1:]:
        aligned = aligned.join(df, how='outer')
    aligned = aligned.dropna()
    return aligned


def sample_gaps(df):
    '''Finds all gaps > 25 minutes between two consequtive samples and returns them as an np.DatetimeIndex.'''
    T = pd.DataFrame()
    T['time'] = df.index[1:]
    T['gap'] = df.index[1:] - df.index[:-1]
    T = T.set_index('time')
    T = T.asfreq(freq='T', method='bfill')
    T = T.asfreq(freq='10T')
    T = T[T['gap'] > np.timedelta64(25, 'm')]
    return T.index

# def fill_gaps(df):
#     df = df.asfreq(freq='T', method='ffill')
#     return df.asfreq(freq='10T')
