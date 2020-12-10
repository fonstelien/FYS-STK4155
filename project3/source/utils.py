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



def sample_gaps(df):
    '''Finds all gaps > 25 minutes between two consequtive samples and returns them in a np.DataFrame.'''
    T = pd.DataFrame()
    T['time'] = df.index[1:]
    T['gap'] = df.index[1:] - df.index[:-1]
    T = T.set_index('time')
    T = T.asfreq(freq='T', method='bfill')
    T = T.asfreq(freq='10T')
    T = T[T['gap'] > np.timedelta64(25, 'm')]
    T['gap'] = 1
    return T

# def fill_gaps(df):
#     df = df.asfreq(freq='T', method='ffill')
#     return df.asfreq(freq='10T')
