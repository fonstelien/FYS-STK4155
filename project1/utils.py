import os
import sys
import glob
import operator as op
import itertools as it
from functools import reduce, partial
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as skl
from sklearn import *

# From the assignment; The function that we are goint to approximate
def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

# Generate samples from the FrankeFunction with some optional noise
def franke_sampler(x, y, noise=.1):
    f = FrankeFunction(x, y).reshape(len(x), 1)
    var_eps = noise * np.var(f)
    z = f + np.sqrt(var_eps)*np.random.randn(len(x), 1)
    return (z, f, var_eps)

# Generate samples from the FrankeFunction with some optional noise
def image_sampler(x, y, img):
    import imageio
    terrain = imageio.imread(img)
    return terrain[x,y]

# Create a mesh with sn uniform randomly scattered points in the rectangle (x_start, y_start), (x_end, y_end). Returned ndarrays have shape (sn, 1)
def randmesh(sn=500, x_start=0., y_start=0., x_end=1., y_end=1., random_state=0):
    np.random.seed(random_state)
    x = np.random.uniform(x_start, x_end, (sn, 1))
    y = np.random.uniform(y_start, y_end, (sn, 1))
    return (x, y)

# Create a mesh with sn uniform randomly scattered points in the rectangle (x_start, y_start), (x_end, y_end). Returned ndarrays have shape (sn, 1)
def randmesh_int(x_end, y_end, sn=500, x_start=0, y_start=0, random_state=0):
    np.random.seed(random_state)
    x = np.random.randint(x_start, x_end, (sn, 1))
    y = np.random.randint(y_start, y_end, (sn, 1))
    return (x, y)

# Make design matrix with polinomial degree np in two variables
def make_design_matrix(x, y, pn=5):
    X = np.ndarray([len(x), int((pn+1)*(pn+2)/2)])

    ex = [0]*int((pn+1)*(pn+2)/2)
    ey = [0]*int((pn+1)*(pn+2)/2)
    kx = ky = 0
    for i in range(pn+1):
        for j in range(i+1):
            ex[kx] = j
            kx += 1
        for j in range(i,-1,-1):
            ey[ky] = j
            ky += 1

    for i, (xi, yi) in enumerate(zip(x, y)):
        X[i,:] = [(xi**px)*(yi**py) for px, py in zip(ex, ey)]

    return X

# Defining some useful functions
def mse(y, y_tilde):
    return np.mean(np.mean((y - y_tilde)**2, axis=1, keepdims=True))

def r2(y, y_tilde):
    return 1 - sum((y - np.mean(y_tilde, axis=1, keepdims=True))**2)/sum((y - np.mean(y))**2)

def bias(f, y_tilde):
    return np.mean((f - np.mean(y_tilde, axis=1, keepdims=True))**2)

def var(y):
    return np.mean(np.var(y, axis=1, keepdims=True))

def best_r2(mse_array, y):
    return 1 - np.min(mse_array)/np.mean(sum((y - np.mean(y))**2)/len(y))

# Preprocessing X
def truncate_to_poly(X, pn):
    p = int((pn+1)*(pn+2)/2)
    return np.copy(X[:,:p])

def scale(X_train, X_test, **kwargs):
    '''Wrapper for skl.preprocessing.StandardScaler. **kwargs are forwarded to StandardScaler'''
    scaler = skl.preprocessing.StandardScaler(**kwargs)
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    X_train[:,0] = 1
    X_test[:,0] = 1
    return (X_train, X_test)


def center(X):
    X = np.copy(X)
    if X.shape[1] == 1:
        return X
    X -= np.mean(X, axis=0, keepdims=True)
    X[:,0] = 1
    return X

def center_scale(X):
    X = np.copy(X)
    if X.shape[1] == 1:
        return X
    means = np.mean(X, axis=0, keepdims=True)
    stds = np.std(X, axis=0, keepdims=True)
    stds[0,0] = 1
    X = (X - means) / stds
    X[:,0] = 1
    return X


def split(y, k):
    '''k-fold splitter function. NOTE: Choose k such that len(z)%k is zero, ie the split is even!'''
    n = len(y)
    s = n//k  # samples in each split
    last_idx = n - n%k  # remove overshooting samples
    test_splits = [list(range(i, i+s)) for i in range(0, last_idx, s)]
    train_splits = [list(set(range(last_idx)) - set(test_split)) for test_split in test_splits]
    return (train_splits, test_splits)
