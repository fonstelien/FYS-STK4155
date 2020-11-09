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
import imageio
from numba import jit, njit, prange


## ==========================================================
## Below: New methods for Project 2

def onehot(targets, classes):
    '''Translates 1D targets array with classes=0,1,2,... into a 2D one-hot np.ndarray.'''
    n = len(targets)
    hot_targets = np.zeros((n, classes))
    for r, i in enumerate(targets):
        hot_targets[r,i] = 1
    return hot_targets

def softmax(z):
    '''Returns an nd.ndarray where softmax has been applies on z.'''
    stability_factor = z.max()
    e = np.exp(z-stability_factor)
    return e/np.sum(e, axis=1, keepdims=True)



## Above: Project 2
## ==========================================================
## Below: Project 1

# From assignment 1:
def FrankeFunction(x,y):
    '''Evaluates the Franke function at x,y'''
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4


def franke_sampler(x, y, var_eps=.01):
    '''Generate samples from the FrankeFunction with some optional noise. Returns tuple=(z, f, eps)'''
    f = FrankeFunction(x, y).reshape(len(x), 1)
    eps = np.sqrt(var_eps)*np.random.randn(len(x), 1)
    z = f + eps
    return (z, f, eps)


def image_sampler(file_name, sn, random_state=0):
    '''Draws sn normally distributed samples from file_name GEOTIF image and returns a tuple=(x, y, z, (x_dim, y_dim, z_norm)), where x, y z, are normalized with x_dim, y_dim, z_norm'''
    img = imageio.imread(file_name)

    x_dim, y_dim = img.shape
    z_norm = img.max()

    np.random.seed(random_state)
    x = np.random.randint(0, x_dim, (sn, 1))
    y = np.random.randint(0, y_dim, (sn, 1))
    z = img[x, y]

    return (x/x_dim, y/y_dim, z/z_norm, (x_dim, y_dim, z_norm))


def randmesh(sn=500, x_start=0., y_start=0., x_end=1., y_end=1., random_state=0):
    '''Create a mesh with sn uniform randomly scattered points in the rectangle (x_start, y_start), (x_end, y_end). Returned ndarrays have shape (sn, 1)'''
    np.random.seed(random_state)
    x = np.random.uniform(x_start, x_end, (sn, 1))
    y = np.random.uniform(y_start, y_end, (sn, 1))
    return (x, y)


def randmesh_int(x_end, y_end, x_start=0, y_start=0, sn=500, random_state=0):
    '''Create a mesh with sn uniform randomly scattered points in the rectangle (x_start, y_start), (x_end, y_end). Returned ndarrays have shape (sn, 1)'''
    np.random.seed(random_state)
    x = np.random.randint(x_start, x_end, (sn, 1))
    y = np.random.randint(y_start, y_end, (sn, 1))
    return (x, y)


def make_design_matrix(x, y, pn=5):
    '''Make design matrix with polinomial degree pn in two variables. Rows are on the form Pn(x,y) = [1,x,y,x^2,xy,y^2,x^3,x^2y,...]'''
    X = np.ndarray([len(x), int((pn+1)*(pn+2)/2)])

    n_terms = int((pn+1)*(pn+2)/2)
    x_exponents = [0]*n_terms
    y_exponents = [0]*n_terms

    xn = yn = 0
    for i in range(pn+1):
        for j in range(i+1):
            y_exponents[yn] = j
            yn += 1
        for j in range(i,-1,-1):
            x_exponents[xn] = j
            xn += 1

    for i, (xi, yi) in enumerate(zip(x, y)):
        X[i,:] = [(xi**xn)*(yi**yn) for xn, yn in zip(x_exponents, y_exponents)]

    return X


# Defining some useful functions
def mse(y, y_tilde):
    return np.mean(np.mean((y - y_tilde)**2, axis=1, keepdims=True))

def r2(y, y_tilde):
    return (1 - sum((y - np.mean(y_tilde, axis=1, keepdims=True))**2)/sum((y - np.mean(y))**2))[0]

def bias(f, y_tilde):
    return np.mean((f - np.mean(y_tilde, axis=1, keepdims=True))**2)

def var(y):
    return np.mean(np.var(y, axis=1, keepdims=True))

def best_r2(mse_array, y):
    return 1 - np.min(mse_array)/np.mean(sum((y - np.mean(y))**2)/len(y))

def beta_hat_confidence_intervals(X, y, var_eps, ci=95):
    '''Returns an ndarray((3, p)) of confidence intervals for the estimators of y on X'''
    std_err_multipliers = {90:1.645, 95:1.96, 98:2.326, 99:2.576}
    std_err_multiplier = std_err_multipliers[ci]
    n, p = X.shape
    intervals = np.ndarray((p, 3))

    XtX_inv = np.linalg.inv(X.T @ X)
    beta_hat = XtX_inv @ X.T @ y
    y_tilde = X @ beta_hat
    var_beta_hat = var_eps*np.sqrt(np.diag(XtX_inv)).reshape(p,1)
    intervals[:,0] = (beta_hat - std_err_multiplier*np.sqrt(var_beta_hat)).ravel()
    intervals[:,1] = beta_hat.ravel()
    intervals[:,2] = (beta_hat + std_err_multiplier*np.sqrt(var_beta_hat)).ravel()

    return intervals.T


# Preprocessing X
def truncate_to_poly(X, pn):
    '''Truncates the design matrix X to right shape for polynomial degree pn'''
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


def split(y, k):
    '''k-fold splitter function. NOTE: Choose k such that len(z)%k is zero, ie the split is even!'''
    n = len(y)
    s = n//k  # samples in each split
    last_idx = n - n%k  # remove overshooting samples
    test_splits = [list(range(i, i+s)) for i in range(0, last_idx, s)]
    train_splits = [list(set(range(last_idx)) - set(test_split)) for test_split in test_splits]
    return (train_splits, test_splits)


def best_lambda_mse(df=DataFrame(), polynomial_orders=[], col_prefix=str()):
    '''Searches df for the lambda which gives the best MSE for each polynomial in polynomial_orders and returns the lambdas and MSEs in as (lambdas, best_mse)'''
    best_mse = np.ndarray(len(polynomial_orders))
    row_best_lambda = df.filter(regex=col_prefix).idxmin()
    lambdas = df['lambda'][row_best_lambda].to_numpy()
    for i, row in enumerate(row_best_lambda):
        pn = polynomial_orders[i]
        best_mse[i] = df.at[row, col_prefix + str(pn)]
    return lambdas, best_mse


@njit(parallel=True)
def run_image_calcs(img, pn, betas):
    '''numba-parallelized part of generate_image()'''
    n_terms = int((pn+1)*(pn+2)/2)
    x_exponents = [0]*n_terms
    y_exponents = [0]*n_terms

    xn = yn = 0
    for i in range(pn+1):
        for j in range(i+1):
            x_exponents[xn] = j
            xn += 1
        for j in range(i,-1,-1):
            y_exponents[yn] = j
            yn += 1

    x_dim, y_dim = img.shape
    xi = np.linspace(0,1,x_dim)
    yj = np.linspace(0,1,y_dim)
    img[:,:] = 0

    for i in prange(x_dim):
        for j in range(y_dim):
            for k in range(n_terms):
                b = betas[k,0]
                xn = x_exponents[k]
                yn = y_exponents[k]
                img[i,j] += b*(xi[i]**xn)*(yj[j]**yn)

def generate_image(pn, xyz_norm, betas):
    '''Generates image with pn-th polynomial Pn(x,y) and coefficient vector betas'''
    x_dim, y_dim, z_norm = xyz_norm
    img = np.ndarray((x_dim, y_dim))
    run_image_calcs(img, pn, betas)
    img = img * z_norm
    img = img.astype(int)
    return img
