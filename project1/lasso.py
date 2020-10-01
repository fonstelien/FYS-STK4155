from utils import *
from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

@ignore_warnings(category=ConvergenceWarning)
def lasso_bootstrap(*arrays, lambdas=[], train_size=.7, bootstraps=30, **kwargs):
    '''Performs LASSO regression with Bootstrap on arrays=(X, z, f). Returns tuple=(train MSE, test MSE, test Bias, test Variance)'''
    train_test_arrays = skl.model_selection.train_test_split(*arrays, train_size=train_size)
    if len(train_test_arrays) < 4:
        X_train, X_test, z_train, z_test = train_test_arrays[:4]
        f_test = z_test
    else:
        X_train, X_test, z_train, z_test, _, f_test = train_test_arrays

    scaler = skl.preprocessing.StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)
    X_train[:,0] = 1
    X_test[:,0] = 1

    num_lambdas = len(lambdas)
    mse_train_buf = np.ndarray(bootstraps)
    mse_train = np.ndarray(num_lambdas)
    mse_test = np.ndarray(num_lambdas)
    bias_test = np.ndarray(num_lambdas)
    var_test = np.ndarray(num_lambdas)

    beta_hat = np.ndarray((X_train.shape[1], 1))
    z_test_tilde = np.ndarray((z_test.shape[0], bootstraps))

    for i, lmd in enumerate(lambdas):
        lasso = skl.linear_model.Lasso(alpha=lmd, fit_intercept=False, **kwargs)

        for bs in range(bootstraps):
            X_resampled, z_resampled = skl.utils.resample(X_train, z_train, random_state=bs)

            lasso.fit(X_resampled, z_resampled)
            beta_hat[:,0] = lasso.coef_

            mse_train_buf[bs] = mse(z_resampled, X_resampled @ beta_hat)
            z_test_tilde[:,bs] = (X_test @ beta_hat).ravel()

        mse_train[i] = np.mean(mse_train_buf)
        mse_test[i] = mse(z_test, z_test_tilde)
        bias_test[i] = bias(f_test, z_test_tilde)
        var_test[i] = var(z_test_tilde)

    return (mse_train, mse_test, bias_test, var_test)


@ignore_warnings(category=ConvergenceWarning)
def lasso_kfold(X, z, lambdas=[], k=50, **kwargs):
    '''Performs LASSO regression with k-fold split on X, z. Returns tuple=(train MSE, test MSE)'''
    splits = split(z, k)

    num_lambdas = len(lambdas)
    mse_train = np.ndarray(num_lambdas)
    mse_test = np.ndarray(num_lambdas)

    mse_train_buf = np.ndarray(k)
    mse_test_buf = np.ndarray(k)

    for i, lmd in enumerate(lambdas):
        lasso = skl.linear_model.Lasso(alpha=lmd, fit_intercept=False, **kwargs)

        for j, (train_split, test_split) in enumerate(zip(*splits)):
            X_train = X[train_split]
            z_train = z[train_split]
            X_test = X[test_split]
            z_test = z[test_split]

            n, p = X_train.shape
            X_train = X_train[:p,:p]
            z_train = z_train[:p]
            X_test = X_test[:p,:p]
            z_test = z_test[:p]

            scaler = skl.preprocessing.StandardScaler()
            scaler.fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)
            X_train[:,0] = 1
            X_test[:,0] = 1

            lasso.fit(X_train, z_train)
            beta_hat = np.ndarray((X_train.shape[1], 1))
            beta_hat[:,0] = lasso.coef_

            mse_train_buf[j] = mse(z_train, X_train @ beta_hat)
            mse_test_buf[j] = mse(z_test, X_test @ beta_hat)

        mse_train[i] = np.mean(mse_train_buf)
        mse_test[i] = np.mean(mse_test_buf)

    return (mse_train, mse_test)
