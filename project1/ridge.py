from utils import *

def ridge_bootstrap(*arrays, lambdas=[], train_size=.5, bootstraps=30):
    '''Performs Ridge regression with Bootstrap on arrays=(X, z, f). Returns tuple=(train MSE, test MSE, test Bias, test Variance)'''
    train_test_arrays = skl.model_selection.train_test_split(*arrays, train_size=train_size)
    if len(train_test_arrays) < 4:
        X_train, X_test, z_train, z_test = train_test_arrays[:4]
        f_test = z_test
    else:
        X_train, X_test, z_train, z_test, _, f_test = train_test_arrays

    n, p = X_train.shape
    X_train = X_train[:p,:p]
    z_train = z_train[:p]
    X_test = X_test[:p,:p]
    z_test = z_test[:p]
    f_test = f_test[:p]

    scaler = skl.preprocessing.StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    X_train[:,0] = 1
    X_test[:,0] = 1

    num_lambdas = len(lambdas)
    mse_train_buf = np.ndarray(bootstraps)
    mse_train = np.ndarray(num_lambdas)
    mse_test = np.ndarray(num_lambdas)
    bias_test = np.ndarray(num_lambdas)
    var_test = np.ndarray(num_lambdas)

    z_test_tilde = np.ndarray((p, bootstraps))

    for i, lmd in enumerate(lambdas):
        L = lmd*np.identity(p)

        for bs in range(bootstraps):
            X_resampled, z_resampled = skl.utils.resample(X_train, z_train, random_state=bs)
            beta_hat = np.linalg.inv(X_resampled.T @ X_resampled + L) @ X_resampled.T @ z_resampled
            mse_train_buf[bs] = mse(z_resampled, X_resampled @ beta_hat)
            z_test_tilde[:,bs] = (X_test @ beta_hat).ravel()

        mse_train[i] = np.mean(mse_train_buf)
        mse_test[i] = mse(z_test, z_test_tilde)
        bias_test[i] = bias(f_test, z_test_tilde)
        var_test[i] = var(z_test_tilde)

    return (mse_train, mse_test, bias_test, var_test)


def ridge_kfold(X, z, lambdas=[], k=50):
    '''Performs Ridge regression with k-fold split on X, z. Returns tuple=(train MSE, test MSE)'''
    splits = split(z, k)

    num_lambdas = len(lambdas)
    mse_train = np.ndarray(num_lambdas)
    mse_test = np.ndarray(num_lambdas)

    mse_train_buf = np.ndarray(k)
    mse_test_buf = np.ndarray(k)

    for i, lmd in enumerate(lambdas):
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

            beta_hat = np.linalg.inv(X_train.T @ X_train + lmd*np.identity(p)) @ X_train.T @ z_train

            mse_train_buf[j] = mse(z_train, X_train @ beta_hat)
            mse_test_buf[j] = mse(z_test, X_test @ beta_hat)

        mse_train[i] = np.mean(mse_train_buf)
        mse_test[i] = np.mean(mse_test_buf)

    return (mse_train, mse_test)
