from utils import *

def ols_bootstrap(*arrays, train_size=.7, bootstraps=30):
    '''Performs Ordinary Least Squares with Bootstrap on arrays=(X, z, f). Returns tuple=(train MSE, test MSE, test Bias, test Variance)'''
    train_test_arrays = skl.model_selection.train_test_split(*arrays, train_size=train_size)
    if len(train_test_arrays) < 4:
        X_train, X_test, z_train, z_test = train_test_arrays[:4]
        f_test = z_test
    else:
        X_train, X_test, z_train, z_test, _, f_test = train_test_arrays

    X_train = center(X_train)
    X_test = center(X_test)

    mse_train = np.ndarray(bootstraps)
    z_test_tilde = np.ndarray((len(z_test), bootstraps))

    for bs in range(bootstraps):
        X_resampled, z_resampled = skl.utils.resample(X_train, z_train, random_state=bs)
        beta_hat = np.linalg.pinv(X_resampled) @ z_resampled
        mse_train = mse(z_resampled, X_resampled @ beta_hat)
        z_test_tilde[:, bs] = (X_test @ beta_hat).ravel()

    mse_train = np.mean(mse_train)
    mse_test = mse(z_test, z_test_tilde)
    bias_test = bias(z_test, z_test_tilde)
    var_test = var(z_test_tilde)

    return (mse_train, mse_test, bias_test, var_test)


def ols_kfold(X, z, k=50):
    '''Performs Ordinary Least Squares with k-fold split on X, z. Returns tuple=(train MSE, test MSE)'''
    splits = split(z, k)

    mse_train = np.ndarray(k)
    mse_test = np.ndarray(k)

    for i, (train_split, test_split) in enumerate(zip(*splits)):
        X_train = X[train_split]
        z_train = z[train_split]
        X_test = X[test_split]
        z_test = z[test_split]

        X_train = center(X_train)
        X_test = center(X_test)

        beta_hat = np.linalg.pinv(X_train) @ z_train
        mse_train[i] = mse(z_train, X_train @ beta_hat)
        mse_test[i] = mse(z_test, X_test @ beta_hat)

    return (np.mean(mse_train), np.mean(mse_test))
