'''Ridge regression methods'''

from utils import *

def run_ridge_bootstrap(X, z, f=None, polynomial_orders=[], lambdas=[], train_size=.5, bootstraps=20):
    '''Runs Ridge regression with Bootstrap resampling on X, z, f on every polynomial up to polynomial_orders and for every lambda in lambdas. Returns DataFrame with columns=["lambda", "train_mse", "test_mse", "test_bias", "test_var"]'''

    ridge_bs_df = DataFrame()
    ridge_bs_df["lambda"] = lambdas

    num_lambdas = len(lambdas)
    mse_train_buf = np.ndarray(bootstraps)
    mse_train = np.ndarray(num_lambdas)
    mse_test = np.ndarray(num_lambdas)
    bias_test = np.ndarray(num_lambdas)
    var_test = np.ndarray(num_lambdas)

    for pn in polynomial_orders:
        Xpn = truncate_to_poly(X, pn)
        if f is None:
            X_train, X_test, z_train, z_test = skl.model_selection.train_test_split(Xpn, z, train_size=train_size)
            f_test = z_test
        else:
            X_train, X_test, z_train, z_test, _, f_test = skl.model_selection.train_test_split(Xpn, z, f, train_size=train_size)

        X_train, X_test = scale(X_train, X_test)

        z_test_tilde = np.ndarray((X_test.shape[0], bootstraps))
        for i, lmd in enumerate(lambdas):
            L = lmd*np.identity(X_train.shape[1])
            for bs in range(bootstraps):
                X_resampled, z_resampled = skl.utils.resample(X_train, z_train, random_state=bs)
                beta_hat = np.linalg.inv(X_resampled.T @ X_resampled + L) @ X_resampled.T @ z_resampled
                mse_train_buf[bs] = mse(z_resampled, X_resampled @ beta_hat)
                z_test_tilde[:,bs] = (X_test @ beta_hat).ravel()

            mse_train[i] = np.mean(mse_train_buf)
            mse_test[i] = mse(z_test, z_test_tilde)
            bias_test[i] = bias(f_test, z_test_tilde)
            var_test[i] = var(z_test_tilde)

        ridge_bs_df[f"train_mse_{pn}"] = mse_train
        ridge_bs_df[f"test_mse_{pn}"] = mse_test
        ridge_bs_df[f"test_bias_{pn}"] = bias_test
        ridge_bs_df[f"test_var_{pn}"] = var_test

    return ridge_bs_df


def run_ridge_kfold(X, z, k=10, polynomial_orders=[], lambdas=[]):
    '''Performs Ridge regression with k-fold resampling on X, z. Returns DataFrame with columns=["lambda", "train_mse", "test_mse"]'''

    ridge_k_df = DataFrame()
    ridge_k_df["lambda"] = lambdas

    splits = split(z, k)
    num_lambdas = len(lambdas)
    mse_train = np.ndarray(num_lambdas)
    mse_test = np.ndarray(num_lambdas)
    mse_train_buf = np.ndarray(k)
    mse_test_buf = np.ndarray(k)

    for pn in polynomial_orders:
        Xpn = truncate_to_poly(X, pn)
        for i, lmd in enumerate(lambdas):
            for j, (train_split, test_split) in enumerate(zip(*splits)):
                X_train = Xpn[train_split]
                z_train = z[train_split]
                X_test = Xpn[test_split]
                z_test = z[test_split]

                X_train, X_test = scale(X_train, X_test)

                L = lmd*np.identity(X_train.shape[1])
                beta_hat = np.linalg.inv(X_train.T @ X_train + L) @ X_train.T @ z_train

                mse_train_buf[j] = mse(z_train, X_train @ beta_hat)
                mse_test_buf[j] = mse(z_test, X_test @ beta_hat)

            mse_train[i] = np.mean(mse_train_buf)
            mse_test[i] = np.mean(mse_test_buf)

        ridge_k_df[f"train_mse_{pn}"] = mse_train
        ridge_k_df[f"test_mse_{pn}"] = mse_test

    return ridge_k_df
