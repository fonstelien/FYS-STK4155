'''Ordinary Least Squares regression methods'''

from utils import *

def run_ols_bootstrap(X, z, f=None, polynomial_orders=[], train_size=.7, bootstraps=20):
    '''Runs Ordinary Least Square regression with Bootstrap resampling on X, z on every polynomial up to max_poly_order. Returns DataFrame with columns=["pol_order", "train_mse", "test_mse", "test_bias", "test_var"]'''

    if f is None:
        f = z

    ols_bs_df = DataFrame(columns=["pol_order", "train_mse", "test_mse", "test_bias", "test_var"])
    mse_train = np.ndarray(bootstraps)

    for pn in polynomial_orders:
        Xpn = truncate_to_poly(X, pn)
        X_train, X_test, z_train, z_test, _, f_test = skl.model_selection.train_test_split(Xpn, z, f, train_size=train_size)
        X_train, X_test = scale(X_train, X_test, with_std=False)

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
        ols_bs_df.loc[pn] = [pn, mse_train, mse_test, bias_test, var_test]

    return ols_bs_df


def run_ols_kfold(X, z, k=10, polynomial_orders=[]):
    '''Runs Ordinary Least Square regression with k-fold resampling on X, z on every polynomial up to max_poly_order. Returns DataFrame with columns=["pol_order", "train_mse", "test_mse"]'''

    ols_k_df = DataFrame(columns=["pol_order", "train_mse", "test_mse"])
    splits = split(z, k)
    mse_train = np.ndarray(k)
    mse_test = np.ndarray(k)

    for pn in polynomial_orders:
        Xpn = truncate_to_poly(X, pn)
        for i, (train_split, test_split) in enumerate(zip(*splits)):
            X_train = Xpn[train_split]
            z_train = z[train_split]
            X_test = Xpn[test_split]
            z_test = z[test_split]

            X_train, X_test = scale(X_train, X_test, with_std=False)

            beta_hat = np.linalg.pinv(X_train) @ z_train
            mse_train[i] = mse(z_train, X_train @ beta_hat)
            mse_test[i] = mse(z_test, X_test @ beta_hat)

        ols_k_df.loc[pn] = [pn, np.mean(mse_train), np.mean(mse_test)]

    return ols_k_df
