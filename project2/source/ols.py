'''Ordinary Least Squares regression methods'''

from utils import *
from StochasticGradientDescent import SGD


def run_ols_kfold(X, z, SGD, k=10, polynomial_orders=[]):
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

            # X_train, X_test = scale(X_train, X_test, with_std=False)
            
            beta_hat = SGD.run(X_train, z_train)
            mse_train[i] = mse(z_train, X_train @ beta_hat)
            mse_test[i] = mse(z_test, X_test @ beta_hat)

        ols_k_df.loc[pn] = [pn, np.mean(mse_train), np.mean(mse_test)]

    return ols_k_df
