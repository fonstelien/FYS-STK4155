'''Ridge regression methods'''

from utils import *


def run_ridge_kfold(X, z, SGD, k=10, polynomial_orders=[], lambdas=[]):
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

                # L = lmd*np.identity(X_train.shape[1])
                # beta_hat = np.linalg.inv(X_train.T @ X_train + L) @ X_train.T @ z_train

                beta_hat = SGD.run(X_train, z_train, lmd=lmd)
                
                mse_train_buf[j] = mse(z_train, X_train @ beta_hat)
                mse_test_buf[j] = mse(z_test, X_test @ beta_hat)

            mse_train[i] = np.mean(mse_train_buf)
            mse_test[i] = np.mean(mse_test_buf)

        ridge_k_df[f"train_mse_{pn}"] = mse_train
        ridge_k_df[f"test_mse_{pn}"] = mse_test

    return ridge_k_df
