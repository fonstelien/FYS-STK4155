from utils import *
from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning


@ignore_warnings(category=ConvergenceWarning)
def lasso_bootstrap(X, z, f=None, polynomial_orders=[], lambdas=[], train_size=.7, bootstraps=20, **kwargs):
    '''Runs LASSO regression with Bootstrap resampling on X, z, f on every polynomial up to polynomial_orders and for every lambda in lambdas. **kwargs are forwarded to skl.linear_model.Lasso. Returns DataFrame with columns=["lambda", "train_mse", "test_mse", "test_bias", "test_var"]'''

    if f is None:
        f = z

    lasso_bs_df = DataFrame()
    lasso_bs_df["lambda"] = lambdas

    num_lambdas = len(lambdas)
    mse_train_buf = np.ndarray(bootstraps)
    mse_train = np.ndarray(num_lambdas)
    mse_test = np.ndarray(num_lambdas)
    bias_test = np.ndarray(num_lambdas)
    var_test = np.ndarray(num_lambdas)

    X_train, X_test, z_train, z_test, _, f_test = skl.model_selection.train_test_split(X, z, f, train_size=train_size, random_state=0)
    X_train, X_test = scale(X_train, X_test)
    for pn in polynomial_orders:
        X_train_pn = truncate_to_poly(X_train, pn)
        X_test_pn = truncate_to_poly(X_test, pn)

        beta_hat = np.ndarray((X_train_pn.shape[1], 1))
        z_test_tilde = np.ndarray((z_test.shape[0], bootstraps))
        for i, lmd in enumerate(lambdas):
            lasso = skl.linear_model.Lasso(alpha=lmd, **kwargs)
            for bs in range(bootstraps):
                X_resampled, z_resampled = skl.utils.resample(X_train_pn, z_train, random_state=bs)
                lasso.fit(X_resampled, z_resampled)
                beta_hat[:,0] = lasso.coef_
                mse_train_buf[bs] = mse(z_resampled, X_resampled @ beta_hat)
                z_test_tilde[:,bs] = (X_test_pn @ beta_hat).ravel()

            mse_train[i] = np.mean(mse_train_buf)
            mse_test[i] = mse(z_test, z_test_tilde)
            bias_test[i] = bias(f_test, z_test_tilde)
            var_test[i] = var(z_test_tilde)

        lasso_bs_df[f"train_mse_{pn}"] = mse_train
        lasso_bs_df[f"test_mse_{pn}"] = mse_test
        lasso_bs_df[f"test_bias_{pn}"] = bias_test
        lasso_bs_df[f"test_var_{pn}"] = var_test

    return lasso_bs_df


@ignore_warnings(category=ConvergenceWarning)
def lasso_kfold(X, z, k=10, polynomial_orders=[], lambdas=[], **kwargs):
    '''Performs LASSO regression with k-fold split on X, z. **kwargs are forwarded to skl.linear_model.Lasso. Returns DataFrame with columns=["lambda", "train_mse", "test_mse"]'''

    lasso_k_df = DataFrame()
    lasso_k_df["lambda"] = lambdas

    splits = split(z, k)
    num_lambdas = len(lambdas)
    mse_train = np.ndarray(num_lambdas)
    mse_test = np.ndarray(num_lambdas)
    mse_train_buf = np.ndarray(k)
    mse_test_buf = np.ndarray(k)

    for pn in polynomial_orders:
        Xpn = truncate_to_poly(X, pn)
        for i, lmd in enumerate(lambdas):
            lasso = skl.linear_model.Lasso(alpha=lmd, **kwargs)
            for j, (train_split, test_split) in enumerate(zip(*splits)):
                X_train = Xpn[train_split]
                z_train = z[train_split]
                X_test = Xpn[test_split]
                z_test = z[test_split]

                X_train, X_test = scale(X_train, X_test)

                lasso.fit(X_train, z_train)
                beta_hat = np.ndarray((X_train.shape[1], 1))
                beta_hat[:,0] = lasso.coef_

                mse_train_buf[j] = mse(z_train, X_train @ beta_hat)
                mse_test_buf[j] = mse(z_test, X_test @ beta_hat)

            mse_train[i] = np.mean(mse_train_buf)
            mse_test[i] = np.mean(mse_test_buf)

        lasso_k_df[f"train_mse_{pn}"] = mse_train
        lasso_k_df[f"test_mse_{pn}"] = mse_test

    return lasso_k_df
