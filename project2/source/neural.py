'''Neural network running methods'''

from utils import *


def run_neural_regression_kfold(X, z, FFNN, k=5, etas=[], lambdas=[], **kwargs):
    '''Trains the FFNN on regression problem with k-fold resampling on X, t and looping over etas, lambdas. kwargs are forwarded to FFNN.train(). Returns 2D np.ndarrays (mse_train, mse_test) indexed on eta,lmd.'''

    splits = split(z, k)
    mse_train = np.ndarray((len(etas), len(lambdas)))
    mse_test = np.ndarray((len(etas), len(lambdas)))
    mse_train_buf = np.ndarray(k)
    mse_test_buf = np.ndarray(k)

    for i, eta in enumerate(etas):
        for j, lmd in enumerate(lambdas):
            for s, (train_split, test_split) in enumerate(zip(*splits)):
                X_train = X[train_split]
                z_train = z[train_split]
                X_test = X[test_split]
                z_test = z[test_split]

                X_train, X_test = scale(X_train, X_test)

                FFNN.train(X_train, z_train, eta0=eta, lmd=lmd, **kwargs)

                mse_train_buf[s] = r2(z_train, FFNN.predict(X_train))
                mse_test_buf[s] = r2(z_test, FFNN.predict(X_test))

                FFNN.reset()

            mse_train[i,j] = np.mean(mse_train_buf)
            mse_test[i,j] = np.mean(mse_test_buf)

    return (mse_train, mse_test)



def run_neural_classification_kfold(X, t, FFNN, t_onehot=None, k=5, etas=[], lambdas=[], **kwargs):
    '''Trains the FFNN on classification problem with k-fold resampling on X, t and looping over etas, lambdas. Use one-hot representation of t in t_onehot for classification problems. kwargs are forwarded to FFNN.train(). Returns 2D np.ndarrays (train_accuracy, test_accuracy) indexed on eta,lmd.'''

    splits = split(t, k)
    train_accuracy = np.ndarray((len(etas), len(lambdas)))
    test_accuracy = np.ndarray((len(etas), len(lambdas)))
    train_accuracy_buf = np.ndarray(k)
    test_accuracy_buf = np.ndarray(k)

    for i, eta in enumerate(etas):
        for j, lmd in enumerate(lambdas):
            for s, (train_split, test_split) in enumerate(zip(*splits)):
                X_train = X[train_split]
                t_train = t[train_split]
                train_targets = t_train if t_onehot is None else t_onehot[train_split]

                X_test = X[test_split]
                t_test = t[test_split]

                X_train, X_test = scale(X_train, X_test, with_std=False, scale_intercept=True)

                FFNN.train(X_train, train_targets, eta0=eta, lmd=lmd, **kwargs)
                pred = FFNN.predict(X_train)
                if not t_onehot is None:
                    pred = pred.argmax(axis=1)
                train_accuracy_buf[s] = skl.metrics.accuracy_score(t_train, pred)

                pred = FFNN.predict(X_test)
                if not t_onehot is None:
                    pred = pred.argmax(axis=1)
                test_accuracy_buf[s] = skl.metrics.accuracy_score(t_test, pred)

                FFNN.reset()

            train_accuracy[i,j] = np.mean(train_accuracy_buf)
            test_accuracy[i,j] = np.mean(test_accuracy_buf)

    return (train_accuracy, test_accuracy)
