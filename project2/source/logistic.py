'''Logistic regression methods'''

from utils import *


def run_logistic_kfold(X, t, t_onehot, SGD, k=5, etas=[], lambdas=[]):
    '''Logistic regression with k-fold resampling on X, t and looping over etas, lambdas. t_onehot is one-hot representation of t. Returns 2D np.ndarrays (train_accuracy, test_accuracy) indexed on eta,lmd.'''

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
                t_hot_train = t_onehot[train_split]

                X_test = X[test_split]
                t_test = t[test_split]
                X_train, X_test = scale(X_train, X_test, with_std=True)

                beta_hat = SGD.run(X_train, t_hot_train, eta0=eta, lmd=lmd)

                pred = softmax(X_train @ beta_hat)
                pred = pred.argmax(axis=1)
                train_accuracy_buf[s] = skl.metrics.accuracy_score(t_train, pred)

                pred = softmax(X_test @ beta_hat)
                pred = pred.argmax(axis=1)
                test_accuracy_buf[s] = skl.metrics.accuracy_score(t_test, pred)

            train_accuracy[i] = np.mean(train_accuracy_buf)
            test_accuracy[i] = np.mean(test_accuracy_buf)

    return (train_accuracy, test_accuracy)
